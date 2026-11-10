#!/usr/bin/env python3
"""
End-to-end ATC scenario runner (LIVE: uses OpenAI, and optionally the running server).

Three actors:
  * ATC          - the system under test. Either the live HTTP server (--server URL) or an
                   in-process ATCLLMService (default), so you can iterate on prompts quickly.
  * Pilot agent  - an LLM playing a pilot with a goal (from the scenario YAML). It reads
                   back clearances, answers questions, and optionally misbehaves (wrong
                   readback, nonexistent runway...) when the scenario says so.
  * Aircraft sim - deterministic: advances position/altitude/on_ground according to the
                   clearances the ATC actually issued (parsed by clearance_parser), so
                   the trigger engine gets realistic state and handoffs can fire.

After the flight a judge LLM scores the ATC transcript against the scenario's rubric
(phraseology, correctness of runway/frequency facts, position discipline, safety,
readback handling) and hard checks are evaluated deterministically (e.g. "a takeoff
clearance was issued", "handoff to approach happened", "no nonexistent runway used").

Usage:
    python e2e/run_e2e.py                               # all scenarios, in-process ATC
    python e2e/run_e2e.py -s sbgr_ifr_departure_ptbr    # one scenario
    python e2e/run_e2e.py --server http://127.0.0.1:5000  # against the running server
    python e2e/run_e2e.py --model gpt-5.4-mini --judge gpt-5.6-luna

Reports: e2e/reports/<timestamp>_<scenario>.md (+ .json)
"""

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from openai import OpenAI  # noqa: E402

from airport_data_service import AirportDataService, haversine_nm, bearing_deg  # noqa: E402
from frequency_service import FrequencyService  # noqa: E402
from session_manager import ATCSession, ATCPhase, ClearanceType, FlightPlan, phase_from_text  # noqa: E402
from atc_triggers import ATCTriggerEngine  # noqa: E402
from clearance_parser import parse_clearances  # noqa: E402

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("e2e")

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
REPORTS_DIR = Path(__file__).parent / "reports"


# ---------------------------------------------------------------------------
# Aircraft simulator
# ---------------------------------------------------------------------------

@dataclass
class SimAircraft:
    lat: float
    lon: float
    alt_ft: float
    elev_ft: float
    heading: float = 90.0
    ias: float = 0.0
    gs: float = 0.0
    vs: float = 0.0
    on_ground: bool = True
    squawk: int = 1200
    mode: str = "parked"     # parked | taxiing | holding | rolling | climbing | cruising | descending | landed
    target_alt_ft: Optional[float] = None   # assigned altitude to level at (None = keep going)
    steer_to_field: bool = False            # cleared approach/landing: turn to the field and fly a glide path
    pattern: bool = False                   # circuit work: slow, 1000 ft AGL, stays around the field
    touch_and_go: bool = False              # next "landing" is a touch-and-go

    def state(self) -> dict:
        return {"latitude": self.lat, "longitude": self.lon, "altitude_ft": self.alt_ft, "heading": self.heading,
                "indicated_airspeed": self.ias, "groundspeed": self.gs, "vertical_speed": self.vs,
                "squawk": self.squawk, "on_ground": self.on_ground}

    def advance(self, seconds: float, airport_lat: float, airport_lon: float):
        """Very small physics: enough to move through phases in a believable way."""
        d = haversine_nm(self.lat, self.lon, airport_lat, airport_lon)
        to_field = bearing_deg(self.lat, self.lon, airport_lat, airport_lon)
        agl = self.alt_ft - self.elev_ft
        if self.mode == "taxiing":
            self.gs = 15
        elif self.mode == "rolling":
            self.gs = min(self.gs + 8 * seconds, 140)
            if self.gs >= 110:
                self.mode, self.on_ground, self.vs = "climbing", False, 1500
        elif self.mode == "climbing":
            self.ias, self.gs, self.vs = (90, 95, 700) if self.pattern else (150, 160, 1500)
            self.alt_ft += self.vs * seconds / 60
            self._steer(d, to_field)
            self._move(seconds)
            level_at = self.target_alt_ft if self.target_alt_ft else self.elev_ft + (1000 if self.pattern else 4000)
            if self.alt_ft >= level_at:
                self.alt_ft, self.mode, self.vs = level_at, "cruising", 0
        elif self.mode == "cruising":
            self.ias, self.gs, self.vs = (90, 95, 0) if self.pattern else (170, 180, 0)
            self._steer(d, to_field)
            if self.steer_to_field and agl > 200 and d <= agl / 318 + 0.3:
                self.mode, self.target_alt_ft = "descending", None     # intercept a ~3 degree path
            self._move(seconds)
        elif self.mode == "descending":
            self.ias, self.gs, self.vs = (80, 80, -500) if self.pattern else (140, 150, -800)
            self._steer(d, to_field)
            if self.steer_to_field and not self.target_alt_ft:
                # crude final approach: descend to arrive at the field
                minutes_left = max(d / self.gs * 60, 0.2)
                self.vs = -min(1500, max(300, agl / minutes_left))
            floor = self.target_alt_ft if self.target_alt_ft else self.elev_ft + 50
            self.alt_ft = max(floor, self.alt_ft + self.vs * seconds / 60)
            step_nm = self.gs * seconds / 3600
            self._move(seconds)
            if self.target_alt_ft and self.alt_ft <= self.target_alt_ft:
                self.mode, self.vs = "cruising", 0
            if not self.target_alt_ft and d < max(0.5, step_nm) and self.alt_ft - self.elev_ft < 300:
                self.steer_to_field = False
                if self.touch_and_go:
                    # touch, then straight back up into the circuit
                    self.touch_and_go, self.mode, self.alt_ft, self.vs = False, "climbing", self.elev_ft + 50, 700
                    self.target_alt_ft = self.elev_ft + 1000
                else:
                    self.mode, self.on_ground, self.gs, self.vs, self.alt_ft = "landed", True, 60, 0, self.elev_ft
        elif self.mode == "landed":
            self.gs = max(0, self.gs - 10 * seconds)
        else:
            self.gs = 0

    def _steer(self, d: float, to_field: float):
        if self.steer_to_field:
            self.heading = to_field
        elif self.pattern and d > 1.2:
            self.heading = (to_field + 75) % 360      # circle the field at ~1-2 nm: a crude circuit

    def _move(self, seconds: float):
        import math
        nm = self.gs * seconds / 3600
        self.lat += nm / 60 * math.cos(math.radians(self.heading))
        self.lon += nm / 60 * math.sin(math.radians(self.heading)) / max(math.cos(math.radians(self.lat)), 0.1)

    def apply_clearance(self, ctype: ClearanceType, value: Optional[int], runway: Optional[str], details: str = ""):
        if ctype == ClearanceType.TAXI and self.mode in ("parked",):
            self.mode = "taxiing"
        elif ctype in (ClearanceType.LINEUP,) and self.mode in ("taxiing", "holding"):
            self.mode = "holding"
        elif ctype == ClearanceType.TAKEOFF and self.mode in ("parked", "taxiing", "holding"):
            self.mode = "rolling"
            self.target_alt_ft = None
            if runway and runway[:2].isdigit():
                self.heading = int(runway[:2]) * 10
        elif ctype == ClearanceType.HEADING and value is not None and not self.on_ground:
            self.heading = value
        elif ctype == ClearanceType.ALTITUDE and value is not None and not self.on_ground:
            self.target_alt_ft = float(value)
            if value < self.alt_ft - 300:
                self.mode = "descending"
            elif value > self.alt_ft + 300:
                self.mode, self.vs = "climbing", 1500
        elif ctype == ClearanceType.APPROACH and not self.on_ground:
            self.steer_to_field = True     # fly to the field; keep any assigned altitude until the glide path
        elif ctype == ClearanceType.LANDING and not self.on_ground:
            self.target_alt_ft = None      # descend to the field
            self.mode = "descending"
            self.steer_to_field = True
            low = details.lower()
            self.touch_and_go = any(k in low for k in ("touch-and-go", "touch and go", "toque e arremetida",
                                                       "the option"))
        elif ctype == ClearanceType.GO_AROUND and not self.on_ground:
            self.mode, self.vs, self.steer_to_field = "climbing", 1500, False
            self.target_alt_ft = self.elev_ft + 2000
        elif ctype == ClearanceType.SQUAWK and value is not None:
            self.squawk = value


# ---------------------------------------------------------------------------
# ATC adapters (in-process or HTTP)
# ---------------------------------------------------------------------------

class InProcessATC:
    def __init__(self, scenario: dict, model: str, fs: FrequencyService, ads: AirportDataService):
        from atc_llm_service import ATCLLMService
        self.language = scenario["language"]
        self.session = ATCSession(scenario["airport"], scenario["callsign"], frequency_service=fs,
                                  airport_data_service=ads, language=self.language,
                                  flight_plan=FlightPlan.from_dict(scenario.get("flight_plan")))
        self.session.set_phase(phase_from_text(scenario.get("initial_position", "Ground")) or ATCPhase.GROUND)
        if scenario.get("metar"):
            pin_metar(self.session, scenario["metar"])
        self.llm = ATCLLMService(model=model, language=self.language, airport_data_service=ads)
        self.engine = ATCTriggerEngine()
        self.model = model

    def say(self, text: str, state: dict) -> dict:
        self.session.update_aircraft_state(**state)
        n = len(self.session.clearances)
        reply = self.llm.generate_response(text, self.session)
        return {"atc_text": reply, "latency_s": self.llm.last_latency_s, "tool_calls": list(self.llm.last_tool_calls),
                "clearances": [c.to_dict() for c in self.session.clearances[n:]],
                "phase": self.session.phase.value, "frequency": self.session.current_frequency}

    def update_state(self, state: dict) -> Optional[dict]:
        self.session.update_aircraft_state(**state)
        call = self.engine.evaluate(self.session)
        if call:
            self.session.add_atc_transmission(call.message)
            if call.new_phase:
                self.session.set_phase(call.new_phase)
            return call.to_dict()
        return None

    def initiative(self, state: dict) -> Optional[dict]:
        self.session.update_aircraft_state(**state)
        n = len(self.session.clearances)
        text = self.llm.controller_initiative(self.session, reason="pilot is silent")
        if not text:
            return None
        if self.session.pending_handoff:
            self.session.set_phase(self.session.pending_handoff)
            self.session.pending_handoff = None
        return {"trigger_type": "initiative", "message": text, "new_phase": self.session.phase.value,
                "latency_s": self.llm.last_latency_s, "tool_calls": list(self.llm.last_tool_calls),
                "clearances": [c.to_dict() for c in self.session.clearances[n:]]}

    def snapshot(self) -> dict:
        return self.session.to_dict()


def pin_metar(session: ATCSession, raw: str):
    """Scenario weather: a fixed METAR instead of the live one, so runway choice is deterministic."""
    import re
    from weather_service import active_runways
    m = re.search(r"\b(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\b", raw)
    wdir = int(m.group(1)) if m and m.group(1) != "VRB" else None
    wspd = int(m.group(2)) if m else 0
    alt = re.search(r"\b([AQ]\d{4})\b", raw)
    rwys = active_runways(wdir, wspd, session.airport.runways if session.airport else None)
    session.remember_fact(f"weather {session.airport_icao}",
                          f"METAR {raw} | wind {wdir if wdir is not None else 'VRB'}@{wspd}kt | "
                          f"altimeter {alt.group(1) if alt else '?'} | runways in use (into wind) "
                          f"{rwys or 'any (calm/variable)'}")


class HTTPATC:
    def __init__(self, scenario: dict, server: str):
        self.server = server.rstrip("/")
        self.language = scenario["language"]
        r = requests.post(f"{self.server}/tune", json={
            "airport_icao": scenario["airport"], "callsign": scenario["callsign"],
            "frequency": scenario.get("initial_position", "Ground"), "language": self.language,
            "flight_plan": scenario.get("flight_plan") or {},
        }, timeout=60)
        r.raise_for_status()
        self.model = requests.get(f"{self.server}/health", timeout=10).json().get("model", "?")

    def say(self, text: str, state: dict) -> dict:
        r = requests.post(f"{self.server}/transmit_text", json={"text": text, "state": state, "audio": False},
                          timeout=180)
        r.raise_for_status()
        return r.json()

    def update_state(self, state: dict, initiative: bool = False) -> Optional[dict]:
        r = requests.post(f"{self.server}/state", params={"initiative": "true" if initiative else "false"},
                          json=state, timeout=180).json()
        if r.get("pending_atc"):
            ack = requests.get(f"{self.server}/pending_atc", params={"audio": "false", "ack": "true"}, timeout=10).json()
            call = ack.get("call") or {}
            call["clearances"] = []
            return call
        return None

    def initiative(self, state: dict) -> Optional[dict]:
        return self.update_state(state, initiative=True)

    def snapshot(self) -> dict:
        return requests.get(f"{self.server}/session", timeout=10).json()


# ---------------------------------------------------------------------------
# Pilot agent
# ---------------------------------------------------------------------------

PILOT_PROMPT = """You are a pilot talking to ATC on the radio in a flight simulator test.
Language of the radio: {language_name}. Callsign: {callsign}. Aircraft: {aircraft}.
Airport: {airport}. Flight plan: {flight_plan}.

Your goal for this session: {goal}
{misbehavior}

Rules:
- Reply with ONE radio transmission only, in {language_name}, standard phraseology, ending with your callsign.
- Read back clearances (runway, altitude, heading, squawk, frequency) when ATC issues them.
- If ATC asks a question (type, intentions, position), answer it.
- If ATC hands you to another frequency, acknowledge with the frequency, then make an initial call
  to the new position on your NEXT transmission (e.g. "Approach, {callsign}, passing two thousand climbing five thousand").
- If you are AIRBORNE and ATC only acknowledged ("roger", "readback correct") and you have nothing NEW to
  say, reply exactly: STANDBY  (you stay silent and keep flying; ATC will call you). Never send "roger" back to a roger.
- On the ground never go silent: keep the flight moving (report ready to taxi, report at the holding point
  ready for departure, call the next position after a handoff).
- When your goal is complete and there is nothing left to say, reply exactly: DONE
- Your current aircraft state: {state}
- The position you are talking to now: {position}
"""

JUDGE_PROMPT = """You are an expert air traffic control instructor evaluating a virtual ATC system.
The radio language is {language_name}. Airport: {airport}. Valid runways: {runways}.
Published frequencies: {frequencies}. Flight plan: {flight_plan}.
Scenario goal (pilot side): {goal}
Scenario notes for the judge: {rubric_notes}

Below is the full transcript. Lines marked [TRIGGER] are ATC-initiated calls generated by rules
(handoffs); the rest of the ATC lines come from the LLM controller.

TRANSCRIPT:
{transcript}

Score the ATC (not the pilot) from 1 to 5 on each dimension and give a one-line justification each:
1. phraseology       - standard ICAO/DECEA wording, numbers, callsign usage, brevity
2. correctness       - only real runways/frequencies/procedures; no invented facts
3. position_discipline - only issued clearances proper to its position (ground/tower/approach...)
4. safety_logic      - sensible sequencing, denials when appropriate, no takeoff without clearance, etc.
5. readback_handling - caught wrong readbacks, answered questions, corrected the pilot when needed
6. flow              - the flight actually progressed; no loops, no stalls, no ignoring the pilot

Return JSON only:
{{"scores": {{"phraseology": n, "correctness": n, "position_discipline": n, "safety_logic": n,
  "readback_handling": n, "flow": n}}, "justifications": {{...same keys...}},
  "worst_atc_line": "...", "best_atc_line": "...", "overall": n, "summary": "2-3 sentences"}}
"""


class PilotAgent:
    def __init__(self, client: OpenAI, model: str, scenario: dict):
        self.client = client
        self.model = model
        self.scenario = scenario
        self.history: list[dict] = []

    def next_transmission(self, atc_text: Optional[str], state: dict, position: str) -> str:
        sc = self.scenario
        misbehavior = ""
        if sc.get("pilot_misbehavior"):
            misbehavior = "Deliberate test behaviour you MUST do once, early: " + sc["pilot_misbehavior"]
        system = PILOT_PROMPT.format(
            language_name="Brazilian Portuguese" if sc.get("pilot_language", sc["language"]) == "pt-br" else "English",
            callsign=sc["callsign"], aircraft=(sc.get("flight_plan") or {}).get("aircraft_type", "unknown"),
            airport=sc["airport"], flight_plan=json.dumps(sc.get("flight_plan") or {}),
            goal=sc["pilot_goal"], misbehavior=misbehavior,
            state=json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in state.items()}),
            position=position,
        )
        if atc_text:
            self.history.append({"role": "user", "content": f"ATC: {atc_text}"})
        elif not self.history:
            self.history.append({"role": "user", "content": "(Frequency is quiet. Make your initial call.)"})
        r = self.client.responses.create(model=self.model, instructions=system, input=self.history,
                                         reasoning={"effort": "low"})
        text = (r.output_text or "").strip().strip('"')
        self.history.append({"role": "assistant", "content": text})
        return text

    def record(self, atc_text: Optional[str], text: str):
        """A scripted line was sent: keep the agent's memory consistent in case it takes over."""
        if atc_text:
            self.history.append({"role": "user", "content": f"ATC: {atc_text}"})
        self.history.append({"role": "assistant", "content": text})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    n: int
    pilot: str
    atc: str
    latency_s: float
    tool_calls: list
    clearances: list
    phase: str
    frequency: str
    trigger: Optional[dict] = None
    sim_mode: str = ""


@dataclass
class Result:
    scenario: str
    atc_model: str
    pilot_model: str
    judge_model: str
    language: str
    airport: str
    turns: list = field(default_factory=list)
    hard_checks: dict = field(default_factory=dict)
    judge: dict = field(default_factory=dict)
    total_latency_s: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    category: str = ""
    known_gap: str = ""
    run: int = 0                 # repeat index (--repeat)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and all(v for k, v in self.hard_checks.items() if not k.startswith("_")) \
            and (self.judge.get("overall", 0) >= 3.5)

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        if self.known_gap:
            return "XPASS" if self.passed else "XFAIL"
        return "PASS" if self.passed else "FAIL"


SIM_FIELDS = {"lat", "lon", "alt_ft", "heading", "ias", "gs", "vs", "on_ground", "squawk", "mode",
              "target_alt_ft", "steer_to_field", "pattern", "touch_and_go"}


def _next_pilot_line(sc: dict, script: list, pilot: "PilotAgent", sim: SimAircraft, atc_text: Optional[str],
                     position: str) -> str:
    """Scripted line first (deterministic adversarial/emergency calls), then the LLM pilot agent."""
    if script:
        item = script.pop(0)
        if isinstance(item, dict):
            for k, v in (item.get("sim") or {}).items():
                if k in SIM_FIELDS:
                    setattr(sim, k, v)
            text = item.get("say", "STANDBY")
        else:
            text = str(item)
        pilot.record(atc_text, text)
        return text
    if sc.get("script_only"):
        return "DONE"
    return pilot.next_transmission(atc_text, sim.state(), position)


def run_scenario(sc: dict, args, client: OpenAI, fs: FrequencyService, ads: AirportDataService,
                 out=print) -> Result:
    airport = ads.get(sc["airport"])
    if not airport:
        raise SystemExit(f"unknown airport {sc['airport']}")
    start = sc.get("start", {})
    sim = SimAircraft(lat=start.get("lat", airport.latitude), lon=start.get("lon", airport.longitude),
                      alt_ft=start.get("alt_ft", airport.elevation_ft or 0), elev_ft=airport.elevation_ft or 0,
                      heading=start.get("heading", 90), on_ground=start.get("on_ground", True),
                      mode=start.get("mode", "parked"), squawk=start.get("squawk", 1200),
                      target_alt_ft=start.get("target_alt_ft"), steer_to_field=start.get("steer_to_field", False))
    sim.pattern = bool(start.get("pattern")) or FlightPlan.from_dict(sc.get("flight_plan")).is_local_pattern()
    if not sim.on_ground:
        sim.ias, sim.gs = (90, 95) if sim.pattern else (150, 160)

    atc = HTTPATC(sc, args.server) if args.server else InProcessATC(sc, args.model, fs, ads)
    pilot = PilotAgent(client, args.pilot, sc)
    res = Result(scenario=sc["name"], atc_model=atc.model, pilot_model=args.pilot, judge_model=args.judge,
                 language=sc["language"], airport=sc["airport"], started_at=datetime.now().isoformat(),
                 category=sc.get("category", ""), known_gap=sc.get("known_gap", ""))

    out(f"\n=== {sc['name']}  [{sc['airport']} {sc['language']}  ATC={atc.model}] ===")
    atc_text: Optional[str] = None
    position = sc.get("initial_position", "Ground")
    max_turns = sc.get("max_turns", 14)
    script = list(sc.get("pilot_script") or [])
    for n in range(1, max_turns + 1):
        pilot_text = _next_pilot_line(sc, script, pilot, sim, atc_text, position)
        if pilot_text.strip().upper() == "DONE":
            out("  pilot: DONE")
            break
        if pilot_text.strip().upper().startswith("STANDBY"):
            # pilot is silent: advance the flight, then let the controller take the initiative
            # (rule-based triggers get the first chance, exactly like the plugin's /state loop)
            call = None
            for _ in range(sc.get("sim_steps_per_turn", 6)):
                sim.advance(sc.get("sim_step_seconds", 20), airport.latitude, airport.longitude)
                call = atc.update_state(sim.state())
                if call:
                    break
            reply = call or atc.initiative(sim.state()) or {}
            atc_text = reply.get("message") or ""
            for c in reply.get("clearances", []):
                sim.apply_clearance(ClearanceType(c["type"]), c.get("value"), c.get("runway"), c.get("details", ""))
            turn = Turn(n, "(silence)", atc_text or "(silence)", reply.get("latency_s", 0), reply.get("tool_calls", []),
                        reply.get("clearances", []), reply.get("new_phase") or position, "", sim_mode=sim.mode,
                        trigger=call)
            res.total_latency_s += turn.latency_s
            res.turns.append(turn)
            out(f"  [{n:02d}] PILOT> (silence)  sim={sim.mode} alt={sim.alt_ft:.0f}")
            out(f"       ATC  > {atc_text or '(silence)'}   ({turn.latency_s:.1f}s, tools={turn.tool_calls}, "
                f"clr={[c['type'] for c in turn.clearances]}{', TRIGGER' if call else ''})")
            if reply.get("new_phase"):
                position = reply["new_phase"]
            if not atc_text:
                atc_text = None
            continue
        reply = atc.say(pilot_text, sim.state())
        atc_text = reply["atc_text"] or "(no reply)"     # intended silence: handoff readback, uncontrolled field
        position = reply.get("phase") or position
        for c in reply.get("clearances", []):
            sim.apply_clearance(ClearanceType(c["type"]), c.get("value"), c.get("runway"), c.get("details", ""))
        turn = Turn(n, pilot_text, atc_text, reply.get("latency_s", 0), reply.get("tool_calls", []),
                    reply.get("clearances", []), reply.get("phase", ""), reply.get("frequency", ""), sim_mode=sim.mode)
        res.total_latency_s += turn.latency_s
        out(f"  [{n:02d}] PILOT> {pilot_text}")
        out(f"       ATC  > {atc_text}   ({turn.latency_s:.1f}s, tools={turn.tool_calls}, "
            f"clr={[c['type'] for c in turn.clearances]}, sim={sim.mode})")

        # let time pass: several state updates so triggers can fire
        for _ in range(sc.get("sim_steps_per_turn", 6)):
            sim.advance(sc.get("sim_step_seconds", 20), airport.latitude, airport.longitude)
            call = atc.update_state(sim.state())
            if call:
                turn.trigger = call
                position = call.get("new_phase") or position
                out(f"       [TRIGGER] {call['message']}")
                atc_text = f"{atc_text}\n{call['message']}"
                break
        res.turns.append(turn)

    res.finished_at = datetime.now().isoformat()
    snap = atc.snapshot()
    res.hard_checks = hard_checks(sc, res, snap, airport)
    res.judge = judge(client, args.judge, sc, res, airport, snap)
    return res


# Words that only show up in a Portuguese transmission (used to check the reply language)
_PT_WORDS = {"pista", "autorizado", "autorizada", "ciente", "chame", "mantenha", "vento", "solo", "torre", "táxi",
             "ponto", "espera", "decolagem", "pouso", "suba", "desça", "desca", "proa", "nível", "nivel", "pés",
             "reporte", "prossiga", "afirmativo", "negativo", "aguarde", "controle", "tráfego", "trafego", "livre",
             "frequência", "frequencia", "após", "apos", "circuito", "perna", "aproximação", "aproximacao",
             "colacionar", "correção", "correcao", "bom", "dia", "via", "para", "em", "com", "de", "nós",
             "colação", "colacao", "correta", "identificado", "acionamento", "autorizados", "estabelecido",
             "confirme", "posição", "posicao", "intenção", "intencao", "taxe", "pátio", "patio", "repita"}
_PT_STRONG = _PT_WORDS - {"via", "para", "em", "com", "de", "dia", "bom", "solo", "controle"}


def reply_language(text: str) -> str:
    words = {w.lower() for w in _tokens_simple(text)}
    strong = len(words & _PT_STRONG)
    weak = len(words & _PT_WORDS)
    return "pt-br" if strong >= 1 and (weak >= 2 or len(words) <= 4) else "en"


def _tokens_simple(text: str) -> list[str]:
    import re
    return re.findall(r"[A-Za-zÀ-ÿ]+", text)


def _has_any(text: str, phrases) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in phrases)


def hard_checks(sc: dict, res: Result, snapshot: dict, airport) -> dict:
    """Deterministic assertions declared in the scenario YAML."""
    issued = {c["type"] for c in snapshot.get("clearances", [])}
    atc_lines = [t.atc for t in res.turns]
    llm_lines = [t.atc for t in res.turns if t.atc not in ("(silence)", "(no reply)")
                 and not (t.pilot == "(silence)" and t.trigger)]
    all_atc = "\n".join(atc_lines + [t.trigger["message"] for t in res.turns if t.trigger]).lower()
    checks = {}
    for want in sc.get("expect_clearances", []):
        checks[f"clearance:{want}"] = want in issued
    for bad in sc.get("forbid_clearances", []):
        checks[f"no_clearance:{bad}"] = bad not in issued
    for phase in sc.get("expect_phases", []):
        checks[f"phase:{phase}"] = any(t.phase == phase or (t.trigger or {}).get("new_phase") == phase for t in res.turns)
    if sc.get("expect_no_invalid_runways", True):
        # Only runways the controller actually CLEARED count (taxi/lineup/takeoff/landing/approach).
        # Quoting a wrong runway while denying it ("negative, runway 27R does not exist") is correct.
        valid = set(airport.runway_idents())
        mentioned = set()
        for line in atc_lines:
            for pc in parse_clearances(line, sc["callsign"]):
                if pc.runway:
                    mentioned.add(pc.runway)
        bad = {m for m in mentioned if m not in valid and m.lstrip("0") not in {v.lstrip("0") for v in valid}}
        checks["no_invalid_runways"] = not bad
        if bad:
            checks["_invalid_runways_seen"] = sorted(bad)
    for phrase in sc.get("expect_phrases", []):
        checks[f"phrase:{phrase}"] = phrase.lower() in all_atc
    for group in sc.get("expect_any", []):
        checks[f"any:{'|'.join(group)}"] = _has_any(all_atc, group)
    for phrase in sc.get("forbid_phrases", []):
        checks[f"forbid:{phrase}"] = phrase.lower() not in all_atc
    for tc in sc.get("turn_checks", []):
        n = tc["turn"]
        turn = next((t for t in res.turns if t.n == n), None)
        text = turn.atc if turn else ""
        if tc.get("any"):
            checks[f"turn{n}:any:{'|'.join(tc['any'])}"] = bool(turn) and _has_any(text, tc["any"])
        for p in tc.get("none", []):
            checks[f"turn{n}:none:{p}"] = bool(turn) and p.lower() not in text.lower()
    used_tools = {tool for t in res.turns for tool in (t.tool_calls or [])}
    if sc.get("expect_tools"):
        checks[f"tools:{'|'.join(sc['expect_tools'])}"] = bool(used_tools & set(sc["expect_tools"]))
    want_lang = sc.get("expect_reply_language", sc["language"])
    wrong_lang = [line for line in llm_lines if line.strip() and reply_language(line) != want_lang]
    checks[f"reply_language:{want_lang}"] = not wrong_lang
    if wrong_lang:
        checks["_wrong_language_lines"] = wrong_lang[:3]
    checks["no_empty_replies"] = all(t.atc.strip() for t in res.turns if t.pilot != "(silence)")
    silent = [t for t in res.turns if t.pilot == "(silence)" and t.sim_mode not in ("parked", "taxiing", "holding")]
    if silent and sc.get("expect_initiative", False):   # silence is often right (waiting for a report)
        checks["controller_initiative_used"] = any(t.atc != "(silence)" for t in silent)
    checks["min_turns"] = len(res.turns) >= sc.get("min_turns", 3)
    return checks


def judge(client: OpenAI, model: str, sc: dict, res: Result, airport, snapshot: dict) -> dict:
    lines = []
    for t in res.turns:
        lines.append(f"PILOT: {t.pilot}")
        lines.append(f"ATC ({t.frequency}): {t.atc.splitlines()[0]}")
        if t.trigger:
            lines.append(f"[TRIGGER] ATC: {t.trigger['message']}")
    prompt = JUDGE_PROMPT.format(
        language_name="Brazilian Portuguese" if sc["language"] == "pt-br" else "English",
        airport=f"{airport.name} ({airport.icao})", runways=", ".join(airport.runway_idents()),
        frequencies=json.dumps({k: v for k, v in (snapshot.get("airport") or {}).items() if k == "icao"}) if False else
        ", ".join(f"{t} {f['frequency_formatted']}" for t, f in _freqs(snapshot).items()),
        flight_plan=json.dumps(sc.get("flight_plan") or {}), goal=sc["pilot_goal"],
        rubric_notes=sc.get("rubric_notes", "none"), transcript="\n".join(lines))
    err = None
    for _ in range(2):                  # JSON mode + one retry: a judge parse error is not an ATC failure
        try:
            r = client.responses.create(model=model, input=prompt, reasoning={"effort": "medium"},
                                        text={"format": {"type": "json_object"}})
            txt = r.output_text.strip()
            txt = txt[txt.find("{"): txt.rfind("}") + 1]
            return json.loads(txt)
        except Exception as e:  # noqa: BLE001
            err = e
    return {"error": f"judge failed: {err}", "overall": 0}


def _freqs(snapshot: dict) -> dict:
    # session.to_dict doesn't expose frequency objects; rebuild from FrequencyService cheaply
    fs = FrequencyService()
    return {t: {"frequency_formatted": f.frequency_formatted()} for t, f in fs.get_all_atc(snapshot["airport_icao"]).items()}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(res: Result, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / (f"{stamp}_{res.scenario}" + (f"_r{res.run}" if res.run else ""))
    (base.with_suffix(".json")).write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False))

    j = res.judge or {}
    scores = j.get("scores", {})
    md = [f"# E2E report: {res.scenario}", "",
          f"- Airport: {res.airport}  Language: {res.language}",
          f"- ATC model: `{res.atc_model}`  Pilot: `{res.pilot_model}`  Judge: `{res.judge_model}`",
          f"- Turns: {len(res.turns)}  Total ATC latency: {res.total_latency_s:.1f}s  "
          f"(avg {res.total_latency_s / max(len(res.turns), 1):.1f}s/turn)",
          f"- Category: {res.category or '-'}" + (f"  Known gap: {res.known_gap}" if res.known_gap else ""),
          f"- **Result: {res.status}**  (judge overall {j.get('overall', '?')}/5)" + (f"  error: {res.error}" if res.error else ""), "",
          "## Hard checks", ""]
    for k, v in res.hard_checks.items():
        if k.startswith("_"):
            md.append(f"- {k[1:]}: {v}")
        else:
            md.append(f"- {'✅' if v else '❌'} {k}")
    md += ["", "## Judge", ""]
    for k, v in scores.items():
        md.append(f"- {k}: **{v}/5** — {j.get('justifications', {}).get(k, '')}")
    md += ["", f"Summary: {j.get('summary', j.get('error', ''))}", "",
           f"Best line: _{j.get('best_atc_line', '')}_", "", f"Worst line: _{j.get('worst_atc_line', '')}_", "",
           "## Transcript", ""]
    for t in res.turns:
        md.append(f"**[{t.n}] PILOT:** {t.pilot}  ")
        md.append(f"**ATC ({t.frequency}):** {t.atc.splitlines()[0]}  ")
        md.append(f"<sub>{t.latency_s:.1f}s · tools {t.tool_calls or '-'} · clearances "
                  f"{[c['type'] + (' ' + c['runway'] if c.get('runway') else '') + (' ' + str(c['value']) if c.get('value') else '') for c in t.clearances] or '-'} · sim {t.sim_mode}</sub>  ")
        if t.trigger:
            md.append(f"**[TRIGGER] ATC:** {t.trigger['message']}  ")
        md.append("")
    base.with_suffix(".md").write_text("\n".join(md))
    return base.with_suffix(".md")


def aggregate(results: list) -> list[dict]:
    """
    One row per scenario across --repeat runs. Majority decides: PASS when more than half the runs
    pass, FLAKY when some (not most) pass, FAIL when none do. Known gaps: XPASS / XFAIL.
    """
    groups: dict[str, list] = {}
    for res, path, _ in results:
        groups.setdefault(res.scenario, []).append((res, path))
    rows = []
    for name, runs in groups.items():
        n = len(runs)
        passes = sum(1 for r, _ in runs if r.passed)
        judges = [r.judge.get("overall") for r, _ in runs if isinstance(r.judge.get("overall"), (int, float))
                  and not r.judge.get("error")]
        first = runs[0][0]
        if any(r.error for r, _ in runs) and passes == 0:
            status = "ERROR"
        elif first.known_gap:
            status = "XPASS" if passes * 2 > n else "XFAIL"
        else:
            status = "PASS" if passes * 2 > n else ("FLAKY" if passes else "FAIL")
        failed: dict[str, int] = {}
        for r, _ in runs:
            for k, v in r.hard_checks.items():
                if not k.startswith("_") and v is False:
                    failed[k] = failed.get(k, 0) + 1
        lat = [r.total_latency_s / max(len(r.turns), 1) for r, _ in runs]
        rows.append({"scenario": name, "category": first.category, "status": status, "passes": passes, "n": n,
                     "judge_mean": sum(judges) / len(judges) if judges else None,
                     "judge_min": min(judges) if judges else None, "judge_max": max(judges) if judges else None,
                     "latency": sum(lat) / len(lat), "failed": failed, "known_gap": first.known_gap,
                     "reports": [p.name for _, p in runs if p],
                     "guard": sum(t.tool_calls.count("position_guard") for r, _ in runs for t in r.turns)})
    return sorted(rows, key=lambda r: (r["category"], r["scenario"]))


def write_summary(results: list, out_dir: Path, meta: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stamp}_SUMMARY.md"
    rows = aggregate(results)
    counts: dict = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    judged = [r["judge_mean"] for r in rows if r["judge_mean"] is not None]
    md = ["# E2E summary", "", meta, "",
          "  ".join(f"**{k}** {v}" for k, v in sorted(counts.items()))
          + (f"  ·  mean judge {sum(judged) / len(judged):.2f}/5" if judged else ""), "",
          "| Status | Category | Scenario | Runs passed | Judge mean (min–max) | Avg latency | Guard fixes | "
          "Failed checks (×runs) / known gap |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        j = "?" if r["judge_mean"] is None else f"{r['judge_mean']:.1f} ({r['judge_min']}–{r['judge_max']})"
        note = ", ".join(f"{k} ×{v}" for k, v in r["failed"].items()) + (f" — gap: {r['known_gap']}" if r["known_gap"] else "")
        link = f"[{r['scenario']}]({r['reports'][0]})" if r["reports"] else r["scenario"]
        md.append(f"| {r['status']} | {r['category']} | {link} | {r['passes']}/{r['n']} | {j} | "
                  f"{r['latency']:.1f}s | {r['guard'] or ''} | {note} |")
    path.write_text("\n".join(md) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--scenario", action="append", help="scenario name(s) (file stem). Default: all")
    ap.add_argument("-c", "--category", action="append", help="only scenarios in these categories")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="scenarios in parallel (in-process mode only)")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--server", help="run against a live server URL instead of in-process")
    ap.add_argument("--model", default=None, help="ATC model for in-process mode (default: config.OPENAI_MODEL)")
    ap.add_argument("--pilot", default="gpt-5.6-luna",
                    help="pilot agent model (gpt-5.4-mini is cheaper but loops/goes silent more often)")
    ap.add_argument("-r", "--repeat", type=int, default=1,
                    help="run every scenario N times and score by majority (LLM runs vary a lot)")
    ap.add_argument("--judge", default="gpt-5.6-luna", help="judge model")
    ap.add_argument("--out", default=str(REPORTS_DIR))
    args = ap.parse_args()

    import config
    args.model = args.model or config.OPENAI_MODEL

    scenarios = []
    for f in sorted(SCENARIOS_DIR.glob("*.yaml")):
        sc = yaml.safe_load(f.read_text())
        sc.setdefault("name", f.stem)
        scenarios.append(sc)
    if args.scenario:
        scenarios = [sc for sc in scenarios if sc["name"] in args.scenario]
    if args.category:
        scenarios = [sc for sc in scenarios if sc.get("category") in args.category]
    if not scenarios:
        raise SystemExit(f"no scenario matched; available: {[f.stem for f in SCENARIOS_DIR.glob('*.yaml')]}")
    if args.list:
        for sc in scenarios:
            print(f"{sc.get('category', '-'):12s} {sc['name']:40s} {sc['airport']} {sc['language']}"
                  + (f"  [known gap: {' '.join(sc['known_gap'].split())[:80]}...]" if sc.get("known_gap") else ""))
        return
    if args.server and args.jobs > 1:
        raise SystemExit("--server holds one session at a time; run it with -j 1")

    client = OpenAI()
    fs = FrequencyService()
    ads = AirportDataService()
    print_lock = threading.Lock()

    def run_one(job):
        sc, run = job
        buf: list[str] = []
        out = buf.append if args.jobs > 1 else print
        t0 = time.time()
        try:
            res = run_scenario(sc, args, client, fs, ads, out=out)
        except Exception as e:  # noqa: BLE001  one broken scenario must not kill the whole suite
            log.exception(f"scenario {sc['name']} crashed")
            res = Result(scenario=sc["name"], atc_model=args.model, pilot_model=args.pilot, judge_model=args.judge,
                         language=sc["language"], airport=sc["airport"], category=sc.get("category", ""),
                         known_gap=sc.get("known_gap", ""), error=f"{type(e).__name__}: {e}")
        res.run = run
        path = write_report(res, Path(args.out))
        with print_lock:
            for line in buf:
                print(line)
            print(f"  -> {res.status}{f' (run {run})' if run else ''}  judge={res.judge.get('overall', '?')}/5  "
                  f"checks={sum(1 for k, v in res.hard_checks.items() if v is True and not k.startswith('_'))}/"
                  f"{sum(1 for k in res.hard_checks if not k.startswith('_'))}  report={path}")
        return res, path, time.time() - t0

    jobs = [(sc, (r + 1) if args.repeat > 1 else 0) for r in range(args.repeat) for sc in scenarios]
    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            summary = list(pool.map(run_one, jobs))
    else:
        summary = [run_one(job) for job in jobs]

    rows = aggregate(summary)
    print(f"\n==== SUMMARY  (ATC {args.model}, pilot {args.pilot}, judge {args.judge}, x{args.repeat}) ====")
    for r in rows:
        j = "?" if r["judge_mean"] is None else f"{r['judge_mean']:.1f} ({r['judge_min']}-{r['judge_max']})"
        print(f"{r['status']:5s}  {r['category']:11s} {r['scenario']:38s} pass={r['passes']}/{r['n']} judge={j} "
              f"lat={r['latency']:.1f}s" + (f" guard={r['guard']}" if r["guard"] else "")
              + (f"  failed={r['failed']}" if r["failed"] else ""))
    meta = f"ATC `{args.model}` · pilot `{args.pilot}` · judge `{args.judge}` · {args.repeat} run(s) per scenario"
    print(f"summary: {write_summary(summary, Path(args.out), meta)}")
    sys.exit(0 if all(r["status"] in ("PASS", "XFAIL", "XPASS") for r in rows) else 1)


if __name__ == "__main__":
    main()
