#!/usr/bin/env python3
"""
Fake X-Plane - drives the REAL server exactly like the X-Plane plugin does, without X-Plane.

It reproduces the plugin's traffic pattern:
  * POST /tune with airport_icao="auto" + position (auto-detect nearest airport)
  * POST /state every STATE_INTERVAL seconds from a scripted flight profile
  * POST /transmit_text (or /transmit with synthesized audio) for pilot calls
  * GET  /pending_atc?ack=true&format=wav when /state reports a pending ATC call, and
    writes the WAV to disk so you can listen to what the plugin would play on COM1.

The flight profile is a list of (seconds, pilot_call | None, aircraft-state-change) steps,
so you can see handoffs fire at the right moments (takeoff -> departure/approach, etc).

Usage:
    python atc_server.py                     # in another terminal
    python e2e/fake_xplane.py                # default profile: SBGR departure, pt-br
    python e2e/fake_xplane.py --profile kjfk --audio   # send real TTS-generated pilot audio through /transmit
    python e2e/fake_xplane.py --server http://host:5000 --interval 1

Exit code 0 when every expected handoff fired.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).parent / "reports" / "fake_xplane"

PROFILES = {
    # (label, {state deltas}, pilot_call or None, expect_trigger_phase or None)
    "sbgr": {
        "language": "pt-br", "callsign": "PT-FXP", "aircraft_type": "C172",
        "flight_plan": {"aircraft_type": "C172", "flight_rules": "IFR", "destination": "SBKP", "route": "DCT",
                        "cruise_altitude_ft": 6000},
        "position": "Solo",
        "start": {"latitude": -23.4356, "longitude": -46.4731, "altitude_ft": 2461, "heading": 90,
                  "indicated_airspeed": 0, "groundspeed": 0, "vertical_speed": 0, "squawk": 2000, "on_ground": True,
                  "com1_freq_hz": 121_700_000},
        "steps": [
            ("parked", {}, "Guarulhos Solo, PT-FXP, Cessna 172 no pátio três, autorização IFR a bordo, "
                           "solicito acionamento e táxi", None),
            ("taxi", {"groundspeed": 15}, "Ciente, PT-FXP", None),
            ("holding", {"groundspeed": 0, "com1_freq_hz": 132_750_000},
             "Guarulhos Torre, PT-FXP, ponto de espera, pronto para decolagem", None),
            ("roll", {"groundspeed": 90, "indicated_airspeed": 80},
             "Autorizado decolagem, PT-FXP", None),
            ("airborne 300", {"on_ground": False, "altitude_ft": 2761, "vertical_speed": 1200, "indicated_airspeed": 120,
                              "groundspeed": 130}, None, None),
            ("climb 1500", {"altitude_ft": 3961, "latitude": -23.40}, None, "approach"),   # -> São Paulo APP
            ("climb 3000", {"altitude_ft": 5461, "latitude": -23.35, "com1_freq_hz": 119_800_000},
             "Controle São Paulo, PT-FXP, passando cinco mil pés subindo seis mil", None),
        ],
    },
    "kjfk": {
        "language": "en", "callsign": "N123FX", "aircraft_type": "C172",
        "flight_plan": {"aircraft_type": "C172", "flight_rules": "VFR", "destination": "KHPN"},
        "position": "Ground",
        "start": {"latitude": 40.6425, "longitude": -73.7790, "altitude_ft": 13, "heading": 310,
                  "indicated_airspeed": 0, "groundspeed": 0, "vertical_speed": 0, "squawk": 1200, "on_ground": True,
                  "com1_freq_hz": 121_650_000},
        "steps": [
            ("parked", {}, "Kennedy Ground, November one two three foxtrot x-ray, Cessna 172 at the GA ramp with "
                           "information Alpha, VFR to White Plains, request taxi", None),
            ("taxi", {"groundspeed": 15}, "Roger, November one two three foxtrot x-ray", None),
            ("holding", {"groundspeed": 0, "com1_freq_hz": 119_100_000},
             "Kennedy Tower, November one two three foxtrot x-ray, holding short, ready for departure", None),
            ("roll", {"groundspeed": 70, "indicated_airspeed": 65}, "Cleared for takeoff, one two three foxtrot x-ray", None),
            ("airborne", {"on_ground": False, "altitude_ft": 500, "vertical_speed": 800, "indicated_airspeed": 90,
                          "groundspeed": 95}, None, None),
            ("climb 1200", {"altitude_ft": 1213, "latitude": 40.66}, None, "departure"),
            ("checkin", {"altitude_ft": 2000, "latitude": 40.70, "com1_freq_hz": 135_900_000},
             "New York Departure, November one two three foxtrot x-ray, two thousand climbing three thousand", None),
        ],
    },
    # Arrival: Approach -> (approach clearance) -> Tower trigger -> landing clearance -> touchdown -> Ground trigger.
    # COM1 starts on whatever Approach frequency /tune reports (the plugin's radio follows the handoffs).
    "kbos_arrival": {
        "language": "en", "callsign": "N543AR", "aircraft_type": "PC12", "airport": "KBOS",
        "flight_plan": {"aircraft_type": "PC12", "flight_rules": "IFR", "departure": "KJFK", "destination": "KBOS"},
        "position": "Approach",
        "start": {"latitude": 42.2400, "longitude": -71.1700, "altitude_ft": 4000, "heading": 40,
                  "indicated_airspeed": 160, "groundspeed": 165, "vertical_speed": 0, "squawk": 4521,
                  "on_ground": False},
        "steps": [
            ("10 nm SW", {}, "Boston Approach, November five four three alpha romeo, four thousand, "
                             "established on the localizer runway four right, request the ILS", None),
            ("6 nm final", {"latitude": 42.2900, "longitude": -71.1050, "altitude_ft": 2000, "vertical_speed": -700,
                            "indicated_airspeed": 130, "groundspeed": 130}, None, "tower"),
            ("4 nm final", {"latitude": 42.3150, "longitude": -71.0700, "altitude_ft": 1300},
             "Boston Tower, November five four three alpha romeo, ILS runway four right, four mile final", None),
            ("touchdown", {"latitude": 42.3560, "longitude": -71.0120, "altitude_ft": 20, "on_ground": True,
                           "vertical_speed": 0, "indicated_airspeed": 90, "groundspeed": 90}, None, None,
             30),   # real time between the Tower handoff and the rollout (30 s handoff cooldown)
            ("rollout", {"latitude": 42.3650, "longitude": -71.0050, "indicated_airspeed": 30, "groundspeed": 30},
             None, "ground"),
        ],
    },
    # Cross-country: tuned at Guarulhos on load, then flies to Campinas and dials Campinas Tower on COM1.
    # A real X-Plane flight does exactly this; the session must follow the radio to the new airport.
    "xcountry": {
        "language": "pt-br", "callsign": "PT-FXQ", "aircraft_type": "C172",
        "flight_plan": {"aircraft_type": "C172", "flight_rules": "VFR", "destination": "SBKP"},
        "position": "Solo",
        "start": {"latitude": -23.4356, "longitude": -46.4731, "altitude_ft": 2461, "heading": 300,
                  "indicated_airspeed": 0, "groundspeed": 0, "vertical_speed": 0, "squawk": 1200, "on_ground": True,
                  "com1_freq_hz": 121_700_000},
        "expect_final_airport": "SBKP",
        "steps": [
            ("near Campinas", {"latitude": -23.0500, "longitude": -47.0200, "altitude_ft": 4500, "on_ground": False,
                               "indicated_airspeed": 110, "groundspeed": 115, "heading": 250,
                               "com1_freq_hz": 118_250_000},
             "Torre Campinas, PT-FXQ, Cessna 172, oito milhas a nordeste, quatro mil e quinhentos pés, "
             "para pouso", None),
        ],
    },
}


class FakeXPlane:
    def __init__(self, server: str, profile: dict, interval: float, audio: bool):
        self.server = server.rstrip("/")
        self.p = profile
        self.interval = interval
        self.audio = audio
        self.state = dict(profile["start"])
        self.fired: list[dict] = []
        OUT.mkdir(parents=True, exist_ok=True)

    def _post(self, path, **payload):
        r = requests.post(f"{self.server}{path}", json=payload, timeout=180)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"raw": r.content[:80]}

    def tune(self):
        code, r = self._post("/tune", airport_icao=self.p.get("airport", "auto"), latitude=self.state["latitude"],
                             longitude=self.state["longitude"], callsign=self.p["callsign"],
                             frequency=self.p["position"], language=self.p["language"],
                             flight_plan=self.p["flight_plan"], state=self.state)
        assert code == 200, r
        s = r["session"]
        if not self.state.get("com1_freq_hz") and s.get("frequency_mhz"):
            self.state["com1_freq_hz"] = int(round(s["frequency_mhz"] * 1_000_000))  # radio on the tuned position
        print(f"[tune] auto -> {s['airport_icao']} {s['airport_name']} | {s['frequency']} {s['frequency_mhz']} "
              f"| runways {s['runways']}")

    def transmit(self, text: str):
        if self.audio:
            # synthesize pilot audio with the server's own TTS provider (OpenAI) and push through /transmit
            from audio_service import AudioService
            wav = AudioService(language=self.p["language"], tts_provider="openai").speak_atc_wav(text)
            r = requests.post(f"{self.server}/transmit", json={"audio": base64.b64encode(wav).decode(),
                                                                 "state": self.state, "format": "wav"}, timeout=180)
            pilot = r.headers.get("X-Pilot-Text", "")
            atc = r.headers.get("X-ATC-Text", "")
            print(f"  PILOT (audio→{r.headers.get('X-STT-Provider')}): {pilot}")
            print(f"  ATC   ({r.headers.get('X-Audio-Provider')}): {atc}")
            if r.headers.get("Content-Type", "").startswith("audio/"):
                (OUT / f"atc_{int(time.time())}.wav").write_bytes(r.content)
        else:
            code, r = self._post("/transmit_text", text=text, state=self.state, audio=True, format="wav")
            print(f"  PILOT: {text}")
            print(f"  ATC  : {r.get('atc_text')}   ({r.get('latency_s')}s, tools={r.get('tool_calls')}, "
                  f"clr={[c['type'] for c in r.get('clearances', [])]}, audio={r.get('audio_provider')})")
            if r.get("audio_base64"):
                (OUT / f"atc_{int(time.time())}.wav").write_bytes(base64.b64decode(r["audio_base64"]))

    def state_tick(self):
        code, r = self._post("/state", **self.state)
        if r.get("pending_atc"):
            ack = requests.get(f"{self.server}/pending_atc", params={"ack": "true", "format": "wav"}, timeout=120).json()
            call = ack.get("call") or {}
            print(f"  [ATC CALL {call.get('trigger_type')}] {call.get('message')}  -> now {ack.get('frequency')} "
                  f"{ack.get('frequency_mhz')}")
            if ack.get("audio_base64"):
                (OUT / f"atc_call_{int(time.time())}.wav").write_bytes(base64.b64decode(ack["audio_base64"]))
            self.fired.append(call)
            if ack.get("frequency_mhz"):
                self.state["com1_freq_hz"] = int(round(ack["frequency_mhz"] * 1_000_000))
        return r

    def run(self) -> bool:
        self.tune()
        expected = [step[3] for step in self.p["steps"] if step[3]]
        for step in self.p["steps"]:
            label, delta, call, expect = step[:4]
            wait_s = step[4] if len(step) > 4 else 0
            self.state.update(delta)
            print(f"\n== {label} == alt={self.state['altitude_ft']:.0f} gs={self.state['groundspeed']} "
                  f"on_ground={self.state['on_ground']}")
            if call:
                self.transmit(call)
            for _ in range(3):
                self.state_tick()
                time.sleep(self.interval)
            t_end = time.time() + wait_s
            while time.time() < t_end:          # keep the plugin's /state cadence while time passes
                self.state_tick()
                time.sleep(2.0)
            if expect:
                got = any(c.get("new_phase") == expect for c in self.fired)
                print(f"  expect handoff to {expect}: {'OK' if got else 'MISSING'}")
        phases = [c.get("new_phase") for c in self.fired]
        ok = all(e in phases for e in expected)
        snap = requests.get(f"{self.server}/session", timeout=10).json()
        if self.p.get("expect_final_airport"):
            got = snap.get("airport_icao")
            print(f"  expect session airport {self.p['expect_final_airport']}: got {got} "
                  f"({snap.get('frequency')}, phase {snap.get('phase')})")
            ok = ok and got == self.p["expect_final_airport"]
        (OUT / "last_session.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False))
        print(f"\nfired handoffs: {phases}  expected: {expected}  -> {'PASS' if ok else 'FAIL'}")
        print(f"audio + session dump in {OUT}")
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:5000")
    ap.add_argument("--profile", default="sbgr", choices=sorted(PROFILES))
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between /state posts (plugin uses 2.0)")
    ap.add_argument("--audio", action="store_true", help="send synthesized pilot audio via /transmit (STT path)")
    args = ap.parse_args()
    try:
        requests.get(f"{args.server}/health", timeout=3).raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"server not reachable at {args.server}: {e}  (start it with: python atc_server.py)")
    ok = FakeXPlane(args.server, PROFILES[args.profile], args.interval, args.audio).run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
