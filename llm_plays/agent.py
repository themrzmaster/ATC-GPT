"""
The pilot: a model with three tools (look / act / report) and one goal, pi_plays_pokemon style.

Turn-based: X-Plane is paused while the model thinks; act(seconds) flies that long and pauses again,
so any model - fast or slow - flies the same aircraft under the same rules. Every turn is logged to
campaigns/<name>/events.jsonl (+ screenshots) and the sim situation is saved after every act, so a
campaign can be resumed. Scoring: missions.MilestoneTracker.
"""

import base64
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .geometry import FT_PER_NM, Runway
from .missions import MilestoneTracker

HISTORY_BLOCK = 12     # history="blocks": trim this many turns at once (prompt-cache friendly)

SYSTEM = """You are the pilot of a Cessna 172 SP in X-Plane 12, a realistic flight simulator. You fly it by hand,
alone, like a licensed private pilot.

Goal: {goal}

How time works: the simulator is PAUSED while you think. Time only passes inside act(), which holds the
controls you set for `seconds` of simulated time and then pauses again. Controls stay where you left them.

Tools:
- look(): the view from the pilot seat (screenshot) plus the instrument readout.
- act(...): pitch, roll, yaw = yoke and rudder, -1..1 (pitch +1 = full back / nose up, roll +1 = right,
  yaw +1 = right rudder). trim: elevator trim -1..1 (+ = nose up). throttle, mixture: 0..1.
  flaps: handle 0, 0.33, 0.67, 1.0 = 0/10/20/30 degrees. brakes: toe brakes 0..1. parking_brake: 0 or 1.
  commands: optional X-Plane command paths. seconds: 0.5 to 5 while airborne, at most 1 below 50 ft above
  the ground (the flare needs continuous small inputs), up to 30 on the ground.
  Returns the instrument readout (and, on short final, the view outside).
- report(message, status): short note for the flight log; status "done" when the goal is achieved,
  "give_up" if it cannot be.
Crashing ends the flight.

Your pilot's operating handbook card (C172 SP):
- Approach: 65-70 KIAS with flaps 30 (70-75 KIAS flaps 10-20), about 1500-1700 rpm, 3-degree glidepath
  (~400 fpm down at 70 kt). Short final 60-65 KIAS. Stall: 40 KIAS flaps 30, 48 KIAS clean.
- Trim the airplane for the speed you want so it flies itself hands-off; then use SMALL yoke inputs
  (a few hundredths to a tenth) and bring the yoke back toward neutral once the attitude is right.
  The yoke commands a rate, not an attitude: holding it keeps the nose / wing moving.
- Flaps (below 110 KIAS): extend one notch at a time. Each notch makes the nose pitch up and the airplane
  balloon and slow down - as the flaps come out, lower the nose (a few degrees) and re-trim nose-down to
  hold your approach speed; in cloud, watch the attitude indicator and VSI for the pitch-up.
- Pitch controls airspeed, power controls the descent rate. Bank no more than 15-20 degrees on final.
- Correct drift with a small bank toward the centerline, then level the wings.
- Stabilised by 500 ft above the runway (on speed, on glidepath, on centerline, landing flaps) or go
  around: full power, pitch ~7 degrees up, flaps to 20 then 10, climb and try again.
- Over the threshold: power to idle, hold the aim point, flare at ~10-20 ft so the mains touch first at
  ~55 kt with little sink (under ~300 fpm; over 600 fpm is a hard landing that damages the airplane);
  keep the nose straight with rudder, then brake to a stop on the centerline.
Look outside on final: the runway view tells you what the numbers can't."""

TOOLS = [
    {"type": "function", "name": "look", "description": "Screenshot of the cockpit view + instrument readout.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "function", "name": "act",
     "description": "Set controls (any subset; the rest stay as they are) and fly for `seconds`, then pause.",
     "parameters": {"type": "object", "properties": {
         "pitch": {"type": "number", "minimum": -1, "maximum": 1},
         "roll": {"type": "number", "minimum": -1, "maximum": 1},
         "yaw": {"type": "number", "minimum": -1, "maximum": 1},
         "trim": {"type": "number", "minimum": -1, "maximum": 1},
         "throttle": {"type": "number", "minimum": 0, "maximum": 1},
         "mixture": {"type": "number", "minimum": 0, "maximum": 1},
         "flaps": {"type": "number", "minimum": 0, "maximum": 1},
         "brakes": {"type": "number", "minimum": 0, "maximum": 1},
         "parking_brake": {"type": "number", "minimum": 0, "maximum": 1},
         "commands": {"type": "array", "items": {"type": "string"}},
         "seconds": {"type": "number", "minimum": 0.5, "maximum": 30}},
         "required": ["seconds"]}},
    {"type": "function", "name": "report", "description": "Write a note to the flight log / declare the result.",
     "parameters": {"type": "object", "properties": {
         "message": {"type": "string"},
         "status": {"type": "string", "enum": ["progress", "done", "give_up"]}},
         "required": ["message", "status"]}},
]


def instruments(st: dict, runway: Runway, approach: str = "rnav", ils: Optional[dict] = None) -> str:
    """
    The text half of what the model sees - ONLY what a real pilot in this seat has (feedback rule: no
    simulator hacks): the C172's gauges as they read (with their lag), the controls in their hands, what
    they feel, and the avionics fitted for this approach: a GPS (GNS 430 style) readout for RNAV, the NAV1
    ILS needles for ILS, nothing extra for a visual approach. No radar altitude, no exact sim wind.
    """
    g = lambda k, alt: st.get(k) if st.get(k) is not None else st.get(alt, 0)  # noqa: E731
    magoff = ((st.get("heading_mag", 0) - st.get("heading_true", 0) + 180) % 360) - 180   # local variation
    ball = st.get("slip_ind") or 0
    ball_txt = "centered" if abs(ball) < 0.5 else f"{abs(ball):.1f} deg {'right' if ball > 0 else 'left'}"
    held = st.get("held_controls") or {}
    lines = [
        f"clock {st.get('sim_time_s', 0):.1f}s{'  PAUSED' if st.get('paused') else ''}"
        + ("  *** CRASHED ***" if st.get("crashed") else ""),
        f"Airspeed indicator {g('ias_ind', 'ias_kt'):.0f} kt | altimeter {st.get('alt_indicated_ft', st['alt_msl_ft']):.0f} ft "
        f"(runway {runway.ident} elevation {runway.elevation_ft:.0f} ft) | vertical speed {g('vs_ind', 'vs_fpm'):+.0f} fpm",
        f"Attitude indicator pitch {g('pitch_ind', 'pitch_deg'):+.1f} deg, bank {g('roll_ind', 'bank_deg'):+.1f} deg "
        f"(+ = right); the nose is moving {st.get('pitch_rate_dps', 0):+.1f} deg/s (+ = up) | heading indicator "
        f"{g('hdg_ind', 'heading_mag'):.0f} (magnetic) | turn coordinator {st.get('turn_ind', 0):+.1f} deg/s, ball {ball_txt}"
        + ("  *** STALL WARNING HORN ***" if st.get("stall_horn") or st.get("stall_warning") else ""),
        f"Engine {'running' if st.get('engine_running') else 'NOT running'} {st.get('engine_rpm') or 0:.0f} rpm | flaps "
        f"indicator {st.get('flaps_deployed', 0) * 30:.0f} deg (handle {st.get('flaps_handle', 0):.2f}) | elevator trim "
        f"{st.get('trim', 0):+.2f} | fuel {st.get('fuel_kg', 0):.0f} kg | parking brake "
        f"{'on' if st.get('parking_brake', 0) > 0.5 else 'off'} | wheels on the ground: {'yes' if st.get('on_ground') else 'no'}",
        "Your hands/feet (held controls): " + (", ".join(f"{k} {v:.2f}" for k, v in held.items()) or "none"),
    ]
    d_ = st.get("during") or {}
    if d_:
        alt_txt = (f"altimeter {d_['alt_min']:.0f}-{d_['alt_max']:.0f} ft" if "alt_min" in d_ else "")
        lines.append(
            f"During the last {d_['seconds']:.1f} s: airspeed {d_['ias_min']:.0f}-{d_['ias_max']:.0f} kt (trend "
            f"{d_['ias_trend_kt_s']:+.1f} kt/s), {alt_txt + ', ' if alt_txt else ''}pitch {d_['pitch_min']:+.0f} to "
            f"{d_['pitch_max']:+.0f} deg, bank {d_['bank_min']:+.0f} to {d_['bank_max']:+.0f} deg, vertical speed "
            f"{d_['vs_min']:+.0f} to {d_['vs_max']:+.0f} fpm, turning {d_['turn_rate_dps']:+.1f} deg/s"
            + (f"; STALL HORN sounded for {d_['stall_horn_s']:.1f} s" if d_.get("stall_horn_s") else ""))
    if approach == "rnav":
        d = runway.deviations(st["lat"], st["lon"], st["alt_msl_ft"])
        dist = math.hypot(d["along_ft"], d["cross_ft"]) / FT_PER_NM
        dtk = (runway.heading_true + magoff) % 360
        trk = (st.get("track_true", 0) + magoff) % 360
        xtk = abs(d["cross_ft"]) / FT_PER_NM
        vdi = d["gs_dots"]
        lines.append(
            f"GPS (GNS 430, RNAV RWY {runway.ident} LPV): RW{runway.ident} {dist:.1f} nm | DTK {dtk:03.0f} | TRK "
            f"{trk:03.0f} | GS {st.get('gs_kt', 0):.0f} kt | XTK {xtk:.2f} nm {'R' if d['cross_ft'] > 0 else 'L'} of "
            f"course | vertical deviation " + ("centered" if abs(vdi) < 0.05 else
                                              f"{abs(vdi):.1f} dots {'ABOVE' if vdi > 0 else 'BELOW'} the glidepath")
            + ("" if d["along_ft"] < 0 else " | past the runway waypoint"))
    elif approach == "ils":
        ident = (ils or {}).get("id", "?")
        freq = (ils or {}).get("freq") or (st.get("nav1_freq", 0) / 100.0)
        if st.get("nav1_has_loc"):
            h = st.get("nav1_hdef", 0) or 0
            loc = ("localizer needle centered" if abs(h) < 0.05 else f"localizer needle {abs(h):.1f} dots "
                   f"{'RIGHT' if h > 0 else 'LEFT'} (the course is to your {'right' if h > 0 else 'left'})")   # +hdef = needle right (calibrated in-sim)
        else:
            loc = "localizer: flag (no signal)"
        if st.get("nav1_has_gs") and st.get("nav1_has_loc"):      # the CDI flags both when the signal is lost
            v = st.get("nav1_vdef", 0) or 0
            gs = ("glideslope needle centered" if abs(v) < 0.05 else f"glideslope needle {abs(v):.1f} dots "
                  f"{'DOWN' if v > 0 else 'UP'} (the glidepath is {'below' if v > 0 else 'above'} you)")
        else:
            gs = "glideslope: flag (no signal)"
        dme = f"DME {st.get('nav1_dme_nm', 0):.1f} nm" if st.get("nav1_has_dme") else "no DME on this ILS"
        mk = " | ".join(n for n, k in (("OUTER MARKER", "marker_outer"), ("MIDDLE MARKER", "marker_middle"),
                                       ("INNER MARKER", "marker_inner")) if st.get(k))
        lines.append(f"NAV1 ILS {ident} {freq:.2f}: {loc} | {gs} | {dme}" + (f" | {mk}" if mk else ""))
    for ev in st.get("events") or []:
        if ev.get("type") == "touchdown":
            sink = -ev["sink_fpm"]
            feel = "smooth" if sink < 200 else "firm" if sink < 400 else "HARD - a heavy jolt"
            lines.append(f"You felt the wheels touch the ground: {feel}.")
    return "\n".join(lines)


def _bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


@dataclass
class RunStats:
    turns: int = 0
    acts: int = 0
    looks: int = 0
    views: int = 0               # automatic outside views on short final
    sim_s: float = 0.0
    wall_s: float = 0.0
    model_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0        # when the endpoint reports it (e.g. a LiteLLM proxy)
    cached_tokens: int = 0       # part of input_tokens served from the provider's prompt cache (cheaper)
    reasoning_tokens: int = 0    # part of output_tokens spent reasoning: shows the requested effort was applied
    status: str = "running"      # running | done | give_up | crashed | turn_limit | time_limit | error
    notes: list = field(default_factory=list)


class Campaign:
    """campaigns/<name>/: meta.json, events.jsonl, shots/, progress.json. Situation saved in X-Plane."""

    def __init__(self, root: Path, name: str):
        self.dir = root / name
        self.name = name
        (self.dir / "shots").mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **data):
        with (self.dir / "events.jsonl").open("a") as fh:
            fh.write(json.dumps({"t": datetime.now().isoformat(timespec="seconds"), "kind": kind, **data},
                                ensure_ascii=False, default=str) + "\n")

    def shot(self, turn: int, jpeg: bytes) -> str:
        p = self.dir / "shots" / f"{turn:04d}.jpg"
        p.write_bytes(jpeg)
        (self.dir / "latest.jpg").write_bytes(jpeg)
        return p.name

    def write(self, name: str, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))

    def read(self, name: str):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else None


class Pilot:
    def __init__(self, client, model: str, sim, mission: dict, campaign: Campaign, tracker: MilestoneTracker,
                 reasoning: Optional[str] = "low", image_detail: str = "auto", keep_turns: int = 24,
                 spectate_every: int = 0, history: str = "sliding", cache_marks: bool = False,
                 images: str = "latest"):
        self.client, self.model, self.sim = client, model, sim
        self.mission, self.campaign, self.tracker = mission, campaign, tracker
        self.reasoning, self.image_detail, self.keep_turns = reasoning, image_detail, keep_turns
        self.spectate_every = spectate_every   # viewer-only screenshot every N acts (never sent to the model)
        # history: "sliding" (v1) drops the oldest turn every turn, which changes the start of every request and
        # defeats prompt caching. "blocks" lets it grow to keep_turns + HISTORY_BLOCK and then drops a whole block,
        # so the prefix is identical for HISTORY_BLOCK turns at a time; the model never sees fewer than keep_turns.
        self.history, self._base = history, 0
        # images="keep" (with history="blocks"): screenshots stay in the history exactly as sent and older ones are
        # swapped for a placeholder only when a block is trimmed. Swapping every turn edits the previous request,
        # and prompt caches only hit when a request extends the previous one unchanged (measured: 1,090 cached
        # tokens per turn with swapping vs ~80% of the prompt without). The model then sees the views since the
        # last trim (<= HISTORY_BLOCK + 1), not only the latest one.
        self.images, self._img_from = images, 0
        # cache_marks: Anthropic only caches up to explicit cache_control points (OpenAI caches automatically)
        self.cache_marks = cache_marks
        # Anthropic models reject forced tool use ("tool_choice any is not supported"): let them choose, and
        # nudge them back to the tools when a turn has none (the sim stays paused, so that costs no sim time)
        self.tool_choice = "auto" if any(k in model.lower() for k in ("claude", "opus", "sonnet", "haiku")) \
            else "required"
        self.approach = mission.get("approach", "rnav")   # what avionics the pilot has: visual | rnav | ils
        self.ils = mission.get("ils_tuned")                # {id, freq, ...} from the setup's NAV1 tuning
        self.max_air_seconds = 5.0              # a pilot's hands don't hold an input for longer than this in the air
        self.final_view_agl_ft = 1000.0         # below this on final, every act result includes the view outside
        self._last_on_ground = False
        self.flare_agl_ft, self.flare_max_seconds = 50.0, 1.0   # below 50 ft AGL, inputs last at most 1 s
        self._last_agl = 1e9
        self.stats = RunStats()
        self.groups: list[list] = []     # conversation, one group per model turn (kept whole when trimming)
        self.first: list = [{"role": "user", "content": [{"type": "input_text",
                                                          "text": "Begin. Start with look()."}]}]

    # -- model I/O ----------------------------------------------------------
    def _input(self) -> list:
        items = list(self.first)
        if self.history == "blocks":
            if len(self.groups) - self._base > self.keep_turns + HISTORY_BLOCK:
                self._base = len(self.groups) - self.keep_turns
                # the prefix changes here anyway: now drop all but the newest screenshot
                self._img_from = max((i for i, g in enumerate(self.groups) if any(_is_image_msg(x) for x in g)),
                                     default=0)
            groups = self.groups[self._base:]
            offset = self._base
        else:
            groups = self.groups[-self.keep_turns:]
            offset = len(self.groups) - len(groups)
        last_img = max((i for i, g in enumerate(groups) if any(_is_image_msg(x) for x in g)), default=-1)
        for i, g in enumerate(groups):
            if self.history == "blocks" and getattr(self, "images", "latest") == "keep":
                show = i + offset >= getattr(self, "_img_from", 0)
            else:
                show = i == last_img
            for x in g:
                if _is_image_msg(x) and not show:
                    items.append({"role": "user", "content": [{"type": "input_text",
                                                               "text": "(older screenshot omitted)"}]})
                else:
                    items.append(x)
        if self.cache_marks:
            _mark_cache(items)
        return items

    def _call(self):
        kw = {"model": self.model, "instructions": SYSTEM.format(goal=self.mission["goal"]),
              "input": self._input(), "tools": TOOLS, "tool_choice": self.tool_choice, "parallel_tool_calls": False}
        if self.reasoning:
            kw["reasoning"] = {"effort": self.reasoning}
        t0 = time.time()
        r = self.client.responses.create(**kw)
        self.stats.model_s += time.time() - t0
        try:
            self.stats.input_tokens += r.usage.input_tokens
            self.stats.output_tokens += r.usage.output_tokens
            det = getattr(r.usage, "input_tokens_details", None)
            self.stats.cached_tokens += int(getattr(det, "cached_tokens", 0) or 0)
            odet = getattr(r.usage, "output_tokens_details", None)
            self.stats.reasoning_tokens += int(getattr(odet, "reasoning_tokens", 0) or 0)
            cost = getattr(r.usage, "cost", None)
            if cost is None and getattr(r.usage, "model_extra", None):
                cost = r.usage.model_extra.get("cost")
            if cost:
                self.stats.cost_usd += float(cost)
        except Exception:  # noqa: BLE001
            pass
        return r

    # -- loop -------------------------------------------------------------------
    def fly(self, max_turns: int, max_sim_s: float, on_turn=None) -> RunStats:
        t_start = time.time()
        while self.stats.status == "running":
            if self.stats.turns >= max_turns:
                self.stats.status = "turn_limit"
                break
            if self.stats.sim_s >= max_sim_s:
                self.stats.status = "time_limit"
                break
            self.stats.turns += 1
            try:
                r = self._call()
            except Exception as e:  # noqa: BLE001
                self.campaign.log("error", where="model", error=str(e))
                self.stats.status = "error"
                self.stats.notes.append(f"model error: {e}")
                break
            group = _clean_output(r.output)
            text = _output_text(r.output)
            if text:
                self.campaign.log("say", turn=self.stats.turns, text=text)
            for item in r.output:
                if getattr(item, "type", "") != "function_call":
                    continue
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out, extra = self._tool(item.name, args)
                group.append({"type": "function_call_output", "call_id": item.call_id, "output": out})
                group += extra
            if not any(getattr(i, "type", "") == "function_call" for i in r.output):
                group.append({"role": "user", "content": [{"type": "input_text", "text":
                              "(The sim is paused. Use one of your tools: look, act or report.)"}]})
            elif self.cache_marks and not (isinstance(group[-1], dict) and group[-1].get("role") == "user"):
                # the proxy drops cache marks on tool results: end each turn with a message the mark can sit on
                group.append({"role": "user", "content": [{"type": "input_text", "text": "(sim paused)"}]})
            self.groups.append(group)
            if on_turn:
                on_turn(self)
        self.stats.wall_s = time.time() - t_start
        return self.stats

    TRACK_COLS = ("t", "lat", "lon", "alt", "agl", "ias", "gs", "vs", "hdg", "pitch", "bank", "thr", "flaps",
                  "yoke_pitch", "yoke_roll", "on_ground")

    def _save_track(self, turn: int, track):
        """10 Hz replay samples from the plugin -> campaigns/<name>/trajectory.jsonl (one row per sample)."""
        if not track:
            return
        with (self.campaign.dir / "trajectory.jsonl").open("a") as fh:
            for row in track:
                fh.write(json.dumps({**dict(zip(self.TRACK_COLS, row)), "turn": turn}) + "\n")

    def _on_short_final(self, st: dict) -> bool:
        if self.approach == "visual" and not st.get("on_ground"):
            return True                          # flying visually: the view outside is the main reference
        if st.get("on_ground") or st.get("agl_ft", 1e9) > self.final_view_agl_ft:
            return False
        d = self.tracker.runway.deviations(st["lat"], st["lon"], st["alt_msl_ft"])
        return -4 * FT_PER_NM < d["along_ft"] < self.tracker.runway.length_ft

    def _tool(self, name: str, args: dict) -> tuple[str, list]:
        turn = self.stats.turns
        try:
            if name == "look":
                self.stats.looks += 1
                st, jpeg = self.sim.look()
                shot = self.campaign.shot(turn, jpeg)
                text = instruments(st, self.tracker.runway, self.approach, self.ils)
                self.campaign.log("look", turn=turn, shot=shot, instruments=text)
                img = {"role": "user", "content": [
                    {"type": "input_text", "text": "Screenshot:"},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
                     "detail": self.image_detail}]}
                return text, [img]

            if name == "act":
                self.stats.acts += 1
                controls = {k: args.get(k) for k in ("pitch", "roll", "yaw", "trim", "throttle", "mixture", "flaps",
                                                     "brakes", "parking_brake")}
                seconds = float(args.get("seconds", 2))
                clamped = ""
                cap = self.max_air_seconds
                if not self._last_on_ground and self._last_agl < self.flare_agl_ft:
                    cap = self.flare_max_seconds          # the last 50 ft: a pilot's hands are continuous
                if not self._last_on_ground and seconds > cap:
                    why = "below 50 ft: flare" if cap == self.flare_max_seconds else "airborne limit"
                    clamped = f"(seconds clamped from {seconds:g} to {cap:g}: {why})\n"
                    seconds = cap
                st = self.sim.act(seconds, commands=args.get("commands"), **controls)
                self._save_track(turn, st.pop("track", None))
                self.stats.sim_s += st.get("flown_s", seconds)
                self._last_on_ground = bool(st.get("on_ground"))
                self._last_agl = float(st.get("agl_ft", 1e9))
                new = self.tracker.update(st, turn)
                limit = self.mission.get("max_go_arounds", 1)
                if self.tracker.p.go_arounds > limit:
                    self.stats.status = "go_around_limit"
                if "missed_approach_at_minimums" in self.tracker.p.reached and self.stats.status == "running":
                    self.stats.status = "done"               # below-minimums scenario flown correctly
                if self.tracker.p.failed and self.stats.status == "running":
                    self.stats.status = "failed"
                    self.stats.notes.append(self.tracker.p.failed)
                text = clamped + instruments(st, self.tracker.runway, self.approach, self.ils)
                extra = []
                if self._on_short_final(st):
                    # a pilot on short final is looking outside: attach the view to every act result
                    try:
                        _, jpeg = self.sim.look()
                        self.stats.views += 1
                        shot = self.campaign.shot(turn, jpeg)
                        extra = [{"role": "user", "content": [
                            {"type": "input_text", "text": "View outside (short final):"},
                            {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
                             "detail": self.image_detail}]}]
                    except Exception as e:  # noqa: BLE001
                        self.campaign.log("error", where="final_view", error=str(e))
                self.campaign.log("act", turn=turn, args=args, instruments=text, milestones=new,
                                  state={k: st.get(k) for k in ("lat", "lon", "alt_msl_ft", "agl_ft", "ias_kt",
                                                                 "vs_fpm", "heading_true", "bank_deg", "pitch_deg",
                                                                 "on_ground", "crashed", "gs_kt", "events")})
                self.campaign.write("progress.json", self.tracker.score())
                try:
                    self.sim.save(self.campaign.name)
                except Exception as e:  # noqa: BLE001
                    self.campaign.log("error", where="save", error=str(e))
                if self.spectate_every and not extra and self.stats.acts % self.spectate_every == 0:
                    try:   # for the human watching the viewer: costs wall time only, no tokens
                        _, jpeg = self.sim.look()
                        self.campaign.shot(turn, jpeg)
                    except Exception as e:  # noqa: BLE001
                        self.campaign.log("error", where="spectate", error=str(e))
                if st.get("crashed"):
                    self.stats.status = "crashed"
                    text += "\nThe aircraft has crashed. The flight is over."
                elif self.stats.status == "failed":
                    text += f"\nCHECKRIDE FAILED: {self.tracker.p.failed}. The flight is over."
                elif self.stats.status == "go_around_limit":
                    text += "\nSecond go-around: the attempt is over."
                return text, extra

            if name == "report":
                msg, status = args.get("message", ""), args.get("status", "progress")
                self.stats.notes.append(msg)
                self.campaign.log("report", turn=turn, message=msg, status=status)
                if status in ("done", "give_up"):
                    self.stats.status = status
                return "logged", []

            return f"unknown tool {name}", []
        except Exception as e:  # noqa: BLE001
            self.campaign.log("error", where=name, error=str(e))
            return f"tool error: {e}", []


def _mark_cache(items: list) -> None:
    """Anthropic prompt caching: mark the last message part (on a copy, so the stored history stays unmarked
    and each request carries exactly one mark)."""
    for i in range(len(items) - 1, -1, -1):
        x = items[i]
        if isinstance(x, dict) and x.get("role") == "user" and x.get("content"):
            parts = [dict(p) for p in x["content"]]
            parts[-1]["cache_control"] = {"type": "ephemeral"}
            items[i] = {**x, "content": parts}
            return


def _output_text(output) -> str:
    """Text the model wrote this turn. Not `response.output_text`: via the LiteLLM proxy, Anthropic reasoning
    arrives as output_text parts with text=None, which makes the SDK helper raise."""
    return "".join(getattr(c, "text", None) or "" for o in output if getattr(o, "type", "") == "message"
                   for c in (getattr(o, "content", None) or [])).strip()


def _clean_output(output) -> list:
    """The turn's output items to send back as history, minus empty text parts (and messages left empty)."""
    items = []
    for o in output:
        if getattr(o, "type", "") == "message":
            parts = [c for c in (getattr(o, "content", None) or []) if getattr(c, "text", None)]
            if not parts:
                continue
            if len(parts) != len(o.content):
                o = o.model_copy(update={"content": parts})
        items.append(o)
    return items


def _is_image_msg(x) -> bool:
    return isinstance(x, dict) and x.get("role") == "user" and any(
        isinstance(c, dict) and c.get("type") == "input_image" for c in x.get("content", []))
