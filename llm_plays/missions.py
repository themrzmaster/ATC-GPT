"""
Missions ("beat the game") and the milestone ladder that scores them.

A mission = a one-line goal for the model + how to set the sim up + a ladder of milestones checked
against the real X-Plane state after every act(). Score = how far up the ladder the model got, plus
landing quality when it lands. Crashing ends the attempt.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from pathlib import Path

from .geometry import FT_PER_NM, Runway, runway_from_db, wrap180

SCENARIOS_FILE = Path(__file__).with_name("scenarios.yaml")

LADDERS = {
    "landing": ["stable_1000", "stable_500", "touchdown_on_runway", "stopped_on_runway"],
    "circuit": ["airborne", "above_800_agl", "downwind", "stable_500", "touchdown_on_runway", "stopped_on_runway"],
    "cross_country": ["airborne", "above_800_agl", "within_5nm_destination", "stable_500", "touchdown_on_runway",
                      "stopped_on_runway"],
}
DEFAULT_LIMITS = {"landing": {"max_turns": 90, "max_sim_s": 480, "max_go_arounds": 1},
                  "circuit": {"max_turns": 300, "max_sim_s": 1200, "max_go_arounds": 1},
                  "cross_country": {"max_turns": 400, "max_sim_s": 2400, "max_go_arounds": 1}}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "weather" \
            and k != "start" else v
    return out


def load_scenarios(path: Path = SCENARIOS_FILE, airport_data=None) -> dict:
    """scenarios.yaml -> {name: mission dict} (the shape the tracker / setup / agent use)."""
    import yaml
    raw = yaml.safe_load(path.read_text()) or {}
    resolved: dict = {}

    def resolve(name, seen=()):
        if name in resolved:
            return resolved[name]
        if name in seen:
            raise ValueError(f"scenario {name}: extends loop")
        sc = dict(raw[name])
        if "extends" in sc:
            parent = resolve(sc.pop("extends"), seen + (name,))
            sc = _merge(parent, sc)
        resolved[name] = sc
        return sc

    return {name: _to_mission(name, resolve(name), airport_data) for name in raw}


def _to_mission(name: str, sc: dict, airport_data=None) -> dict:
    kind = sc.get("type", "landing")
    if kind not in LADDERS:
        raise ValueError(f"scenario {name}: unknown type {kind}")
    start = dict(sc.get("start") or {})
    setup = {"airport": sc["airport"], "runway": str(sc["runway"]), "on_runway": kind != "landing", **start}
    setup.setdefault("distance_nm", 5.0)
    setup.setdefault("speed_kt", 75)
    dest = sc.get("destination") or {"airport": sc["airport"], "runway": str(sc["runway"])}
    limits = {**DEFAULT_LIMITS[kind], **(sc.get("limits") or {})}
    approach = sc.get("approach", "rnav")
    if approach not in ("visual", "rnav", "ils"):
        raise ValueError(f"scenario {name}: approach must be visual | rnav | ils")
    minimums = sc.get("minimums_ft", {"visual": None, "rnav": 250, "ils": 200}[approach])
    m = {"type": kind, "setup": setup, "destination": (dest["airport"], str(dest["runway"])),
         "ladder": LADDERS[kind], "weather": sc.get("weather") or {}, "approach": approach,
         "reported_weather": sc.get("reported_weather"),
         "minimums_ft": minimums, **limits}
    # weather below minimums: the right answer is the missed approach, flown at the decision altitude
    m["expect_missed"] = bool(minimums) and not runway_visible_at_minimums(m)
    if m["expect_missed"]:
        m["ladder"] = ["stable_1000", "missed_approach_at_minimums"]
    m["goal"] = sc.get("goal") or _goal(m, airport_data)
    return m


def _airport_name(icao: str, airport_data=None) -> str:
    try:
        if airport_data is None:
            from airport_data_service import get_airport_data_service
            airport_data = get_airport_data_service()
        ap = airport_data.get(icao)
        return f"{ap.name} ({icao})" if ap else icao
    except Exception:  # noqa: BLE001
        return icao


def _goal(m: dict, airport_data=None) -> str:
    s, (dest, drwy) = m["setup"], m["destination"]
    here = _airport_name(s["airport"], airport_data)
    if m["type"] == "landing":
        goal = (f"You are on a {s['distance_nm']:g} nm final for runway {s['runway']} at {here}. Land on runway "
                f"{s['runway']} and bring the aircraft to a full stop on the runway.")
    elif m["type"] == "circuit":
        goal = (f"You are lined up on runway {s['runway']} at {here}, engine running. Take off, fly one traffic "
                f"circuit and land back on runway {s['runway']}, then stop on the runway.")
    else:
        goal = (f"You are lined up on runway {s['runway']} at {here}, engine running. Fly to "
                f"{_airport_name(dest, airport_data)}, land on runway {drwy} and stop on the runway.")
    wx = weather_report(m.get("reported_weather") or m.get("weather") or {})   # what the ATIS said
    rwy = m["destination"][1]
    elev = _field_elev(m["destination"][0], airport_data)
    da = f"decision altitude {elev + m['minimums_ft']:.0f} ft ({m['minimums_ft']} ft above the runway)" \
        if m.get("minimums_ft") and elev is not None else ""
    appr = {"visual": f" Visual approach to runway {rwy}: no approach guidance, fly by what you see.",
            "rnav": f" Approach: RNAV (GPS) runway {rwy} with LPV vertical guidance on your GPS" + (f", {da}" if da else "") + ".",
            "ils": f" Approach: ILS runway {rwy}, tuned on NAV1" + (f", {da}" if da else "") + "."}[m["approach"]]
    mins = (" If you cannot see the runway at the decision altitude, fly the missed approach (go around)."
            if m.get("minimums_ft") else "")
    return goal + appr + (f" Latest weather report (ATIS): {wx}." if wx else "") + mins


def _field_elev(icao: str, airport_data=None):
    try:
        if airport_data is None:
            from airport_data_service import get_airport_data_service
            airport_data = get_airport_data_service()
        ap = airport_data.get(icao)
        return float(ap.elevation_ft) if ap and ap.elevation_ft is not None else None
    except Exception:  # noqa: BLE001
        return None


def runway_visible_at_minimums(m: dict) -> bool:
    """Would the runway be in sight at the decision altitude? (ceiling above minimums, >= 1/2 sm visibility)"""
    if not m.get("minimums_ft"):
        return True
    w = m.get("weather") or {}
    ceilings = [c["base_ft_agl"] for c in (w.get("clouds") or []) if c.get("coverage", 0) >= 0.6]
    return (min(ceilings) if ceilings else 1e9) > m["minimums_ft"] and w.get("visibility_sm", 10) >= 0.5


def weather_report(w: dict) -> str:
    """ATIS-style line, what a real pilot would have heard before the approach."""
    parts = []
    if w.get("wind_kt"):
        parts.append(f"wind {int(w.get('wind_from_deg', 0)):03d} at {w['wind_kt']:g} kt"
                     + (f" gusting {w['wind_kt'] + w['gust_kt']:g}" if w.get("gust_kt") else ""))
    elif w:
        parts.append("wind calm")
    if w.get("visibility_sm"):
        parts.append(f"visibility {w['visibility_sm']:g} sm")
    for c in w.get("clouds") or []:
        cov = c.get("coverage", 0.5)
        word = "overcast" if cov >= 0.85 else "broken" if cov >= 0.6 else "scattered" if cov >= 0.3 else "few"
        parts.append(f"{word} {c['base_ft_agl']:g} ft")
    if w.get("rain"):
        parts.append("rain")
    if w.get("turbulence"):
        parts.append("turbulence reported")
    return ", ".join(parts)


def apply_weather(sim, w: dict, field_elev_ft: float, lon: float):
    """Set X-Plane 12 weather (switches from real weather to a manual preset first)."""
    kt = 0.514444
    sim.set({"sim/weather/region/weather_preset": 0})           # leaves Real Weather: region datarefs now stick
    wind, gust = float(w.get("wind_kt", 0)), float(w.get("gust_kt", 0))
    vals = {
        "sim/weather/region/wind_direction_degt": [float(w.get("wind_from_deg", 0))] * 13,
        "sim/weather/region/wind_speed_msc": [wind * kt] * 13,
        "sim/weather/region/shear_speed_msc": [gust * kt] * 13,
        "sim/weather/region/turbulence": [float(w.get("turbulence", 0))] * 13,
        "sim/weather/region/visibility_reported_sm": float(w.get("visibility_sm", 20)),
        "sim/weather/region/rain_percent": float(w.get("rain", 0)),
        "sim/weather/region/variability_pct": 0.0,
    }
    clouds = (w.get("clouds") or [])[:3]
    base = [((field_elev_ft + c["base_ft_agl"]) * 0.3048) for c in clouds] + [6000.0] * (3 - len(clouds))
    vals["sim/weather/region/cloud_base_msl_m"] = base
    vals["sim/weather/region/cloud_tops_msl_m"] = [b + 500.0 for b in base]
    vals["sim/weather/region/cloud_coverage_percent"] = [float(c.get("coverage", 0.5)) for c in clouds] + [0.0] * (3 - len(clouds))
    vals["sim/weather/region/cloud_type"] = [1.0 if c.get("coverage", 0.5) >= 0.85 else 2.0 for c in clouds] + [2.0] * (3 - len(clouds))
    if w.get("time_local"):
        hh, mm = (int(x) for x in str(w["time_local"]).split(":"))
        tz = round(lon / 15.0)                                   # rough local offset from longitude
        vals["sim/time/zulu_time_sec"] = float(((hh - tz) % 24) * 3600 + mm * 60)
    vals["sim/weather/region/update_immediately"] = 1
    sim.set(vals)


MISSIONS = load_scenarios()


@dataclass
class Touchdown:
    sink_fpm: float
    along_ft: float
    cross_ft: float
    bank_deg: float
    ias_kt: float
    on_runway: bool


@dataclass
class Progress:
    reached: dict = field(default_factory=dict)       # milestone -> turn number
    touchdown: Optional[Touchdown] = None
    crashed: bool = False
    unstable_at: list = field(default_factory=list)   # gate names flown unstable
    go_arounds: int = 0                                # low on final (< 300 ft) then climbed back above 600 ft
    failed: str = ""                                   # checkride-style fail reason (the attempt ends)
    last_airborne_vs: float = 0.0

    def to_dict(self) -> dict:
        return {"reached": self.reached, "crashed": self.crashed, "unstable_at": self.unstable_at,
                "go_arounds": self.go_arounds, "failed": self.failed,
                "touchdown": self.touchdown.__dict__ if self.touchdown else None}


class MilestoneTracker:
    def __init__(self, mission: dict, airport_data=None):
        self.mission = mission
        icao, rwy = mission["destination"]
        self.runway: Runway = runway_from_db(icao, rwy, airport_data)
        self.p = Progress()
        self._was_airborne = False
        self._was_airborne_ever = False
        self._low_on_final = False
        self._unstable_since_gate = False
        self._high_crossing = False
        self._going_around = False
        self._turn = 0
        self._gates_checked: set = set()

    def restore(self, prog: dict):
        """Continue a campaign: milestones/gates/go-arounds as they were (the fail verdict is re-judged)."""
        self.p.reached = dict(prog.get("reached") or {})
        self.p.unstable_at = list(prog.get("unstable_at") or [])
        self.p.go_arounds = int(prog.get("go_arounds") or 0)
        self._gates_checked = set(self.p.unstable_at) | {g for g in ("stable_1000", "stable_500") if g in self.p.reached}
        self._unstable_since_gate = "stable_500" in self.p.unstable_at
        self._was_airborne = self._was_airborne_ever = True

    def dev(self, st: dict) -> dict:
        return self.runway.deviations(st["lat"], st["lon"], st["alt_msl_ft"])

    def update(self, st: dict, turn: int) -> list[str]:
        """Feed the state after an act(); returns milestones newly reached."""
        new = []
        self._turn = turn
        before = set(self.p.reached)
        d = self.dev(st)
        airborne = not st.get("on_ground")

        def hit(name, cond):
            if cond and name not in self.p.reached and name in self.mission["ladder"]:
                self.p.reached[name] = turn
                new.append(name)

        if st.get("crashed"):
            self.p.crashed = True
        hit("airborne", airborne)
        hit("above_800_agl", airborne and st.get("agl_ft", 0) > 800)
        hit("downwind", airborne and "above_800_agl" in self.p.reached
            and abs(wrap180(st.get("track_true", 0) - (self.runway.heading_true + 180))) < 30
            and 0.5 < abs(d["cross_ft"]) / FT_PER_NM < 2.5)
        hit("within_5nm_destination", airborne and math.hypot(d["along_ft"], d["cross_ft"]) / FT_PER_NM < 5)

        # stabilised-approach gates (height above threshold, on final: before the threshold, inside 5 nm)
        on_final = -5 * FT_PER_NM < d["along_ft"] < 0 and airborne
        for gate, h in (("stable_1000", 1000), ("stable_500", 500)):
            if on_final and gate not in self._gates_checked and d["height_above_threshold_ft"] <= h:
                self._gates_checked.add(gate)
                stable = (abs(d["gs_dots"]) <= 1.0 and abs(d["loc_dots"]) <= 1.0
                          and 55 <= st.get("ias_kt", 0) <= 90 and st.get("vs_fpm", 0) > -1000
                          and abs(st.get("bank_deg", 0)) < 20)
                hit(gate, stable)
                if not stable:
                    self.p.unstable_at.append(gate)
                    if gate == "stable_500":
                        self._unstable_since_gate = True

        # first touchdown, captured frame-exact by the plugin ("events"); fall back to state edges
        for ev in st.get("events") or []:
            if ev.get("type") == "touchdown" and self.p.touchdown is None and self._was_airborne_ever:
                td = self.runway.deviations(ev["lat"], ev["lon"], self.runway.elevation_ft)
                self.p.touchdown = Touchdown(sink_fpm=ev["sink_fpm"], along_ft=td["along_ft"],
                                             cross_ft=td["cross_ft"], bank_deg=ev["bank_deg"],
                                             ias_kt=ev["ias_kt"], on_runway=td["on_runway_surface"])
        if self._was_airborne and not airborne and self.p.touchdown is None:
            self.p.touchdown = Touchdown(sink_fpm=self.p.last_airborne_vs, along_ft=d["along_ft"],
                                         cross_ft=d["cross_ft"], bank_deg=st.get("bank_deg", 0.0),
                                         ias_kt=st.get("ias_kt", 0.0), on_runway=d["on_runway_surface"])
        self._checkride(st, d, airborne)
        # go-around: got low on final, then climbed away again (a new approach attempt begins)
        hat = d["height_above_threshold_ft"]
        if airborne and -4 * FT_PER_NM < d["along_ft"] < self.runway.length_ft and hat < 300:
            self._low_on_final = True
        if airborne and self._low_on_final and hat > 600:
            if not self._high_crossing:
                self.p.go_arounds += 1
            self._high_crossing = True
            self._low_on_final = False
            self._gates_checked.clear()          # the next approach gets its own stabilisation gates
            self._unstable_since_gate = False
        if d["along_ft"] < -0.5 * FT_PER_NM:
            self._high_crossing = False
        if airborne:
            self.p.last_airborne_vs = st.get("vs_fpm", 0.0)
            self._was_airborne_ever = True
        self._check_touchdown()
        hit("touchdown_on_runway", not airborne and self.p.touchdown is not None and self.p.touchdown.on_runway
            and not st.get("crashed"))
        hit("stopped_on_runway", not airborne and st.get("gs_kt", 99) < 3 and d["on_runway_surface"]
            and "touchdown_on_runway" in self.p.reached and not st.get("crashed"))
        self._was_airborne = airborne
        return new + [m for m in self.p.reached if m not in before and m not in new]

    # checkride limits (private-pilot-examiner style: outside these the flight is over)
    MAX_BANK, MAX_PITCH_UP, MAX_PITCH_DOWN, MAX_HORN_S = 60.0, 30.0, -25.0, 3.0

    def _checkride(self, st: dict, d: dict, airborne: bool):
        if self.p.failed or not airborne:
            return
        env = st.get("during") or {}
        bank = max(abs(env.get("bank_min", st.get("bank_deg", 0))), abs(env.get("bank_max", st.get("bank_deg", 0))))
        pmax, pmin = env.get("pitch_max", st.get("pitch_deg", 0)), env.get("pitch_min", st.get("pitch_deg", 0))
        if bank > self.MAX_BANK:
            self.p.failed = f"loss of control: bank {bank:.0f} deg"
        elif pmax > self.MAX_PITCH_UP or pmin < self.MAX_PITCH_DOWN:
            self.p.failed = f"loss of control: pitch {pmax:+.0f}/{pmin:+.0f} deg"
        elif env.get("stall_horn_s", 0) > self.MAX_HORN_S:
            self.p.failed = f"loss of control: stall horn {env['stall_horn_s']:.1f} s"
        hat, along = d["height_above_threshold_ft"], d["along_ft"]
        if self.p.failed:
            return
        # a go-around by intent: full power and climbing away while on final (at any height). While climbing
        # away the approach-deviation rules don't apply (being "above the glidepath" is the point).
        climbing_away = (st.get("throttle", 0) or 0) >= 0.9 and st.get("vs_fpm", 0) > 300
        on_final = -4 * FT_PER_NM < along < self.runway.length_ft
        if on_final and climbing_away and not self._going_around:
            self._going_around = True
            mins = self.mission.get("minimums_ft")
            # at the decision height (not an earlier go-around from an unstable approach, which is just a go-around)
            if mins and along < 0 and mins - 50 <= hat <= mins + 150 and "missed_approach_at_minimums" in self.mission["ladder"]:
                self.p.reached.setdefault("missed_approach_at_minimums", self._turn)
            if not self._high_crossing and "missed_approach_at_minimums" not in self.p.reached:
                self.p.go_arounds += 1
            self._high_crossing = True
            self._gates_checked.clear()
            self._unstable_since_gate = False
        elif self._going_around and st.get("vs_fpm", 0) < -300 and (st.get("throttle", 1) or 1) < 0.8:
            self._going_around = False            # descending on a new approach
        if self._going_around:
            return
        # below 200 ft on final an examiner only stops a DANGEROUS continuation; a controlled approach that
        # is merely high/fast may continue (the unstable 500 ft gate and a long touchdown cost points)
        danger = self._dangerous_short_final(st, d) if (0 > along > -2 * FT_PER_NM and hat < 200) else ""
        mins = self.mission.get("minimums_ft")
        if mins and not runway_visible_at_minimums(self.mission) and along < 0 and hat < mins - 50:
            self.p.failed = (f"descended below the decision altitude ({mins} ft above the runway) without the runway "
                             f"in sight: missed approach required")
        elif danger:
            self.p.failed = f"continued a dangerous approach below 200 ft instead of going around: {danger}"
        elif -5 * FT_PER_NM < along < 0 and abs(d["cross_ft"]) > 1.5 * FT_PER_NM:
            self.p.failed = f"lost the approach: {abs(d['cross_ft']) / FT_PER_NM:.1f} nm off the centerline"
        elif -3 * FT_PER_NM < along < 0 and abs(d["glidepath_dev_ft"]) > 800:
            self.p.failed = f"lost the approach: {d['glidepath_dev_ft']:+.0f} ft off the glidepath inside 3 nm"
        # crossing the threshold still high = missed approach (counts as the go-around)
        if along > 0 and hat > 400 and not self._high_crossing:
            self._high_crossing = True           # one go-around per approach, whichever rule sees it first
            self.p.go_arounds += 1
            self._gates_checked.clear()
            self._unstable_since_gate = False

    HARD_LANDING_FPM = -600.0          # beyond this a C172 lands hard enough to need an inspection

    def _check_touchdown(self):
        td = self.p.touchdown
        if td and not self.p.failed and td.sink_fpm < self.HARD_LANDING_FPM:
            self.p.failed = f"hard landing: touched down at {td.sink_fpm:.0f} fpm (likely damage)"

    def _dangerous_short_final(self, st: dict, d: dict) -> str:
        why = []
        if st.get("vs_fpm", 0) < -1000:
            why.append(f"sinking {st['vs_fpm']:.0f} fpm")
        if not 55 <= st.get("ias_kt", 0) <= 80:
            why.append(f"{st.get('ias_kt', 0):.0f} kt")
        if abs(st.get("bank_deg", 0)) > 15:
            why.append(f"bank {st['bank_deg']:+.0f} deg")
        if abs(d["glidepath_dev_ft"]) > 300:
            why.append(f"{d['glidepath_dev_ft']:+.0f} ft off the glidepath")
        if abs(d["cross_ft"]) > 1.5 * self.runway.width_ft:
            why.append(f"{abs(d['cross_ft']):.0f} ft off the centerline")
        return ", ".join(why)

    def score(self) -> dict:
        ladder = self.mission["ladder"]
        n = sum(1 for m in ladder if m in self.p.reached)
        base = 70.0 * n / len(ladder)
        landing = 0.0
        td = self.p.touchdown
        if td and td.on_runway and not self.p.crashed:
            landing = 30.0
            landing -= min(15.0, max(0.0, (-td.sink_fpm - 300) / 40))          # firm > 300 fpm
            tdz_m = td.along_ft * 0.3048
            landing -= 0 if 60 <= tdz_m <= 750 else min(8.0, abs(tdz_m - 400) / 100)
            landing -= min(7.0, max(0.0, abs(td.cross_ft) * 0.3048 - 3) * 0.7)  # >3 m off centerline
            if "stable_500" in self.p.unstable_at:
                landing -= 8.0                                                   # pressed on unstable
        elif self.mission.get("expect_missed") and "missed_approach_at_minimums" in self.p.reached \
                and not self.p.crashed:
            landing = 30.0         # below minimums the missed approach IS the good outcome: same 30 points
        total = 0.0 if (self.p.crashed or self.p.failed) else round(base + landing, 1)
        return {"score": total, "milestones": f"{n}/{len(ladder)}", "crashed": self.p.crashed,
                "result": ("CRASH" if self.p.crashed else "FAIL" if self.p.failed else
                           "PASS" if n == len(ladder) else
                           "LANDED" if "stopped_on_runway" in self.p.reached else "INCOMPLETE"),
                "landing_points": round(landing, 1), **self.p.to_dict()}


def start_situation_name(mission_name: str) -> str:
    return f"start_{mission_name}"


def setup_mission(sim, mission: dict, airport_data=None, mission_name: str = ""):
    """
    Put the sim at the mission start, paused. Preferred: a saved X-Plane situation per mission
    (llmplays_start_<mission>.sit, captured once with `python -m llm_plays.run --save-start <mission>`):
    deterministic and includes a running engine. Fallback: load the C172 and place it (engine may be off -
    X-Plane's auto-start proved unreliable, see GAPS.md).
    """
    from .sim import C172_PATH
    s = mission["setup"]
    rw = runway_from_db(s["airport"], s["runway"], airport_data)
    for attempt in range(2):
        if not (mission_name and sim.load(start_situation_name(mission_name))):
            break
        _setup_after_load(sim, mission, rw, airport_data)
        # The first load after an X-Plane start also loads the scenery and can take ~10 s longer, with the
        # airplane flying uncontrolled meanwhile (seen: 94 kt, 430 ft low, zooming). Such a start isn't the
        # scenario: reload once (the scenery is cached by then).
        bad = _start_off_nominal(sim.state(), mission, rw)
        if not bad:
            return rw
        print(f"start state off nominal ({bad}); reloading the start situation")
    else:
        raise RuntimeError(f"start state still off nominal after a reload: {bad}")
    return _setup_by_placement(sim, mission, rw, airport_data)


def _start_off_nominal(st: dict, mission: dict, rw) -> str:
    s = mission["setup"]
    if s.get("on_runway"):
        return "" if st.get("on_ground") else "not on the ground"
    d = rw.deviations(st["lat"], st["lon"], st["alt_msl_ft"])
    exp_gs = s.get("height_offset_ft", 0)
    probs = []
    if not exp_gs and abs(d["gs_dots"]) > 0.8:
        probs.append(f"{d['gs_dots']:+.1f} dots off the glidepath")
    if abs((st.get("ias_kt") or 0) - s.get("speed_kt", 75)) > 12:
        probs.append(f"{st.get('ias_kt', 0):.0f} kt")
    if abs(st.get("vs_fpm") or 0) > 1200:
        probs.append(f"{st.get('vs_fpm', 0):+.0f} fpm")
    return ", ".join(probs)


def _setup_after_load(sim, mission: dict, rw, airport_data=None):
    s = mission["setup"]
    sim.wait_ready(settle_s=10)          # a situation load reloads the aircraft: let it settle
    apply_weather(sim, mission.get("weather") or {}, rw.elevation_ft, rw.threshold_lon)
    # X-Plane doesn't restore a *running* engine from a situation (it windmills and dies), so run
    # the same fixed start sequence right after the load: identical every time.
    if s.get("on_runway"):
        st = sim.start_engine(5, mixture=1.0, throttle=0.1, parking_brake=1.0)
        sim.act(0.1, throttle=0.0)
    else:
        st = sim.start_engine(2.5, mixture=1.0, throttle=0.5, pitch=0.1, roll=0, yaw=0)
        sim.act(0.01, pitch=0.0)
    if not st.get("engine_running"):
        raise RuntimeError("engine did not start after loading the start situation")
    _tune_approach(sim, mission, airport_data)
    sim.pause()


def _setup_by_placement(sim, mission: dict, rw, airport_data=None):
    from .sim import C172_PATH
    s = mission["setup"]
    # Swapping aircraft and then placing it straight away deadlocked XPPython3 (seen twice): only swap
    # when needed, then wait until the sim answers state requests again before placing it.
    if (sim.state().get("aircraft_file") or "").strip().upper() != "C172":   # ICAO type of the loaded aircraft
        sim.load_aircraft(C172_PATH)
        sim.wait_ready(settle_s=20)
    apply_weather(sim, mission.get("weather") or {}, rw.elevation_ft, rw.threshold_lon)
    # Placing the aircraft resets the engine to off, so start it right after placing (see
    # XPlaneSim.start_engine for the C172 fuel-selector fix). Airborne: place a few seconds of flight
    # before the start point so the aircraft arrives there with the engine running.
    if s.get("on_runway"):
        lat, lon = rw.from_local(150, 0)
        sim.place(lat, lon, rw.elevation_ft * 0.3048, rw.heading_true, 0)
        st = sim.start_engine(6, mixture=1.0, throttle=0.1, pitch=0, roll=0, yaw=0, flaps=0.0, brakes=0.0,
                              parking_brake=1.0)
        sim.act(0.1, throttle=0.0)
    else:
        spd = s.get("speed_kt", 75)
        lead_s = 3.5
        along = -s["distance_nm"] * FT_PER_NM - spd * 1.68781 * lead_s
        hdg_off = s.get("heading_offset_deg", 0)
        lat, lon = rw.from_local(along, s.get("cross_offset_ft", 0))
        ele_ft = rw.elevation_ft + rw.glidepath_height_ft(along) + s.get("height_offset_ft", 0) + 30
        sim.place(lat, lon, ele_ft * 0.3048, (rw.heading_true + hdg_off) % 360, spd * 0.514444)
        st = sim.start_engine(lead_s - 0.5, mixture=1.0, throttle=0.5, pitch=0.1, roll=0, yaw=0, flaps=0.0,
                              brakes=0.0, parking_brake=0.0)
        sim.act(0.01, pitch=0.0)
    if not st.get("engine_running"):
        raise RuntimeError("engine did not start during mission setup")
    _tune_approach(sim, mission, airport_data)
    return rw


def _tune_approach(sim, mission: dict, airport_data=None):
    """ILS scenarios: tune NAV1 to the runway's localizer from X-Plane's nav database (like the pilot would)."""
    if mission.get("approach") != "ils":
        return
    rw = runway_from_db(*mission["destination"], airport_data)
    r = sim._post("/tune_ils", {"lat": rw.threshold_lat, "lon": rw.threshold_lon, "course_true": rw.heading_true},
                  timeout=180)
    if not r.get("tuned"):
        raise RuntimeError(f"ILS scenario but {r.get('error')}")
    mission["ils_tuned"] = r
    for i in range(8):    # receivers only update while the sim runs and slew ~2 s after tuning: settle first
        if sim.act(0.5).get("nav1_has_loc") and i >= 4:
            break


def _stabilise(sim, seconds: float, target_pitch: float, step: float = 0.5):
    """Setup only (not the pilot): crude attitude hold in 0.5 s steps while the engine starts."""
    st = sim.state()
    t = 0.0
    while t < seconds:
        pitch_cmd = max(-0.4, min(0.4, 0.05 + 0.06 * (target_pitch - st.get("pitch_deg", 0))))
        roll_cmd = max(-0.3, min(0.3, -0.04 * st.get("bank_deg", 0)))
        st = sim.act(step, pitch=pitch_cmd, roll=roll_cmd)
        t += step
    sim.act(0.01, pitch=0.0, roll=0.0)      # hand over with the yoke centred


def mission_names() -> list[str]:
    return list(MISSIONS)


def check_fn(name: str) -> Callable:
    return MISSIONS[name]
