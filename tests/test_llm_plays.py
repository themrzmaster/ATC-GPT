"""Offline tests for LLM-plays-X-Plane: mission tracker/scoring and the agent loop (fake sim + fake model)."""

import json
import types

import numpy as np
import pytest

from llm_plays.geometry import FT_PER_NM
from llm_plays.missions import MISSIONS, MilestoneTracker


def _state(rw, along_ft, cross_ft=0.0, hat_ft=None, **kw):
    lat, lon = rw.from_local(along_ft, cross_ft)
    hat = rw.glidepath_height_ft(along_ft) if hat_ft is None else hat_ft
    st = {"lat": lat, "lon": lon, "alt_msl_ft": rw.elevation_ft + hat, "agl_ft": hat, "ias_kt": 70,
          "vs_fpm": -400, "bank_deg": 0, "track_true": rw.heading_true, "on_ground": 0, "crashed": 0, "gs_kt": 68}
    st.update(kw)
    return st


@pytest.fixture
def tracker(airport_data):
    return MilestoneTracker(MISSIONS["land_sbbh13"], airport_data)


def test_perfect_approach_and_landing_scores_high(tracker):
    rw = tracker.runway
    for along in (-3.5 * FT_PER_NM, -3.2 * FT_PER_NM, -1.6 * FT_PER_NM, -1.4 * FT_PER_NM, -0.3 * FT_PER_NM):
        tracker.update(_state(rw, along), 1)
    lat, lon = rw.from_local(1000, 5)
    td = {"type": "touchdown", "sink_fpm": -180, "bank_deg": 0.5, "pitch_deg": 4, "ias_kt": 55, "lat": lat, "lon": lon}
    tracker.update(_state(rw, 1400, 5, hat_ft=0, on_ground=1, gs_kt=40, events=[td]), 2)
    tracker.update(_state(rw, 2400, 5, hat_ft=0, on_ground=1, gs_kt=1), 3)
    s = tracker.score()
    assert s["milestones"] == "4/4" and not s["crashed"]
    assert s["touchdown"]["sink_fpm"] == -180 and s["score"] >= 95


def test_unstable_gate_and_hard_landing_cost_points(tracker):
    rw = tracker.runway
    tracker.update(_state(rw, -1.2 * FT_PER_NM, hat_ft=900), 1)           # 900 ft where the glidepath is ~430: way high
    tracker.update(_state(rw, -1.4 * FT_PER_NM), 2)
    lat, lon = rw.from_local(3500, 0)
    tracker.update(_state(rw, 3600, hat_ft=0, on_ground=1, gs_kt=50,
                          events=[{"type": "touchdown", "sink_fpm": -750, "bank_deg": 2, "pitch_deg": 0,
                                   "ias_kt": 70, "lat": lat, "lon": lon}]), 3)
    s = tracker.score()
    assert "stable_1000" in s["unstable_at"]
    assert s["landing_points"] < 15 and s["score"] < 75


def test_crash_scores_zero(tracker):
    rw = tracker.runway
    tracker.update(_state(rw, -1.4 * FT_PER_NM), 1)
    tracker.update(_state(rw, -0.5 * FT_PER_NM, hat_ft=0, on_ground=1, crashed=1), 2)
    assert tracker.score()["score"] == 0 and tracker.score()["crashed"]


def test_touchdown_short_of_runway_is_not_on_runway(tracker):
    rw = tracker.runway
    tracker.update(_state(rw, -0.3 * FT_PER_NM), 1)
    lat, lon = rw.from_local(-600, 0)
    tracker.update(_state(rw, -500, hat_ft=0, on_ground=1, gs_kt=40,
                          events=[{"type": "touchdown", "sink_fpm": -200, "bank_deg": 0, "pitch_deg": 3,
                                   "ias_kt": 55, "lat": lat, "lon": lon}]), 2)
    s = tracker.score()
    assert "touchdown_on_runway" not in s["reached"] and s["landing_points"] == 0


# ---------------------------------------------------------------------------
# Agent loop with a fake sim and a scripted fake model
# ---------------------------------------------------------------------------

class FakeSim:
    def __init__(self, rw):
        self.rw, self.along, self.saved = rw, -2 * FT_PER_NM, []

    def look(self):
        return _state(self.rw, self.along), b"\xff\xd8fakejpeg"

    def act(self, seconds, commands=None, **controls):
        self.along += 120 * seconds
        st = _state(self.rw, self.along)
        st["flown_s"] = seconds
        return st

    def save(self, name):
        self.saved.append(name)


def _call(name, args, i):
    return types.SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id=f"c{i}")


class FakeClient:
    """Scripts the model: look, act x2, look, report done. Records what it was sent."""
    def __init__(self):
        self.script = [("look", {}), ("act", {"pitch": 0.1, "throttle": 0.4, "seconds": 3}),
                       ("act", {"seconds": 2}), ("look", {}), ("report", {"message": "landed", "status": "done"})]
        self.inputs = []
        self.responses = self

    def create(self, **kw):
        self.inputs.append(kw["input"])
        i = len(self.inputs)
        name, args = self.script[i - 1]
        return types.SimpleNamespace(output=[_call(name, args, i)], output_text="",
                                     usage=types.SimpleNamespace(input_tokens=100, output_tokens=10))


def test_agent_loop_logs_saves_and_keeps_only_latest_screenshot(tmp_path, tracker):
    from llm_plays.agent import Campaign, Pilot
    sim, client = FakeSim(tracker.runway), FakeClient()
    camp = Campaign(tmp_path, "t1")
    pilot = Pilot(client, "fake-model", sim, MISSIONS["land_sbbh13"], camp, tracker)
    stats = pilot.fly(max_turns=10, max_sim_s=600)
    assert stats.status == "done" and stats.acts == 2 and stats.looks == 2 and stats.sim_s == 5
    assert sim.saved == ["t1", "t1"]                              # situation saved after every act
    last_input = client.inputs[-1]
    images = [x for x in last_input if isinstance(x, dict) and x.get("role") == "user"
              and any(c.get("type") == "input_image" for c in x["content"])]
    assert len(images) == 1                                       # older screenshots trimmed from context
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "t1" / "events.jsonl").read_text().splitlines()]
    assert kinds.count("act") == 2 and "report" in kinds
    assert (tmp_path / "t1" / "latest.jpg").exists() and (tmp_path / "t1" / "progress.json").exists()


def test_instruments_readout_is_what_a_pilot_has(tracker):
    """Feedback rule: only what a real C172 pilot sees/hears/feels - no radar altitude, no sim wind."""
    from llm_plays.agent import instruments
    st = _state(tracker.runway, -2 * FT_PER_NM, cross_ft=300, hat_ft=800, held_controls={"throttle": 0.4},
                agl_ft=812, wind_from_deg=40, wind_kt=11, alt_indicated_ft=3400, heading_mag=132, heading_true=109,
                ias_ind=71, vs_ind=-450, hdg_ind=132, slip_ind=1.2, turn_ind=0.5)
    text = instruments(st, tracker.runway, "rnav")
    for word in ("Airspeed indicator 71 kt", "altimeter 3400 ft", "vertical speed -450", "heading indicator 132",
                 "turn coordinator", "ball 1.2 deg right", "GPS (GNS 430", "XTK", "R of course", "dots ABOVE",
                 "Your hands/feet"):
        assert word in text, word
    for hack in ("AGL", "812", "Wind", "wind 40", "11 kt", "milestone"):
        assert hack not in text, hack


def test_approach_modes_show_only_the_fitted_avionics(tracker):
    from llm_plays.agent import instruments
    # X-Plane sign conventions, measured in the sim: 1500 ft LEFT of the localizer -> hdef +1.86 (needle right),
    # above the glideslope -> vdef +2.5 (needle down)
    st = _state(tracker.runway, -2 * FT_PER_NM, cross_ft=300, nav1_has_loc=1, nav1_has_gs=1, nav1_hdef=-0.8,
                nav1_vdef=-0.5, nav1_has_dme=0, marker_outer=1)
    visual = instruments(st, tracker.runway, "visual")
    assert "GPS" not in visual and "NAV1" not in visual and "XTK" not in visual
    ils = instruments(st, tracker.runway, "ils", {"id": "ISO", "freq": 109.7})
    assert "NAV1 ILS ISO 109.70" in ils and "localizer needle 0.8 dots LEFT (the course is to your left)" in ils
    assert "glideslope needle 0.5 dots UP (the glidepath is above you)" in ils and "OUTER MARKER" in ils
    assert "GPS" not in ils


def test_touchdown_is_felt_not_measured(tracker):
    from llm_plays.agent import instruments
    lat, lon = tracker.runway.from_local(900, 0)
    for sink, feel in ((-150, "smooth"), (-330, "firm"), (-900, "HARD")):
        text = instruments(_state(tracker.runway, 900, hat_ft=0, on_ground=1, events=[
            {"type": "touchdown", "sink_fpm": sink, "bank_deg": 0, "pitch_deg": 3, "ias_kt": 50, "lat": lat,
             "lon": lon}]), tracker.runway)
        assert feel in text and str(abs(sink)) not in text


def test_decision_altitude_rule(airport_data):
    from llm_plays.missions import load_scenarios, runway_visible_at_minimums
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkdtemp()) / "s.yaml"
    p.write_text("""
fog_ok:  {type: landing, airport: SBSP, runway: "35L", approach: ils, weather: {visibility_sm: 1, clouds: [{base_ft_agl: 300, coverage: 1}]}}
fog_bad: {extends: fog_ok, weather: {visibility_sm: 0.25, clouds: [{base_ft_agl: 100, coverage: 1}]}}
""")
    sc = load_scenarios(p, airport_data)
    assert runway_visible_at_minimums(sc["fog_ok"]) and not runway_visible_at_minimums(sc["fog_bad"])
    assert "ILS runway 35L" in sc["fog_bad"]["goal"] and "decision altitude" in sc["fog_bad"]["goal"]
    t = MilestoneTracker(sc["fog_bad"], airport_data)
    rw = t.runway
    t.update(_state(rw, -1.2 * FT_PER_NM, throttle=0.4), 1)
    assert not t.p.failed
    t.update(_state(rw, -0.5 * FT_PER_NM, hat_ft=120, throttle=0.4), 2)          # below DA 200-50, no runway in sight
    assert "decision altitude" in t.p.failed
    ok = MilestoneTracker(sc["fog_ok"], airport_data)
    ok.update(_state(ok.runway, -0.5 * FT_PER_NM, hat_ft=120, throttle=0.4), 1)  # runway visible: may continue
    assert not ok.p.failed


def test_airborne_act_is_capped_at_five_seconds_and_short_final_attaches_the_view(tmp_path, tracker):
    from llm_plays.agent import Campaign, Pilot
    sim = FakeSim(tracker.runway)
    sim.along = -1.0 * FT_PER_NM                 # ~370 ft above the runway: short final
    seen = []
    orig_act = sim.act

    def act(seconds, commands=None, **controls):
        seen.append((seconds, controls.get("trim")))
        return orig_act(seconds, commands, **controls)
    sim.act = act

    class Client(FakeClient):
        def __init__(self):
            super().__init__()
            self.script = [("act", {"trim": 0.1, "seconds": 20}), ("report", {"message": "x", "status": "give_up"})]
    client = Client()
    pilot = Pilot(client, "fake", sim, MISSIONS["land_sbbh13"], Campaign(tmp_path, "t2"), tracker)
    stats = pilot.fly(max_turns=5, max_sim_s=600)
    assert seen == [(5.0, 0.1)]                                   # 20 s request clamped in the air, trim passed
    assert stats.views == 1                                       # view outside attached on short final
    out = [x for x in client.inputs[-1] if isinstance(x, dict) and x.get("type") == "function_call_output"][0]
    assert "clamped from 20 to 5" in out["output"]
    assert any(isinstance(x, dict) and x.get("role") == "user" and any(
        c.get("type") == "input_image" for c in x["content"]) for x in client.inputs[-1])


def test_instruments_show_trim(tracker):
    from llm_plays.agent import instruments
    assert "elevator trim +0.12" in instruments(_state(tracker.runway, -2 * FT_PER_NM, trim=0.12), tracker.runway)


def test_instruments_report_what_happened_during_the_act_and_the_stall_horn(tracker):
    from llm_plays.agent import instruments
    during = {"seconds": 3.0, "ias_min": 41, "ias_max": 60, "agl_min": 280, "pitch_min": -2, "pitch_max": 18,
              "bank_min": -4, "bank_max": 22, "vs_min": -300, "vs_max": 900, "stall_horn_s": 1.4,
              "ias_trend_kt_s": -6.3, "turn_rate_dps": 2.0}
    text = instruments(_state(tracker.runway, -1 * FT_PER_NM, during=during, stall_horn=True,
                              pitch_rate_dps=2.5, roll_rate_dps=-1.0), tracker.runway)
    assert "During the last 3.0 s: airspeed 41-60 kt (trend -6.3 kt/s)" in text
    assert "STALL HORN sounded for 1.4 s" in text and "STALL WARNING HORN" in text
    assert "nose is moving +2.5 deg/s" in text


def test_go_around_is_counted_and_the_attempt_ends_after_the_limit(tmp_path, tracker):
    rw = tracker.runway
    tracker.update(_state(rw, -0.3 * FT_PER_NM, hat_ft=150), 1)       # low on final
    tracker.update(_state(rw, 0.3 * FT_PER_NM, hat_ft=700), 2)        # climbing away = go-around
    assert tracker.p.go_arounds == 1
    tracker.update(_state(rw, -3 * FT_PER_NM, hat_ft=900), 3)          # back on a new final
    tracker.update(_state(rw, -0.2 * FT_PER_NM, hat_ft=120), 3)
    tracker.update(_state(rw, 0.2 * FT_PER_NM, hat_ft=650), 4)
    assert tracker.p.go_arounds == 2 and tracker.score()["go_arounds"] == 2


@pytest.mark.parametrize("during,reason", [
    ({"bank_min": -70, "bank_max": 5, "pitch_min": 0, "pitch_max": 5, "stall_horn_s": 0}, "bank 70"),
    ({"bank_min": 0, "bank_max": 5, "pitch_min": -2, "pitch_max": 35, "stall_horn_s": 0}, "pitch +35"),
    ({"bank_min": 0, "bank_max": 5, "pitch_min": -2, "pitch_max": 10, "stall_horn_s": 4.2}, "stall horn 4.2"),
])
def test_checkride_loss_of_control_fails_the_attempt(tracker, during, reason):
    tracker.update(_state(tracker.runway, -3 * FT_PER_NM, during=during), 1)
    assert reason in tracker.p.failed and tracker.score()["score"] == 0 and tracker.score()["result"] == "FAIL"


def test_checkride_fails_continuing_an_unstable_approach_but_not_a_go_around(tracker, airport_data):
    rw = tracker.runway
    tracker.update(_state(rw, -1.4 * FT_PER_NM, hat_ft=450, ias_kt=95), 1)   # 95 kt at the 500 ft gate: unstable
    assert "stable_500" in tracker.p.unstable_at and not tracker.p.failed
    tracker.update(_state(rw, -0.3 * FT_PER_NM, hat_ft=150), 2)          # below 200 ft, controlled: may continue
    assert not tracker.p.failed
    tracker.update(_state(rw, -0.25 * FT_PER_NM, hat_ft=120, ias_kt=92, vs_fpm=-1300), 3)   # dangerous: fail
    assert "dangerous approach" in tracker.p.failed and "sinking -1300" in tracker.p.failed
    t2 = MilestoneTracker(MISSIONS["land_sbbh13"], airport_data)
    t2.update(_state(rw, -1.4 * FT_PER_NM, hat_ft=450, ias_kt=95), 1)
    t2.update(_state(rw, -1.0 * FT_PER_NM, hat_ft=700), 2)               # went around instead
    assert not t2.p.failed


def test_checkride_lost_approach_and_high_threshold_crossing(tracker, airport_data):
    rw = tracker.runway
    tracker.update(_state(rw, -3 * FT_PER_NM, cross_ft=2 * FT_PER_NM), 1)
    assert "off the centerline" in tracker.p.failed
    t2 = MilestoneTracker(MISSIONS["land_sbbh13"], airport_data)
    t2.update(_state(rw, 500, hat_ft=600), 1)                             # over the runway, still 600 ft up
    assert t2.p.go_arounds == 1 and not t2.p.failed


def test_resume_restores_gates_and_rejudges_a_fail(tracker, airport_data):
    t = MilestoneTracker(MISSIONS["land_sbbh13"], airport_data)
    t.restore({"reached": {"stable_1000": 26}, "unstable_at": ["stable_500"], "go_arounds": 0,
               "failed": "continued an unstable approach below 200 ft instead of going around"})
    assert t.p.failed == "" and t._gates_checked == {"stable_1000", "stable_500"} and t._unstable_since_gate


def test_hard_landing_fails_but_touchdown_is_still_recorded(tracker):
    rw = tracker.runway
    tracker.update(_state(rw, -0.2 * FT_PER_NM, hat_ft=60), 1)
    lat, lon = rw.from_local(1800, 60)
    tracker.update(_state(rw, 1800, 60, hat_ft=0, on_ground=1, gs_kt=50,
                          events=[{"type": "touchdown", "sink_fpm": -943, "bank_deg": 1, "pitch_deg": 2,
                                   "ias_kt": 53, "lat": lat, "lon": lon}]), 2)
    s = tracker.score()
    assert "hard landing" in s["failed"] and s["result"] == "FAIL" and s["touchdown"]["sink_fpm"] == -943


def test_flare_cap_one_second_below_50_ft(tmp_path, tracker):
    from llm_plays.agent import Campaign, Pilot
    sim = FakeSim(tracker.runway)
    seen = []
    orig = sim.act
    sim.act = lambda seconds, commands=None, **c: (seen.append(seconds), orig(seconds, commands, **c))[1]

    class Client(FakeClient):
        def __init__(self):
            super().__init__()
            self.script = [("act", {"seconds": 3}), ("act", {"seconds": 3}),
                           ("report", {"message": "x", "status": "give_up"})]
    pilot = Pilot(Client(), "fake", sim, MISSIONS["land_sbbh13"], Campaign(tmp_path, "t3"), tracker)
    pilot._last_agl = 30.0                                     # as if the previous state was 30 ft AGL
    sim.along = -0.05 * FT_PER_NM
    pilot.fly(max_turns=5, max_sim_s=600)
    assert seen[0] == 1.0


def test_every_scenario_resolves_to_real_runways(airport_data):
    from llm_plays.geometry import runway_from_db
    from llm_plays.missions import load_scenarios
    scenarios = load_scenarios(airport_data=airport_data)
    assert len(scenarios) >= 10
    for name, m in scenarios.items():
        s = m["setup"]
        runway_from_db(s["airport"], s["runway"], airport_data)            # start runway exists
        runway_from_db(*m["destination"], airport_data)                     # destination runway exists
        assert m["goal"] and m["ladder"] and m["max_turns"] > 0, name


def test_extends_overrides_and_weather_report(tmp_path, airport_data):
    from llm_plays.missions import load_scenarios, weather_report
    p = tmp_path / "s.yaml"
    p.write_text("""
base: {type: landing, airport: SBBH, runway: "13", start: {distance_nm: 5}, limits: {max_turns: 90}}
windy: {extends: base, weather: {wind_from_deg: 21, wind_kt: 10, gust_kt: 5}, limits: {max_turns: 50}}
far: {extends: base, start: {distance_nm: 8}}
""")
    sc = load_scenarios(p, airport_data)
    assert sc["windy"]["max_turns"] == 50 and sc["windy"]["setup"]["distance_nm"] == 5
    assert "wind 021 at 10 kt gusting 15" in sc["windy"]["goal"]
    assert sc["far"]["setup"]["distance_nm"] == 8 and "8 nm final" in sc["far"]["goal"]
    assert weather_report({"visibility_sm": 3, "clouds": [{"base_ft_agl": 900, "coverage": 0.9}], "rain": 0.3}) \
        == "wind calm, visibility 3 sm, overcast 900 ft, rain"


def test_go_around_from_500_ft_is_not_a_lost_approach(tracker):
    rw = tracker.runway
    tracker.update(_state(rw, -1.4 * FT_PER_NM, hat_ft=850, throttle=0.4), 1)          # high, unstable
    tracker.update(_state(rw, -1.2 * FT_PER_NM, hat_ft=700, throttle=1.0, vs_fpm=1000), 2)   # full power, climbing
    tracker.update(_state(rw, -0.9 * FT_PER_NM, hat_ft=1300, throttle=1.0, vs_fpm=1100), 3)  # 800+ ft "high"
    assert not tracker.p.failed and tracker.p.go_arounds == 1


def test_below_minimums_the_missed_approach_is_the_pass(airport_data):
    m = dict(MISSIONS["land_sbsp35l_ils_below_mins"])
    assert m["expect_missed"] and m["ladder"] == ["stable_1000", "missed_approach_at_minimums"]
    assert not MISSIONS["land_sbsp35l_ils_fog"]["expect_missed"]
    t = MilestoneTracker(m, airport_data)
    rw = t.runway
    t.update(_state(rw, -3 * FT_PER_NM, throttle=0.4), 1)
    t.update(_state(rw, -0.7 * FT_PER_NM, hat_ft=215, throttle=0.4), 2)                # at the DA, nothing in sight
    new = t.update(_state(rw, -0.6 * FT_PER_NM, hat_ft=260, throttle=1.0, vs_fpm=700), 3)  # missed approach
    assert "missed_approach_at_minimums" in new
    t.update(_state(rw, 500, hat_ft=600, throttle=1.0, vs_fpm=700), 4)                 # overflies the runway high
    assert not t.p.failed and t.p.go_arounds == 0


def test_needles_read_centered_near_zero(tracker):
    from llm_plays.agent import instruments
    st = _state(tracker.runway, -2 * FT_PER_NM, nav1_has_loc=1, nav1_has_gs=1, nav1_hdef=0.01, nav1_vdef=-0.02)
    ils = instruments(st, tracker.runway, "ils", {"id": "ISO", "freq": 109.7})
    assert "localizer needle centered" in ils and "glideslope needle centered" in ils
    lost = instruments(dict(st, nav1_has_loc=0), tracker.runway, "ils", {"id": "ISO", "freq": 109.7})
    assert "localizer: flag" in lost and "glideslope: flag" in lost


def test_sim_notification_boxes_are_painted_over():
    from PIL import Image, ImageDraw
    from llm_plays.overlays import hide_sim_overlays
    im = Image.new("RGB", (1280, 536), (180, 176, 172))                      # fog
    d = ImageDraw.Draw(im)
    d.rectangle((0, 360, 1280, 536), fill=(2, 1, 0))                         # glareshield / panel
    d.rectangle((1100, 380, 1280, 420), fill=(22, 25, 27))                   # dark avionics, runs to the edge
    for y in (16, 133):                                                      # two stacked notifications
        d.rectangle((985, y, 1262, y + 110), fill=(21, 25, 26))
        d.text((995, y + 10), "The following scenery package has a problem", fill=(255, 255, 255))
        d.rectangle((1195, y + 88, 1255, y + 102), fill=(20, 110, 200))      # "Understood"
    d.text((1140, 3), "PAUSED", fill=(255, 255, 255))
    out, n = hide_sim_overlays(im)
    a, b = np.asarray(im).astype(int), np.asarray(out).astype(int)
    assert n >= 1
    fog = np.array([180, 176, 172])
    assert np.abs(b[5:250, 985:1262] - fog).max() < 12                       # boxes and text gone
    assert np.abs(b[0:16, 1100:1280] - fog).max() < 12                       # PAUSED gone
    assert (a[360:] == b[360:]).all()                                        # panel untouched


def test_atis_can_differ_from_the_real_weather():
    m = MISSIONS["land_sbsp35l_ils_below_mins"]
    assert "visibility 0.5 sm, overcast 200 ft" in m["goal"]          # the pilot was told: legal to start
    assert "0.25" not in m["goal"] and "100 ft" not in m["goal"]
    assert m["expect_missed"]                                          # judged on the real weather
    assert m["weather"]["clouds"][0]["base_ft_agl"] == 100


def test_poh_teaches_the_flap_pitch_up():
    from llm_plays.agent import SYSTEM
    assert "one notch at a time" in SYSTEM and "lower the nose" in SYSTEM


def test_off_nominal_start_is_detected(airport_data):
    from llm_plays.missions import _start_off_nominal
    m = MISSIONS["land_sbsp35l_ils_fog"]
    t = MilestoneTracker(m, airport_data)
    rw = t.runway
    good = _state(rw, -4.7 * FT_PER_NM, ias_kt=78, vs_fpm=-430)
    assert _start_off_nominal(good, m, rw) == ""
    # the bad start seen after an X-Plane restart: 94 kt, ~430 ft low, zooming
    bad = _state(rw, -4.4 * FT_PER_NM, hat_ft=rw.glidepath_height_ft(-4.4 * FT_PER_NM) - 430, ias_kt=94, vs_fpm=764)
    assert "dots off the glidepath" in _start_off_nominal(bad, m, rw) and "94 kt" in _start_off_nominal(bad, m, rw)


def test_early_go_around_is_not_the_missed_approach_at_minimums(airport_data):
    # gpt-6-sol v2: unstable (flap balloon), went around at 768 ft - that is not "missed approach at the DA"
    m = MISSIONS["land_sbsp35l_ils_below_mins"]
    t = MilestoneTracker(m, airport_data)
    rw = t.runway
    t.update(_state(rw, -1.74 * FT_PER_NM, hat_ft=731, throttle=0.49, vs_fpm=-350), 1)
    new = t.update(_state(rw, -1.65 * FT_PER_NM, hat_ft=768, throttle=1.0, vs_fpm=1062), 2)
    assert "missed_approach_at_minimums" not in new and "missed_approach_at_minimums" not in t.p.reached


def test_empty_text_parts_from_the_proxy_are_tolerated():
    from openai.types.responses import ResponseOutputMessage, ResponseFunctionToolCall
    from openai.types.responses.response_output_text import ResponseOutputText
    from llm_plays.agent import _clean_output, _output_text
    empty = ResponseOutputText.model_construct(type="output_text", text=None, annotations=[])
    said = ResponseOutputText.model_construct(type="output_text", text="Runway ahead.", annotations=[])
    msg = ResponseOutputMessage.model_construct(type="message", id="m", role="assistant", status="completed",
                                                content=[empty, said])
    only_empty = ResponseOutputMessage.model_construct(type="message", id="m2", role="assistant",
                                                       status="completed", content=[empty])
    call = ResponseFunctionToolCall.model_construct(type="function_call", call_id="c", name="look", arguments="{}")
    out = [only_empty, msg, call]
    assert _output_text(out) == "Runway ahead."
    cleaned = _clean_output(out)
    assert len(cleaned) == 2 and [c.text for c in cleaned[0].content] == ["Runway ahead."]


def test_block_history_keeps_the_prefix_stable_and_never_shows_less():
    from llm_plays.agent import HISTORY_BLOCK, Pilot
    p = Pilot.__new__(Pilot)
    p.first, p.keep_turns, p.history, p._base, p.cache_marks = [{"role": "user", "content": []}], 24, "blocks", 0, False
    p.groups = []
    prefixes = []
    for t in range(60):
        p.groups.append([{"type": "function_call_output", "call_id": str(t), "output": str(t)}])
        inp = p._input()
        assert len(inp) - 1 >= min(t + 1, 24)                       # never fewer than keep_turns turns
        prefixes.append(inp[1]["call_id"])
    changes = sum(1 for a, b in zip(prefixes, prefixes[1:]) if a != b)
    assert changes <= 60 // HISTORY_BLOCK                           # the start changes once per block, not per turn


def test_cache_mark_goes_on_a_copy_of_the_last_message():
    from llm_plays.agent import _mark_cache
    msg = {"role": "user", "content": [{"type": "input_text", "text": "(sim paused)"}]}
    items = [{"role": "user", "content": [{"type": "input_text", "text": "Begin"}]},
             {"type": "function_call_output", "call_id": "1", "output": "x"}, msg]
    _mark_cache(items)
    assert items[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in msg["content"][-1]                 # stored history stays unmarked


def test_kept_screenshots_never_edit_a_sent_request_between_trims():
    import json
    from llm_plays.agent import Pilot
    p = Pilot.__new__(Pilot)
    p.first, p.keep_turns, p.history, p._base, p.cache_marks = [{"role": "user", "content": []}], 24, "blocks", 0, False
    p.images, p._img_from, p.groups = "keep", 0, []
    img = lambda t: {"role": "user", "content": [{"type": "input_text", "text": f"view {t}"},
                                                  {"type": "input_image", "image_url": f"data:{t}"}]}
    prev, extended, trims = None, 0, 0
    for t in range(50):
        p.groups.append([{"type": "function_call_output", "call_id": str(t), "output": str(t)}, img(t)])
        cur = [json.dumps(x, sort_keys=True) for x in p._input()]
        if prev is not None:
            if cur[:len(prev)] == prev:
                extended += 1                       # this request extends the previous one unchanged: cache hit
            else:
                trims += 1
        prev = cur
    assert trims <= 3 and extended >= 45
    # right after a trim only the newest screenshot is shown
    shown = [x for x in p._input() if isinstance(x, dict) and any(c.get("type") == "input_image" for c in x.get("content", []))]
    assert 1 <= len(shown) <= 13
