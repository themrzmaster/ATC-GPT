"""Cross-country flow fixes from the 2026-11-07 live flight SBCF -> SBCB (BE58, IFR, pt-br)."""
import json

PLAN_SBCB = {"aircraft_type": "BE58", "flight_rules": "IFR", "departure": "SBCF", "destination": "SBCB",
             "route": "MAVGU UZ24 VUNLI", "cruise_altitude_ft": 10000}
AIR = {"altitude_ft": 9000, "on_ground": False, "groundspeed": 190, "heading": 120, "com1_freq_hz": 120200000}


def _tune(api, **kw):
    body = {"airport_icao": "SBCF", "callsign": "N45XS", "frequency": "Approach", "language": "pt-br",
            "flight_plan": PLAN_SBCB}
    body.update(kw)
    return api.post("/tune", json=body).get_json()


def test_ifr_arrival_at_afis_field_gets_area_controller(make_session):
    """Cabo Frio publishes only AFIS: the IFR arrival was told "no ATC service, self-announce on 123.45"."""
    s = make_session("SBCB", "N45XS", language="pt-br", **PLAN_SBCB)
    s.update_aircraft_state(latitude=-22.8, longitude=-42.4, altitude_ft=6000, on_ground=False)
    text = s.get_situation_summary()
    assert s.area_control() and "area radar controller" in text and "NO ATC SERVICE" not in text
    assert "132.150" in text                                  # the AFIS is named, not invented

    vfr = make_session("SBCB", "PT-ABC", language="pt-br", flight_rules="VFR", destination="SBCB")
    vfr.update_aircraft_state(latitude=-22.8, longitude=-42.4, altitude_ft=3000, on_ground=False)
    assert not vfr.area_control() and "NO ATC SERVICE" in vfr.get_situation_summary()


def test_arrival_moves_session_to_destination_without_atc(api):
    _tune(api)
    sess = api.server.session_manager.get_active_session()
    sess.add_pilot_transmission("Belo Horizonte, N45XS, nível 100")
    api.post("/state", json={"latitude": -21.5, "longitude": -43.9, **AIR})          # far: stays at SBCF
    assert api.server.session_manager.get_active_session().airport_icao == "SBCF"
    api.post("/state", json={"latitude": -22.75, "longitude": -42.55, **AIR})        # ~28 nm W of SBCB
    new = api.server.session_manager.get_active_session()
    assert new.airport_icao == "SBCB" and new.phase.value == "approach" and new.area_control()
    assert any("nível 100" in m.content for m in new.conversation)                  # same controller, same story


def test_destination_with_approach_gets_real_handoff(api):
    plan = dict(PLAN_SBCB, destination="SBGL")
    _tune(api, flight_plan=plan)
    r = api.post("/state", json={"latitude": -22.35, "longitude": -43.25, **AIR}).get_json()
    call = r["pending_atc"]
    assert call and call["trigger_type"] == "handoff" and abs(call["new_frequency_mhz"] - 119.35) < 0.01
    assert "chame Controle" in call["message"] and "um um nove decimal três cinco" in call["message"]
    assert api.server.session_manager.get_active_session().airport_icao == "SBCF"   # until the pilot retunes


def test_no_tower_handoff_for_area_control(make_session):
    from atc_triggers import ATCTriggerEngine
    from session_manager import ATCPhase, ClearanceType
    s = make_session("SBCB", "N45XS", language="pt-br", **PLAN_SBCB)
    s.set_phase(ATCPhase.APPROACH)
    s.update_aircraft_state(latitude=-22.92, longitude=-42.15, altitude_ft=1500, on_ground=False)
    s.add_clearance(ClearanceType.APPROACH, "autorizado aproximação RNP Z pista 10")
    assert ATCTriggerEngine()._check_tower_handoff(s, ATCTriggerEngine().get_trigger_state(s)) is None


def test_auto_tune_in_the_air_uses_flight_plan_airport(api):
    """Server restart mid-flight: the plugin re-tune landed on São João del-Rei (SNJR), plan lost."""
    near_snjr = {"latitude": -21.08, "longitude": -44.11, **AIR}
    r = api.post("/tune", json={"airport_icao": "auto", "callsign": "N45XS", "frequency": "auto",
                                "language": "pt-br", "flight_plan": PLAN_SBCB, "state": near_snjr}).get_json()
    assert r["session"]["airport_icao"] == "SBCF" and r["session"]["phase"] == "approach"
    on_ground = dict(near_snjr, on_ground=True, groundspeed=0)
    r = api.post("/tune", json={"airport_icao": "auto", "callsign": "N45XS", "frequency": "auto",
                                "flight_plan": PLAN_SBCB, "state": on_ground}).get_json()
    assert r["session"]["airport_icao"] != "SBCF" and r["session"]["phase"] == "ground"


def test_flight_plan_survives_retune(api, tmp_path, monkeypatch):
    monkeypatch.setattr(api.server.config, "FLIGHT_PLAN_STORE", str(tmp_path / "plans.json"))
    _tune(api)
    r = api.post("/tune", json={"airport_icao": "auto", "callsign": "N45XS", "frequency": "auto",
                                "flight_plan": {"aircraft_type": "BE58"},
                                "state": {"latitude": -21.08, "longitude": -44.11, **AIR}}).get_json()
    assert r["session"]["flight_plan"]["destination"] == "SBCB"
    assert json.loads((tmp_path / "plans.json").read_text())["N45XS"]["plan"]["route"] == "MAVGU UZ24 VUNLI"


def test_fms_plan_on_state_updates_session(api):
    api.post("/tune", json={"airport_icao": "SBCF", "callsign": "N45XS", "frequency": "Ground"})
    api.post("/state", json={"latitude": -19.63, "longitude": -43.97, "on_ground": True,
                             "flight_plan": {"departure": "SBCF", "destination": "SBCB", "route": "MAVGU UZ24 VUNLI",
                                             "cruise_altitude_ft": 10000, "flight_rules": "IFR"}})
    fp = api.server.session_manager.get_active_session().flight_plan
    assert (fp.destination, fp.flight_rules, fp.cruise_altitude_ft) == ("SBCB", "IFR", 10000)


def test_fms_entries_to_plan():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "xplane_plugin"))
    from PI_VirtualATC import fms_plan_from_entries
    e = [(1, "SBCF", 0), (512, "MAVGU", 10000), (512, "VUNLI", 10000), (1, "SBCB", 0)]
    assert fms_plan_from_entries(e) == {"departure": "SBCF", "destination": "SBCB", "route": "MAVGU VUNLI",
                                        "cruise_altitude_ft": 10000}
    assert fms_plan_from_entries([(1, "SBCF", 0)]) is None
    assert fms_plan_from_entries([(512, "CB003", 3000), (1, "SBCB", 0)])["departure"] == ""


def test_aguarde_is_not_a_hold():
    from clearance_parser import parse_clearances
    from session_manager import ClearanceType
    assert not [c for c in parse_clearances("N45XS, aguarde.", "N45XS") if c.type == ClearanceType.HOLD]
    assert [c for c in parse_clearances("N45XS, mantenha posição.", "N45XS") if c.type == ClearanceType.HOLD]


class _FakeCharts:
    def __init__(self):
        self.fetches = 0

    def fetch_pdf(self, icao, chart_id):
        self.fetches += 1
        return {"name": "RNP Z RWY 10"}, b"%PDF-fake"

    def get_charts_by_category(self, icao, cat):
        return [{"id": "abc", "name": "RNP Z RWY 10", "rwys": ["10"]}]


class _FakeResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        assert kw["input"][0]["content"][0]["type"] == "input_file"
        return type("R", (), {"output_text": "IAF CB003 6800 ft; FAF 2000 ft; MDA 560 ft"})()


def test_chart_digest_cached(tmp_path, monkeypatch, make_session):
    import atc_llm_service as m
    monkeypatch.setattr(m, "CHART_DIGEST_DIR", tmp_path)
    svc = m.ATCLLMService.__new__(m.ATCLLMService)
    svc.charts, svc.model, svc.reasoning_effort, svc.last_tool_calls = _FakeCharts(), "fake", None, []
    svc.client = type("C", (), {"responses": _FakeResponses()})()
    s = make_session("SBCB", "N45XS")
    out = json.loads(svc._execute_tool("read_chart", {"icao": "SBCB", "chart_id": "abc"}, s))
    assert "CB003" in out["digest"]
    svc.prefetch_chart_digests("SBCB")                       # already on disk: no second model call
    json.loads(svc._execute_tool("read_chart", {"icao": "SBCB", "chart_id": "abc"}, s))
    assert svc.client.responses.calls == 1 and svc.charts.fetches == 1


def test_nearby_metar_skips_non_icao_idents(make_session):
    import atc_llm_service as m
    asked = []

    class AWC:
        def get_metar_taf(self, icao, ap):
            raise RuntimeError("timeout")

        def metars(self, ids):
            asked.extend(ids)
            return {}

    class WX:
        def get_metar_taf(self, icao):
            raise RuntimeError("500")

    from airport_data_service import AirportDataService
    svc = m.ATCLLMService.__new__(m.ATCLLMService)
    svc.awc, svc.wx, svc._airport_data = AWC(), WX(), AirportDataService()
    svc._weather_with_fallback("SBCF")
    assert asked and all(len(i) == 4 and i.isalpha() for i in asked)


def test_no_frequency_handoff_does_not_loop(api):
    """Live: Approach -> Tower at a field with no Tower frequency repeated "permaneça nesta frequência,
    contato radar" every 30 s because COM1 kept snapping the session back to Approach."""
    from datetime import datetime, timedelta
    from session_manager import ClearanceType
    api.post("/tune", json={"airport_icao": "SBME", "callsign": "N45XS", "frequency": "Approach",
                            "language": "pt-br", "flight_plan": dict(PLAN_SBCB, destination="SBME")})
    sess = api.server.session_manager.get_active_session()
    if sess.get_handoff_frequency(sess.phase.__class__.TOWER):
        sess.airport_frequencies.pop("TWR")            # make it a field with Approach but no Tower
    sess.add_clearance(ClearanceType.APPROACH, "autorizado aproximação pista 06")
    app_hz = int(round(sess.current_frequency_mhz * 1e6))
    st = {"latitude": sess.airport.latitude + 0.05, "longitude": sess.airport.longitude, "altitude_ft": 1500,
          "on_ground": False, "groundspeed": 100, "com1_freq_hz": app_hz}
    for _ in range(3):
        r = api.post("/state", json=st).get_json()
        assert r["pending_atc"] is None
        api.server.trigger_engine.get_trigger_state(sess).last_handoff_time = datetime.now() - timedelta(minutes=2)


def test_arrival_transfer_does_not_ping_pong(api):
    """Live 2026-11-08 SBRJ->SBCB: on Rio APP 119.35 the transfer to SBCB was undone by the radio-follow
    (119.35 = SBRJ APP) every 2 s, losing the conversation on each bounce."""
    _tune(api, airport_icao="SBRJ", flight_plan=dict(PLAN_SBCB, departure="SBRJ"))
    api.server.session_manager.get_active_session().add_pilot_transmission("Controle Rio, N45XS, 7000 pés")
    st = {"latitude": -22.87, "longitude": -42.75, "altitude_ft": 7000, "on_ground": False,
          "groundspeed": 150, "com1_freq_hz": 119350000}
    for _ in range(3):
        api.post("/state", json=st)
    s = api.server.session_manager.get_active_session()
    assert s.airport_icao == "SBCB" and any("7000" in m.content for m in s.conversation)
