"""Position guard: clearances a position may not issue are caught in code, not left to the prompt."""

import pytest

import position_rules
from session_manager import ATCPhase, ClearanceType, FlightPlan


@pytest.mark.parametrize("phase,text,bad", [
    (ATCPhase.APPROACH, "N54321, roger emergency, cleared to land runway zero four right", True),
    (ATCPhase.APPROACH, "N54321, cleared ILS runway zero four right approach, maintain three thousand", False),
    (ATCPhase.GROUND, "N123AB, runway one six, cleared for takeoff", True),
    (ATCPhase.GROUND, "N123AB, taxi to runway one six via Alpha, hold short", False),
    (ATCPhase.CLEARANCE, "N123AB, taxi to runway one six via Alpha", True),
    (ATCPhase.TOWER, "N123AB, runway one six, cleared to land", False),
    (ATCPhase.DEPARTURE, "N123AB, contact Departure on one three five decimal niner", True),   # itself
])
def test_forbidden_clearances_by_position(make_session, phase, text, bad):
    s = make_session("KJFK" if "Departure" in text else "KBOS", "N54321" if "N54321" in text else "N123AB")
    s.set_phase(phase)
    assert bool(position_rules.check(s, text)) is bad


def test_sector_handoff_between_approach_frequencies_is_allowed(make_session):
    s = make_session("KBOS", "N54321")
    s.set_phase(ATCPhase.APPROACH)       # current APP frequency is one of several published sectors
    other = "one two seven decimal two" if s.current_frequency_mhz != 127.2 else "one two zero decimal six"
    assert not position_rules.check(s, f"N54321, contact Boston Approach {other}")


def test_tower_may_not_send_circuit_traffic_to_approach(make_session):
    s = make_session("SBMT", "PR-XYZ", language="pt-br", aircraft_type="C152", destination="SBMT",
                     remarks="circuito")
    s.set_phase(ATCPhase.TOWER)
    s.update_aircraft_state(latitude=-23.49, longitude=-46.63, altitude_ft=3400, on_ground=False)
    assert position_rules.check(s, "PR-XYZ, chame Controle em um um nove decimal oito.")
    # after a go-around the same handoff is legitimate (e.g. IFR missed approach)
    s.add_clearance(ClearanceType.GO_AROUND, "PR-XYZ, arremeta")
    assert not position_rules.check(s, "PR-XYZ, chame Controle em um um nove decimal oito.")


def test_fallback_hands_off_to_the_owner(make_session):
    s = make_session("KBOS", "N54321")
    s.set_phase(ATCPhase.APPROACH)
    v = position_rules.check(s, "N54321, cleared to land runway four right")
    msg = position_rules.fallback(s, v)
    assert msg and "contact Tower" in msg


class _Resp:
    def __init__(self, text):
        self.output_text, self.output, self.usage = text, [], None


class _Client:
    """Scripted Responses API: first the model breaks the rule, then it fixes (or doesn't)."""
    def __init__(self, replies):
        self.replies, self.inputs = list(replies), []
        self.responses = self

    def create(self, **kw):
        self.inputs.append(kw.get("input"))
        return _Resp(self.replies.pop(0))


def _service(replies):
    from atc_llm_service import ATCLLMService
    svc = ATCLLMService(client=_Client(replies))
    svc._prefetch_weather = lambda session: None
    return svc


def test_guard_regenerates_with_the_reason(make_session):
    s = make_session("KBOS", "N54321")
    s.set_phase(ATCPhase.APPROACH)
    svc = _service(["N54321, cleared to land runway four right.",
                    "N54321, cleared ILS runway four right approach, contact Tower one two eight decimal eight."])
    out = svc.generate_response("Boston Approach, N54321, established ILS four right", s)
    assert "cleared ILS" in out and "position_guard" in svc.last_tool_calls
    assert "may not issue a landing clearance" in str(svc.client.inputs[-1])
    assert not s.has_clearance(ClearanceType.LANDING)


def test_guard_falls_back_to_handoff_when_model_insists(make_session):
    s = make_session("KHPN", "N123AB")
    s.set_phase(ATCPhase.GROUND)
    svc = _service(["N123AB, runway one six, cleared for takeoff.", "N123AB, cleared for takeoff runway one six."])
    out = svc.generate_response("Westchester Ground, N123AB, holding short one six, ready", s)
    assert out.startswith("N123AB, contact Tower on one one eight decimal five seven five")
    assert not s.has_clearance(ClearanceType.TAKEOFF) and s.pending_handoff == ATCPhase.TOWER


def test_outbound_departure_levelling_off_is_not_handed_back_to_approach(make_session):
    from atc_triggers import ATCTriggerEngine
    s = make_session("KJFK", "N12345", aircraft_type="C172", flight_rules="IFR", destination="KBOS")
    s.set_phase(ATCPhase.DEPARTURE)
    eng = ATCTriggerEngine()
    for lat in (40.70, 40.90, 41.00, 41.05):          # out past 20 nm, then level cruise inside 25 nm
        s.update_aircraft_state(latitude=lat, longitude=-73.60, altitude_ft=7000, vertical_speed=0,
                                on_ground=False, groundspeed=180)
        assert eng.evaluate(s) is None


def test_inbound_arrival_still_handed_to_approach(make_session):
    from atc_triggers import ATCTriggerEngine
    s = make_session("KBOS", "N54321", aircraft_type="PC12", flight_rules="IFR", destination="KBOS")
    s.set_phase(ATCPhase.CENTER)
    eng = ATCTriggerEngine()
    s.update_aircraft_state(latitude=41.95, longitude=-71.45, altitude_ft=9000, vertical_speed=-800, on_ground=False)
    assert eng.evaluate(s) is None                     # ~31 nm: outside the ring
    s.update_aircraft_state(latitude=42.10, longitude=-71.30, altitude_ft=8500)
    call = eng.evaluate(s)                             # 20 nm and closing
    assert call and call.new_phase == ATCPhase.APPROACH
