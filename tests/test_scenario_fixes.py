"""Offline tests for the fixes that came out of the 2026-09-28 e2e scenario sweep."""

import pytest

from airport_data_service import Runway
from clearance_parser import frequencies_in, parse_clearances
from session_manager import ATCPhase, ClearanceType, FlightPlan
from weather_service import WeatherService, _altimeter_repr, active_runways


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "PT-ABC, autorizado SBKP via direto, suba e mantenha seis mil pés, transponder quatro cinco dois um.",
    "PT-MZA, autorizado Santos Dumont, rota direta, saída OMNI, subida inicial cinco mil pés, transponder 4521.",
    "PT-ABC, autorizado ao aeródromo de Campinas conforme plano, transponder 2301.",
])
def test_portuguese_ifr_clearance_is_recognised(text):
    assert ClearanceType.IFR in {c.type for c in parse_clearances(text, "PT-ABC")}


@pytest.mark.parametrize("text", [
    "PT-ABC, autorizado decolagem pista um zero esquerda, saída OMNI, transponder 4521",
    "PT-ABC, autorizado pouso pista dois oito direita",
    "PT-ABC, autorizado a decolagem pista 10L, mantenha seis mil",
    "PT-ABC, autorizado ILS pista 28R, mantenha quatro mil até estabelecido",
])
def test_portuguese_non_ifr_autorizado_is_not_an_ifr_clearance(text):
    assert ClearanceType.IFR not in {c.type for c in parse_clearances(text, "PT-ABC")}


@pytest.mark.parametrize("text", [
    "N54321, hold present position, right turns, maintain seven thousand, expect further clearance at 1435",
    "N54321, hold as published at BOSOX, maintain six thousand",
    "PT-ABC, mantenha espera sobre o VOR, curvas à direita",
])
def test_holding_instructions_are_parsed(text):
    assert ClearanceType.HOLD in {c.type for c in parse_clearances(text, "N54321")}


def test_frequencies_in_written_and_spoken():
    assert frequencies_in("One two one decimal seven, PT-ABC") == [121700]
    assert frequencies_in("Tower 118.575, N123AB") == [118575]
    assert frequencies_in("um três dois decimal sete cinco, PT-ABC") == [132750]
    assert frequencies_in("descend and maintain three thousand, squawk one two zero zero") == []


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def test_handoff_readback_is_detected_and_does_not_move_session(make_session):
    s = make_session("SBGR", "PT-ABC", language="pt-br")
    s.set_phase(ATCPhase.CLEARANCE)
    reply = "PT-ABC, colação correta, chame Solo em um dois um decimal sete."
    for pc in parse_clearances(reply, "PT-ABC"):
        s.add_clearance(pc.type, reply, position=pc.position, frequency_khz=pc.frequency_khz)
    assert s.pending_handoff == ATCPhase.GROUND
    assert s.is_handoff_readback("Um dois um decimal sete, PT-ABC")
    assert not s.is_handoff_readback("Solo Guarulhos, PT-ABC, pronto para o táxi")
    assert s.phase == ATCPhase.CLEARANCE          # readback happens on the old frequency


def test_llm_service_stays_silent_on_handoff_readback(make_session):
    from atc_llm_service import ATCLLMService

    class NoCallsClient:
        @property
        def responses(self):
            raise AssertionError("the model must not be called for a handoff readback")

    s = make_session("KHPN", "N123AB")
    reply = "N123AB, contact Tower one one eight decimal five seven five."
    for pc in parse_clearances(reply, "N123AB"):
        s.add_clearance(pc.type, reply, position=pc.position, frequency_khz=pc.frequency_khz)
    llm = ATCLLMService(client=NoCallsClient())
    assert llm.generate_response("One one eight decimal five seven five, N123AB", s) == ""
    assert s.pending_handoff == ATCPhase.TOWER    # still pending: the next call is to Tower
    assert s.conversation[-1].role == "pilot"


@pytest.mark.parametrize("squawk,word", [(7700, "EMERGENCY"), (7600, "RADIO FAILURE"), (7500, "UNLAWFUL")])
def test_emergency_squawk_raises_radar_alert(make_session, squawk, word):
    s = make_session("KBOS", "N54321")
    s.update_aircraft_state(latitude=42.1, longitude=-71.3, altitude_ft=7000, squawk=squawk, on_ground=False)
    assert "RADAR ALERT" in s.get_situation_summary() and word in s.get_situation_summary()


def test_no_radar_alert_for_normal_squawk(make_session):
    s = make_session("KBOS", "N54321")
    s.update_aircraft_state(squawk=4521)
    assert "RADAR ALERT" not in s.get_situation_summary()


def test_uncontrolled_field_is_flagged(make_session):
    s = make_session("SDTK", "PR-ABC", language="pt-br")
    assert not s.has_atc_service()
    assert "NO ATC SERVICE" in s.get_situation_summary()
    assert make_session("SBMT", "PR-ABC").has_atc_service()


def test_flight_plan_with_only_departure_is_empty(make_session):
    s = make_session("KHPN", "N123AB")          # session fills departure=KHPN
    assert s.flight_plan.departure == "KHPN"
    assert s.flight_plan.is_empty()
    assert "No flight plan on file" in s.get_situation_summary()
    assert not FlightPlan(aircraft_type="C172").is_empty()


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def _rwy(low, high, hl, hh):
    return Runway(low, high, 10000, 150, "ASP", hl, hh)


def test_active_runways_into_wind():
    sbgr = [_rwy("10L", "28R", 95, 275), _rwy("10R", "28L", 95, 275)]
    assert active_runways(60, 10, sbgr) == ["10L", "10R"]
    assert active_runways(270, 12, sbgr) == ["28L", "28R"]
    assert active_runways(60, 2, sbgr) == []            # calm: no preference
    assert active_runways("VRB", 4, sbgr) == []


def test_active_runways_without_true_headings_uses_ident():
    assert active_runways(350, 8, [_rwy("16", "34", None, None)]) == ["34"]


def test_altimeter_units_by_region():
    assert _altimeter_repr(1009.6, "KJFK") == "A2981"
    assert _altimeter_repr(1018, "SBGR") == "Q1018"
    assert _altimeter_repr(None, "SBGR") is None


def test_awc_record_is_normalised(airport_data):
    rec = {"icaoId": "SBGR", "rawOb": "METAR SBGR 281300Z 06010KT 9999 Q1018", "wdir": 60, "wspd": 10,
           "altim": 1018, "fltCat": "VFR"}
    w = WeatherService._normalise("SBGR", rec, airport_data.get("SBGR"))
    assert w["metar"]["raw"].startswith("METAR SBGR")
    assert w["metar"]["altimeter"]["repr"] == "Q1018"
    assert w["active_runways"] == ["10L", "10R"]


# ---------------------------------------------------------------------------
# Server: the radio follows the pilot to another airport
# ---------------------------------------------------------------------------

def test_com1_on_another_airports_tower_moves_the_session(api):
    api.post("/tune", json={"airport_icao": "SBGR", "callsign": "PT-FXQ", "frequency": "Solo",
                            "language": "pt-br", "flight_plan": {"destination": "SBKP"}})
    r = api.post("/state", json={"latitude": -23.05, "longitude": -47.02, "altitude_ft": 4500,
                                 "on_ground": False, "com1_freq_hz": 118_250_000}).get_json()
    assert r["phase"] == "tower"
    s = api.get("/session").get_json()
    assert s["airport_icao"] == "SBKP"
    assert s["callsign"] == "PT-FXQ" and s["language"] == "pt-br"
    assert s["flight_plan"]["destination"] == "SBKP"


def test_shared_approach_frequency_does_not_move_the_session(api):
    # 119.8 is São Paulo APP at SBGR, SBSP and SBMT alike: stay at the tuned airport
    api.post("/tune", json={"airport_icao": "SBMT", "callsign": "PR-XYZ", "frequency": "Torre", "language": "pt-br"})
    api.post("/state", json={"latitude": -23.45, "longitude": -46.60, "altitude_ft": 4500,
                             "on_ground": False, "com1_freq_hz": 119_800_000})
    s = api.get("/session").get_json()
    assert s["airport_icao"] == "SBMT" and s["phase"] == "approach"


def test_unknown_frequency_far_from_everything_keeps_session(api):
    api.post("/tune", json={"airport_icao": "KHPN", "callsign": "N123AB"})
    api.post("/state", json={"latitude": 41.07, "longitude": -73.70, "com1_freq_hz": 122_800_000})
    assert api.get("/session").get_json()["airport_icao"] == "KHPN"


def test_frequencies_in_ignores_headings_and_keeps_round_frequencies():
    assert frequencies_in("N123AB, turn heading one two zero, contact tower 118.575", "N123AB") == [118575]
    assert frequencies_in("One two one decimal zero, N777AB", "N777AB") == [121000]   # SBGR delivery


# ---------------------------------------------------------------------------
# Second sweep: readback kinds, addressed station, malformed output
# ---------------------------------------------------------------------------

def _pending(s, reply):
    for pc in parse_clearances(reply, s.callsign):
        s.add_clearance(pc.type, reply, position=pc.position, frequency_khz=pc.frequency_khz)


def test_full_readback_with_frequency_goes_to_the_old_controller(make_session):
    s = make_session("SBGR", "N777AB")
    s.set_phase(ATCPhase.CLEARANCE)
    _pending(s, "N777AB, cleared to Galeão, climb five thousand, squawk 4521, contact Ground one two one decimal seven.")
    assert s.handoff_readback_kind("Ground one two one decimal seven, N777AB") == "short"
    assert s.handoff_readback_kind("Cleared to Galeão, climb five thousand, squawk four five two one, "
                                   "Ground one two one decimal seven, N777AB") == "full"
    assert s.handoff_readback_kind("Guarulhos Ground, N777AB, request taxi") is None


def test_pilot_addressing_tower_is_answered_by_tower_without_radio(make_session):
    s = make_session("SBGR", "PT-ABC", language="pt-br")
    s.set_phase(ATCPhase.GROUND)
    assert s.follow_addressed_station("Torre, PT-ABC, no ponto de espera, pronto para decolagem") == ATCPhase.TOWER
    assert s.phase == ATCPhase.TOWER
    assert s.follow_addressed_station("PT-ABC, pronto") is None
    # Brazil has no DEP position: "Controle" / "Departure" goes to APP
    assert s.follow_addressed_station("Departure, PT-ABC, passando três mil") == ATCPhase.APPROACH


def test_radio_wins_over_addressed_station(make_session):
    s = make_session("SBGR", "PT-ABC", language="pt-br")
    s.sync_phase_to_frequency(121_700_000)             # COM1 on Ground
    assert s.follow_addressed_station("Torre, PT-ABC, pronto") is None
    assert s.phase == ATCPhase.GROUND


@pytest.mark.parametrize("text,ok", [
    (">tagger", False), ("<think>", False), ("N123AB, **cleared** to land", False), ("roger", False),
    ("N123AB, cleared to land runway one six.", True), ("PT-ABC, ciente.", True),
])
def test_looks_like_radio(text, ok):
    from atc_llm_service import looks_like_radio
    assert looks_like_radio(text) is ok


def test_ground_handoff_survives_touchdown_inside_the_handoff_cooldown(make_session):
    from datetime import datetime, timedelta
    from atc_triggers import ATCTriggerEngine
    s = make_session("KBOS", "N543AR")
    s.set_phase(ATCPhase.TOWER)
    s.add_clearance(ClearanceType.LANDING, "N543AR, runway four right, cleared to land", runway="04R")
    eng = ATCTriggerEngine()
    s.update_aircraft_state(latitude=42.33, longitude=-71.03, altitude_ft=500, on_ground=False, groundspeed=130)
    eng.evaluate(s)
    eng.get_trigger_state(s).last_handoff_time = datetime.now()          # Tower handoff just happened
    s.update_aircraft_state(latitude=42.355, longitude=-71.012, altitude_ft=20, on_ground=True, groundspeed=90)
    assert eng.evaluate(s) is None                                        # touchdown inside cooldown
    s.update_aircraft_state(groundspeed=30)
    eng.get_trigger_state(s).last_handoff_time = datetime.now() - timedelta(seconds=31)
    call = eng.evaluate(s)
    assert call and call.new_phase == ATCPhase.GROUND


def test_stray_trailing_token_is_stripped():
    from atc_llm_service import strip_leaked_token
    assert strip_leaked_token("N54321, readback correct. erotikk") == "N54321, readback correct."
    assert strip_leaked_token("N54321, cleared to land runway four right.") == "N54321, cleared to land runway four right."


def test_runway_comma_taxi_is_a_taxi_clearance():
    types = {c.type for c in parse_clearances("N123AB, runway one six, taxi, hold short of runway one six", "N123AB")}
    assert ClearanceType.TAXI in types


def test_english_crew_in_a_portuguese_session_gets_english_handoffs(make_session):
    from atc_llm_service import pilot_speaks_english
    from atc_triggers import handoff_message
    assert pilot_speaks_english("Guarulhos Ground, American nine thirty, ready to taxi with information Alpha")
    assert not pilot_speaks_english("Solo Guarulhos, PT-ABC, pronto para o táxi")
    s = make_session("SBGR", "AAL930", language="pt-br")
    s.add_pilot_transmission("Guarulhos Tower, American nine thirty, holding short runway one zero left, ready")
    msg, _ = handoff_message(s, ATCPhase.APPROACH)
    assert "contact Approach on one one nine decimal eight" in msg
    s.add_pilot_transmission("Torre, PT-ABC, pronto")
    assert "chame Controle" in handoff_message(s, ATCPhase.APPROACH)[0]


def test_stt_punctuated_callsign_is_stripped_and_readback_is_short(make_session):
    # real XPlane run 2026-09-28: STT returned the readback with commas inside the callsign
    from clearance_parser import strip_callsign
    t = "Um, um, oito, decimal zero, November quatro cinco, X-Ray Sierra"
    assert "Sierra" not in strip_callsign(t, "N45XS")
    s = make_session("SBBH", "N45XS", language="pt-br")
    s.set_phase(ATCPhase.GROUND)
    _pending(s, "November quatro cinco X-ray Sierra, chame Torre em um um oito decimal zero.")
    assert s.handoff_readback_kind(t) == "short"


@pytest.mark.parametrize("text,rwy", [
    ("November quatro cinco X-Ray Sierra, aproximação visual pista um três autorizada, desça e mantenha "
     "quatro mil e quinhentos pés", "13"),
    ("PT-ABC, aproximação ILS pista dois oito direita autorizada, chame Torre", "28R"),
])
def test_portuguese_postfix_approach_clearance(text, rwy):
    cs = [c for c in parse_clearances(text, "N45XS") if c.type == ClearanceType.APPROACH]
    assert cs and cs[0].runway == rwy


def test_no_ground_handoff_for_touchdown_far_from_the_field(make_session):
    # real X-Plane 2026-09-28: aircraft crashed 2.2 nm short after "autorizado pouso" -> got "livre da pista"
    from atc_triggers import ATCTriggerEngine
    s = make_session("SBBH", "N45XS", language="pt-br")
    s.set_phase(ATCPhase.TOWER)
    s.add_clearance(ClearanceType.LANDING, "N45XS, autorizado pouso pista um três", runway="13")
    eng = ATCTriggerEngine()
    s.update_aircraft_state(latitude=-19.83, longitude=-44.00, altitude_ft=3500, on_ground=False, groundspeed=100)
    eng.evaluate(s)
    s.update_aircraft_state(latitude=-19.832, longitude=-43.995, altitude_ft=2650, on_ground=True, groundspeed=0)
    assert eng.evaluate(s) is None                         # ~2 nm out: no "livre da pista"
    s.update_aircraft_state(latitude=-19.8475, longitude=-43.9585, altitude_ft=2590, on_ground=False, groundspeed=80)
    eng.evaluate(s)
    s.update_aircraft_state(on_ground=True, groundspeed=30)
    call = eng.evaluate(s)
    assert call and call.new_phase == ATCPhase.GROUND      # real touchdown on the runway still works
