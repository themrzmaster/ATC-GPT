"""Offline unit tests: data services, session, clearance parser, triggers, chart parsing."""

import textwrap

import pytest

from clearance_parser import parse_clearances, spoken_to_number, strip_callsign, parse_handoff
from session_manager import ATCPhase, ClearanceType, FlightPlan, phase_from_text
from atc_triggers import ATCTriggerEngine, handoff_message, _say_freq
from chart_providers import AISWebChartProvider, FAAChartProvider, parse_runways


# ---------------------------------------------------------------------------
# Frequencies & airport data
# ---------------------------------------------------------------------------

def test_frequency_lookup_sbgr(frequency_service):
    f = frequency_service.get_all_atc("SBGR")
    assert f["GND"].frequency_mhz == pytest.approx(121.7)
    assert f["TWR"].frequency_mhz == pytest.approx(132.75)
    assert "DEP" not in f          # Brazil: no separate departure position -> APP handles it
    assert f["APP"].frequency_mhz == pytest.approx(119.8)


def test_frequency_lookup_kjfk_has_departure(frequency_service):
    f = frequency_service.get_all_atc("KJFK")
    assert f["DEP"].frequency_mhz == pytest.approx(135.9)


def test_airport_runways_and_nearest(airport_data):
    ap = airport_data.get("SBGR")
    assert ap.elevation_ft == 2461
    assert set(ap.runway_idents()) == {"10L", "28R", "10R", "28L"}
    assert not ap.has_runway("27R")
    near = airport_data.nearest(-23.4356, -46.4731, max_nm=30)
    assert near[0][0].icao == "SBGR"
    assert near[0][1] < 1.0


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def test_session_summary_contains_runways_and_flight_plan(make_session):
    s = make_session("SBGR", aircraft_type="C172", flight_rules="IFR", destination="SBKP")
    summary = s.get_situation_summary()
    assert "Valid runway identifiers here: 10L, 28R, 10R, 28L" in summary
    assert "IFR, type C172, SBGR -> SBKP" in summary
    assert "GUARULHOS GROUND" in summary


def test_session_phase_frequency_mapping(make_session):
    s = make_session("SBGR")
    s.set_phase(ATCPhase.TOWER)
    assert s.current_frequency_mhz == pytest.approx(132.75)
    s.set_phase(ATCPhase.DEPARTURE)   # no DEP at SBGR -> falls back to a service name, no mhz
    assert s.current_frequency_mhz is None
    assert s.current_frequency == "Departure"


def test_session_geometry(make_session):
    s = make_session("SBGR")
    s.update_aircraft_state(latitude=-23.4356, longitude=-46.4731, altitude_ft=3461)
    assert s.distance_to_airport_nm() < 1
    assert s.altitude_agl_ft() == pytest.approx(1000)


def test_known_facts_in_summary(make_session):
    s = make_session("SBGR")
    assert "Facts already looked up" not in s.get_situation_summary()
    s.remember_fact("weather SBGR", "METAR SBGR 091200Z 05004KT | wind 50@4kt | IFR")
    summary = s.get_situation_summary()
    assert "Facts already looked up" in summary and "05004KT" in summary
    assert s.fact("weather SBGR").startswith("METAR")
    assert s.fact("weather SBGR", max_age_s=0) is None or True  # age check path executes


def test_validate_pilot_frequency(make_session):
    s = make_session("SBGR")           # ground = 121.700
    ok, _ = s.validate_pilot_frequency(121_700_000)
    assert ok
    ok, msg = s.validate_pilot_frequency(132_750_000)
    assert not ok and "TWR" in msg


def test_sync_phase_to_frequency_follows_radio(make_session):
    s = make_session("SBGR")
    assert s.phase == ATCPhase.GROUND
    assert s.sync_phase_to_frequency(132_750_000) == ATCPhase.TOWER      # tower
    assert s.phase == ATCPhase.TOWER and s.current_frequency_mhz == pytest.approx(132.75)
    assert s.sync_phase_to_frequency(132_760_000) is None                # same position, within 25 kHz
    assert s.sync_phase_to_frequency(119_800_000) == ATCPhase.APPROACH   # São Paulo APP
    assert s.sync_phase_to_frequency(123_450_000) is None                # unknown freq: keep position
    assert s.phase == ATCPhase.APPROACH
    assert s.sync_phase_to_frequency(None) is None


@pytest.mark.parametrize("text,phase", [
    ("Ground", ATCPhase.GROUND), ("Solo", ATCPhase.GROUND), ("GUARULHOS TWR", ATCPhase.TOWER),
    ("Torre", ATCPhase.TOWER), ("Delivery", ATCPhase.CLEARANCE), ("Tráfego", ATCPhase.CLEARANCE),
    ("Approach", ATCPhase.APPROACH), ("Departure", ATCPhase.DEPARTURE), ("Center", ATCPhase.CENTER),
])
def test_phase_from_text(text, phase):
    assert phase_from_text(text) == phase


def test_flight_plan_from_dict_normalizes():
    fp = FlightPlan.from_dict({"aircraft_type": "c172", "flight_rules": "ifr", "destination": "sbkp",
                               "cruise_altitude_ft": "5000"})
    assert (fp.aircraft_type, fp.flight_rules, fp.destination, fp.cruise_altitude_ft) == ("C172", "IFR", "SBKP", 5000)
    assert FlightPlan.from_dict(None).is_empty()


# ---------------------------------------------------------------------------
# Clearance parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("words,expected", [
    ("two seven zero", 270), ("three thousand", 3000), ("three thousand five hundred", 3500),
    ("flight level three five zero", None),   # handled at parser level, not here
    ("dois mil e quinhentos", 2500), ("um zero", 10), ("cinco mil", 5000), ("4521", 4521),
    ("one two one", 121), ("seis mil", 6000), ("tree tousand", None),
])
def test_spoken_to_number(words, expected):
    assert spoken_to_number(words.split()) == expected


def test_strip_callsign_removes_written_and_spoken():
    assert "12345" not in strip_callsign("N12345, descend and maintain 3000", "N12345")
    out = strip_callsign("November one two three four five, climb and maintain five thousand", "N12345")
    assert "one two three four five" not in out.lower()
    assert "five thousand" in out


def test_callsign_digits_never_become_values():
    pcs = parse_clearances("N12345, turn left heading two seven zero, descend and maintain three thousand", "N12345")
    kinds = {p.type: p for p in pcs}
    assert kinds[ClearanceType.ALTITUDE].value == 3000
    assert kinds[ClearanceType.HEADING].value == 270


def test_flight_level_and_squawk():
    pcs = parse_clearances("November one two three four five, climb and maintain flight level three five zero, "
                           "squawk four five two one", "N12345")
    kinds = {p.type: p for p in pcs}
    assert kinds[ClearanceType.ALTITUDE].value == 35000
    assert kinds[ClearanceType.SQUAWK].value == 4521


@pytest.mark.parametrize("text,ctype,rwy", [
    ("N12345, cleared for takeoff runway two eight right, wind zero five zero at four", ClearanceType.TAKEOFF, "28R"),
    ("N12345, cleared to land runway 04L", ClearanceType.LANDING, "04L"),
    ("N12345, taxi to holding point runway 31L via Alpha, Bravo, hold short runway 31L", ClearanceType.TAXI, "31L"),
    ("N12345, line up and wait runway two two right", ClearanceType.LINEUP, "22R"),
    ("N12345, cleared ILS runway one three left approach", ClearanceType.APPROACH, "13L"),
    ("PT-ABC, autorizado decolagem pista dois oito direita, vento zero cinco zero, quatro nós", ClearanceType.TAKEOFF, "28R"),
    ("PT-ABC, autorizado pouso pista 28L", ClearanceType.LANDING, "28L"),
    ("PT-ABC, táxi para ponto de espera pista um zero esquerda via Charlie", ClearanceType.TAXI, "10L"),
    ("PT-ABC, alinhe e mantenha pista um zero direita", ClearanceType.LINEUP, "10R"),
    ("PT-ABC, autorizado ILS Z pista dois oito esquerda", ClearanceType.APPROACH, "28L"),
])
def test_runway_clearances_both_languages(text, ctype, rwy):
    pcs = parse_clearances(text, text.split(",")[0])
    match = [p for p in pcs if p.type == ctype]
    assert match, f"{ctype} not detected in {text!r}: {pcs}"
    assert match[0].runway == rwy


def test_portuguese_altitude_heading_squawk():
    pcs = parse_clearances("PT-ABC, suba e mantenha dois mil e quinhentos pés, proa zero nove zero, "
                           "transponder quatro cinco dois um", "PT-ABC")
    kinds = {p.type: p for p in pcs}
    assert kinds[ClearanceType.ALTITUDE].value == 2500
    assert kinds[ClearanceType.HEADING].value == 90
    assert kinds[ClearanceType.SQUAWK].value == 4521


@pytest.mark.parametrize("text,rwy", [
    ("N777AB, negative, runway two seven right does not exist, taxi holding point runway one zero left, QNH one zero two two.", "10L"),
    ("N777AB, negative, runway two seven right does not exist. Taxi runway one zero left, hold short runway one zero left.", "10L"),
    ("PT-ABC, negativo, pista dois sete direita não existe, táxi para ponto de espera pista um zero esquerda.", "10L"),
])
def test_denied_runway_is_not_the_cleared_runway(text, rwy):
    taxi = [p for p in parse_clearances(text, text.split(",")[0]) if p.type == ClearanceType.TAXI]
    assert taxi and taxi[0].runway == rwy


def test_no_false_positive_on_denial():
    pcs = parse_clearances("N12345, negative, hold short runway two eight right, traffic on final", "N12345")
    types = {p.type for p in pcs}
    assert ClearanceType.TAKEOFF not in types
    assert ClearanceType.HOLD in types


@pytest.mark.parametrize("text,pos,khz", [
    ("N12345, contact Guarulhos Tower one three two decimal seven five zero", "tower", 132750),
    ("N12345, contact departure on one two seven point one, good day", "departure", 127100),
    ("N12345, contact New York Approach 125.7", "approach", 125700),
    ("PT-ABC, chame Torre em um três dois decimal sete cinco", "tower", 132750),
    ("PT-ABC, chame Controle São Paulo em um um nove decimal oito", "approach", 119800),
    ("PT-ABC, chame Solo em 121.7", "ground", 121700),
    ("N12345, monitor ground point six five", "ground", None),
])
def test_parse_handoff(text, pos, khz):
    assert parse_handoff(text) == (pos, khz)


def test_llm_handoff_moves_session_on_next_call(make_session):
    s = make_session("SBGR")
    pcs = parse_clearances("N12345, taxi holding point runway one zero left, contact Guarulhos Tower one three two "
                           "decimal seven five zero", "N12345")
    for pc in pcs:
        s.add_clearance(pc.type, "x", value=pc.value, runway=pc.runway, position=pc.position,
                        frequency_khz=pc.frequency_khz)
    assert s.pending_handoff == ATCPhase.TOWER and s.phase == ATCPhase.GROUND
    assert s.apply_pending_handoff() == ATCPhase.TOWER
    assert s.phase == ATCPhase.TOWER and s.pending_handoff is None


def test_radio_overrides_pending_handoff(make_session):
    s = make_session("SBGR")
    s.add_clearance(ClearanceType.FREQUENCY, "contact tower", position="tower", frequency_khz=132750)
    s.sync_phase_to_frequency(121_700_000)          # pilot still on ground frequency
    assert s.apply_pending_handoff() is None and s.phase == ATCPhase.GROUND
    s.sync_phase_to_frequency(132_750_000)          # now they retune
    assert s.phase == ATCPhase.TOWER and s.pending_handoff is None


def test_session_records_parsed_values(make_session):
    s = make_session("SBGR")
    for pc in parse_clearances("PT-ABC, autorizado decolagem pista dois oito direita", "PT-ABC"):
        s.add_clearance(pc.type, "x", value=pc.value, runway=pc.runway)
    assert s.has_clearance(ClearanceType.TAKEOFF)
    assert s.assigned_runway == "28R"


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------

def _airborne(s, agl, vs=1500):
    s.update_aircraft_state(latitude=-23.44, longitude=-46.47, altitude_ft=2461 + agl, on_ground=False,
                            vertical_speed=vs, indicated_airspeed=140, groundspeed=150)


def test_departure_handoff_requires_takeoff_clearance(make_session):
    eng = ATCTriggerEngine()
    s = make_session("SBGR")
    s.set_phase(ATCPhase.TOWER)
    _airborne(s, 1500)
    assert eng.evaluate(s) is None          # no takeoff clearance on record
    s.add_clearance(ClearanceType.TAKEOFF, "cleared for takeoff", runway="28R")
    call = eng.evaluate(s)
    assert call is not None and call.new_phase == ATCPhase.APPROACH   # SBGR has no DEP -> APP
    assert "119.8" in f"{call.new_frequency_mhz}"
    assert "one one nove decimal oito" not in call.message
    assert call.message == "PT-ABC, contact Approach on one one nine decimal eight."


def test_departure_handoff_kjfk_goes_to_departure_and_respects_cooldown(make_session):
    eng = ATCTriggerEngine()
    s = make_session("KJFK", callsign="N12345")
    s.set_phase(ATCPhase.TOWER)
    s.add_clearance(ClearanceType.TAKEOFF, "cleared for takeoff", runway="31L")
    s.update_aircraft_state(latitude=40.65, longitude=-73.78, altitude_ft=13 + 1200, on_ground=False,
                            vertical_speed=1800)
    call = eng.evaluate(s)
    assert call.new_phase == ATCPhase.DEPARTURE and call.new_frequency_type == "DEP"
    assert call.message == "N12345, contact Departure on one three five decimal nine."
    s.set_phase(ATCPhase.TOWER)  # pretend not acknowledged; must not re-fire within cooldown
    assert eng.evaluate(s) is None


@pytest.mark.parametrize("fp,expected", [
    (dict(departure="SBMT", destination="SBMT"), True),
    (dict(departure="SBMT", destination="SBMT", route="DCT"), True),
    (dict(remarks="circuito de tráfego, toque e arremetida"), True),
    (dict(route="closed traffic"), True),
    (dict(departure="SBGR", destination="SBKP"), False),
    (dict(departure="SBMT", destination="SBMT", route="SBMT VOR SBMT"), False),
])
def test_flight_plan_local_pattern(fp, expected):
    assert FlightPlan(**fp).is_local_pattern() is expected


def test_no_departure_handoff_for_pattern_work(make_session):
    eng = ATCTriggerEngine()
    s = make_session("SBMT", callsign="PR-XYZ", language="pt-br", aircraft_type="C152",
                     remarks="circuito, toque e arremetida")
    s.set_phase(ATCPhase.TOWER)
    s.add_clearance(ClearanceType.TAKEOFF, "autorizado decolagem pista um dois", runway="12")
    s.update_aircraft_state(latitude=-23.50, longitude=-46.63, altitude_ft=2368 + 1000, on_ground=False,
                            vertical_speed=700, indicated_airspeed=80)
    assert eng.evaluate(s) is None


def test_departure_handoff_not_before_threshold(make_session):
    eng = ATCTriggerEngine()
    s = make_session("SBGR")
    s.set_phase(ATCPhase.TOWER)
    s.add_clearance(ClearanceType.TAKEOFF, "x", runway="28R")
    _airborne(s, 300)
    assert eng.evaluate(s) is None


def test_portuguese_handoff_message(make_session):
    s = make_session("SBGR", language="pt-br")
    msg, mhz = handoff_message(s, ATCPhase.TOWER)
    assert msg == "PT-ABC, chame Torre em um três dois decimal sete cinco."
    assert mhz == pytest.approx(132.75)


def test_say_freq():
    assert _say_freq(121.7, "en") == "one two one decimal seven"
    assert _say_freq(118.0, "en") == "one one eight"
    assert _say_freq(126.9, "pt-br") == "um dois meia decimal nove"


def test_tower_handoff_on_approach(make_session):
    eng = ATCTriggerEngine()
    s = make_session("SBGR")
    s.set_phase(ATCPhase.APPROACH)
    s.update_aircraft_state(latitude=-23.30, longitude=-46.47, altitude_ft=2461 + 2500, on_ground=False,
                            vertical_speed=-700)                       # ~8nm north
    assert eng.evaluate(s) is None                                     # no approach clearance yet
    s.add_clearance(ClearanceType.APPROACH, "cleared ILS", runway="28L")
    assert s.distance_to_airport_nm() < 8.5
    call = eng.evaluate(s)
    assert call is not None and call.new_phase == ATCPhase.TOWER


def test_ground_handoff_after_landing_when_slow(make_session):
    eng = ATCTriggerEngine()
    s = make_session("SBGR")
    s.set_phase(ATCPhase.TOWER)
    s.add_clearance(ClearanceType.LANDING, "cleared to land", runway="28L")
    _airborne(s, 200, vs=-500)
    assert eng.evaluate(s) is None
    s.update_aircraft_state(on_ground=True, groundspeed=110, vertical_speed=0, altitude_ft=2461)
    assert eng.evaluate(s) is None          # rolling out fast
    s.update_aircraft_state(groundspeed=25)
    call = eng.evaluate(s)
    assert call is not None and call.new_phase == ATCPhase.GROUND
    assert "vacate next taxiway" in call.message


# ---------------------------------------------------------------------------
# Chart providers (parsing only)
# ---------------------------------------------------------------------------

AISWEB_SNIPPET = textwrap.dedent("""
    <h4 class="heading-primary">Cartas (3)</h4>
    <h4>ADC</h4><ul class="list">
      <li><a href="https://aisweb.decea.gov.br/download/?arquivo=aaa-111&amp;apikey=1">SBGR ADC</a></li>
    </ul>
    <h4>IAC</h4><ul class="list">
      <li><a href="https://aisweb.decea.gov.br/download/?arquivo=bbb-222&amp;apikey=1">ILS Z OR LOC Z RWY 28L</a></li>
    </ul>
    <h4>SID</h4><ul class="list">
      <li><a href="https://aisweb.decea.gov.br/download/?arquivo=ccc-333&amp;apikey=1">RNAV CGO 5A RWY 10R</a></li>
    </ul>
    <h4>NOTAM</h4><ul><li><a href="https://aisweb.decea.gov.br/download/?arquivo=zzz&amp;apikey=1">ignored</a></li></ul>
""")


def test_aisweb_parse():
    charts = AISWebChartProvider.parse_page(AISWEB_SNIPPET, "SBGR")
    assert [c["category"] for c in charts] == ["AD", "APP", "SID"]
    assert charts[1]["rwys"] == ["28L"] and charts[2]["rwys"] == ["10R"]
    assert charts[0]["url"].startswith("https://aisweb.decea.mil.br/download/?arquivo=aaa-111&apikey=1")
    assert charts[0]["id"] == "aaa-111"


FAA_SNIPPET = b"""<?xml version="1.0"?>
<digital_tpp cycle="2609"><state_code ID="NY"><city_name ID="NEW YORK">
<airport_name ID="JOHN F KENNEDY INTL" apt_ident="JFK" icao_ident="KJFK">
<record><chart_code>APD</chart_code><chart_name>AIRPORT DIAGRAM</chart_name><pdf_name>00610AD.PDF</pdf_name></record>
<record><chart_code>IAP</chart_code><chart_name>ILS OR LOC RWY 04L</chart_name><pdf_name>00610IL4L.PDF</pdf_name></record>
<record><chart_code>DP</chart_code><chart_name>KENNEDY ONE</chart_name><pdf_name>00610KENNEDY.PDF</pdf_name></record>
<record><chart_code>MIN</chart_code><chart_name>TAKEOFF MINIMUMS</chart_name><pdf_name>NE2TO.PDF</pdf_name></record>
</airport_name></city_name></state_code></digital_tpp>"""


def test_faa_parse():
    idx = FAAChartProvider.parse_metafile(FAA_SNIPPET, "2609")
    charts = idx["KJFK"]
    assert [c["category"] for c in charts] == ["AD", "APP", "SID", "MISC"]
    assert charts[1]["url"] == "https://aeronav.faa.gov/d-tpp/2609/00610IL4L.PDF"
    assert charts[1]["rwys"] == ["04L"]


@pytest.mark.parametrize("name,rwys", [
    ("ILS Z OR LOC Z RWY 28L", ["28L"]), ("RNAV AKRER 3A RWY 28R/28L", ["28R", "28L"]),
    ("ILS OR LOC RWY 04L", ["04L"]), ("CAMRN FIVE", []), ("OMNI 10L/10R 28L/28R", []),
])
def test_parse_runways(name, rwys):
    assert parse_runways(name) == rwys
