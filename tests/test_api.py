"""Offline API tests against the Flask app with fake LLM/audio (see conftest.api)."""

import base64


def test_health_without_session(api):
    r = api.get("/health").get_json()
    assert r["status"] == "ok" and r["session"] is None


def test_transmit_requires_session(api):
    r = api.post("/transmit_text", json={"text": "hello"})
    assert r.status_code == 400


def test_tune_ground_pt_br_with_flight_plan(api):
    r = api.post("/tune", json={
        "airport_icao": "sbgr", "callsign": "PT-ABC", "frequency": "Solo", "language": "pt-br",
        "flight_plan": {"aircraft_type": "C172", "flight_rules": "IFR", "destination": "SBKP"},
        "state": {"latitude": -23.4356, "longitude": -46.4731, "altitude_ft": 2461, "on_ground": True},
    })
    assert r.status_code == 200, r.get_json()
    s = r.get_json()["session"]
    assert s["airport_icao"] == "SBGR" and s["phase"] == "ground"
    assert s["frequency_mhz"] == 121.7 and s["language"] == "pt-br"
    assert s["runways"] == ["10L", "28R", "10R", "28L"]
    assert s["flight_plan"]["destination"] == "SBKP"
    full = api.get("/session").get_json()
    assert full["distance_nm"] < 1


def test_tune_auto_picks_nearest(api):
    r = api.post("/tune", json={"airport_icao": "auto", "latitude": 40.64, "longitude": -73.78,
                                "callsign": "N12345", "frequency": "Delivery"}).get_json()
    assert r["session"]["airport_icao"] == "KJFK" and r["session"]["phase"] == "clearance"


def test_tune_rejects_bad_language(api):
    r = api.post("/tune", json={"airport_icao": "KJFK", "language": "fr"})
    assert r.status_code == 400


def test_transmit_text_returns_audio_and_clearances(api):
    api.post("/tune", json={"airport_icao": "SBGR", "callsign": "PT-ABC", "frequency": "Tower"})
    api.fake_llm.replies = ["PT-ABC, cleared for takeoff runway two eight right, wind calm."]
    r = api.post("/transmit_text", json={"text": "Tower, PT-ABC ready for departure 28R"}).get_json()
    assert r["atc_text"].startswith("PT-ABC, cleared for takeoff")
    assert base64.b64decode(r["audio_base64"]).startswith(b"ID3")
    assert r["audio_provider"] == "fake"
    assert [c["type"] for c in r["clearances"]] == ["takeoff"]
    assert r["clearances"][0]["runway"] == "28R"


def test_transmit_text_audio_false_and_wav(api):
    api.post("/tune", json={"airport_icao": "SBGR"})
    r = api.post("/transmit_text", json={"text": "radio check", "audio": False}).get_json()
    assert r["audio_base64"] is None
    r = api.post("/transmit_text", json={"text": "radio check", "format": "wav"}).get_json()
    assert base64.b64decode(r["audio_base64"]).startswith(b"RIFF")


def test_transmit_text_survives_tts_failure(api, monkeypatch):
    api.post("/tune", json={"airport_icao": "SBGR"})

    def boom(text, language=None):
        raise RuntimeError("quota_exceeded")
    monkeypatch.setattr(api.fake_audio, "speak_atc", boom)
    r = api.post("/transmit_text", json={"text": "radio check"})
    assert r.status_code == 200
    assert r.get_json()["audio_base64"] is None and r.get_json()["atc_text"]


def test_transmit_audio_roundtrip_headers(api):
    api.post("/tune", json={"airport_icao": "SBGR", "callsign": "PT-ABC"})
    api.fake_audio.transcript = "Solo, PT-ABC, pronto para táxi"
    api.fake_llm.replies = ["PT-ABC, táxi para ponto de espera pista dois oito direita via Charlie."]
    r = api.post("/transmit", json={"audio": base64.b64encode(b"fake-wav").decode(), "format": "wav",
                                    "state": {"com1_freq_hz": 121_700_000}})
    assert r.status_code == 200
    assert r.mimetype == "audio/wav" and r.data.startswith(b"RIFF")
    assert r.headers["X-Pilot-Text"] == "Solo, PT-ABC, pronto para táxi"   # latin-1 safe: plain
    assert r.headers["X-ATC-Text"].startswith("PT-ABC, táxi")
    assert api.get("/session").get_json()["assigned"]["runway"] == "28R"


def test_header_safe_wraps_non_latin1(api):
    from atc_server import _header_safe
    assert _header_safe("plain") == "plain"
    wrapped = _header_safe("turn → left")
    assert wrapped.startswith("b64:")
    assert base64.b64decode(wrapped[4:]).decode() == "turn → left"


def test_retuning_com1_switches_position(api):
    api.post("/tune", json={"airport_icao": "SBGR", "frequency": "Ground"})
    r = api.post("/state", json={"com1_freq_hz": 132_750_000}).get_json()       # pilot tunes tower
    assert r["phase"] == "tower" and r["frequency"] == "GUARULHOS TWR"
    api.post("/transmit", json={"audio": base64.b64encode(b"x").decode(),
                                "state": {"com1_freq_hz": 119_800_000}})        # tunes approach mid-call
    assert api.get("/session").get_json()["phase"] == "approach"
    assert not api.fake_llm.calls[-1].startswith("[")                          # no mismatch warning


def test_unknown_frequency_is_flagged_to_llm(api):
    api.post("/tune", json={"airport_icao": "SBGR", "frequency": "Ground"})
    api.post("/transmit", json={"audio": base64.b64encode(b"x").decode(),
                                "state": {"com1_freq_hz": 123_450_000}})        # not an SBGR position
    assert api.fake_llm.calls[-1].startswith("[Pilot is on 123.450")
    assert api.get("/session").get_json()["phase"] == "ground"


def test_state_triggers_handoff_and_ack_changes_phase(api):
    api.post("/tune", json={"airport_icao": "KJFK", "callsign": "N12345", "frequency": "Tower"})
    api.fake_llm.replies = ["N12345, runway three one left, cleared for takeoff."]
    api.post("/transmit_text", json={"text": "ready for departure", "audio": False})

    r = api.post("/state", json={"latitude": 40.65, "longitude": -73.78, "altitude_ft": 1500,
                                 "on_ground": False, "vertical_speed": 1500}).get_json()
    assert r["pending_atc"]["trigger_type"] == "handoff"
    assert r["pending_atc"]["new_phase"] == "departure"

    # poll without ack keeps it pending
    p = api.get("/pending_atc?audio=false").get_json()
    assert p["pending"] and api.get("/health").get_json()["pending_atc"]

    # ack with wav audio: phase + frequency change, call cleared, logged in conversation
    p = api.get("/pending_atc?ack=true&format=wav").get_json()
    assert p["phase"] == "departure" and p["frequency_mhz"] == 135.9
    assert base64.b64decode(p["audio_base64"]).startswith(b"RIFF")
    assert api.get("/pending_atc").get_json()["pending"] is False
    conv = api.get("/conversation").get_json()["messages"]
    assert conv[-1]["role"] == "atc" and "contact Departure" in conv[-1]["content"]


def test_playground_follows_llm_handoff_without_radio(api):
    api.post("/tune", json={"airport_icao": "SBGR", "callsign": "N777AB", "frequency": "Ground"})
    api.fake_llm.replies = ["N777AB, taxi holding point runway one zero left, contact Guarulhos Tower one three "
                            "two decimal seven five zero."]
    api.post("/transmit_text", json={"text": "ready to taxi", "audio": False})
    assert api.get("/session").get_json()["pending_handoff"] == "tower"
    r = api.post("/transmit_text", json={"text": "Tower, N777AB, holding short 10L", "audio": False}).get_json()
    assert r["phase"] == "tower" and r["frequency"] == "GUARULHOS TWR"


def test_initiative_delivers_llm_call_during_silence(api):
    api.post("/tune", json={"airport_icao": "KBOS", "callsign": "N54321", "frequency": "Approach"})
    api.fake_llm.initiative_replies = [None, "N54321, turn left heading three zero zero, cleared ILS runway "
                                             "three three left approach, maintain three thousand until established."]
    st = {"latitude": 42.2, "longitude": -71.1, "altitude_ft": 3000, "on_ground": False, "groundspeed": 150}
    r = api.post("/state?initiative=true", json=st).get_json()
    assert r["pending_atc"] is None and api.fake_llm.initiative_calls == 1          # controller stayed quiet
    r = api.post("/state?initiative=true", json=st).get_json()
    assert r["pending_atc"]["trigger_type"] == "initiative"
    assert "cleared ILS" in r["pending_atc"]["message"]
    p = api.get("/pending_atc?ack=true&audio=false").get_json()
    conv = api.get("/conversation").get_json()["messages"]
    assert conv[-1]["role"] == "atc" and "cleared ILS" in conv[-1]["content"]
    sess = api.get("/session").get_json()
    assert sess["assigned"]["heading"] == 300 and sess["assigned"]["runway"] == "33L"
    # without force and with a fresh transmission, the timer gate keeps the LLM quiet
    api.post("/transmit_text", json={"text": "wilco", "audio": False})
    api.post("/state", json=st)
    assert api.fake_llm.initiative_calls == 2


def test_reset_clears_everything(api):
    api.post("/tune", json={"airport_icao": "SBGR"})
    api.post("/reset")
    assert api.get("/session").status_code == 404
    assert api.get("/health").get_json()["session"] is None


def test_lookup_endpoints(api):
    assert api.get("/frequencies/SBGR").get_json()["frequencies"]["TWR"]["frequency_mhz"] == 132.75
    assert api.get("/airport/SBGR").get_json()["elevation_ft"] == 2461
    assert api.get("/airport/XXXX").status_code == 404
    near = api.get("/nearest_airport?lat=-23.43&lon=-46.47").get_json()["airports"]
    assert near[0]["icao"] == "SBGR"
    assert api.get("/nearest_airport").status_code == 400


def test_standby_gets_answered_unprompted(api):
    """Live 2026-11-07: Approach said "aguarde" to a route-change request and never came back."""
    from datetime import datetime, timedelta
    api.post("/tune", json={"airport_icao": "SBCF", "callsign": "N45XS", "frequency": "Approach", "language": "pt-br"})
    sess = api.server.session_manager.get_active_session()
    sess.standby_request = "solicito direto MAVGU, aerovia UZ24"
    sess.standby_at = datetime.now() - timedelta(seconds=30)
    api.fake_llm.initiative_replies = ["N45XS, autorizado direto MAVGU, aerovia UZ24, VUNLI, direto Cabo Frio."]
    st = {"latitude": -19.75, "longitude": -43.98, "altitude_ft": 8500, "on_ground": False, "groundspeed": 150}
    r = api.post("/state", json=st).get_json()
    assert r["pending_atc"]["trigger_type"] == "initiative" and "MAVGU" in r["pending_atc"]["message"]
    assert sess.standby_at is None and api.fake_llm.initiative_calls == 1
    api.get("/pending_atc?ack=true&audio=false")
    api.post("/state", json=st)                     # answered once; no repeat callback
    assert api.fake_llm.initiative_calls == 1


def test_is_standby():
    from atc_llm_service import is_standby
    assert is_standby("N45XS, solicitação recebida, aguarde.")
    assert is_standby("N45XS, solicitação ainda em coordenação, aguarde.")
    assert is_standby("N123, stand by")
    assert not is_standby("N45XS, autorizado direto MAVGU.")


def test_strip_non_latin_garbage():
    from atc_llm_service import strip_leaked_token
    assert strip_leaked_token("N45XS, pista em uso três quatro.уаҩ") == "N45XS, pista em uso três quatro."


def test_outbound_prompt_says_departure(make_session):
    from atc_llm_service import ATCLLMService
    s = make_session("SBCF", "N45XS", language="pt-br", aircraft_type="BE58", flight_rules="IFR",
                     departure="SBCF", destination="SBCB")
    s.update_aircraft_state(latitude=-19.9, longitude=-44.0, altitude_ft=10500, on_ground=False)
    svc = ATCLLMService.__new__(ATCLLMService); svc.language = "pt-br"
    assert "DECOLOU de SBCF" in svc._build_system_prompt(s)


def _quiet_airborne_session(api):
    from datetime import datetime, timedelta
    api.post("/tune", json={"airport_icao": "SBCF", "callsign": "N45XS", "frequency": "Approach", "language": "pt-br"})
    sess = api.server.session_manager.get_active_session()
    sess.last_transmission = datetime.now() - timedelta(minutes=10)
    api.server._last_initiative_at = None
    return sess


def test_no_radio_checks_while_sim_paused(api):
    """Live 2026-11-07: sim paused mid-air, ATC kept calling "como me escuta? acione ident" for an hour."""
    from datetime import datetime, timedelta
    sess = _quiet_airborne_session(api)
    api.fake_llm.initiative_replies = ["N45XS, como me escuta?"]
    st = {"latitude": -20.81, "longitude": -44.08, "altitude_ft": 10400, "on_ground": False, "groundspeed": 170}
    api.post("/state", json=st)
    api.get("/pending_atc?ack=true&audio=false")
    calls = api.fake_llm.initiative_calls
    sess.position_changed_at = datetime.now() - timedelta(seconds=60)   # same lat/lon for a minute
    api.server._last_initiative_at = None
    assert api.post("/state", json=st).get_json()["pending_atc"] is None
    assert api.fake_llm.initiative_calls == calls


def test_radio_checks_stop_after_unanswered(api):
    from datetime import datetime, timedelta
    sess = _quiet_airborne_session(api)
    api.fake_llm.initiative_replies = ["N45XS, como me escuta?"] * 5
    for i in range(4):
        api.server._last_initiative_at = datetime.now() - timedelta(minutes=5)
        api.post("/state", json={"latitude": -20.8 - i / 100, "longitude": -44.08, "altitude_ft": 10400,
                                 "on_ground": False, "groundspeed": 170})
        api.get("/pending_atc?ack=true&audio=false")
    assert api.fake_llm.initiative_calls == 2 and sess.unanswered_initiatives == 2
    sess.add_pilot_transmission("Belo Horizonte, N45XS, nível 100")
    assert sess.unanswered_initiatives == 0
