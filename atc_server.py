"""
ATC Backend Server - connects the X-Plane plugin (or the web playground) to the AI services.

Endpoints:
    GET  /health                 - status + which providers/model are active
    POST /tune                   - create a session: airport, callsign, position, language, flight plan
    POST /transmit               - pilot audio (base64) -> ATC audio (mp3 or wav) + text headers
    POST /transmit_text          - pilot text -> ATC text (+ audio_base64 unless audio=false)
    POST /state                  - aircraft state update; evaluates ATC-initiated triggers
    GET  /pending_atc            - poll/ack ATC-initiated call (optionally with audio)
    GET  /session                - current session (incl. clearances, assigned values)
    GET  /conversation           - conversation history
    POST /reset                  - end session
    GET  /frequencies/<ICAO>     - published ATC frequencies
    GET  /airport/<ICAO>         - airport info (runways, elevation, position)
    GET  /nearest_airport?lat&lon - nearest airports to a position (plugin auto-tune)
    GET  /charts/<ICAO>[?category=APP] - chart listing from the official providers

Run with: python atc_server.py
"""

import base64
import io
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

import config
from audio_service import AudioService
from atc_llm_service import ATCLLMService
from session_manager import SessionManager, ATCPhase, FlightPlan, phase_from_text
from frequency_service import FrequencyService
from airport_data_service import AirportDataService, haversine_nm
from chart_providers import ChartService
from atc_triggers import trigger_engine, ATCCall, TriggerType
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("ATC-Server")

HOST = config.ATC_SERVER_HOST
PORT = config.ATC_SERVER_PORT
DEBUG = config.ATC_DEBUG
STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

# -----------------------------------------------------------------------------
# Services (lazy so the server boots without keys for offline tests)
# -----------------------------------------------------------------------------

session_manager = SessionManager()
current_language = config.ATC_LANGUAGE
_lock = threading.Lock()

audio_service: Optional[AudioService] = None
llm_service: Optional[ATCLLMService] = None
frequency_service: Optional[FrequencyService] = None
airport_data_service: Optional[AirportDataService] = None
chart_service: Optional[ChartService] = None
pending_atc_call: Optional[ATCCall] = None
_last_initiative_at: Optional[datetime] = None


def get_audio_service() -> AudioService:
    global audio_service
    if audio_service is None:
        audio_service = AudioService(language=current_language)
    audio_service.language = current_language
    return audio_service


def get_llm_service(language: Optional[str] = None) -> ATCLLMService:
    global llm_service, current_language
    lang = language or current_language
    if llm_service is None or llm_service.language != lang:
        llm_service = ATCLLMService(language=lang, chart_service=get_chart_service(),
                                    airport_data_service=get_airport_data_service())
        current_language = lang
    return llm_service


def get_frequency_service() -> FrequencyService:
    global frequency_service
    if frequency_service is None:
        frequency_service = FrequencyService(auto_download=True)
    return frequency_service


def get_airport_data_service() -> AirportDataService:
    global airport_data_service
    if airport_data_service is None:
        airport_data_service = AirportDataService()
    return airport_data_service


def get_chart_service() -> ChartService:
    global chart_service
    if chart_service is None:
        chart_service = ChartService()
    return chart_service


_prefetched: set = set()


def _prefetch_charts(icao: Optional[str]):
    """Digest the destination's approach charts in the background, so read_chart is instant on arrival."""
    if not icao or icao in _prefetched or os.environ.get("ATC_CHART_PREFETCH", "true").lower() != "true":
        return
    _prefetched.add(icao)
    svc = get_llm_service()
    threading.Thread(target=svc.prefetch_chart_digests, args=(icao, "APP"), daemon=True).start()


def init_services():
    logger.info("Loading airport frequencies (OurAirports)...")
    get_frequency_service()
    logger.info("Loading airport database (OurAirports)...")
    get_airport_data_service()
    logger.info("Data loaded.")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

STATE_FIELDS = ("latitude", "longitude", "altitude_ft", "heading", "indicated_airspeed", "groundspeed",
                "vertical_speed", "squawk", "com1_freq_hz", "on_ground")


RETUNE_RADIUS_NM = 60
_ATC_FREQ_TYPES = ("CLD", "GND", "TWR", "DEP", "APP")


# -- flight plans that survive a server restart / plugin re-tune ---------------------------------

def _load_plan_store() -> dict:
    if not config.FLIGHT_PLAN_STORE:
        return {}
    try:
        return json.loads(Path(config.FLIGHT_PLAN_STORE).read_text())
    except (OSError, ValueError):
        return {}


def _remember_plan(callsign: str, fp: FlightPlan):
    """Keep the latest plan with a destination per callsign (memory + disk)."""
    if not config.FLIGHT_PLAN_STORE or not fp.destination:
        return
    store = _load_plan_store()
    store[callsign.upper()] = {"plan": fp.to_dict(), "saved_at": datetime.now().isoformat()}
    try:
        Path(config.FLIGHT_PLAN_STORE).parent.mkdir(parents=True, exist_ok=True)
        Path(config.FLIGHT_PLAN_STORE).write_text(json.dumps(store, indent=1))
    except OSError as e:
        logger.warning(f"could not save flight plan: {e}")


def _recall_plan(callsign: str) -> Optional[FlightPlan]:
    item = _load_plan_store().get(callsign.upper())
    if not item:
        return None
    try:
        age_h = (datetime.now() - datetime.fromisoformat(item["saved_at"])).total_seconds() / 3600
    except (KeyError, ValueError):
        return None
    return FlightPlan.from_dict(item["plan"]) if age_h <= config.FLIGHT_PLAN_REUSE_H else None


def _pick_airport_for_plan(fp: FlightPlan, state: dict, nearest_icao: str) -> str:
    """
    Auto-tune: on the ground, the nearest field. In the air with a flight plan, the nearer of departure
    and destination (a re-tune mid-flight used to land on whatever small field was below: SNJR, SDMC).
    """
    if not fp.destination or state.get("on_ground", True):
        return nearest_icao
    lat, lon = state.get("latitude"), state.get("longitude")
    best = None
    for icao in (fp.departure, fp.destination):
        ap = get_airport_data_service().get(icao) if icao else None
        if ap and lat is not None:
            d = haversine_nm(float(lat), float(lon), ap.latitude, ap.longitude)
            if best is None or d < best[0]:
                best = (d, icao)
    return best[1] if best else nearest_icao


def _auto_phase(session, state: dict) -> ATCPhase:
    """Initial position when the plugin tunes "auto" (COM1 then overrides it if it matches a position)."""
    if state.get("on_ground", True):
        return ATCPhase.GROUND
    fp = session.flight_plan
    if fp.destination and fp.destination != session.airport_icao and fp.departure == session.airport_icao:
        return ATCPhase.DEPARTURE if session.get_handoff_frequency(ATCPhase.DEPARTURE) else ATCPhase.APPROACH
    return ATCPhase.APPROACH


def _transfer_to_destination(session):
    """
    Inside ARRIVAL_TRANSFER_NM of a destination that has no Approach/Tower of its own (or whose Approach
    is this same frequency), the same controller keeps the flight but now works the arrival there:
    the session moves to the destination (runways, charts, weather) with the conversation intact.
    A destination with its own Approach/Tower on another frequency gets a real handoff instead
    (atc_triggers._check_destination_handoff) and the radio-follow moves the session when the pilot dials it.
    """
    global pending_atc_call
    fp, st = session.flight_plan, session.aircraft_state
    dest = fp.destination
    if not dest or dest == session.airport_icao or st.on_ground or not (st.latitude or st.longitude):
        return session
    ap = get_airport_data_service().get(dest)
    if not ap or haversine_nm(st.latitude, st.longitude, ap.latitude, ap.longitude) > config.ARRIVAL_TRANSFER_NM:
        return session
    freqs = get_frequency_service().get_all_atc(dest)
    unit = freqs.get("APP") or freqs.get("TWR")
    if unit and not (session.current_frequency_mhz and abs(unit.frequency_mhz - session.current_frequency_mhz) < 0.01):
        return session
    old = session
    new = session_manager.create_session(
        dest, old.callsign, frequency_service=get_frequency_service(),
        airport_data_service=get_airport_data_service(), language=old.language, flight_plan=old.flight_plan)
    new.aircraft_state = old.aircraft_state
    new.position_changed_at = old.position_changed_at
    new.conversation = old.conversation
    new.clearances = old.clearances
    new.last_transmission = old.last_transmission
    new.unanswered_initiatives = old.unanswered_initiatives
    for k in ("assigned_altitude", "assigned_heading", "assigned_speed", "assigned_squawk"):
        setattr(new, k, getattr(old, k))
    new.set_phase(ATCPhase.APPROACH)
    # the pilot stays on the frequency they were working: radio-follow must not bounce them back
    com1 = st.com1_freq_hz
    new.worked_freq_mhz = (com1 / 1e6) if com1 else old.current_frequency_mhz
    if not new.current_frequency_mhz:
        new.current_frequency_mhz = new.worked_freq_mhz
    trigger_engine.reset_session(old)
    if pending_atc_call is not None and pending_atc_call.trigger_type == TriggerType.INITIATIVE:
        pending_atc_call = None
    logger.info(f"arrival: {old.airport_icao} -> {dest} ({'area control, no ATC unit there' if not unit else 'same frequency'})")
    return new


def _apply_state(session, data: dict):
    """Update the aircraft and follow the radio. Returns the active session (it can change airport)."""
    session.update_aircraft_state(**{k: data.get(k) for k in STATE_FIELDS})
    if data.get("flight_plan") and session.flight_plan.update_from(data["flight_plan"]):
        logger.info(f"flight plan updated from the aircraft FMS: {session.flight_plan.summary()}")
        _remember_plan(session.callsign, session.flight_plan)
        _prefetch_charts(session.flight_plan.destination)
    session = _follow_radio_to_other_airport(session, data.get("com1_freq_hz"))
    session = _transfer_to_destination(session)
    # Follow the radio: retuning COM1 to another published position switches the controller.
    new_phase = session.sync_phase_to_frequency(data.get("com1_freq_hz"))
    if new_phase:
        logger.info(f"COM1 retuned -> now talking to {session.current_frequency} ({new_phase.value})")
    return session


def _follow_radio_to_other_airport(session, freq_hz: Optional[int]):
    """
    The plugin tunes the departure airport once. On a cross-country flight the pilot dials the
    destination's Tower/Approach: if COM1 matches no position here but matches a published ATC
    position of a nearby airport, move the session there (same callsign, language, flight plan).
    """
    global pending_atc_call
    if not freq_hz or session.phase_for_frequency(int(freq_hz)) is not None:
        return session
    worked = getattr(session, "worked_freq_mhz", None)
    if worked and abs(int(freq_hz) / 1e6 - worked) < 0.013:
        return session   # the arrival controller kept the pilot on this frequency (no ping-pong back)
    st = session.aircraft_state
    if not (st.latitude or st.longitude):
        return session
    for ap, _d in get_airport_data_service().nearest(st.latitude, st.longitude, max_nm=RETUNE_RADIUS_NM,
                                                     min_runway_ft=0)[:25]:
        if ap.icao == session.airport_icao:
            continue
        m = get_frequency_service().match_frequency(ap.icao, int(freq_hz) / 1_000_000, tolerance=0.013)
        if not m or m[0] not in _ATC_FREQ_TYPES:
            continue
        old = session
        new = session_manager.create_session(
            ap.icao, old.callsign, frequency_service=get_frequency_service(),
            airport_data_service=get_airport_data_service(), language=old.language, flight_plan=old.flight_plan)
        new.aircraft_state = old.aircraft_state
        new.known_facts = {k: v for k, v in old.known_facts.items() if ap.icao in k}
        trigger_engine.reset_session(old)
        pending_atc_call = None
        logger.info(f"COM1 {int(freq_hz) / 1e6:.3f} is {ap.icao} {m[0]}: session moved {old.airport_icao} -> {ap.icao}")
        return new
    return session


def _make_audio(text: str, fmt: str) -> Optional[bytes]:
    """TTS that never takes the request down: returns None on failure or when disabled."""
    if not config.TTS_ENABLED:
        return None
    try:
        svc = get_audio_service()
        return svc.speak_atc_wav(text) if fmt == "wav" else svc.speak_atc(text)
    except Exception as e:  # noqa: BLE001
        logger.error(f"TTS failed: {e}")
        return None


def _evaluate_triggers(session, force_initiative: bool = False):
    global pending_atc_call, _last_initiative_at
    if pending_atc_call is not None:
        return
    call = trigger_engine.evaluate(session)
    if call:
        pending_atc_call = call
        logger.info(f"ATC trigger: {call.message}")
        return
    now = datetime.now()
    if session.sim_frozen():
        return  # sim paused: no stand-by callbacks or radio checks into a frozen cockpit
    if session.standby_at and (now - session.standby_at).total_seconds() >= config.STANDBY_CALLBACK_S:
        # the controller said "stand by" on a request: come back with the decision, unprompted
        waited = (now - session.standby_at).total_seconds()
        request_text = session.standby_request
        session.standby_at = None          # one callback per stand-by (a new stand-by re-arms it)
        _last_initiative_at = now
        reason = (f"{waited:.0f}s ago you told the pilot to stand by on this request: \"{request_text}\". "
                  "Call the pilot now with your decision (approve, amend or deny); do not say stand by again")
        try:
            text = get_llm_service().controller_initiative(session, reason=reason)
        except Exception as e:  # noqa: BLE001
            logger.error(f"standby callback failed: {e}")
            return
        if text:
            session.conversation.pop()
            pending_atc_call = ATCCall(TriggerType.INITIATIVE, text, new_phase=session.pending_handoff)
            session.pending_handoff = None
        return
    if not (config.INITIATIVE_ENABLED or force_initiative):
        return
    quiet_s = (now - session.last_transmission).total_seconds()
    since_last = (now - _last_initiative_at).total_seconds() if _last_initiative_at else 1e9
    airborne_or_moving = (not session.aircraft_state.on_ground) or session.aircraft_state.groundspeed > 5
    answered = session.unanswered_initiatives < config.INITIATIVE_MAX_UNANSWERED
    if force_initiative or (quiet_s >= config.INITIATIVE_SILENCE_S and answered
                            and since_last >= config.INITIATIVE_MIN_INTERVAL_S and airborne_or_moving):
        _last_initiative_at = now
        reason = f"{quiet_s:.0f}s of radio silence"
        try:
            text = get_llm_service().controller_initiative(session, reason=reason)
        except Exception as e:  # noqa: BLE001
            logger.error(f"initiative failed: {e}")
            return
        if text:
            # controller spoke: the session already has the transmission; deliver as a pending call
            session.conversation.pop()  # /pending_atc ack re-adds it (keeps ordering with audio playback)
            session.unanswered_initiatives += 1
            pending_atc_call = ATCCall(TriggerType.INITIATIVE, text, new_phase=session.pending_handoff)
            session.pending_handoff = None


def _require_session():
    session = session_manager.get_active_session()
    if not session:
        return None, (jsonify({"error": "No active session. Call /tune first."}), 400)
    return session, None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    session = session_manager.get_active_session()
    return jsonify({
        "status": "ok",
        "service": "atc-server",
        "model": llm_service.model if llm_service else config.OPENAI_MODEL,
        "reasoning_effort": config.OPENAI_REASONING_EFFORT,
        "language": current_language,
        "tts": {"provider": config.TTS_PROVIDER, "enabled": config.TTS_ENABLED,
                "last_used": audio_service.last_tts_provider if audio_service else None},
        "stt": {"provider": config.STT_PROVIDER,
                "last_used": audio_service.last_stt_provider if audio_service else None},
        "session": {"airport": session.airport_icao, "callsign": session.callsign,
                    "phase": session.phase.value} if session else None,
        "pending_atc": pending_atc_call is not None,
    })


@app.route("/tune", methods=["POST"])
def tune():
    """
    Create/switch session.

    Request JSON:
        airport_icao   required (or "auto" with latitude/longitude to pick the nearest)
        callsign       default N12345
        frequency      Delivery|Ground|Tower|Departure|Approach|Center (or pt-br names)
        language       "en" | "pt-br"
        flight_plan    {aircraft_type, flight_rules, departure, destination, route, cruise_altitude_ft, remarks}
        state          optional initial aircraft state
    """
    global current_language, pending_atc_call
    try:
        data = request.get_json(silent=True) or {}
        icao = (data.get("airport_icao") or "").upper()
        if not icao:
            return jsonify({"error": "airport_icao required"}), 400

        callsign = data.get("callsign") or config.DEFAULT_CALLSIGN
        fp = FlightPlan.from_dict(data.get("flight_plan"))
        if not fp.destination:
            saved = _recall_plan(callsign)
            if saved:
                saved.aircraft_type = fp.aircraft_type or saved.aircraft_type
                fp = saved
                logger.info(f"reusing the last flight plan of {callsign}: {fp.summary()}")
        state = data.get("state") or {}

        if icao == "AUTO":
            lat = data.get("latitude", state.get("latitude"))
            lon = data.get("longitude", state.get("longitude"))
            if lat is None or lon is None:
                return jsonify({"error": "latitude/longitude required for airport_icao=auto"}), 400
            near = get_airport_data_service().nearest(float(lat), float(lon))
            if not near:
                return jsonify({"error": "no airport within range"}), 404
            icao = _pick_airport_for_plan(fp, {"latitude": lat, "longitude": lon, **state}, near[0][0].icao)

        frequency = data.get("frequency") or "Ground"
        language = (data.get("language") or current_language).lower()
        if language not in ("en", "pt-br"):
            return jsonify({"error": "language must be 'en' or 'pt-br'"}), 400

        with _lock:
            if language != current_language:
                logger.info(f"ATC language: {current_language} -> {language}")
                current_language = language
                get_llm_service(language)

            old = session_manager.get_active_session()
            if old:
                trigger_engine.reset_session(old)
            pending_atc_call = None

            session = session_manager.create_session(
                icao, callsign,
                frequency_service=get_frequency_service(),
                airport_data_service=get_airport_data_service(),
                language=language,
                flight_plan=fp,
            )
            if frequency.lower() == "auto":
                session.set_phase(_auto_phase(session, state))
            else:
                session.set_phase(phase_from_text(frequency) or ATCPhase.GROUND)
            if state:
                session = _apply_state(session, state)
            _remember_plan(callsign, session.flight_plan)
        _prefetch_charts(session.flight_plan.destination)

        return jsonify({"status": "ok", "session": _session_brief(session)})
    except Exception as e:  # noqa: BLE001
        logger.exception("Error in /tune")
        return jsonify({"error": str(e)}), 500


def _session_brief(session) -> dict:
    return {
        "airport_icao": session.airport_icao,
        "airport_name": session.airport.name if session.airport else None,
        "runways": session.airport.runway_idents() if session.airport else [],
        "callsign": session.callsign,
        "frequency": session.current_frequency,
        "frequency_mhz": session.current_frequency_mhz,
        "phase": session.phase.value,
        "language": session.language,
        "flight_plan": session.flight_plan.to_dict(),
    }


@app.route("/transmit", methods=["POST"])
def transmit():
    """
    Pilot audio -> ATC audio.

    Request JSON: {"audio": <base64>, "state": {...}, "format": "mp3"|"wav"}
    Response: audio bytes; headers X-ATC-Text, X-Pilot-Text, X-Audio-Provider.
    """
    try:
        data = request.get_json(silent=True) or {}
        session, err = _require_session()
        if err:
            return err
        if "audio" not in data:
            return jsonify({"error": "No audio provided"}), 400

        fmt = (data.get("format") or "mp3").lower()
        freq_warning = None
        if data.get("state"):
            session = _apply_state(session, data["state"])
            if data["state"].get("com1_freq_hz"):
                ok, msg = session.validate_pilot_frequency(int(data["state"]["com1_freq_hz"]))
                if not ok:
                    freq_warning = msg

        audio_bytes = base64.b64decode(data["audio"])
        if config.SAVE_LAST_TRANSMIT:
            try:  # keep the last pilot recording for mic debugging
                Path(".cache").mkdir(exist_ok=True)
                Path(".cache/last_transmit.wav").write_bytes(audio_bytes)
            except OSError:
                pass
        audio_svc = get_audio_service()
        pilot_text = audio_svc.transcribe(audio_bytes, language=session.language)
        if not pilot_text.strip():
            return jsonify({"error": "Could not transcribe audio"}), 400

        llm_input = pilot_text
        if freq_warning:
            llm_input = f"[{freq_warning}. The pilot is not on your frequency; respond only with a brief '{session.callsign}, wrong frequency' style correction, nothing else.] {pilot_text}"

        with _lock:
            atc_text = get_llm_service().generate_response(llm_input, session)

        atc_audio = _make_audio(atc_text, fmt) if atc_text else None   # "" = controller stays silent
        if atc_audio is None:
            resp = jsonify({"pilot_text": pilot_text, "atc_text": atc_text, "audio_base64": None})
        else:
            resp = send_file(io.BytesIO(atc_audio), mimetype="audio/wav" if fmt == "wav" else "audio/mpeg",
                             as_attachment=True, download_name=f"atc_response.{fmt}")
        resp.headers["X-ATC-Text"] = _header_safe(atc_text)
        resp.headers["X-Pilot-Text"] = _header_safe(pilot_text)
        resp.headers["X-Audio-Provider"] = audio_svc.last_tts_provider or "none"
        resp.headers["X-STT-Provider"] = audio_svc.last_stt_provider or "none"
        return resp
    except Exception as e:  # noqa: BLE001
        logger.exception("Error in /transmit")
        return jsonify({"error": str(e)}), 500


def _header_safe(text: str) -> str:
    """HTTP headers must be latin-1; base64-encode anything else."""
    try:
        text.encode("latin-1")
        return text.replace("\n", " ")
    except UnicodeEncodeError:
        return "b64:" + base64.b64encode(text.encode("utf-8")).decode()


@app.route("/transmit_text", methods=["POST"])
def transmit_text():
    """
    Text-only transmission.

    Request JSON: {"text": "...", "state": {...}, "audio": true|false, "format": "mp3"|"wav"}
    Response JSON: {pilot_text, atc_text, audio_base64|null, latency_s, tool_calls, clearances}
    """
    try:
        data = request.get_json(silent=True) or {}
        if not data.get("text"):
            return jsonify({"error": "No text provided"}), 400
        session, err = _require_session()
        if err:
            return err
        if data.get("state"):
            session = _apply_state(session, data["state"])

        want_audio = data.get("audio", True) not in (False, "false", 0)
        fmt = (data.get("format") or "mp3").lower()

        with _lock:
            llm = get_llm_service()
            n_before = len(session.clearances)
            atc_text = llm.generate_response(data["text"], session)
            new_clearances = [c.to_dict() for c in session.clearances[n_before:]]

        audio = _make_audio(atc_text, fmt) if want_audio and atc_text else None
        return jsonify({
            "pilot_text": data["text"],
            "atc_text": atc_text,
            "audio_base64": base64.b64encode(audio).decode() if audio else None,
            "audio_format": fmt if audio else None,
            "audio_provider": (audio_service.last_tts_provider if audio_service and audio else None),
            "latency_s": round(llm.last_latency_s, 2),
            "tool_calls": llm.last_tool_calls,
            "clearances": new_clearances,
            "phase": session.phase.value,
            "frequency": session.current_frequency,
        })
    except Exception as e:  # noqa: BLE001
        logger.exception("Error in /transmit_text")
        return jsonify({"error": str(e)}), 500


@app.route("/state", methods=["POST"])
def update_state():
    """Aircraft state update (plugin flight loop). Returns any pending ATC-initiated call."""
    try:
        data = request.get_json(silent=True) or {}
        session, err = _require_session()
        if err:
            return err
        force = request.args.get("initiative", "false").lower() == "true"
        with _lock:
            session = _apply_state(session, data)
            _evaluate_triggers(session, force_initiative=force)
        return jsonify({
            "status": "ok",
            "phase": session.phase.value,
            "frequency": session.current_frequency,
            "distance_nm": session.distance_to_airport_nm(),
            "pending_atc": pending_atc_call.to_dict() if pending_atc_call else None,
        })
    except Exception as e:  # noqa: BLE001
        logger.exception("Error in /state")
        return jsonify({"error": str(e)}), 500


@app.route("/pending_atc", methods=["GET"])
def get_pending_atc():
    """
    Poll / acknowledge the pending ATC-initiated call.
    Query: audio=true|false, format=mp3|wav, ack=true|false
    """
    global pending_atc_call
    try:
        include_audio = request.args.get("audio", "true").lower() == "true"
        fmt = request.args.get("format", "mp3").lower()
        acknowledge = request.args.get("ack", "false").lower() == "true"

        if pending_atc_call is None:
            return jsonify({"pending": False, "call": None})

        session = session_manager.get_active_session()
        call = pending_atc_call
        out = {"pending": True, "call": call.to_dict()}

        if include_audio:
            audio = _make_audio(call.message, fmt)
            out["audio_base64"] = base64.b64encode(audio).decode() if audio else None
            out["audio_format"] = fmt if audio else None

        if acknowledge and session:
            with _lock:
                session.add_atc_transmission(call.message)
                if call.new_phase:
                    session.set_phase(call.new_phase)
                pending_atc_call = None
            logger.info(f"ATC call acknowledged: {call.message}")
            out["phase"] = session.phase.value
            out["frequency"] = session.current_frequency
            out["frequency_mhz"] = session.current_frequency_mhz
        return jsonify(out)
    except Exception as e:  # noqa: BLE001
        logger.exception("Error in /pending_atc")
        return jsonify({"error": str(e)}), 500


@app.route("/session", methods=["GET"])
def get_session():
    session = session_manager.get_active_session()
    if not session:
        return jsonify({"error": "No active session"}), 404
    return jsonify(session.to_dict())


@app.route("/conversation", methods=["GET"])
def get_conversation():
    session = session_manager.get_active_session()
    if not session:
        return jsonify({"error": "No active session"}), 404
    return jsonify({"airport_icao": session.airport_icao, "callsign": session.callsign,
                    "messages": [m.to_dict() for m in session.conversation]})


@app.route("/reset", methods=["POST"])
def reset():
    global pending_atc_call
    session = session_manager.get_active_session()
    if session:
        trigger_engine.reset_session(session)
        session_manager.end_session(session.callsign, session.airport_icao)
    pending_atc_call = None
    return jsonify({"status": "ok", "message": "Session reset"})


@app.route("/frequencies/<icao>", methods=["GET"])
def get_frequencies(icao: str):
    try:
        freqs = get_frequency_service().get_all_atc(icao.upper())
        return jsonify({"airport_icao": icao.upper(), "frequencies": {
            t: {"frequency_mhz": f.frequency_mhz, "frequency_formatted": f.frequency_formatted(),
                "description": f.description} for t, f in freqs.items()}})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/airport/<icao>", methods=["GET"])
def get_airport(icao: str):
    ap = get_airport_data_service().get(icao)
    if not ap:
        return jsonify({"error": f"unknown airport {icao.upper()}"}), 404
    return jsonify(ap.to_dict())


@app.route("/nearest_airport", methods=["GET"])
def nearest_airport():
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
    except (KeyError, ValueError):
        return jsonify({"error": "lat and lon query params required"}), 400
    max_nm = float(request.args.get("max_nm", 50))
    near = get_airport_data_service().nearest(lat, lon, max_nm=max_nm)[:10]
    return jsonify({"airports": [{"icao": ap.icao, "name": ap.name, "distance_nm": round(d, 1),
                                  "type": ap.type, "runways": ap.runway_idents()} for ap, d in near]})


@app.route("/charts/<icao>", methods=["GET"])
def get_charts(icao: str):
    cat = request.args.get("category")
    svc = get_chart_service()
    charts = svc.get_charts_by_category(icao, cat) if cat else svc.list_charts(icao)
    prov = svc.provider_for(icao)
    return jsonify({"airport_icao": icao.upper(), "provider": prov.name if prov else None,
                    "count": len(charts), "charts": charts})


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    init_services()
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    Virtual ATC Server                         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Web Playground: http://{HOST}:{PORT}/                          ║
    ║  Model: {config.OPENAI_MODEL:<20s}  Language: {current_language:<6s}          ║
    ║  TTS: {config.TTS_PROVIDER:<10s} STT: {config.STT_PROVIDER:<10s}                      ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║    POST /tune            POST /transmit    POST /transmit_text║
    ║    POST /state           GET  /pending_atc GET  /session      ║
    ║    GET  /frequencies/X   GET  /airport/X   GET  /charts/X     ║
    ║    GET  /nearest_airport?lat&lon           POST /reset        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False, threaded=True)
