"""
PI_VirtualATC.py - X-Plane Virtual ATC Plugin (XPPython3)

Push-to-talk AI ATC for X-Plane 12.

Installation:
    Copy this file to: X-Plane 12/Resources/plugins/PythonPlugins/PI_VirtualATC.py
    Bind a key/button to command "virtualatc/ptt" (Settings > Keyboard, search "Virtual ATC").

Requirements:
    - XPPython3 (https://xppython3.readthedocs.io)
    - pyaudio  (pip install pyaudio  -- into XPPython3's python, see its docs)
    - requests (pip install requests)
    - ATC backend server running (python atc_server.py), default http://127.0.0.1:5000

What this plugin does:
    - Hold PTT: records the mic (16 kHz mono).  Release: sends audio + aircraft state to /transmit.
    - Plays the ATC reply on the COM1 radio bus via xp.playPCMOnBus (WAV from the server).
    - Every 2 s sends aircraft state to /state (in a worker thread, never blocking the sim).
    - Handles ATC-initiated calls (handoffs) returned by /pending_atc and plays them.
    - Auto-tunes the session on load (airport_icao="auto": nearest field on the ground; in the air, the
      nearer of the FMS departure/destination) and re-tunes when the server restarts.
    - Reads the flight plan loaded in the X-Plane FMS/GPS (departure, destination, route, cruise
      altitude) and sends it with /tune and on every change, so the controller knows where you go.

Configuration: virtualatc_config.txt next to this file (env vars ATC_SERVER_URL / ATC_LANGUAGE override):
    server_url=http://127.0.0.1:5000
    language=en            # or pt-br
    mic_device=            # pyaudio input index (blank = default mic)
    test_wav=virtualatc_test.wav   # command virtualatc/test_transmit sends this file as the pilot
    flight_rules=IFR       # IFR | VFR | blank (unknown): sent with the FMS flight plan
    callsign=              # e.g. N1323 (blank = the aircraft's tail number)
"""

import base64
import io
import os
import queue
import threading
import time
import wave
from typing import Optional

# ---------------------------------------------------------------------------
# X-Plane imports (with a mock so the module imports outside the sim for tests)
# ---------------------------------------------------------------------------
try:
    from XPPython3 import xp
    IN_XPLANE = True
except ImportError:
    IN_XPLANE = False

    class _MockXP:
        WidgetClass_MainWindow = 1
        WidgetClass_Button = 1
        WidgetClass_Caption = 1
        Msg_PushButtonPressed = 1
        CommandBegin = 0
        CommandContinue = 1
        CommandEnd = 2
        AudioRadioCom1 = 1
        AudioUI = 6

        def log(self, msg): print(f"[XP] {msg}")
        def createWidget(self, *a): return 1
        def destroyWidget(self, *a): pass
        def addWidgetCallback(self, *a): pass
        def setWidgetDescriptor(self, *a): pass
        def findDataRef(self, name): return name
        def getDatad(self, ref): return 0.0
        def getDataf(self, ref): return 0.0
        def getDatai(self, ref): return 0
        def getDatas(self, ref, offset=0, count=256): return "N12345"
        def createFlightLoop(self, *a, **k): return 1
        def scheduleFlightLoop(self, *a, **k): pass
        def destroyFlightLoop(self, *a): pass
        def createCommand(self, *a): return 1
        def registerCommandHandler(self, *a): pass
        def unregisterCommandHandler(self, *a): pass
        def playPCMOnBus(self, *a, **k): return 1
        def stopAudio(self, *a): pass
    xp = _MockXP()

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """key=value lines in virtualatc_config.txt next to this file (env vars override)."""
    cfg = {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "virtualatc_config.txt")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip().lower()] = v.strip()
    except OSError:
        pass
    return cfg


CONFIG = _load_config()
ATC_SERVER_URL = os.environ.get("ATC_SERVER_URL", CONFIG.get("server_url", "http://127.0.0.1:5000"))
ATC_LANGUAGE = os.environ.get("ATC_LANGUAGE", CONFIG.get("language", "en"))
MIC_DEVICE = CONFIG.get("mic_device")           # pyaudio input device index, or blank for default
TEST_WAV = CONFIG.get("test_wav", "virtualatc_test.wav")   # played through /transmit by virtualatc/test_transmit
FLIGHT_RULES = CONFIG.get("flight_rules", "").strip().upper()   # IFR | VFR | "" (unknown)
CALLSIGN = CONFIG.get("callsign", "").strip().upper()            # overrides the aircraft tail number
FMS_READ_EVERY = 40              # flight-loop ticks (0.25 s) between FMS reads
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16 if pyaudio else None
STATE_UPDATE_INTERVAL = 2.0      # seconds between /state posts
PENDING_REPLAY_GUARD = 5.0       # seconds before another ATC-initiated call can be handled

DATAREFS = {
    "latitude": "sim/flightmodel/position/latitude",
    "longitude": "sim/flightmodel/position/longitude",
    "elevation": "sim/flightmodel/position/elevation",          # meters MSL
    "heading": "sim/flightmodel/position/psi",                  # true heading
    "indicated_airspeed": "sim/flightmodel/position/indicated_airspeed",
    "groundspeed": "sim/flightmodel/position/groundspeed",      # m/s
    "vertical_speed": "sim/flightmodel/position/vh_ind_fpm",
    "squawk": "sim/cockpit/radios/transponder_code",
    "com1_freq": "sim/cockpit2/radios/actuators/com1_frequency_hz_833",  # Hz, 8.33 aware
    "com1_freq_legacy": "sim/cockpit/radios/com1_freq_hz",     # 10 kHz units (fallback)
    "on_ground": "sim/flightmodel/failures/onground_any",
    "tail_number": "sim/aircraft/view/acf_tailnum",
    "icao_type": "sim/aircraft/view/acf_ICAO",
}


# ---------------------------------------------------------------------------
# FMS flight plan
# ---------------------------------------------------------------------------

def fms_plan_from_entries(entries, airport_type: int = 1) -> Optional[dict]:
    """
    [(nav_type, nav_id, altitude_ft), ...] from the X-Plane FMS -> flight plan dict, or None when no
    destination is loaded. Departure = first entry if it is an airport, destination = last airport.
    """
    pts = [(t, (nid or "").strip().upper(), int(alt or 0)) for t, nid, alt in entries if (nid or "").strip()]
    airports = [i for i, (t, _n, _a) in enumerate(pts) if t == airport_type]
    if not airports:
        return None
    dest_i = airports[-1]
    dep_i = 0 if (airports[0] == 0 and dest_i > 0) else None
    if dep_i is None and dest_i == 0:
        return None                       # only a departure airport loaded: no destination yet
    start = 1 if dep_i == 0 else 0
    route = [n for _t, n, _a in pts[start:dest_i]][:25]
    alts = [a for _t, _n, a in pts if a >= 1000]
    return {
        "departure": pts[0][1] if dep_i == 0 else "",
        "destination": pts[dest_i][1],
        "route": " ".join(route) if route else "DCT",
        "cruise_altitude_ft": max(alts) if alts else None,
    }


# ---------------------------------------------------------------------------
# Audio Recorder
# ---------------------------------------------------------------------------

class AudioRecorder:
    """Records microphone audio to WAV bytes using PyAudio (background thread)."""

    def __init__(self):
        self.pa = pyaudio.PyAudio() if pyaudio else None
        self.stream = None
        self.frames: list[bytes] = []
        self.is_recording = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if not self.pa:
            xp.log("VirtualATC: cannot record, pyaudio not available")
            return False
        if self.is_recording:
            return False
        self.frames = []
        self.is_recording = True
        # Try the configured device first, then the default device; mono 16k is ideal,
        # but some Windows devices (and RDP-redirected audio) only open stereo or 44.1/48 kHz.
        devices = []
        if MIC_DEVICE not in (None, ""):
            devices.append(int(MIC_DEVICE))
        devices.append(None)
        attempts = [(d, ch, rate) for d in devices for ch in (1, 2) for rate in (16000, 44100, 48000)]
        last_err = None
        for dev, ch, rate in attempts:
            kwargs = {"input_device_index": dev} if dev is not None else {}
            try:
                self.stream = self.pa.open(format=AUDIO_FORMAT, channels=ch, rate=rate, input=True,
                                           frames_per_buffer=CHUNK_SIZE, **kwargs)
                self.channels, self.rate = ch, rate
                xp.log(f"VirtualATC: recording (device={dev if dev is not None else 'default'}, {ch}ch, {rate}Hz)")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.stream = None
        if not self.stream:
            xp.log(f"VirtualATC: failed to start recording on any device: {last_err}")
            self.is_recording = False
            return False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def _loop(self):
        while self.is_recording and self.stream:
            try:
                self.frames.append(self.stream.read(CHUNK_SIZE, exception_on_overflow=False))
            except Exception as e:  # noqa: BLE001
                xp.log(f"VirtualATC: recording error: {e}")
                break

    def stop(self) -> bytes:
        self.is_recording = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
            self.stream = None
        if not self.frames:
            return b""
        pcm = b"".join(self.frames)
        if getattr(self, "channels", 1) == 2:
            pcm = _downmix_stereo(pcm)
        if getattr(self, "rate", SAMPLE_RATE) != SAMPLE_RATE:
            pcm = _resample_pcm16(pcm, self.rate, SAMPLE_RATE)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.pa.get_sample_size(AUDIO_FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)
        self.frames = []
        return buf.getvalue()

    def cleanup(self):
        if self.is_recording:
            self.stop()
        if self.pa:
            self.pa.terminate()


def _resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampler for 16-bit mono PCM (stdlib only)."""
    if src_rate == dst_rate:
        return pcm
    import array
    src = array.array("h")
    src.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    n_src = len(src)
    n_dst = int(n_src * dst_rate / src_rate)
    dst = array.array("h", [0]) * n_dst
    ratio = src_rate / dst_rate
    for i in range(n_dst):
        pos = i * ratio
        j = int(pos)
        frac = pos - j
        a = src[j]
        b = src[j + 1] if j + 1 < n_src else a
        dst[i] = int(a + (b - a) * frac)
    return dst.tobytes()


def _downmix_stereo(pcm: bytes) -> bytes:
    """Interleaved 16-bit stereo -> mono (average)."""
    import array
    st = array.array("h")
    st.frombytes(pcm[: len(pcm) - (len(pcm) % 4)])
    mono = array.array("h", [0]) * (len(st) // 2)
    for i in range(len(mono)):
        mono[i] = (st[2 * i] + st[2 * i + 1]) // 2
    return mono.tobytes()


# ---------------------------------------------------------------------------
# Audio Player (X-Plane sound bus)
# ---------------------------------------------------------------------------

class AudioPlayer:
    """
    Plays 16-bit mono PCM on the COM1 radio bus using the XPLM 4 sound API.

    X-Plane's API wants raw PCM, so the server is asked for WAV (format="wav") and
    the header is stripped here. Playback must be issued from the main thread, so
    worker threads enqueue buffers and the flight loop drains the queue.
    """

    def __init__(self):
        self.queue: "queue.Queue[tuple[bytes, int]]" = queue.Queue()
        self.current = None

    def enqueue_wav(self, wav_bytes: bytes):
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                rate = wf.getframerate()
                pcm = wf.readframes(wf.getnframes())
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    xp.log("VirtualATC: unexpected WAV format, expected 16-bit mono")
            self.queue.put((pcm, rate))
        except Exception as e:  # noqa: BLE001
            xp.log(f"VirtualATC: bad audio from server: {e}")

    def pump(self):
        """Call from the flight loop (main thread)."""
        if self.current is not None:
            return  # one transmission at a time (radio realism)
        try:
            pcm, rate = self.queue.get_nowait()
        except queue.Empty:
            return
        try:
            # playPCMOnBus(audioBuffer, bufferSize, soundFormat=bytes per frame, freqHz, numChannels,
            #              loop, audioType, callback, refCon)
            self.current = xp.playPCMOnBus(pcm, len(pcm), 2, rate, 1, 0, xp.AudioRadioCom1,
                                           self._on_done, None)
            xp.log(f"VirtualATC: playing {len(pcm) // 2 / rate:.1f}s on COM1 bus")
            if not self.current:
                self.current = None
        except Exception as e:  # noqa: BLE001
            xp.log(f"VirtualATC: playPCMOnBus failed: {e}")
            self.current = None

    def _on_done(self, refcon, is_done):
        self.current = None

    def stop(self):
        if self.current:
            try:
                xp.stopAudio(self.current)
            except Exception:  # noqa: BLE001
                pass
            self.current = None


# ---------------------------------------------------------------------------
# Backend Client
# ---------------------------------------------------------------------------

class ATCBackendClient:
    def __init__(self, base_url: str = ATC_SERVER_URL):
        self.base_url = base_url.rstrip("/")

    def _ok(self) -> bool:
        return requests is not None

    def health(self) -> bool:
        if not self._ok():
            return False
        try:
            return requests.get(f"{self.base_url}/health", timeout=3).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def tune(self, airport_icao: str, callsign: str, frequency: str = "auto",
             language: str = ATC_LANGUAGE, lat: float = None, lon: float = None,
             flight_plan: Optional[dict] = None, state: Optional[dict] = None) -> dict:
        if not self._ok():
            return {"error": "requests not available"}
        payload = {"airport_icao": airport_icao, "callsign": callsign, "frequency": frequency,
                   "language": language, "flight_plan": flight_plan or {}}
        if lat is not None:
            payload.update({"latitude": lat, "longitude": lon})
        if state:
            payload["state"] = state
        try:
            return requests.post(f"{self.base_url}/tune", json=payload, timeout=30).json()
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    def transmit(self, audio_bytes: bytes, state: dict) -> tuple[Optional[bytes], str, str]:
        """Returns (wav_bytes|None, atc_text, pilot_text)."""
        if not self._ok():
            return None, "", ""
        try:
            r = requests.post(f"{self.base_url}/transmit",
                              json={"audio": base64.b64encode(audio_bytes).decode(), "state": state,
                                    "format": "wav"},
                              timeout=90)
            if r.status_code != 200:
                xp.log(f"VirtualATC: transmit failed {r.status_code}: {r.text[:200]}")
                return None, "", ""
            atc_text = _decode_header(r.headers.get("X-ATC-Text", ""))
            pilot_text = _decode_header(r.headers.get("X-Pilot-Text", ""))
            if r.headers.get("Content-Type", "").startswith("application/json"):
                return None, r.json().get("atc_text", atc_text), pilot_text  # TTS unavailable
            return r.content, atc_text, pilot_text
        except Exception as e:  # noqa: BLE001
            xp.log(f"VirtualATC: transmit error: {e}")
            return None, "", ""

    def update_state(self, state: dict) -> dict:
        if not self._ok():
            return {}
        try:
            r = requests.post(f"{self.base_url}/state", json=state, timeout=5)
            if r.status_code == 400:
                return {"_no_session": True}       # server restarted: the plugin must /tune again
            return r.json() if r.status_code == 200 else {}
        except Exception:  # noqa: BLE001
            return {}

    def get_pending_atc(self, acknowledge: bool = True) -> dict:
        if not self._ok():
            return {}
        try:
            r = requests.get(f"{self.base_url}/pending_atc",
                             params={"audio": "true", "format": "wav", "ack": "true" if acknowledge else "false"},
                             timeout=60)
            return r.json() if r.status_code == 200 else {}
        except Exception:  # noqa: BLE001
            return {}


def _decode_header(v: str) -> str:
    if v.startswith("b64:"):
        try:
            return base64.b64decode(v[4:]).decode("utf-8")
        except Exception:  # noqa: BLE001
            return v
    return v


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class PythonInterface:
    def __init__(self):
        self.Name = "Virtual ATC v1.1"
        self.Sig = "virtualatc.xppython3"
        self.Desc = "AI-powered virtual ATC with push-to-talk"

        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        self.backend = ATCBackendClient()

        self.is_recording = False
        self.is_enabled = False
        self.callsign = "N12345"
        self.aircraft_type = ""
        self.session: dict = {}

        self.dataref_handles: dict = {}
        self.widget_window = None
        self.ptt_button = None
        self.status_label = None
        self.freq_label = None
        self.flight_loop_id = None
        self.ptt_command = None
        self.test_command = None
        self.ptt_toggle_command = None

        # Worker thread state
        self._state_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_state: dict = {}
        self._state_lock = threading.Lock()
        self._pending_guard_until = 0.0
        self._retune_after = 0.0
        self._fms_plan: Optional[dict] = None     # read on the main thread (flight loop)
        self._fms_sent: Optional[dict] = None     # last plan the server has
        self._fms_tick = 0
        self._fms_error_logged = False
        self._status_queue: "queue.Queue[str]" = queue.Queue()

    # -- lifecycle -------------------------------------------------------

    def XPluginStart(self):
        xp.log(f"VirtualATC: starting {self.Name}")
        return self.Name, self.Sig, self.Desc

    def XPluginEnable(self):
        for name, path in DATAREFS.items():
            ref = xp.findDataRef(path)
            if ref:
                self.dataref_handles[name] = ref
            elif name not in ("com1_freq", "com1_freq_legacy"):
                xp.log(f"VirtualATC: dataref not found {path}")

        self._read_aircraft_identity()
        self._create_ui()
        self._create_ptt_command()

        self.flight_loop_id = xp.createFlightLoop(self._flight_loop_callback)
        xp.scheduleFlightLoop(self.flight_loop_id, 0.25)   # main-thread pump: audio + status

        self._stop_event.clear()
        self._state_thread = threading.Thread(target=self._state_worker, daemon=True)
        self._state_thread.start()

        self.is_enabled = True               # the state worker tunes as soon as server + position are there
        xp.log("VirtualATC: enabled")
        return 1

    def XPluginDisable(self):
        self.is_enabled = False
        self._stop_event.set()
        if self.is_recording:
            self.recorder.stop()
            self.is_recording = False
        self.recorder.cleanup()
        self.player.stop()
        if self.flight_loop_id:
            xp.destroyFlightLoop(self.flight_loop_id)
            self.flight_loop_id = None
        if self.widget_window:
            xp.destroyWidget(self.widget_window, 1)
            self.widget_window = None
        if self.ptt_command:
            xp.unregisterCommandHandler(self.ptt_command, self._ptt_command_handler, 1, None)
        if getattr(self, "test_command", None):
            xp.unregisterCommandHandler(self.test_command, self._test_command_handler, 1, None)
        if getattr(self, "ptt_toggle_command", None):
            xp.unregisterCommandHandler(self.ptt_toggle_command, self._ptt_toggle_command_handler, 1, None)
        xp.log("VirtualATC: disabled")

    def XPluginStop(self):
        pass

    def XPluginReceiveMessage(self, inFromWho, inMessage, inParam):
        # XPLM_MSG_PLANE_LOADED (102) for user aircraft (param 0): re-read identity
        if inMessage == 102 and inParam == 0:
            self._read_aircraft_identity()

    def _read_aircraft_identity(self):
        try:
            if CALLSIGN:
                self.callsign = CALLSIGN
            elif "tail_number" in self.dataref_handles:
                tail = xp.getDatas(self.dataref_handles["tail_number"])
                if tail and tail.strip():
                    self.callsign = tail.strip()
            if "icao_type" in self.dataref_handles:
                t = xp.getDatas(self.dataref_handles["icao_type"])
                if t and t.strip():
                    self.aircraft_type = t.strip()
        except Exception as e:  # noqa: BLE001
            xp.log(f"VirtualATC: identity read failed: {e}")

    # -- UI ---------------------------------------------------------------

    def _create_ui(self):
        self.widget_window = xp.createWidget(100, 300, 460, 180, 1, "Virtual ATC", 1, 0,
                                             xp.WidgetClass_MainWindow)
        self.ptt_button = xp.createWidget(110, 280, 260, 258, 1, "Talk (toggle)", 0,
                                          self.widget_window, xp.WidgetClass_Button)
        self.status_label = xp.createWidget(270, 280, 450, 258, 1, "Connecting...", 0,
                                            self.widget_window, xp.WidgetClass_Caption)
        self.freq_label = xp.createWidget(110, 250, 450, 228, 1, "Not tuned", 0,
                                          self.widget_window, xp.WidgetClass_Caption)
        xp.addWidgetCallback(self.widget_window, self._widget_callback)

    def _widget_callback(self, inMessage, inWidget, inParam1, inParam2):
        try:
            if inMessage == xp.Msg_PushButtonPressed and inParam1 == self.ptt_button:
                xp.log("VirtualATC: PTT button clicked in the plugin window")
                self._toggle_ptt()
                return 1
        except Exception as e:  # noqa: BLE001
            xp.log(f"VirtualATC: widget error: {e}")
        return 0

    def _set_status(self, text: str):
        """Thread-safe: queued and applied on the main thread."""
        self._status_queue.put(text)

    def _apply_status(self):
        while True:
            try:
                text = self._status_queue.get_nowait()
            except queue.Empty:
                return
            if text.startswith("FREQ:"):
                if self.freq_label:
                    xp.setWidgetDescriptor(self.freq_label, text[5:])
            elif self.status_label:
                xp.setWidgetDescriptor(self.status_label, text)

    # -- PTT ----------------------------------------------------------------

    def _create_ptt_command(self):
        self.ptt_command = xp.createCommand("virtualatc/ptt", "Virtual ATC push-to-talk (hold)")
        xp.registerCommandHandler(self.ptt_command, self._ptt_command_handler, 1, None)
        self.test_command = xp.createCommand("virtualatc/test_transmit",
                                             "Virtual ATC: send virtualatc_test.wav as the pilot (testing)")
        xp.registerCommandHandler(self.test_command, self._test_command_handler, 1, None)
        self.ptt_toggle_command = xp.createCommand(
            "virtualatc/ptt_toggle", "Virtual ATC push-to-talk (toggle: press to talk, press again to send)")
        xp.registerCommandHandler(self.ptt_toggle_command, self._ptt_toggle_command_handler, 1, None)

    def _ptt_toggle_command_handler(self, inCommand, inPhase, inRefcon):
        if inPhase == xp.CommandBegin:
            xp.log("VirtualATC: PTT toggle (command virtualatc/ptt_toggle)")
            self._toggle_ptt()
        return 0

    def _test_command_handler(self, inCommand, inPhase, inRefcon):
        if inPhase == xp.CommandBegin:
            self.send_test_wav()
        return 0

    def send_test_wav(self, path: Optional[str] = None) -> bool:
        """Send a WAV from disk through /transmit (lets a remote tester exercise STT->LLM->TTS->COM1)."""
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), TEST_WAV)
        try:
            with open(path, "rb") as fh:
                audio = fh.read()
        except OSError as e:
            xp.log(f"VirtualATC: test wav not found: {e}")
            self._set_status("No test wav")
            return False
        self._set_status("Test transmit...")
        with self._state_lock:
            state = dict(self._latest_state)
        threading.Thread(target=self._send_and_play, args=(audio, state), daemon=True).start()
        return True

    def _ptt_command_handler(self, inCommand, inPhase, inRefcon):
        if inPhase == xp.CommandBegin:
            xp.log("VirtualATC: PTT pressed (command virtualatc/ptt)")
            self._start_transmission()
        elif inPhase == xp.CommandEnd:
            self._end_transmission()
        return 0

    def _toggle_ptt(self):
        if self.is_recording:
            self._end_transmission()
        else:
            self._start_transmission()

    def _start_transmission(self):
        if self.is_recording:
            return
        self.player.stop()  # you can't hear ATC while you key the mic
        if self.recorder.start():
            self.is_recording = True
            self._set_status("Transmitting...")
            xp.setWidgetDescriptor(self.ptt_button, "Release to send")
        else:
            self._set_status("Mic unavailable")

    def _end_transmission(self):
        if not self.is_recording:
            return
        self.is_recording = False
        xp.setWidgetDescriptor(self.ptt_button, "Talk (toggle)")
        audio = self.recorder.stop()
        if len(audio) < 4000:  # < ~0.1 s
            self._set_status("Too short")
            return
        self._set_status("Waiting for ATC...")
        state = self._read_state()
        threading.Thread(target=self._send_and_play, args=(audio, state), daemon=True).start()

    def _send_and_play(self, audio: bytes, state: dict):
        wav, atc_text, pilot_text = self.backend.transmit(audio, state)
        if pilot_text:
            xp.log(f"VirtualATC PILOT: {pilot_text}")
        if atc_text:
            xp.log(f"VirtualATC ATC:   {atc_text}")
            self._set_status(atc_text[:60])
        else:
            self._set_status("No response")
        if wav:
            self.player.enqueue_wav(wav)

    # -- aircraft state ----------------------------------------------------

    def _read_state(self) -> dict:
        """Read datarefs. Must run on the main thread (flight loop)."""
        h = self.dataref_handles
        s = {}
        if "latitude" in h:
            s["latitude"] = xp.getDatad(h["latitude"])
        if "longitude" in h:
            s["longitude"] = xp.getDatad(h["longitude"])
        if "elevation" in h:
            s["altitude_ft"] = xp.getDatad(h["elevation"]) * 3.28084
        if "heading" in h:
            s["heading"] = xp.getDataf(h["heading"])
        if "indicated_airspeed" in h:
            s["indicated_airspeed"] = xp.getDataf(h["indicated_airspeed"])
        if "groundspeed" in h:
            s["groundspeed"] = xp.getDataf(h["groundspeed"]) * 1.94384
        if "vertical_speed" in h:
            s["vertical_speed"] = xp.getDataf(h["vertical_speed"])
        if "squawk" in h:
            s["squawk"] = xp.getDatai(h["squawk"])
        if "com1_freq" in h:
            # despite the name, com1_frequency_hz_833 is in kHz (e.g. 121600)
            s["com1_freq_hz"] = xp.getDatai(h["com1_freq"]) * 1_000
        elif "com1_freq_legacy" in h:
            s["com1_freq_hz"] = xp.getDatai(h["com1_freq_legacy"]) * 10_000
        if "on_ground" in h:
            s["on_ground"] = xp.getDatai(h["on_ground"]) > 0
        return s

    # -- main-thread pump ------------------------------------------------

    def _read_fms_plan(self) -> Optional[dict]:
        """Flight plan loaded in the FMS/GPS (main thread only). XP 12.1+ API first, then the legacy one."""
        try:
            n, get = 0, None
            fpl = getattr(xp, "Fpl_Pilot_Primary", None)
            if fpl is not None and hasattr(xp, "countFMSFlightPlanEntries"):
                n = xp.countFMSFlightPlanEntries(fpl)
                get = lambda i: xp.getFMSFlightPlanEntryInfo(fpl, i)  # noqa: E731
            if not n:
                n, get = xp.countFMSEntries(), xp.getFMSEntryInfo
            entries = []
            for i in range(n or 0):
                info = get(i)
                entries.append((getattr(info, "type", 0), getattr(info, "navID", ""), getattr(info, "altitude", 0)))
            return fms_plan_from_entries(entries, getattr(xp, "Nav_Airport", 1))
        except Exception as e:  # noqa: BLE001
            if not self._fms_error_logged:
                self._fms_error_logged = True
                xp.log(f"VirtualATC: FMS read failed ({e}); no flight plan from the FMS")
            return None

    def _plan_payload(self) -> dict:
        plan = {"aircraft_type": self.aircraft_type}
        if FLIGHT_RULES:
            plan["flight_rules"] = FLIGHT_RULES
        if self._fms_plan:
            plan.update(self._fms_plan)
        return plan

    def _flight_loop_callback(self, sinceLast, elapsedTime, counter, refCon):
        if not self.is_enabled:
            return 0
        try:
            with self._state_lock:
                self._latest_state = self._read_state()
            if self._fms_tick % FMS_READ_EVERY == 0:
                self._fms_plan = self._read_fms_plan()
            self._fms_tick += 1
            self.player.pump()
            self._apply_status()
        except Exception as e:  # noqa: BLE001
            xp.log(f"VirtualATC: flight loop error: {e}")
        return 0.25

    # -- worker thread ------------------------------------------------------

    def _connect_and_tune(self):
        """One attempt; the state worker retries every 10 s until a session exists (live 2026-11-08: the
        server was restarting while X-Plane loaded, the one-shot startup tune failed and never came back)."""
        if not self.backend.health():
            self._set_status("Server offline - retrying")
            return
        with self._state_lock:
            st = dict(self._latest_state)
        if not (st.get("latitude") or st.get("longitude")):
            self._set_status("Waiting for a flight")      # main menu: no position yet
            return
        plan = self._plan_payload()
        res = self.backend.tune("auto", self.callsign, "auto", ATC_LANGUAGE,
                                lat=st.get("latitude"), lon=st.get("longitude"), flight_plan=plan, state=st)
        if "session" in res:
            self._fms_sent = self._fms_plan
            if self._fms_plan:
                xp.log(f"VirtualATC: FMS flight plan sent: {self._fms_plan}")
            self.session = res["session"]
            f = self.session
            mhz = f" {f['frequency_mhz']:.3f}" if f.get("frequency_mhz") else ""
            self._set_status(f"Tuned {f['airport_icao']}")
            self._set_status(f"FREQ:{f['airport_icao']} {f['frequency']}{mhz}")
            xp.log(f"VirtualATC: tuned {f}")
        else:
            self._set_status(f"Tune failed: {res.get('error', '?')[:40]}")

    def _state_worker(self):
        """Posts state every STATE_UPDATE_INTERVAL s and handles ATC-initiated calls."""
        while not self._stop_event.wait(STATE_UPDATE_INTERVAL):
            if not self.session:
                if time.time() > self._retune_after:
                    self._retune_after = time.time() + 10
                    self._connect_and_tune()
                continue
            with self._state_lock:
                st = dict(self._latest_state)
            if not st:
                continue
            fms = self._fms_plan
            if fms and fms != self._fms_sent:
                st["flight_plan"] = self._plan_payload()
            resp = self.backend.update_state(st)
            if fms and "flight_plan" in st and resp.get("status") == "ok":
                self._fms_sent = fms
                xp.log(f"VirtualATC: FMS flight plan updated: {fms}")
            if resp.get("_no_session") and time.time() > self._retune_after:
                self._retune_after = time.time() + 15
                xp.log("VirtualATC: server has no session (restarted?) - tuning again")
                self._connect_and_tune()
                continue
            if resp.get("pending_atc") and time.time() > self._pending_guard_until:
                self._pending_guard_until = time.time() + PENDING_REPLAY_GUARD
                data = self.backend.get_pending_atc(acknowledge=True)
                call = (data or {}).get("call") or {}
                if call.get("message"):
                    xp.log(f"VirtualATC ATC [{call.get('trigger_type')}]: {call['message']}")
                    self._set_status(call["message"][:60])
                    if data.get("frequency"):
                        mhz = f" {data['frequency_mhz']:.3f}" if data.get("frequency_mhz") else ""
                        self._set_status(f"FREQ:{self.session.get('airport_icao', '')} {data['frequency']}{mhz}")
                    if data.get("audio_base64"):
                        self.player.enqueue_wav(base64.b64decode(data["audio_base64"]))
