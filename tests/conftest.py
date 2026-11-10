"""
Shared fixtures. Everything here is OFFLINE: no OpenAI, no ElevenLabs, no chart hosts.

- frequency/airport data come from the .cache/ CSVs (downloaded once by the server).
- the LLM and audio services are replaced by fakes so the Flask API can be exercised
  deterministically.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ATC_TTS_PROVIDER", "openai")
os.environ.setdefault("ATC_TTS_ENABLED", "false")
os.environ.setdefault("ATC_SAVE_LAST_TRANSMIT", "false")   # tests must not overwrite the mic-debug wav
os.environ.setdefault("ATC_FLIGHT_PLAN_STORE", "")           # no persisted flight plans leaking between tests
os.environ.setdefault("ATC_CHART_PREFETCH", "false")


@pytest.fixture(scope="session")
def frequency_service():
    from frequency_service import FrequencyService
    return FrequencyService(auto_download=True)


@pytest.fixture(scope="session")
def airport_data():
    from airport_data_service import AirportDataService
    return AirportDataService()


@pytest.fixture
def make_session(frequency_service, airport_data):
    from session_manager import ATCSession, FlightPlan

    def _make(icao="SBGR", callsign="PT-ABC", language="en", **fp):
        return ATCSession(icao, callsign, frequency_service=frequency_service,
                          airport_data_service=airport_data, language=language,
                          flight_plan=FlightPlan(**fp) if fp else None)
    return _make


class FakeLLM:
    """Stands in for ATCLLMService: scripted replies, records what it was asked."""

    def __init__(self, language="en"):
        self.language = language
        self.model = "fake"
        self.replies: list[str] = []
        self.calls: list[str] = []
        self.last_latency_s = 0.01
        self.last_tool_calls: list[str] = []
        self.last_usage = {}
        self.initiative_replies: list = []   # None = stays quiet
        self.initiative_calls = 0

    def controller_initiative(self, session, reason=""):
        from clearance_parser import parse_clearances
        self.initiative_calls += 1
        text = self.initiative_replies.pop(0) if self.initiative_replies else None
        if not text:
            return None
        session.add_atc_transmission(text)
        for pc in parse_clearances(text, session.callsign):
            session.add_clearance(pc.type, text, value=pc.value, runway=pc.runway,
                                  position=pc.position, frequency_khz=pc.frequency_khz)
        return text

    def generate_response(self, pilot_text, session):
        from clearance_parser import parse_clearances
        self.calls.append(pilot_text)
        session.apply_pending_handoff()
        reply = self.replies.pop(0) if self.replies else f"{session.callsign}, roger."
        session.add_pilot_transmission(pilot_text)
        session.add_atc_transmission(reply)
        for pc in parse_clearances(reply, session.callsign):
            session.add_clearance(pc.type, reply, value=pc.value, runway=pc.runway,
                                  position=pc.position, frequency_khz=pc.frequency_khz)
        return reply


class FakeAudio:
    language = "en"
    last_tts_provider = "fake"
    last_stt_provider = "fake"

    def __init__(self):
        self.transcript = "Ground, PT-ABC, request taxi"

    def transcribe(self, audio_bytes, language=None):
        return self.transcript

    def speak_atc(self, text, language=None):
        return b"ID3fake-mp3-" + text.encode()

    def speak_atc_wav(self, text, language=None):
        from audio_service import pcm_to_wav
        return pcm_to_wav(b"\x00\x00" * 1600, 16000)


@pytest.fixture
def api(monkeypatch, frequency_service, airport_data):
    """Flask test client with fake LLM + audio wired in."""
    import atc_server
    fake_llm = FakeLLM()
    fake_audio = FakeAudio()
    monkeypatch.setattr(atc_server, "llm_service", fake_llm)
    monkeypatch.setattr(atc_server, "audio_service", fake_audio)
    monkeypatch.setattr(atc_server, "frequency_service", frequency_service)
    monkeypatch.setattr(atc_server, "airport_data_service", airport_data)
    monkeypatch.setattr(atc_server, "pending_atc_call", None)
    monkeypatch.setattr(atc_server, "current_language", "en")
    monkeypatch.setattr(atc_server, "_last_initiative_at", None)
    monkeypatch.setattr(atc_server.config, "TTS_ENABLED", True)

    def _get_llm(language=None):
        fake_llm.language = language or fake_llm.language
        return fake_llm
    monkeypatch.setattr(atc_server, "get_llm_service", _get_llm)
    monkeypatch.setattr(atc_server, "get_audio_service", lambda: fake_audio)

    atc_server.session_manager.sessions.clear()
    atc_server.session_manager.active_session_id = None
    atc_server.trigger_engine.trigger_states.clear()

    client = atc_server.app.test_client()
    client.fake_llm = fake_llm
    client.fake_audio = fake_audio
    client.server = atc_server
    return client
