"""
Audio Service - STT and TTS for ATC communications with provider fallback.

Providers:
    ElevenLabs  - scribe_v1 (STT), eleven_multilingual_v2 (TTS). Best voice quality.
    OpenAI      - gpt-4o-transcribe (STT), gpt-4o-mini-tts (TTS). Always available with
                  the OpenAI key we already need for the LLM. ~1s STT, ~3-4s TTS.

Selection (config.TTS_PROVIDER / STT_PROVIDER):
    "auto"        try ElevenLabs first (if key set), fall back to OpenAI on ANY error.
                  A provider that fails is benched for FALLBACK_COOLDOWN_S so a dead
                  quota doesn't cost a round-trip on every transmission.
    "elevenlabs"  ElevenLabs only (errors propagate).
    "openai"      OpenAI only.

Usage:
    service = AudioService()
    text = service.transcribe(audio_bytes, language="en")
    mp3 = service.speak_atc("November one two three, cleared for takeoff runway three one left")
    wav = service.speak_atc_wav(...)     # 16kHz mono PCM WAV for X-Plane playPCMOnBus
"""

import io
import logging
import os
import time
import wave
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("Audio")

FALLBACK_COOLDOWN_S = 300  # after a provider fails, skip it for 5 minutes

# ATC radio delivery instructions for OpenAI's steerable TTS
OPENAI_ATC_STYLE = {
    "en": "You are an air traffic controller speaking on a VHF radio. Deliver quickly, "
          "clipped and monotone, professional, no emotion, no pauses between items. "
          "Pronounce digits individually and use ICAO phonetics exactly as written.",
    "pt-br": "Voce e um controlador de trafego aereo falando no radio VHF. Fale rapido, "
             "seco e monotono, profissional, sem emocao. Pronuncie digitos individualmente "
             "e o alfabeto fonetico exatamente como escrito. Sotaque brasileiro.",
}


class ProviderError(RuntimeError):
    pass


class AudioService:
    """STT/TTS with ElevenLabs -> OpenAI fallback."""

    ELEVEN_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
    ELEVEN_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",
        tts_provider: Optional[str] = None,
        stt_provider: Optional[str] = None,
        openai_client=None,
        language: str = "en",
    ):
        # Lazy config import so this module stays importable in the X-Plane plugin tree
        try:
            import config as cfg
        except ImportError:  # pragma: no cover
            cfg = None

        self.eleven_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", self.DEFAULT_VOICE_ID)
        self.model_id = model_id
        self.language = language

        self.tts_provider = (tts_provider or (cfg.TTS_PROVIDER if cfg else "auto")).lower()
        self.stt_provider = (stt_provider or (cfg.STT_PROVIDER if cfg else "auto")).lower()
        self.openai_tts_model = cfg.OPENAI_TTS_MODEL if cfg else "gpt-4o-mini-tts"
        self.openai_tts_voice = cfg.OPENAI_TTS_VOICE if cfg else "onyx"
        self.openai_stt_model = cfg.OPENAI_STT_MODEL if cfg else "gpt-4o-transcribe"

        if not self.eleven_key and self.tts_provider == "auto":
            self.tts_provider = "openai"
        if not self.eleven_key and self.stt_provider == "auto":
            self.stt_provider = "openai"
        if self.tts_provider == "elevenlabs" and not self.eleven_key:
            raise ValueError("ELEVENLABS_API_KEY not set but ATC_TTS_PROVIDER=elevenlabs")

        self._openai = openai_client
        self._eleven_benched_until = 0.0
        self.last_tts_provider: Optional[str] = None
        self.last_stt_provider: Optional[str] = None

        logger.info(f"AudioService: tts={self.tts_provider} stt={self.stt_provider}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @property
    def openai(self):
        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI()
        return self._openai

    def _eleven_available(self) -> bool:
        return bool(self.eleven_key) and time.time() >= self._eleven_benched_until

    def _bench_eleven(self, why: Exception):
        self._eleven_benched_until = time.time() + FALLBACK_COOLDOWN_S
        logger.warning(f"ElevenLabs failed ({why}); using OpenAI audio for {FALLBACK_COOLDOWN_S}s")

    @staticmethod
    def _lang_code(language: str) -> str:
        return "pt" if language.lower().startswith("pt") else "en"

    # ------------------------------------------------------------------
    # Speech-to-text
    # ------------------------------------------------------------------

    def transcribe(self, audio_data: bytes, language: Optional[str] = None) -> str:
        lang = self._lang_code(language or self.language)
        order = {
            "elevenlabs": ["elevenlabs"],
            "openai": ["openai"],
            "auto": ["elevenlabs", "openai"] if self._eleven_available() else ["openai"],
        }[self.stt_provider]

        last_err: Optional[Exception] = None
        for provider in order:
            try:
                if provider == "elevenlabs":
                    text = self._eleven_stt(audio_data, lang)
                else:
                    text = self._openai_stt(audio_data, lang)
                self.last_stt_provider = provider
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                if provider == "elevenlabs" and self.stt_provider == "auto":
                    self._bench_eleven(e)
                else:
                    raise
        raise ProviderError(f"STT failed: {last_err}")

    def _eleven_stt(self, audio_data: bytes, lang: str) -> str:
        resp = requests.post(
            self.ELEVEN_STT_URL,
            headers={"xi-api-key": self.eleven_key},
            files={"file": ("audio.wav", audio_data, "audio/wav")},
            data={"model_id": "scribe_v1", "language_code": lang, "tag_audio_events": "false"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    def _openai_stt(self, audio_data: bytes, lang: str) -> str:
        prompt = ("Aviation radio, ICAO phraseology and phonetic alphabet. Callsigns like "
                  "November one two three four five, PT-ABC, TAM three four five six.")
        # openai accepts (filename, bytes); extension drives format sniffing
        ext = "webm" if audio_data[:4] == b"\x1a\x45\xdf\xa3" else "wav"
        tr = self.openai.audio.transcriptions.create(
            model=self.openai_stt_model,
            file=(f"pilot.{ext}", audio_data),
            language=lang,
            prompt=prompt,
        )
        return tr.text or ""

    def transcribe_file(self, file_path: str | Path) -> str:
        return self.transcribe(Path(file_path).read_bytes())

    # ------------------------------------------------------------------
    # Text-to-speech
    # ------------------------------------------------------------------

    def text_to_speech(
        self,
        text: str,
        output_format: str = "mp3",     # "mp3" | "pcm16k"
        speed: float = 1.1,
        stability: float = 0.7,
        similarity_boost: float = 0.75,
        language: Optional[str] = None,
    ) -> bytes:
        lang = (language or self.language).lower()
        order = {
            "elevenlabs": ["elevenlabs"],
            "openai": ["openai"],
            "auto": ["elevenlabs", "openai"] if self._eleven_available() else ["openai"],
        }[self.tts_provider]

        last_err: Optional[Exception] = None
        for provider in order:
            try:
                if provider == "elevenlabs":
                    audio = self._eleven_tts(text, output_format, speed, stability, similarity_boost)
                else:
                    audio = self._openai_tts(text, output_format, lang)
                self.last_tts_provider = provider
                return audio
            except Exception as e:  # noqa: BLE001
                last_err = e
                if provider == "elevenlabs" and self.tts_provider == "auto":
                    self._bench_eleven(e)
                else:
                    raise
        raise ProviderError(f"TTS failed: {last_err}")

    def _eleven_tts(self, text, output_format, speed, stability, similarity_boost) -> bytes:
        fmt = "pcm_16000" if output_format == "pcm16k" else "mp3_44100_128"
        resp = requests.post(
            f"{self.ELEVEN_TTS_URL}/{self.voice_id}",
            headers={"xi-api-key": self.eleven_key, "Content-Type": "application/json"},
            params={"output_format": fmt},
            json={
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {"stability": stability, "similarity_boost": similarity_boost, "speed": speed},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.content

    def _openai_tts(self, text, output_format, lang) -> bytes:
        style = OPENAI_ATC_STYLE["pt-br" if lang.startswith("pt") else "en"]
        fmt = "pcm" if output_format == "pcm16k" else "mp3"
        resp = self.openai.audio.speech.create(
            model=self.openai_tts_model,
            voice=self.openai_tts_voice,
            input=text,
            instructions=style,
            response_format=fmt,
        )
        data = resp.content
        if output_format == "pcm16k":
            # OpenAI pcm is 24kHz 16-bit mono; downsample to 16kHz for X-Plane
            data = _resample_pcm16(data, 24000, 16000)
        return data

    # ------------------------------------------------------------------
    # ATC helpers
    # ------------------------------------------------------------------

    def speak_atc(self, text: str, language: Optional[str] = None) -> bytes:
        """ATC-style MP3."""
        return self.text_to_speech(text, output_format="mp3", language=language)

    def speak_atc_wav(self, text: str, language: Optional[str] = None) -> bytes:
        """ATC-style 16kHz mono 16-bit WAV (what the X-Plane plugin feeds to playPCMOnBus)."""
        pcm = self.text_to_speech(text, output_format="pcm16k", language=language)
        return pcm_to_wav(pcm, 16000)

    def speak(self, text: str) -> bytes:
        return self.speak_atc(text)

    def speak_to_file(self, text: str, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.write_bytes(self.speak_atc_wav(text) if out.suffix.lower() == ".wav" else self.speak_atc(text))
        return out


# ----------------------------------------------------------------------
# PCM utilities (stdlib only, so the plugin can vendor them)
# ----------------------------------------------------------------------

def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Simple linear-interpolation resampler for 16-bit mono PCM (good enough for radio)."""
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


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python audio_service.py speak 'text' [out.mp3|out.wav] [en|pt-br]")
        print("  python audio_service.py transcribe input.wav [en|pt-br]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "speak":
        text = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else "atc_response.mp3"
        lang = sys.argv[4] if len(sys.argv) > 4 else "en"
        svc = AudioService(language=lang)
        svc.speak_to_file(text, out)
        print(f"Saved {out} via {svc.last_tts_provider}")
    elif cmd == "transcribe":
        lang = sys.argv[3] if len(sys.argv) > 3 else "en"
        svc = AudioService(language=lang)
        print(svc.transcribe_file(sys.argv[2]), f"(via {svc.last_stt_provider})")
