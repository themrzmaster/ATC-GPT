"""
Configuration for Virtual ATC System.

Set environment variables or modify this file directly.
"""

import os
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass


# -----------------------------------------------------------------------------
# API Keys
# -----------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")


# -----------------------------------------------------------------------------
# Server Configuration
# -----------------------------------------------------------------------------

ATC_SERVER_HOST = os.environ.get("ATC_SERVER_HOST", "127.0.0.1")
ATC_SERVER_PORT = int(os.environ.get("ATC_SERVER_PORT", "5000"))
ATC_DEBUG = os.environ.get("ATC_DEBUG", "false").lower() == "true"


# -----------------------------------------------------------------------------
# ElevenLabs Configuration
# -----------------------------------------------------------------------------

# Voice ID for ATC responses
# Default: "21m00Tcm4TlvDq8ikWAM" (Rachel - clear female voice)
# Alternatives:
#   "ErXwobaYiN019PkySvjV" - Antoni (male)
#   "VR6AewLTigWG4xSOukaG" - Arnold (male, deeper)
#   "pNInz6obpgDQGcFmaJgB" - Adam (male, professional)
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# TTS Model
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")


# -----------------------------------------------------------------------------
# Audio provider selection
# -----------------------------------------------------------------------------

# "auto"       - ElevenLabs if key set and working, fall back to OpenAI on any error (quota, 401, ...)
# "elevenlabs" - ElevenLabs only
# "openai"     - OpenAI only (gpt-4o-mini-tts / gpt-4o-transcribe)
TTS_PROVIDER = os.environ.get("ATC_TTS_PROVIDER", "auto").lower()
STT_PROVIDER = os.environ.get("ATC_STT_PROVIDER", "auto").lower()
OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "onyx")
OPENAI_STT_MODEL = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-transcribe")

# Set to "false" to skip TTS entirely (text-only, fastest for testing)
TTS_ENABLED = os.environ.get("ATC_TTS_ENABLED", "true").lower() == "true"


# -----------------------------------------------------------------------------
# OpenAI Configuration
# -----------------------------------------------------------------------------

# gpt-5.6-luna: best phraseology/judgement in our probes (see GAPS.md). ~10s/turn with tools.
# gpt-5.4-mini: ~2s/turn, acceptable phraseology - use for the e2e "pilot" agent or low-latency mode.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")

# Reasoning effort for the ATC model ("none" | "minimal" | "low" | "medium" | "high").
# Lower = faster radio turnaround. Empty string = provider default.
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "low")


# -----------------------------------------------------------------------------
# ATC Configuration
# -----------------------------------------------------------------------------

# Default airport for new sessions
DEFAULT_AIRPORT = os.environ.get("ATC_DEFAULT_AIRPORT", "KJFK")

# Default callsign if not read from aircraft
DEFAULT_CALLSIGN = os.environ.get("ATC_DEFAULT_CALLSIGN", "N12345")

# How often to poll aircraft state (seconds)
STATE_UPDATE_INTERVAL = float(os.environ.get("ATC_STATE_INTERVAL", "2.0"))

# Controller initiative: during radio silence, ask the model whether a controller would
# transmit now (vectors, descent, approach clearance, circuit instructions...).
INITIATIVE_ENABLED = os.environ.get("ATC_INITIATIVE_ENABLED", "true").lower() == "true"
INITIATIVE_SILENCE_S = float(os.environ.get("ATC_INITIATIVE_SILENCE_S", "40"))   # quiet time before asking
INITIATIVE_MIN_INTERVAL_S = float(os.environ.get("ATC_INITIATIVE_INTERVAL_S", "60"))  # between asks
# after this many controller calls with no pilot reply, stay quiet until the pilot talks again
INITIATIVE_MAX_UNANSWERED = int(os.environ.get("ATC_INITIATIVE_MAX_UNANSWERED", "2"))
STANDBY_CALLBACK_S = float(os.environ.get("ATC_STANDBY_CALLBACK_S", "15"))  # answer a "stand by" after this

# Cross-country: within this distance of the flight-plan destination the flight is worked as an arrival
# there (handoff to its Approach/Tower if published, else the same controller keeps it on frequency).
ARRIVAL_TRANSFER_NM = float(os.environ.get("ATC_ARRIVAL_TRANSFER_NM", "40"))
# Last flight plan per callsign, so a server restart / plugin re-tune mid-flight keeps the destination.
# Empty = don't persist. Reused only if newer than FLIGHT_PLAN_REUSE_H and the new tune has no destination.
FLIGHT_PLAN_STORE = os.environ.get("ATC_FLIGHT_PLAN_STORE", ".cache/last_flight_plans.json")
FLIGHT_PLAN_REUSE_H = float(os.environ.get("ATC_FLIGHT_PLAN_REUSE_H", "6"))
# Keep the last pilot recording at .cache/last_transmit.wav (mic debugging)
SAVE_LAST_TRANSMIT = os.environ.get("ATC_SAVE_LAST_TRANSMIT", "true").lower() == "true"
# OpenAI request timeout for radio replies (s); chart digests get CHART_DIGEST_TIMEOUT_S
LLM_TIMEOUT_S = float(os.environ.get("ATC_LLM_TIMEOUT_S", "25"))
CHART_DIGEST_TIMEOUT_S = float(os.environ.get("ATC_CHART_DIGEST_TIMEOUT_S", "120"))

# ATC Language: "en" for English, "pt-br" for Brazilian Portuguese
# Affects ATC phraseology and responses
ATC_LANGUAGE = os.environ.get("ATC_LANGUAGE", "en")


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_config():
    """Check that required configuration is present."""
    errors = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not set")
    if not ELEVENLABS_API_KEY and TTS_PROVIDER == "elevenlabs":
        errors.append("ELEVENLABS_API_KEY not set (required when ATC_TTS_PROVIDER=elevenlabs)")

    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        print("\nSet environment variables or create a .env file.")
        return False

    return True


if __name__ == "__main__":
    print("Configuration:")
    print(f"  OPENAI_API_KEY: {'set' if OPENAI_API_KEY else 'NOT SET'}")
    print(f"  ELEVENLABS_API_KEY: {'set' if ELEVENLABS_API_KEY else 'NOT SET'}")
    print(f"  Server: {ATC_SERVER_HOST}:{ATC_SERVER_PORT}")
    print(f"  OpenAI Model: {OPENAI_MODEL}")
    print(f"  ElevenLabs Voice: {ELEVENLABS_VOICE_ID}")
    print(f"  Default Airport: {DEFAULT_AIRPORT}")
    print()
    validate_config()
