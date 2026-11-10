# Virtual ATC System Architecture

An LLM-driven air traffic controller for X-Plane 12. The sim (or the web playground, or the
test harness) talks to a local Flask server; the server keeps one ATC session per aircraft,
asks OpenAI for the controller's words with tools for real aeronautical data, and returns
audio.

```
┌──────────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────┐
│ X-Plane 12 + XPPython3       │   │ Web playground (static/)     │   │ e2e harness (e2e/)   │
│ PI_VirtualATC.py             │   │ text or mic, state sliders   │   │ pilot agent + judge  │
│ PTT → mic → /transmit (wav)  │   │                              │   │ fake_xplane.py       │
│ flight loop → /state (2s)    │   │                              │   │                      │
│ /pending_atc → COM1 bus      │   │                              │   │                      │
└──────────────┬───────────────┘   └──────────────┬───────────────┘   └──────────┬───────────┘
               └──────────────────────────────────┼──────────────────────────────┘
                                                  ▼  HTTP (localhost:5000)
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ atc_server.py (Flask)                                                                     │
│  /tune  /transmit  /transmit_text  /state  /pending_atc  /session  /charts  /airport ...  │
│                                                                                           │
│  session_manager.py   one ATCSession: airport, flight plan, phase (Delivery/Ground/Tower/ │
│                       Departure/Approach/Center), aircraft state, clearances, history     │
│  atc_llm_service.py   OpenAI Responses API + tools; parses clearances from the reply      │
│  clearance_parser.py  EN/PT-BR phraseology → structured clearances (runway, alt, hdg...)  │
│  atc_triggers.py      rules for ATC-initiated calls (handoffs by phase/geometry)          │
│  audio_service.py     STT/TTS: ElevenLabs → OpenAI fallback; WAV 16k for X-Plane          │
│                                                                                           │
│  data: frequency_service.py (OurAirports freqs) · airport_data_service.py (OurAirports    │
│        airports+runways, nearest, distance) · chart_providers.py (AISWEB BR, FAA d-TPP US)│
│        airport_charts_service.py (airportwxbot METAR/TAF/NOTAM)                           │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

## Request flow: one pilot transmission

1. Plugin/pilot posts audio (or text) plus the current aircraft state.
2. `_apply_state` updates the session and **follows the radio**: if COM1 is on another published
   position of this airport (within 25 kHz), the session switches to that position. If COM1 matches
   no position here but matches a published ATC position of an airport within 60 nm (the destination
   Tower on a cross-country), the session moves to that airport (same callsign/language/plan).
3. Without a radio (playground, e2e): a call that addresses a station ("Guarulhos Torre, ...") goes
   to that position; otherwise a pending "contact Tower on ..." is applied. A bare readback of the
   handoff frequency ("121.7, PT-ABC") gets no reply (it happens on the old frequency); a full
   clearance readback that also contains it is still checked by the old controller.
4. STT (ElevenLabs `scribe_v1`, or OpenAI `gpt-4o-transcribe` when ElevenLabs fails / is unset).
5. `ATCLLMService.generate_response` builds the system prompt from the session: airport name,
   elevation, **valid runway list**, published frequencies, flight plan, position, distance/AGL,
   assigned values, clearances issued so far, last 12 messages. Calls the model (default
   `gpt-5.6-luna`, reasoning effort `low`) with tools:
   `get_charts`, `get_chart_by_category`, `read_chart` (PDF downloaded server-side and attached
   as base64), `get_weather`, `get_notams`, `get_airport_info`. Tool loop capped at 4 rounds.
6. The reply is parsed by `clearance_parser` (taxi/lineup/takeoff/landing/approach with runway;
   altitude/heading/speed/squawk with values; handoffs with target position+frequency) and
   recorded on the session. A handoff sets `pending_handoff`.
7. TTS → mp3 (playground) or 16 kHz mono WAV (plugin, `xp.playPCMOnBus` on the COM1 bus).
   TTS failure never fails the request: text comes back with `audio_base64: null`.

## ATC-initiated calls (triggers)

`/state` runs `ATCTriggerEngine.evaluate` on every update (2 s from the plugin). Rules:

| Rule | Condition | Call |
|---|---|---|
| Tower → Departure/Approach | on Tower, takeoff clearance recorded, airborne, > 800 ft AGL, climbing | "contact Departure on 1xx.x" (Brazil: → Approach when no DEP exists) |
| Departure/Center → Approach | inbound, within 25 nm, not climbing | "contact Approach ..." |
| Approach → Tower | approach clearance recorded, < 8 nm, < 3500 AGL | "contact Tower ..." |
| Tower → Ground | landing clearance recorded, touched down, < 60 kt | "vacate next taxiway, contact Ground ..." |

30 s minimum between handoffs. Messages are generated in the session language with spoken
frequencies ("one three two decimal seven five" / "um três dois decimal sete cinco"). The call
waits in `pending_atc_call` until the client acknowledges (`/pending_atc?ack=true`), which
appends it to the conversation and switches the session position.

## Session state

```python
ATCSession
  airport_icao, airport (name, lat/lon, elevation, runways), airport_frequencies {CLD,GND,TWR,DEP,APP,ATIS}
  callsign, language, flight_plan (aircraft_type, flight_rules, departure, destination, route, cruise)
  phase: CLEARANCE|GROUND|TOWER|DEPARTURE|APPROACH|CENTER, current_frequency(+mhz)
  aircraft_state (lat, lon, alt, hdg, ias, gs, vs, squawk, com1, on_ground)
  clearances[] (type, details, value, runway), assigned_{runway,altitude,heading,speed,squawk}
  pending_handoff, conversation[]
```

## X-Plane plugin

- Datarefs: position/elevation/psi/IAS/GS/VS, transponder, `com1_frequency_hz_833`, on-ground,
  tail number (callsign) and ICAO type.
- Main thread (flight loop, 4 Hz): read datarefs, pump audio queue, apply status text.
- Worker thread: `/state` every 2 s; when a pending ATC call is reported, fetch + ack it with WAV
  and enqueue for playback. Never blocks the sim.
- PTT: command `virtualatc/ptt` (hold) or the window button (toggle). Keying the mic stops any
  ATC audio in progress.
- On load: waits for the server, then `/tune` with `airport_icao="auto"` (nearest airport with a
  runway ≥ 3000 ft) and the aircraft type from the model.

## Data sources

| Need | Source | Notes |
|---|---|---|
| Frequencies | OurAirports `airport-frequencies.csv` | cached weekly in `.cache/` |
| Airports, runways | OurAirports `airports.csv`, `runways.csv` | cached monthly; 48k airports |
| Charts BR | AISWEB (DECEA) aerodrome page scrape | public PDFs; cached 7 d |
| Charts US | FAA d-TPP metafile XML (16 MB, per 28-day cycle) | public PDFs |
| METAR/TAF | aviationweather.gov (`weather_service.py`) | public JSON, US + Brazil; cached 5 min; prefetched into the prompt; active runway computed from wind |
| NOTAM, METAR fallback | airportwxbot.com | its METAR returned 500 everywhere on 2026-09-28; chart PDFs are behind Cloudflare |

## Testing

- `tests/` (offline, <1 s): parser, session, triggers, chart parsing, Flask API with fake LLM/audio.
- `e2e/run_e2e.py` (live): 34 scenario YAMLs in 12 categories (ground, position, readback,
  departure, pattern, arrival, emergency, safety, adversarial, language, enroute, edge) → LLM pilot
  agent (or a scripted pilot for deterministic adversarial/emergency calls) ↔ ATC, simulated
  aircraft (glide path, circuits, touch-and-go) feeding the trigger engine, pinned METARs, LLM judge
  + deterministic hard checks (clearances issued/forbidden, per-turn phrases, reply language, valid
  runways) → per-scenario Markdown/JSON + a SUMMARY matrix. `-j 6` runs in parallel, `-c` filters
  by category, `known_gap:` marks expected failures (XFAIL).
- `e2e/fake_xplane.py` (live): drives the running server with the plugin's exact traffic pattern
  and a scripted flight; saves the WAVs the plugin would play.
- `scripts/box.sh` (real sim): deploys the plugin to the Windows X-Plane box, launches X-Plane in
  its desktop session, sets COM1 and fires `virtualatc/test_transmit` through X-Plane's UDP
  interface, tails the XPPython3 log.

## Configuration

See `config.py` / `.env.example`. Model, reasoning effort, language, TTS/STT provider policy,
voice, server host/port.
