# ATC-GPT — Virtual ATC for X-Plane

An LLM-powered air traffic controller you talk to with push-to-talk. Backend = Flask + OpenAI
(`gpt-5.6-luna` by default) with tools for real charts, weather and airport data. Frontend = an
XPPython3 plugin for X-Plane 12, or the web playground for testing without the sim.
English (ICAO) and Brazilian Portuguese (DECEA) phraseology.

## Quick start

```bash
make install                 # venv + requirements
cp .env.example .env         # add OPENAI_API_KEY (ElevenLabs optional)
make serve                   # http://127.0.0.1:5000  (web playground)
```

First start downloads OurAirports data (airports, runways, frequencies, ~17 MB) into `.cache/`.

In the playground: enter an ICAO, a callsign, a flight plan (type / rules / destination), pick the
position (Delivery, Ground, Tower, ...), click **Tune**, then type or hold the PTT button.

## Testing

| Command | What | Needs keys? | Time |
|---|---|---|---|
| `make test` | 72 offline unit + API tests (parser, triggers, session, chart parsing, Flask endpoints with fake LLM/TTS) | no | 1 s |
| `make e2e` | LLM **pilot agent** flies each scenario in `e2e/scenarios/*.yaml` against the ATC; a simulated aircraft advances state so handoffs fire; an LLM **judge** scores phraseology / correctness / safety; hard checks assert clearances, phases, valid runways. Reports in `e2e/reports/` | OpenAI | ~2–4 min / scenario |
| `make e2e S=sbgr_ifr_departure_ptbr` | one scenario | OpenAI | |
| `make e2e-server` | same scenarios but through the running HTTP server | OpenAI + `make serve` | |
| `make fake-xplane P=sbgr` | drives the running server **exactly like the plugin** (auto-tune by position, `/state` loop, `/transmit_text`, `/pending_atc` ack) through a scripted flight and saves the WAV the plugin would play on COM1 to `e2e/reports/fake_xplane/` | OpenAI + `make serve` | ~2 min |
| `make fake-xplane P=kjfk AUDIO=1` | same, but synthesizes pilot **audio** and pushes it through `/transmit` (STT path) | | |

Scenarios today: JFK IFR departure (EN, Delivery→Ground→Tower→Departure), SBGR IFR departure (PT-BR),
Campo de Marte VFR pattern (PT-BR), Boston ILS arrival with a wrong altitude readback (EN),
and an adversarial "pilot reads back runway 27R at SBGR" test.

## X-Plane plugin

```bash
cp xplane_plugin/virtualatc_config.example.txt xplane_plugin/virtualatc_config.txt   # git-ignored, edit it
cp xplane_plugin/PI_VirtualATC.py xplane_plugin/virtualatc_config.txt "/path/to/X-Plane 12/Resources/plugins/PythonPlugins/"
```

Requires [XPPython3](https://xppython3.readthedocs.io) (4.2, Python 3.12) plus `pyaudio` and
`requests` in that Python. Edit `virtualatc_config.txt` next to the plugin:

```
server_url=http://127.0.0.1:5000   # or http://<server LAN/VPN ip>:5055 when the server runs elsewhere
language=pt-br                     # or en
mic_device=                        # pyaudio input index, blank = default
test_wav=virtualatc_test.wav       # optional 16 kHz mono WAV; virtualatc/test_transmit sends it as the pilot
flight_rules=IFR                   # IFR | VFR | blank: sent with the flight plan read from the FMS
callsign=                          # radio callsign, blank = the aircraft tail number
```

Bind a key/joystick button to `virtualatc/ptt` (hold to talk) or `virtualatc/ptt_toggle` (press to
talk, press again to send). The plugin auto-tunes on load (nearest field on the ground; in the air the
nearer of the FMS departure/destination), posts aircraft state every 2 s from a worker thread, follows
COM1 (tune Tower, you get Tower), and plays ATC replies on the COM1 bus via `xp.playPCMOnBus`.

**Flight plan = whatever is loaded in the FMS/GPS.** Load a `.fms` route (e.g. from SimBrief / Little
Navmap into `Output/FMS plans/`) before calling Delivery: the plugin sends departure, destination,
route and cruise altitude to the controller and re-sends on every change. No manual `/tune` needed.
Cross-country, inside 40 nm of the destination (`ATC_ARRIVAL_TRANSFER_NM`) ATC hands you to the
destination's Approach/Tower if one is published, or - for AFIS-only / uncontrolled fields such as
SBCB - keeps working your IFR arrival itself on the same frequency.

**Verified in X-Plane 12.4 on 2026-09-09** (Windows box over Tailscale, Baron 58 at SBBH): plugin
load, auto-tune, state loop, pilot WAV → STT → controller → TTS → COM1 playback (~6 s), COM1
retune switching Ground → Tower. The mic could not be opened inside an RDP session (no audio
endpoint there); it needs a real seat at the machine.

### Remote testing on the Windows box

`scripts/box.sh` drives the box over SSH without touching its keyboard. Put `BOX_HOST` and `BOX_USER`
(optionally `BOX_KEY`, `BOX_PYTHON`, `BOX_KNOWN_HOSTS`) in `scripts/box.local.sh` (git-ignored):

```bash
ATC_SERVER_HOST=0.0.0.0 ATC_SERVER_PORT=5055 make serve   # on the Mac (port 5000 is macOS AirPlay)
scripts/box.sh deploy      # copy plugin + config + test wavs, py_compile with X-Plane's Python
scripts/box.sh start       # launch X-Plane in the logged-in desktop session (scheduled task)
scripts/box.sh log         # VirtualATC lines from XPPython3Log.txt (tunes ~60 s after start)
scripts/box.sh com1 121600 # set COM1 via X-Plane's UDP DREF interface
scripts/box.sh test        # fire virtualatc/test_transmit via UDP CMND -> full loop through the sim
scripts/box.sh stop
```

## Configuration (`.env`)

| Var | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | | required |
| `OPENAI_MODEL` | `gpt-5.6-luna` | `gpt-5.4-mini` is ~5x faster with acceptable phraseology |
| `OPENAI_REASONING_EFFORT` | `low` | none/minimal/low/medium/high |
| `ATC_LANGUAGE` | `en` | `en` or `pt-br` (also per-session via `/tune`) |
| `ATC_TTS_PROVIDER` / `ATC_STT_PROVIDER` | `auto` | `auto` = ElevenLabs, fall back to OpenAI on any error |
| `ATC_TTS_ENABLED` | `true` | `false` for text-only |
| `ELEVENLABS_API_KEY` | | optional |
| `OPENAI_TTS_VOICE` | `onyx` | OpenAI TTS voice |

## Data sources

- Charts: **AISWEB/DECEA** (Brazil, official) and **FAA d-TPP** (USA, official). PDFs are fetched
  server-side and attached to the model as base64 (`chart_providers.py`).
- Weather / NOTAMs: airportwxbot.com.
- Airports, runways, frequencies: OurAirports open data (`airport_data_service.py`, `frequency_service.py`).

## API

See the docstring at the top of `atc_server.py`. Main flow: `POST /tune` → `POST /transmit`
(audio) or `POST /transmit_text` → periodic `POST /state` → `GET /pending_atc?ack=true` when a
handoff is pending.

## Known gaps / roadmap

See [GAPS.md](GAPS.md).
