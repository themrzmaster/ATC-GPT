# ATC-GPT — Status, gaps and roadmap

## 2026-11-07 live flight SBCF → SBCB (BE58, IFR via UZ24, pt-br) — fixes

Flown end to end in X-Plane 12 (clearance, taxi, takeoff, en route, RNP Z RWY 10, go-around). What broke
and what changed (tests: `tests/test_flight_flow.py`, `tests/test_api.py`):
- **Flight plan only by hand**: every server restart / plugin re-tune lost it (auto-tune landed on SNJR,
  SDMC). Now the plugin reads the FMS (`fms_plan_from_entries`, XP 12.1 API then legacy) and sends it with
  `/tune` and `/state`; the server keeps the last plan per callsign (`.cache/last_flight_plans.json`, 6 h)
  and an airborne auto-tune picks the nearer of departure/destination, phase from the situation.
- **AFIS-only destination** (Cabo Frio: only "RDO" 132.15): the IFR arrival got "no ATC service,
  self-announce on 123.45". Now `area_control()`: IFR + airborne + no Tower/Approach → the radar
  controller keeps it on frequency, clears the approach and gives "pouso a critério do piloto".
- **No arrival handoff**: the departure's Approach worked the flight to the destination. Now inside
  `ARRIVAL_TRANSFER_NM` (40): handoff to the destination's APP/TWR if published, else a silent transfer
  of the session (conversation kept).
- **"Chame Centro Brasília/Curitiba, frequência conforme carta"** twice: no Center is simulated, so IFR
  stays on frequency en route.
- **Handoff to a position with no frequency looped** ("permaneça nesta frequência, contato radar" every
  30 s): COM1 snapped the session back. Suppressed when COM1 is followed.
- **Radio checks into a paused sim** (9 in an hour): no initiative when the aircraft is frozen in the air;
  at most 2 unanswered calls, then quiet until the pilot speaks.
- **Stand-by never answered**: "aguarde" now gets a callback with the decision after 15 s; prompt forbids
  stand-by without a decision. Bare "aguarde" is no longer parsed as a HOLD.
- **Outbound flight on Approach 120.2 got approach instructions** for its own departure field: outbound note.
- **One hung OpenAI call held the server lock 17 min**: timeout 25 s (`ATC_LLM_TIMEOUT_S`), 1 retry.
- **read_chart took 55 s** (200 KB PDF attached to the radio reply): now a text digest per chart, cached
  in `.cache/chart_digests/`, destination APP charts prefetched in the background at tune / FMS change.
- Nearby-METAR lookup sent OurAirports idents (`BR-3282`) and got 400 for the whole batch: ICAO ids only.
- Stray Cyrillic glyphs from luna stripped; "November" (not "Novembro") for N-registrations; flight levels
  above the transition altitude.
- Tests no longer overwrite `.cache/last_transmit.wav`.

Still open from this flight:
- Server restart mid-flight still loses the conversation and clearances (plan survives).
- AFIS station itself isn't simulated (the area controller gives landing information instead).
- FMS read untested in the sim (code path verified with the mock; check `VirtualATC: FMS flight plan
  sent` in XPPython3Log.txt on the next flight). Approach procedure (APP R10-Z) isn't in the FMS API.
- Initiative still calls every ~60 s on a quiet frequency while flying (capped at 2 unanswered).
- Weather sources for SBCF flaky (aviationweather timeouts, airportwxbot 500).

## 2026-09-28 scenario sweep (pre-X-Plane validation)

34 live scenarios (`make e2e`, 12 categories) + 4 plugin-contract profiles (`make fake-xplane`,
now incl. `kbos_arrival` and `xcountry`) + 136 offline tests. Baseline → after fixes:
**8 → 22 PASS + 1 XFAIL of 34; plugin profiles 2/4 → 4/4.** Latest matrix: `e2e/reports/*_SUMMARY.md`.

Fixed (all covered by `tests/test_scenario_fixes.py`):
- airportwxbot METAR returned 500 for every airport (10–20 s turns, "say requested runway" loops) →
  `weather_service.py` (aviationweather.gov), nearby-station fallback in one request, METAR prefetched
  into the prompt, runway in use computed from wind.
- Cross-country: COM1 on another airport's Tower/Approach now moves the session there (was answering
  "Guarulhos Solo" on Campinas Tower).
- Tower→Ground after landing was lost forever if touchdown happened inside the 30 s handoff cooldown.
- Playground/e2e: the station the pilot addresses answers ("Torre, ..." was answered by Ground, which
  cleared a takeoff); a bare handoff readback gets no reply; a full readback is still checked.
- Emergency squawks (7700/7600/7500) flagged as a radar alert (Approach used to ignore 7700).
- Uncontrolled fields (no ATC frequency): no more fake controller issuing clearances at Paraty.
- Foreign crew speaking English in a PT-BR session gets English replies and handoffs.
- Prompt: no invented traffic (JFK Tower held 4× for phantom "traffic on final"), no "prossiga para
  decolagem", holds with EFC, emergency duties per position, missed-approach handling, Approach→Tower
  timing, destination frequencies in the prompt (was inventing "Campinas 120.3"; real 121.4).
- Parser: PT IFR clearances ("autorizado SBKP via..."), "runway 16, taxi", "hold present position".
- Output guard: luna occasionally emits junk (">tagger", "... correct. erotikk") — retried/stripped.
- `FlightPlan.is_empty()` was never true (departure auto-filled), so "no flight plan" never showed.

Still open after the sweep (LLM judgement, not plumbing — judge 3/5 cases):
- Touch-and-go: Tower gives "cleared to land" instead of "cleared touch-and-go" (EN pattern).
- Ground asks for "exact ramp location" in loops at big airports (no ramp/taxiway data → gap 2).
- 7600: says "roger radio failure" (the aircraft can't have told it) and clears to land from Approach.
- Enroute: no Center data, so Departure has nowhere to hand off to.
- Diversion in the playground (no radio) can't switch airport by name (XFAIL; the plugin can, via COM1).
- Plugin sends no flight plan (only aircraft type): in the sim the controller assumes VFR/unknown
  intentions until the pilot says otherwise. Reading the FMS destination would fix it.
- Run-to-run variance is real: a scenario can go 5/5 → 3/5 between runs; judge with `-j 6` twice
  before trusting a regression.

### Round 2 (same day): model comparison, position guard, repeat scoring

- **gpt-6-luna vs gpt-5.6-luna** on the 12 non-passing scenarios, run side by side: 5 vs 6 passing,
  mean judge 3.4 vs 3.2 — no real difference; both make the same class of mistake (a position issuing
  a clearance it doesn't own). Kept gpt-5.6-luna (`OPENAI_MODEL` switches it). gpt-6-luna was ~2×
  faster on a single probe but not in the suite.
- **Position guard** (`position_rules.py`): after every reply, clearances are checked against the
  position (Approach never clears to land, Ground never clears takeoff, Delivery never taxis, no
  self-handoff, circuit traffic never handed to Approach). A violation regenerates once with the reason;
  if the model insists, the aircraft is handed to the owning position. Fired on ~4% of replies.
- **Emergency squawks**: silent 7700 now gets "confirm squawking 7700, say nature of emergency";
  7600 gets "if you hear this transmission, squawk ident" (no "roger" to a silent aircraft).
- **Trigger fix**: an IFR departure levelling off inside 25 nm on its way elsewhere was handed back to
  this airport's Approach. Approach handoff now requires "landing here" and closing on the field.
- **No invented Center frequencies**: "contact <Name> Center, frequency on your chart".
- **Harness**: pilot agent is now gpt-5.6-luna (mini looped/went silent), `-r 3` majority scoring
  (PASS / FLAKY / FAIL), guard-fix counts in the summary.

Final (3 runs per scenario, `20260928_113027_SUMMARY.md`): **25 PASS, 5 FLAKY, 2 FAIL, 2 XFAIL,
mean judge 4.00/5**; plugin profiles 4/4.

Remaining, in order of value:
1. Touch-and-go (EN) never says "cleared touch-and-go" (0/3) — the model defaults to "cleared to land".
   Candidate for a deterministic rule: circuit plan + pilot asked for touch-and-go → rewrite.
2. Tower phraseology polish (judge): wind in every takeoff/landing clearance, fewer "colação correta",
   require the base/final reports that were asked for before clearing to land.
3. FLAKY PT-BR ones (sbgr ILS, sbgr delivery readback, PAN PAN approach clearance, SBMT circuit) swing
   2–5 between runs: same model, same prompt. More structure (JSON output with clearances) would
   cut this.
4. Center/ACC data (XFAIL enroute) and playground cross-airport by name (XFAIL diversion).

### Real X-Plane run (2026-09-28, Windows box, SBBH Pampulha, Baron 58 N45XS, PT-BR)

Verified in the real sim (plugin → STT → luna → TTS → `playPCMOnBus` on COM1, every call audible):
auto-tune to SBBH; COM1 retune via UDP DREF switches Ground ↔ Tower; taxi clearance (runway from live
wind, "negativo, pista três um" when the pilot said 13); "chame Torre" handoff; bare frequency readback
on the old frequency gets no reply; Tower "autorizado decolagem pista três um" (1.7 s). Replies 1.3–7 s.

Found and fixed there (not visible in the e2e harness):
- STT punctuates callsigns ("cinco, X-Ray Sierra") → callsign stripping failed → handoff readback was
  answered. Spoken-callsign regex now allows `,` and `.`.
- The plugin tuned only once at X-Plane load: a server restart left it posting into "no session" until
  X-Plane restarted. It now re-tunes on HTTP 400 from /state (verified: back in ~2 s).
- `box.sh` pinned the box host key in /private/tmp (wiped on reboot) → `~/.ssh/known_hosts_atc_windows_box`.

Airborne chain, same day (after the BIOS update), with `PI_ATCTestTools` placing the aircraft
(`scripts/box.sh place ...`; UDP `PREL` is ignored by X-Plane 12, the SDK call works):
Tower "autorizado decolagem pista três um" → placed on climb-out → **Tower→Controle trigger from real
datarefs** ("chame Controle em um dois zero decimal dois", SBBH APP 120.2) → COM1 120.2 → Controle
check-in → "autorizado aproximação visual pista um três" → placed on final → **Approach→Tower trigger**
("chame Torre em um um oito decimal zero"; repeated by initiative while COM1 stayed on APP) → COM1 118.0 →
"autorizado pouso pista um três" → placed on the runway → **Tower→Ground trigger** ("livre da pista, chame
Solo em um dois um decimal meia"). Every call was played on the COM1 bus.

Found and fixed during the airborne run:
- PT postfix approach clearance ("aproximação visual pista um três autorizada") wasn't parsed, so the
  Approach→Tower trigger could never fire for it.
- An off-field touchdown (the unmanned Baron crashed 2.2 nm short after the landing clearance) produced
  "livre da pista, chame Solo". Touchdown now has to be within 1.5 nm of the field.

Test-rig notes: nobody flies the aircraft, so it crashes ~20 s after each placement (X-Plane logs "Plane
crashed" and ignores commands for a few seconds; `scripts/box.sh cmd sim/operation/reset_flight` clears
it). `atctest/ap_hold` does not hold the Baron. The ATC only needs a few `/state` ticks per step.

Still unverified in the sim: mic/PTT recording (needs a person at the machine) and a flight flown by a
human end to end.

---

# 2026-09-09 deep dive

## TL;DR

The core idea works: `gpt-5.6-luna` behaves like a competent controller when it is given a
flight strip (type, rules, destination), the real runway list, published frequencies and a
clear position. In the probes it refused a VFR departure in IFR weather, demanded an IFR
clearance before taxi at SBGR, refused to act as Tower while on the Ground frequency, and
caught a pilot reading back a runway that doesn't exist. Latency with `reasoning=low` is
1.5–6 s per transmission (tool calls add ~2 s each), which is usable on a radio.

Before this session the project could not actually complete a flight: chart PDFs were dead,
TTS was out of credits, the clearance parser recorded the callsign as the assigned altitude,
Portuguese could never trigger a handoff, the session did not follow the pilot's radio, and
the X-Plane plugin never played audio. All of those are fixed and covered by tests; what is
left is listed under **Open gaps**.

## What was verified live (2026-09-09)

| Check | Result |
|---|---|
| OpenAI key, `gpt-5.6-luna` available | yes (also `gpt-5.6-sol`, `gpt-5.6-terra`, 5.5, 5.4-mini...) |
| Responses API + tools on luna | ok, `reasoning` item + `function_call` |
| ElevenLabs | key valid, **quota 121000/121000 used** → every TTS call is 401 `quota_exceeded`; STT still works |
| airportwxbot chart listing | ok (SBGR 58, KJFK 43) |
| airportwxbot chart **PDF** | **403 Cloudflare "Just a moment" for every non-browser client** (curl, requests, OpenAI's fetcher). Dead. |
| airportwxbot METAR/TAF/NOTAM | ok |
| AISWEB (DECEA) aerodrome page + PDFs | ok, public, 62 charts for SBGR (24 IAC, 15 SID, 9 STAR...) |
| FAA d-TPP metafile + PDFs | ok, public, cycle 2609 (38 charts for KJFK) |
| luna reading an ILS plate (FAA, by URL) | correct: course 044°, DA 231 ft, tower 123.9, missed climb 2000 |
| luna reading SBGR ADC (AISWEB, base64) | correct: elevation 2461, both runways with lengths, TWR/GND freqs |
| OpenAI TTS `gpt-4o-mini-tts` + STT `gpt-4o-transcribe` | EN and PT-BR round-trip exact; TTS 3–4 s, STT ~1 s |
| Offline test suite | 96 tests, 0.8 s |
| `e2e/fake_xplane.py` SBGR profile (plugin traffic pattern) | PASS: taxi → COM1 retune to Tower → takeoff → **automatic handoff to São Paulo APP 119.8** → check-in → climb |
| `e2e/fake_xplane.py` KJFK profile | PASS: taxi 22L → tower → takeoff → **handoff to New York Departure 135.9** |
| `e2e/run_e2e.py` (LLM pilot + judge), see table below | 5 scenarios |
| **Real X-Plane 12.4** (Windows box, `scripts/box.sh`) | plugin loads, auto-tunes SBBH, pilot WAV → ATC audio on COM1 bus in ~6 s, COM1 retune follows to Tower |

E2E scenario results (ATC `gpt-5.6-luna` low effort, pilot `gpt-5.4-mini`, judge luna):

| Scenario | Hard checks | Judge | Avg latency | Notes |
|---|---|---|---|---|
| kjfk_ifr_departure_en | 11/11 | 4/5 | 3.7 s | Delivery clearance with squawk + dep freq, taxi 22R, takeoff, rule-based handoff to Departure |
| sbgr_ifr_departure_ptbr | 10/10 | 4/5 | 3.0 s | Tráfego → Solo → Torre; **read the AISWEB AGMC chart** and taxied "via Alfa, Golf, Charlie"; handoff to Controle 119.8 |
| wrong_runway_readback_en | 8/8 | 3–4/5 (3 runs) | 2.5–3.2 s | every run: "negative, runway two seven right does not exist", then correct 10L. Lower judge scores came from the gpt-5.4-mini *pilot* looping "continuing taxi" instead of calling ready; use `--pilot gpt-5.6-luna` for a stricter pilot |
| sbmt_vfr_pattern_ptbr | 8/8 | 3/5 | 3.1 s | full circuit, but cleared a full-stop landing instead of the requested touch-and-go (see gap 10) |
| kbos_ils_arrival_en | 8/10 | 3/5 | 6.1 s | readback 5/5, position 5/5: wrong altitude readback caught; controller vectors unprompted via **initiative** (10 calls). Never reached the approach clearance/tower handoff: the e2e sim can't intercept a localizer, so "established" never happens (gap 10). Initiative calls were 8–17 s before the facts cache (weather/chart list re-fetched each time); after it, expect ~3–4 s. |

Model comparison on the same SBGR ground call (first turn, includes tool calls):

| Model | Latency | Reply quality |
|---|---|---|
| gpt-5.6-luna | 10.7 s (default effort) / 1.5–6 s (`low`) | best judgement, terse, correct position discipline |
| gpt-5.5 | 14.4 s | good |
| gpt-5.1 | 5.1 s | ok, invented taxiway names |
| gpt-5.4-mini | 1.7 s | acceptable phraseology, weaker rule-following; used for the e2e pilot agent |

Recommendation: keep `gpt-5.6-luna` with `OPENAI_REASONING_EFFORT=low` (now the default).

## Fixed in this session

- **Model** → `gpt-5.6-luna` default, reasoning effort configurable (`low`).
- **Charts** → `chart_providers.py`: AISWEB (BR) + FAA d-TPP (US) with disk caching; PDFs are
  downloaded server-side and attached as base64 so OpenAI never has to fetch them. `read_chart`
  failures no longer 500 the whole transmission.
- **Audio** → `audio_service.py` provider policy `auto`: ElevenLabs, fall back to OpenAI on any
  error (benched 5 min). WAV 16 kHz output for X-Plane. TTS failure returns text with
  `audio_base64: null` instead of an HTTP 500.
- **Clearance parsing** → `clearance_parser.py`: callsign stripped first (the old regex set
  `assigned_altitude = 12345` from `N12345`), spoken numbers in EN and PT-BR ("dois mil e
  quinhentos", "flight level three five zero"), runways ("dois oito direita" → 28R), handoff
  target + frequency ("chame Torre em um três dois decimal sete cinco" → tower, 132750 kHz).
- **Portuguese handoffs** → triggers were English-only keyword matches; now language-aware
  messages with spoken frequencies, and PT-BR clearances are recognised so triggers fire.
- **Flight plan + airport data in context** → `FlightPlan` (type/rules/dest/route/cruise) on
  `/tune`; airport name, elevation, valid runway list, published frequencies, distance/bearing
  and AGL in the situation summary. The controller now corrects nonexistent runways.
- **Clearance Delivery** position added (`Delivery`/`Tráfego`).
- **Follow the radio** → retuning COM1 to another published position switches the session
  (`sync_phase_to_frequency`). Without a radio (playground/e2e) the controller's own
  "contact X" moves the session on the pilot's next call (`pending_handoff`).
- **Triggers** use real geometry (distance to field, AGL) instead of raw MSL altitude; Brazil
  hands Tower → Approach when no DEP frequency exists.
- **Plugin** → real playback on the COM1 bus (`xp.playPCMOnBus`), `/state` moved off the flight
  loop into a worker thread (the old code did a blocking HTTP call inside the sim loop), 8.33
  kHz COM1 dataref, auto-tune to nearest airport, aircraft type from the model, callsign refresh
  on plane load, status label widget.
- **Server** → thread lock around session mutation, `/airport`, `/nearest_airport`, `/charts`,
  richer `/health`, `wav|mp3` format selection, non-latin-1 headers base64-wrapped.
- **Controller initiative** → the model used to speak only when the pilot did, so an approach
  controller never vectored or cleared the ILS on its own (Boston scenario: 14 turns of
  "roger"). Now `/state` asks the model during radio silence (40 s quiet, ≥ 60 s between asks,
  airborne or moving) whether a controller would transmit; a non-`NONE` answer is delivered like
  a handoff through `/pending_atc`. `ATC_INITIATIVE_*` in config; `?initiative=true` forces it.
- **Weather at small fields** → airportwxbot 500s for SBMT; `get_weather` now falls back to the
  nearest airport with a METAR (within 40 nm) and says so.
- **Pattern work** → flights whose plan is a circuit (dest = dep, "circuito", "touch and go"...)
  are never handed off to Departure/Approach after takeoff.
- **Known-facts cache** → weather, chart lists and airport info fetched once are injected into
  the prompt ("Facts already looked up this session"), so the model stops re-calling tools on
  every turn (this was doubling initiative latency).
- **Tests** → `tests/` (96 offline), `e2e/run_e2e.py` (pilot agent + sim + judge, 5 scenarios;
  pilot can go silent with `STANDBY` to exercise initiative), `e2e/fake_xplane.py`
  (plugin-faithful driver), `Makefile`.

## Open gaps (ranked by impact on "feels like real ATC")

### 1. No other traffic
The controller only knows about one aircraft. Real ATC is 80% sequencing. Options:
- Read X-Plane AI/multiplayer traffic datarefs (`sim/multiplayer/position/plane<N>_*`) in the
  plugin, send them in `/state`, and put a "traffic" table in the situation summary so the LLM
  can issue traffic advisories, "number two following the Embraer on final", and hold-shorts.
- Or synthesise fake traffic server-side (a few scripted aircraft on the same runway) for
  atmosphere. Cheaper, less consistent with what the pilot sees out the window.

### 2. Procedures are read from PDFs, not structured data
Charts work again, but the LLM reads a plate every time it needs a fix. It's slow (~4 s), costs
tokens, and can misread. X-Plane ships **CIFP** (`Resources/default data/CIFP/<ICAO>.dat`,
ARINC-424) with every SID/STAR/approach waypoint, altitude and course, and `apt.dat` with
taxiways. The plugin can upload the airport's CIFP + taxiway graph to the server once per
session; then `get_procedure(name)` returns exact fixes and the controller can vector to the
real FAF and issue real taxi routes ("via Alpha, Bravo, hold short 10L"). This is the single
biggest realism win after traffic, and it also gives taxi routing (today the model invents or
omits taxiway names).

### 3. Latency and the "radio feel"
1.5–6 s per reply is acceptable but the first turn at a new airport can hit 10 s (weather +
airport info tool calls). Improvements, in order of cheapness:
- Prefetch METAR + airport info at `/tune` and inject them into the prompt (no tool round-trip).
- Stream the model and start TTS on the first sentence.
- Play a short radio click / "stand by" immediately on PTT release.
- OpenAI Realtime (`gpt-realtime-2.1`) for speech-to-speech would remove STT and TTS hops
  entirely, but tool-calling for charts inside Realtime needs its own design.

### 4. Position/state machine is still LLM-driven
Whether a pilot is "holding short", "lined up", "airborne" comes from clearances parsed out of
the controller's words. The parser is good and tested, but a structured output from the model
(`{"transmission": "...", "clearances": [...]}`) would be more robust than regex over
phraseology. Cheap to add with the Responses API `text.format` JSON schema; keep the parser
as the safety net.

### 5. Readback verification is prompt-only
The controller is told to catch wrong readbacks and it does in tests, but nothing checks it
deterministically. `clearance_parser` can parse the *pilot's* readback too; the server can then
compare runway/altitude/heading/squawk/frequency against the last clearance and inject a
"readback mismatch: pilot said 27R, cleared 10L" note into the prompt, or even generate the
correction without an LLM call.

### 6. Single session, single aircraft, global mutable state
`pending_atc_call`, `current_language`, `llm_service` are module globals; one pilot per server.
Fine for a desktop sim companion. If this ever serves several sims (a VA, a classroom), move
state into `SessionManager` keyed by callsign and pass a session id from the plugin.

### 7. Weather, ATIS, NOTAMs
- Weather comes from airportwxbot (real world). X-Plane's weather may differ (custom weather,
  time travel). The plugin could send the sim's wind/QNH datarefs so ATC reports what the pilot
  actually sees. Needs a "wind" table in `/state`.
- No ATIS. Generating an ATIS text + audio from METAR (`information Alpha`) is a 30-line
  feature and pilots expect it.
- NOTAM endpoint returns an empty list for SBGR; check whether airportwxbot actually has BR
  NOTAMs or switch to AISWEB's NOTAM page.

### 8. Voice quality
OpenAI `gpt-4o-mini-tts` with the "VHF radio, clipped, monotone" instruction sounds decent; it
is not the ElevenLabs voice. A band-pass filter (300–3400 Hz) + slight noise on the PCM before
`playPCMOnBus` would sell the radio effect regardless of provider (stdlib `array` math, no
dependencies). Also: different voices per position (ground vs tower vs approach).

### 9. Plugin: verified in the sim, mic still unproven
Ran in X-Plane 12.4 on the Windows box (XPPython3 4.2.1 / Python 3.12) over Tailscale on
2026-09-09: load, auto-tune (SBBH, Baron 58 N45XS), 2 s state loop, `virtualatc/test_transmit`
WAV → OpenAI STT → luna → OpenAI TTS → `playPCMOnBus` on the COM1 bus (7 s clip, ~6 s after the
call), COM1 retune 121.6 → 118.0 switching the session Ground → Tower. Found and fixed there:
`playPCMOnBus` argument order, `com1_frequency_hz_833` is in kHz, mono-only mic open.
Still open:
- **Mic**: inside an RDP session pyaudio finds no input endpoint ("no default output device"), so
  push-to-talk recording is unverified. Someone at the machine needs to hold PTT once; the recorder
  now tries configured device → default device × mono/stereo × 16/44.1/48 kHz.
- Joystick button 706 is bound to `virtualatc/ptt` on that box and something clicks the plugin
  window's button at startup (two recording attempts per launch, logged as "PTT button clicked" now).
- Tower approved a start-up request instead of sending the pilot to Ground first (minor
  position-discipline slip; add to the per-position rules).
- The old January plugin on the box used a separate `audio_recorder_server.py` process to keep
  pyaudio out of X-Plane; if in-process recording stutters the sim, revive that design
  (`xplane_plugin/from_box/` has the client side).

### 10. Small things
- Touch-and-go phraseology: at SBMT the tower cleared "autorizado pouso" for a requested toque e
  arremetida and repeated the clearance on final (judge 3/5). Add the pattern-work phrase set to
  the PT-BR prompt ("autorizado toque e arremetida pista XX", "após, autorizado pouso completo").
- Initiative is timer-based; better triggers are geometric (crossing the FAF, reaching the
  assigned altitude, abeam the field on downwind) so the controller calls at the right moment.
- The e2e aircraft sim is crude (no turns toward the runway, no localizer intercept); the ILS
  arrival can't actually get "established", so an approach clearance is a judgement call by the
  model. A tiny nav model (fly heading, intercept final course) would make that scenario crisp.
- Delivery/Ground/Tower share one LLM persona; per-position prompt sections (what each may
  do) would tighten position discipline further.
- Handoff message when the target has no published frequency ("remain this frequency, radar
  contact") is a fudge for small BR fields with no APP; better to hand off to the nearest APP
  in the frequency table (e.g. SBMT → São Paulo APP 119.8 exists; other fields won't).
- OurAirports frequencies have gaps/aliases (JFK ground listed as "GND ALT"); consider
  X-Plane `apt.dat` frequencies from the plugin as the primary source.
- Trigger thresholds (800 ft AGL, 25/8 nm) are guesses; tune per airport class.
- Judge scoring in e2e uses luna; keep a small golden set of transcripts to detect prompt
  regressions without paying for a judge every time.

## How to keep it honest

```bash
make test                       # every commit, 1 s
make e2e S=wrong_runway_readback_en   # cheapest live check (~1 min, ~10 turns)
make serve && make fake-xplane  # plugin contract, listen to e2e/reports/fake_xplane/*.wav
make e2e                        # full suite before changing prompts/models (~10 min)
```
