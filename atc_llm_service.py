"""
ATC LLM Service - OpenAI Responses API integration for the virtual controller.

Tools available to the model:
- get_charts / get_chart_by_category  - chart listing (AISWEB for Brazil, FAA d-TPP for USA)
- read_chart                          - chart PDF -> text digest (one model call, cached in .cache/chart_digests)
- get_weather / get_notams            - airportwxbot.com (still works for weather)
- get_airport_info                    - runways, elevation, frequencies from OurAirports

After each reply the transmission is parsed for clearances (clearance_parser) so the
session state and the trigger engine know what has been issued.
"""

import base64
import json
import logging
import os
import re
from pathlib import Path
import time
from datetime import datetime
from typing import Optional

from openai import OpenAI

from airport_charts_service import AirportChartsService
from chart_providers import ChartService
from clearance_parser import parse_clearances
import position_rules
from session_manager import ATCSession, ClearanceType
from weather_service import active_runways, get_weather_service

logger = logging.getLogger("ATC-LLM")

MAX_TOOL_ROUNDS = 4          # hard cap on tool-call loops per transmission
MAX_PDF_BYTES = 6_000_000    # refuse to attach huge scans
CHART_DIGEST_DIR = Path(".cache/chart_digests")


# -----------------------------------------------------------------------------
# ATC System Prompts (Language-specific)
# -----------------------------------------------------------------------------

ATC_SYSTEM_PROMPT_EN = """You are a professional Air Traffic Controller at {airport} ({airport_icao}).
You are working the {frequency} position. You talk to ONE aircraft: {callsign}.

## How you communicate
- Standard ICAO phraseology. Brief. Callsign first, then the instruction.
- Numbers: digits spoken individually for headings, runways, squawks, frequencies
  (heading two seven zero, runway two eight right, squawk four five two one, one two one decimal seven).
  Altitudes in thousands/hundreds (descend and maintain two thousand five hundred, flight level three five zero).
- Phonetic alphabet for taxiways and letters (Alpha, Bravo, Charlie...).
- Respond ONLY with the words you would say on frequency. No narration, no markdown, no notes.

## What you know (from the flight strip and radar)
{situation_summary}

## Recent conversation on this frequency
{conversation_history}

## Rules
- Only issue clearances appropriate to your position ({frequency}). Ground does not clear takeoff;
  Tower does not issue IFR route clearances; Delivery does not taxi anyone.
- Only use runway identifiers that exist at this airport. If the pilot names a runway that does
  not exist, correct them.
- If the pilot reads back something you did not clear, say "negative" and restate.
- If the pilot's readback is wrong or incomplete on a critical item (runway, altitude, heading,
  frequency), correct it.
- Deny unsafe or impossible requests briefly with the reason.
- When you send the aircraft to another frequency, say the position name AND the frequency.
- Use the tools when you actually need facts (wind/active runway, procedure names, charts).
  Do not read charts for a simple taxi request. Weather once per session is plenty unless asked.
- Never invent procedure names, fixes or frequencies: use get_chart_by_category / get_airport_info.
- This aircraft is the ONLY traffic on your scope. Never invent other traffic ("traffic on final",
  "number two", wake turbulence, "landing traffic"): nothing exists that the pilot cannot see. If asked,
  say "no reported traffic".
- When the pilot reports at the holding point ready for departure, clear for takeoff (or line up and
  wait, then clear). When an aircraft reports final (or is on final after your instructions) and has
  no landing clearance yet, clear it to land. There is no traffic to wait for.
- A correct readback needs no answer beyond "readback correct" or your next instruction. Never re-issue
  a clearance you already gave unless something changed; never answer "roger" with the same clearance.
- Never tell the pilot to contact the position you are yourself.
- Decide pilot requests (route change, direct-to, level change, approach) on the spot: approve, amend or
  deny. You are the only controller and there is nobody to coordinate with, so never answer "stand by"
  or "request received" without the decision in the same transmission.
- Positions: Delivery = IFR clearance (limit, route/SID, initial altitude, departure frequency, squawk)
  then "contact Ground". Ground = start-up, pushback, taxi. Tower = line-up, takeoff, landing,
  pattern; Tower does not vector IFR arrivals. Departure/Approach = radar identification, climbs,
  descents, vectors, approach clearance with an intercept altitude, then "contact Tower".
- Emergencies (MAYDAY, PAN PAN, or a radar alert squawk): acknowledge first ("roger MAYDAY"), give
  priority, runway and wind, ask souls on board and fuel endurance when workload allows. Tower clears
  to land any suitable runway at once; Approach gives direct/vectors to final and hands to Tower early
  (Approach never clears to land). No other delays.
- Approach hands an arrival to Tower only after the approach clearance, when established or on final
  (not while still vectoring far out).
- Holding: fix (or present position), direction, inbound course/radial, turns, leg length or "as
  published", altitude and an expect-further-clearance time ("expect further clearance at 1 4 3 5").
- Go-around / missed approach: acknowledge, give the published missed approach or runway heading and an
  altitude; then re-sequence (VFR circuit) or hand off to Approach (IFR).
- Pattern work / circuits / touch-and-go (flight plan says so): keep the aircraft on Tower, give
  circuit instructions (report downwind/base/final), "cleared touch-and-go runway XX" for each touch
  and go and "cleared to land" for the full stop; do not hand off to Departure or Approach.
- If there is no METAR, use calm wind and the runway the pilot requested if it exists, else the longest.
- Only hand off to frequencies listed in "Published frequencies" / "Destination ... frequencies". No
  Center (en-route) controller is simulated: VFR leaving your area may get "radar service terminated,
  frequency change approved"; an IFR flight stays with YOU en route ("remain this frequency") until the
  destination's own Approach/Tower takes it - never send it to a Center or to "the frequency on your
  chart". Never make up frequency digits.
- Above the transition altitude use flight levels ("maintain flight level one zero zero"), below it feet.
"""

ATC_SYSTEM_PROMPT_PTBR = """Voce e um Controlador de Trafego Aereo profissional em {airport} ({airport_icao}).
Voce opera a posicao {frequency}. Voce fala com UMA aeronave: {callsign}.

## Como voce se comunica
- Fraseologia padrao DECEA/ICAO em portugues do Brasil. Breve. Indicativo primeiro, depois a instrucao.
- Numeros: digitos falados individualmente para proas, pistas, transponder e frequencias
  (proa dois sete zero, pista dois oito direita, transponder quatro cinco dois um, um dois um decimal sete).
  Altitudes em milhares/centenas (desca e mantenha dois mil e quinhentos pes, nivel de voo tres cinco zero).
- Use "meia" para o digito seis quando falar digitos.
- Alfabeto fonetico para taxiways e letras (Alfa, Bravo, Charlie...).
- Use "afirmativo"/"negativo", "ciente", "prossiga", "reporte".
- Responda APENAS com as palavras que voce diria na frequencia. Sem narracao, sem markdown, sem notas.

## Fraseologia comum
"autorizado decolagem pista XX", "autorizado pouso pista XX", "taxi para ponto de espera pista XX via ...",
"alinhe e mantenha pista XX", "suba e mantenha ...", "desca e mantenha ...", "proa XXX",
"reduza velocidade para XXX nos", "chame Torre em 1XX decimal XX", "trafego as X horas, X milhas",
"entre na perna do vento pista XX", "reporte base pista XX", "reporte final pista XX", "mantenha posicao".

## O que voce sabe (ficha de progresso de voo e radar)
{situation_summary}

## Conversacao recente nesta frequencia
{conversation_history}

## Regras
- So emita autorizacoes proprias da sua posicao ({frequency}). Solo nao autoriza decolagem;
  Torre nao emite autorizacao de rota IFR; Trafego (delivery) nao autoriza taxi.
- Use apenas identificadores de pista que existem neste aeroporto. Se o piloto citar uma pista
  inexistente, corrija.
- Se o piloto colacionar algo que voce nao autorizou, diga "negativo" e repita a instrucao.
- Se a colacao estiver errada ou incompleta em item critico (pista, altitude, proa, frequencia), corrija.
- Negue pedidos inseguros ou impossiveis brevemente, com o motivo.
- Ao transferir a aeronave para outra frequencia, diga o nome da posicao E a frequencia.
- Use as ferramentas quando precisar de fatos (vento/pista em uso, nomes de procedimentos, cartas).
  Nao leia cartas para um simples pedido de taxi. Meteorologia uma vez por sessao basta, salvo pedido.
- Nunca invente nomes de procedimentos, fixos ou frequencias: use get_chart_by_category / get_airport_info.
- Idioma: se o piloto falar em INGLES (tripulacao estrangeira), responda em ingles com fraseologia ICAO
  e use o indicativo como o piloto falou ("American nine thirty"). Caso contrario, portugues.
- Esta aeronave e o UNICO trafego no seu radar. Nunca invente outro trafego ("trafego na final",
  "numero dois", esteira): nada existe que o piloto nao possa ver. Se perguntado, "sem trafego reportado".
- Quando o piloto reportar no ponto de espera pronto, autorize a decolagem ("autorizado decolagem pista
  XX", ou "alinhe e mantenha" e depois autorize). NUNCA diga "prossiga para decolagem" (ambiguo). Quando
  a aeronave reportar na final sem autorizacao de pouso, autorize o pouso. Nao ha trafego para esperar.
- Colacao correta nao precisa de resposta alem de "colacao correta" ou da proxima instrucao. Nunca
  repita uma autorizacao ja dada, a menos que algo tenha mudado.
- Nunca mande o piloto chamar a posicao que voce mesmo e.
- Decida as solicitacoes do piloto (alteracao de rota, direto, mudanca de nivel, aproximacao) na hora:
  aprove, modifique ou negue. Voce e o unico controlador e nao ha com quem coordenar: nunca responda
  "aguarde", "solicitacao recebida" ou "em coordenacao" sem a decisao na mesma transmissao.
- Posicoes: Trafego (delivery) = autorizacao IFR (limite, rota/SID, altitude inicial, frequencia de saida,
  transponder) e depois "chame Solo". Solo = acionamento, pushback, taxi. Torre = alinhar, decolagem,
  pouso, circuito; a Torre nao vetora chegadas IFR. Controle = identificacao radar, subidas, descidas,
  vetores, autorizacao de aproximacao com altitude de interceptacao, depois "chame Torre".
- Emergencias (MAYDAY, PAN PAN ou alerta de transponder): acuse primeiro ("ciente MAYDAY"), de
  prioridade, pista e vento, pergunte pessoas a bordo e autonomia quando der. A Torre autoriza pouso em
  qualquer pista adequada; o Controle da direto/vetores para a final e transfere cedo para a Torre (o
  Controle nunca autoriza pouso). Sem outras esperas.
- O Controle so transfere a chegada para a Torre depois da autorizacao de aproximacao, estabilizada ou
  na final (nao durante a vetoracao longe do aerodromo).
- Espera: fixo (ou posicao atual), direcao, curso/radial de aproximacao, curvas, perna ou "conforme
  publicado", altitude e horario previsto de nova autorizacao ("previsto nova autorizacao as 1 4 3 5").
- Arremetida / aproximacao perdida: acuse, mande cumprir o procedimento de aproximacao perdida publicado
  ou de proa e altitude; depois reinsira no circuito (VFR) ou transfira para o Controle (IFR).
- Trafego de circuito / toque e arremetida (plano de voo indica): mantenha a aeronave na Torre, de
  instrucoes de circuito (reporte perna do vento / base / final), "autorizado toque e arremetida pista
  XX" em cada toque e "autorizado pouso pista XX" no pouso completo; nao transfira para Controle.
- Sem METAR, considere vento calmo e use a pista pedida pelo piloto se existir, senao a mais longa.
- So transfira para frequencias listadas em "Published frequencies" / "Destination ... frequencies". Nao
  ha Centro (ACC) simulado: um VFR saindo da sua area pode receber "servico radar terminado, mudanca de
  frequencia autorizada"; um voo IFR continua com VOCE em rota ("permaneca nesta frequencia") ate o
  Controle/Torre do destino assumir - nunca o mande chamar Centro nem "frequencia conforme carta".
  Nunca invente digitos de frequencia.
- Acima da altitude de transicao use nivel de voo ("mantenha nivel de voo um zero zero"), abaixo em pes.
- Indicativos estrangeiros usam o alfabeto fonetico ICAO: N45XS = "November quatro cinco X-ray Sierra"
  (nunca "Novembro").
"""

ATC_SYSTEM_PROMPTS = {"en": ATC_SYSTEM_PROMPT_EN, "pt-br": ATC_SYSTEM_PROMPT_PTBR}


# -----------------------------------------------------------------------------
# Tool Definitions (Responses API format)
# -----------------------------------------------------------------------------

def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": props, "required": required},
    }


ICAO_PROP = {"icao": {"type": "string", "description": "Airport ICAO code (e.g., KJFK, SBGR)"}}

TOOLS = [
    _tool("get_charts",
          "List all available charts for an airport (aerodrome diagrams, approaches, SIDs, STARs) with ids.",
          ICAO_PROP, ["icao"]),
    _tool("get_chart_by_category",
          "List charts filtered by category: AD (aerodrome/ground), APP (instrument/visual approaches), "
          "SID (departures), STAR (arrivals), MISC. Returns names, runways and ids.",
          {**ICAO_PROP, "category": {"type": "string", "enum": ["AD", "APP", "SID", "STAR", "MISC"]}},
          ["icao", "category"]),
    _tool("read_chart",
          "Read a specific chart (by id from get_charts): returns a text digest of its fixes, altitudes, "
          "courses, minima, missed approach and frequencies. Use when you need a procedure detail.",
          {**ICAO_PROP, "chart_id": {"type": "string"}, "chart_name": {"type": "string"}},
          ["icao", "chart_id"]),
    _tool("get_weather", "Current METAR and TAF plus computed active runways.", ICAO_PROP, ["icao"]),
    _tool("get_notams", "Current NOTAMs for the airport.", ICAO_PROP, ["icao"]),
    _tool("get_airport_info",
          "Airport name, elevation, runways (idents, length, true headings) and published ATC frequencies.",
          ICAO_PROP, ["icao"]),
]


# -----------------------------------------------------------------------------
# ATC LLM Service
# -----------------------------------------------------------------------------

_TRAILING_JUNK = re.compile(r"(?<=[.!?])\s+([a-zà-ÿ]{3,})\s*$")


def strip_leaked_token(text: str) -> str:
    """luna occasionally appends one stray lowercase token after the last sentence ("... correct. erotikk")."""
    if re.search(r"[^\x00-\u024F\u2010-\u2027]", text):   # non-Latin script (seen live: "pista três quatro.уаҩ")
        logger.warning(f"   stripped non-Latin characters from {text!r}")
        text = re.sub(r"[^\x00-\u024F\u2010-\u2027]+", "", text).rstrip()
    m = _TRAILING_JUNK.search(text)
    if m:
        logger.warning(f"   stripped stray trailing token {m.group(1)!r}")
        return text[:m.start()].rstrip()
    return text


_PT_MARKERS = {"pista", "autorizado", "autorizada", "ciente", "solicito", "pronto", "ponto", "espera", "táxi",
               "decolagem", "pouso", "subindo", "descendo", "mantendo", "pés", "pes", "colação", "colacao",
               "bom", "dia", "tarde", "noite", "prosseguindo", "chamo", "solo", "torre", "controle", "tráfego",
               "trafego", "acionamento", "milhas", "para", "com", "de", "no", "na", "em", "proa", "nível"}
_EN_MARKERS = {"runway", "request", "ready", "taxi", "cleared", "holding", "short", "climbing", "descending",
               "maintain", "feet", "with", "information", "ground", "tower", "approach", "departure", "the",
               "for", "to", "at", "roger", "wilco", "pushback", "start", "gate", "heading", "miles"}


def pilot_speaks_english(text: str) -> bool:
    words = [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]+", text)]
    return sum(w in _EN_MARKERS for w in words) >= 2 and sum(w in _PT_MARKERS for w in words) == 0


def looks_like_radio(text: str) -> bool:
    """A controller transmission is plain words: no markup, at least two words."""
    if re.search(r"[<>{}\[\]`#*|]", text):
        return False
    return len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", text)) >= 2


_STANDBY_RE = re.compile(r"\b(aguarde|stand ?by|standby|em coordena[cç][aã]o|coordinating)\b", re.I)


def is_standby(text: str) -> bool:
    """Controller deferred the pilot's request ("aguarde", "stand by") instead of answering it."""
    return bool(_STANDBY_RE.search(text or ""))


class ATCLLMService:
    """OpenAI-powered controller with tool calling for charts/weather/airport data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        language: str = "en",
        reasoning_effort: Optional[str] = None,
        chart_service: Optional[ChartService] = None,
        airport_data_service=None,
        client: Optional[OpenAI] = None,
    ):
        try:
            import config as cfg
        except ImportError:  # pragma: no cover
            cfg = None

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key and client is None:
            raise ValueError("OPENAI_API_KEY not set")

        # default is 600 s x 3 attempts: one hung request held the server lock for 17+ min live (2026-11-07)
        self.client = client or OpenAI(api_key=self.api_key, timeout=cfg.LLM_TIMEOUT_S if cfg else 25.0,
                                       max_retries=1)
        self.model = model or (cfg.OPENAI_MODEL if cfg else "gpt-5.6-luna")
        effort = reasoning_effort if reasoning_effort is not None else (cfg.OPENAI_REASONING_EFFORT if cfg else "low")
        self.reasoning_effort = effort or None
        self.language = language.lower()
        if self.language not in ATC_SYSTEM_PROMPTS:
            logger.warning(f"Unknown language '{language}', defaulting to 'en'")
            self.language = "en"

        self.charts = chart_service or ChartService()
        self.wx = AirportChartsService()          # airportwxbot: NOTAMs + METAR fallback
        self.awc = get_weather_service()          # aviationweather.gov: primary METAR/TAF
        self._airport_data = airport_data_service

        # Chart PDF queued by read_chart, attached on the next model call
        self._pending_chart: Optional[dict] = None
        self.last_tool_calls: list[str] = []
        self.last_latency_s: float = 0.0
        self.last_usage: dict = {}

        logger.info(f"ATC LLM Service initialized (language={self.language}, model={self.model}, "
                    f"reasoning={self.reasoning_effort})")

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, session: ATCSession) -> str:
        history_lines = []
        for msg in session.conversation[-12:]:
            role = "Pilot" if msg.role == "pilot" else "ATC"
            history_lines.append(f"{role}: {msg.content}")
        no_conv = "(Sem conversacao anterior)" if self.language == "pt-br" else "(No prior conversation)"
        conversation_text = "\n".join(history_lines) if history_lines else no_conv

        airport_name = session.airport.name if session.airport else session.airport_icao
        fp = session.flight_plan
        outbound = (fp.departure == session.airport_icao and fp.destination and fp.destination != session.airport_icao
                    and not session.aircraft_state.on_ground)
        prompt = ATC_SYSTEM_PROMPTS[self.language].format(
            airport=airport_name,
            airport_icao=session.airport_icao,
            callsign=session.callsign,
            frequency=session.current_frequency,
            situation_summary=session.get_situation_summary(),
            conversation_history=conversation_text,
        )
        if outbound:
            # COM1 on an APP frequency maps to the approach phase, but this aircraft just departed
            prompt += ("\n\nIMPORTANTE: esta aeronave DECOLOU de {d} e esta saindo para {t}. Voce e o controle de "
                       "SAIDA: subida, rota, proa e transferencia para o Centro. NUNCA a vetore nem a desca para "
                       "aproximacao ou pouso em {d}.".format(d=session.airport_icao, t=fp.destination)
                       if self.language == "pt-br" else
                       "\n\nIMPORTANT: this aircraft DEPARTED {d} and is outbound to {t}. You are DEPARTURE control: "
                       "climb, route, heading and handoff to Center. NEVER vector or descend it for an approach or "
                       "landing at {d}.".format(d=session.airport_icao, t=fp.destination))
        return prompt

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, arguments: dict, session: ATCSession) -> str:
        logger.info(f"🔧 TOOL CALL: {name} {json.dumps(arguments)}")
        self.last_tool_calls.append(name)
        icao = (arguments.get("icao") or session.airport_icao).upper()
        try:
            if name == "get_charts":
                charts = self.charts.list_charts(icao)
                for cat in ("APP", "SID", "STAR"):
                    names = [c["name"] for c in charts if c["category"] == cat]
                    if names:
                        session.remember_fact(f"{cat} procedures {icao}", "; ".join(names[:25]))
                return json.dumps([{"id": c["id"], "name": c["name"], "category": c["category"],
                                    "runways": c["rwys"]} for c in charts])

            if name == "get_chart_by_category":
                charts = self.charts.get_charts_by_category(icao, arguments["category"])
                if charts:
                    session.remember_fact(f"{arguments['category']} charts {icao}",
                                          "; ".join(f"{c['name']} [id {c['id']}]" for c in charts[:25]))
                return json.dumps([{"id": c["id"], "name": c["name"], "runways": c["rwys"]} for c in charts])

            if name == "read_chart":
                # A 200 KB PDF attached to the radio reply took 55 s live (2026-11-07, RNP Z RWY 10 SBCB).
                # Read it once into a text digest (cached on disk, prefetched for the destination).
                chart_name, digest = self.chart_digest(icao, arguments["chart_id"])
                session.remember_fact(f"chart {chart_name} {icao}", digest[:900])
                return json.dumps({"chart_name": chart_name, "digest": digest})

            if name == "get_weather":
                w, source = self._weather_with_fallback(icao)
                if w is None:
                    return json.dumps({"error": f"no METAR available for {icao} or nearby",
                                       "advice": "assume calm wind, choose the runway from the flight plan or ask the pilot"})
                metar = w.get("metar") or {}
                self._remember_weather(session, icao, w, source)
                return json.dumps({
                    "metar_station": source,
                    "note": None if source == icao else f"{icao} has no METAR; this is the nearest station ({source})",
                    "metar_raw": metar.get("raw", "N/A"),
                    "taf_raw": (w.get("taf") or {}).get("raw", "N/A"),
                    "flight_rules": metar.get("flight_rules", "N/A"),
                    "wind": f"{(metar.get('wind_direction') or {}).get('value', '?')}@"
                            f"{(metar.get('wind_speed') or {}).get('value', '?')}kt",
                    "altimeter": (metar.get("altimeter") or {}).get("repr"),
                    "active_runways": w.get("active_runways", []),
                })

            if name == "get_notams":
                return json.dumps(self.wx.get_notams(icao))

            if name == "get_airport_info":
                svc = self._airport_data
                if svc is None:
                    from airport_data_service import get_airport_data_service
                    svc = get_airport_data_service()
                ap = svc.get(icao)
                if not ap:
                    return json.dumps({"error": f"unknown airport {icao}"})
                info = ap.to_dict()
                freqs = session.airport_frequencies if icao == session.airport_icao else \
                    session._freq_service.get_all_atc(icao)       # other airports too (diversions)
                info["frequencies"] = {t: f.frequency_formatted() for t, f in freqs.items()}
                session.remember_fact(f"airport {icao}", ap.summary_line())
                return json.dumps(info)

            return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as e:  # noqa: BLE001
            logger.error(f"   tool {name} failed: {e}")
            return json.dumps({"error": str(e)})

    def _weather_with_fallback(self, icao: str) -> tuple[Optional[dict], str]:
        """
        METAR for icao: aviationweather.gov first, airportwxbot second; else the nearest
        airport (within 40 nm) that reports one - fetched in ONE request, so a field without a
        METAR costs one round-trip instead of five slow failures.
        """
        svc = self._airport_data
        if svc is None:
            from airport_data_service import get_airport_data_service
            svc = get_airport_data_service()
        ap = svc.get(icao)
        try:
            w = self.awc.get_metar_taf(icao, ap)
            if w:
                return w, icao
        except Exception as e:  # noqa: BLE001
            logger.warning(f"   aviationweather.gov failed for {icao} ({e}); trying airportwxbot")
            try:
                return self.wx.get_metar_taf(icao), icao
            except Exception as e2:  # noqa: BLE001
                logger.warning(f"   airportwxbot failed for {icao} ({e2})")
        if not ap:
            return None, icao
        # OurAirports idents like "BR-3282" are not METAR stations and made aviationweather.gov 400 the lot
        near = [n for n, _d in svc.nearest(ap.latitude, ap.longitude, max_nm=40, min_runway_ft=3000)[:8]
                if n.icao != icao and re.fullmatch(r"[A-Z]{4}", n.icao)]
        try:
            found = self.awc.metars([n.icao for n in near])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"   nearby METAR lookup failed ({e})")
            return None, icao
        for n in near:                       # nearest first
            if n.icao in found:
                w = self.awc.get_metar_taf(n.icao, n)
                if w:
                    # runways in use are this airport's, from the neighbour's wind
                    wind = w["metar"]["wind_direction"]["value"], w["metar"]["wind_speed"]["value"]
                    w["active_runways"] = active_runways(wind[0], wind[1], ap.runways)
                    return w, n.icao
        return None, icao

    @staticmethod
    def _remember_weather(session: ATCSession, icao: str, w: dict, source: str):
        metar = w.get("metar") or {}
        wind = (f"{(metar.get('wind_direction') or {}).get('value', '?')}@"
                f"{(metar.get('wind_speed') or {}).get('value', '?')}kt")
        session.remember_fact(f"weather {icao}",
                              f"METAR {metar.get('raw', 'N/A')} | wind {wind} | {metar.get('flight_rules', '?')} "
                              f"| altimeter {(metar.get('altimeter') or {}).get('repr')} "
                              f"| runways in use (into wind) {w.get('active_runways') or 'any (calm/variable)'}"
                              + (f" (nearest station {source})" if source != icao else ""))

    def _prefetch_weather(self, session: ATCSession):
        """Put the METAR in the prompt before the first model call (saves a tool round-trip)."""
        if session.fact(f"weather {session.airport_icao}", max_age_s=1800):
            return
        try:
            w, source = self._weather_with_fallback(session.airport_icao)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"   weather prefetch failed: {e}")
            return
        if w:
            self._remember_weather(session, session.airport_icao, w, source)
        else:
            session.remember_fact(f"weather {session.airport_icao}",
                                  "no METAR here or nearby: assume calm wind, use the runway the pilot requests "
                                  "or the longest runway")

    _DIGEST_PROMPT = (
        "You are preparing a radar controller's quick reference from this aeronautical chart. Plain text, "
        "compact, only what is printed: procedure name and runway; IAF/IF/FAF names with the minimum or "
        "crossing altitude at each; transitions; final approach course; DA/MDA; missed approach procedure "
        "(route and altitude); MSA; transition altitude/level; frequencies printed on the chart.")

    def chart_digest(self, icao: str, chart_id: str) -> tuple[str, str]:
        """(chart name, text digest) for a chart PDF - one model call per chart, then from disk."""
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{icao}_{chart_id}")
        path = CHART_DIGEST_DIR / f"{safe}.txt"
        try:
            name, _, text = path.read_text(encoding="utf-8").partition("\n")
            if text.strip():
                return name, text.strip()
        except OSError:
            pass
        chart, pdf = self.charts.fetch_pdf(icao, chart_id)
        if len(pdf) > MAX_PDF_BYTES:
            raise ValueError(f"chart too large ({len(pdf)} bytes)")
        t0 = time.time()
        b64 = base64.b64encode(pdf).decode()
        try:
            import config as cfg
            timeout = cfg.CHART_DIGEST_TIMEOUT_S
        except ImportError:  # pragma: no cover
            timeout = 120.0
        client = self.client.with_options(timeout=timeout) if hasattr(self.client, "with_options") else self.client
        kwargs = {"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else {}
        resp = client.responses.create(model=self.model, input=[{"role": "user", "content": [
            {"type": "input_file", "filename": f"{chart['name']}.pdf", "file_data": f"data:application/pdf;base64,{b64}"},
            {"type": "input_text", "text": self._DIGEST_PROMPT}]}], **kwargs)
        text = (resp.output_text or "").strip()
        if not text:
            raise ValueError("empty chart digest")
        try:
            CHART_DIGEST_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{chart['name']}\n{text}", encoding="utf-8")
        except OSError:
            pass
        logger.info(f"   📄 chart digest {icao} {chart['name']} ({len(pdf)//1024} KB -> {len(text)} chars, "
                    f"{time.time() - t0:.1f}s)")
        return chart["name"], text

    def prefetch_chart_digests(self, icao: str, category: str = "APP", limit: int = 8):
        """Background: digest a destination's approach charts before the pilot asks for one."""
        try:
            charts = self.charts.get_charts_by_category(icao, category)[:limit]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"chart prefetch {icao}: {e}")
            return
        for c in charts:
            try:
                self.chart_digest(icao, c["id"])
            except Exception as e:  # noqa: BLE001
                logger.warning(f"chart prefetch {icao} {c.get('name')}: {e}")

    def _chart_message(self) -> Optional[dict]:
        if not self._pending_chart:
            return None
        chart = self._pending_chart
        self._pending_chart = None
        b64 = base64.b64encode(chart["pdf"]).decode()
        return {
            "role": "user",
            "content": [
                {"type": "input_file", "filename": f"{chart['name']}.pdf",
                 "file_data": f"data:application/pdf;base64,{b64}"},
                {"type": "input_text", "text": f"Attached: the '{chart['name']}' chart. Use it for exact "
                                               f"fixes, altitudes, courses, minima and frequencies."},
            ],
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def _create(self, **kwargs):
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return self.client.responses.create(model=self.model, **kwargs)

    def generate_response(self, pilot_transmission: str, session: ATCSession) -> str:
        """Generate the controller's reply to one pilot transmission (records both in the session)."""
        t0 = time.time()
        self.last_tool_calls = []
        logger.info(f"📻 PILOT ({session.airport_icao}/{session.current_frequency}): {pilot_transmission}")

        readback = session.handoff_readback_kind(pilot_transmission)
        if readback == "short":
            # frequency-change readback on the old frequency: nobody answers it
            session.add_pilot_transmission(pilot_transmission)
            self.last_latency_s = time.time() - t0
            self.last_usage = {}
            logger.info("   (handoff readback on the old frequency - no reply)")
            return ""
        # a full readback is still checked by the old controller; otherwise follow the pilot
        switched = None if readback else (session.follow_addressed_station(pilot_transmission)
                                          or session.apply_pending_handoff())
        if switched:
            logger.info(f"   ↪ pilot moved to {session.current_frequency} ({switched.value}) after handoff")
        session.add_pilot_transmission(pilot_transmission)
        self._prefetch_weather(session)
        instructions = self._build_system_prompt(session)
        hint = ""
        if session.language == "pt-br" and pilot_speaks_english(pilot_transmission):
            hint = "\n[The pilot is speaking ENGLISH (foreign crew): reply in English, ICAO phraseology.]"
        input_messages: list = [{"role": "user", "content": [
            {"type": "input_text", "text": f"Pilot transmission: {pilot_transmission}{hint}"}]}]

        response = self._create(instructions=instructions, input=input_messages, tools=TOOLS)

        rounds = 0
        while self._has_tool_calls(response) and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            tool_outputs = []
            for item in response.output:
                if item.type == "function_call":
                    try:
                        args = json.loads(item.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = self._execute_tool(item.name, args, session)
                    tool_outputs.append({"type": "function_call_output", "call_id": item.call_id,
                                         "output": result})
            input_messages = list(response.output) + tool_outputs
            chart_msg = self._chart_message()
            if chart_msg:
                input_messages.append(chart_msg)
            response = self._create(instructions=instructions, input=input_messages, tools=TOOLS)

        atc_response = (response.output_text or "").strip()
        if not atc_response:
            # Model ended on a tool call we refused to loop further on: every function_call in the
            # input MUST get an output or the API rejects the request, so answer them with an error.
            input_messages = list(response.output) + self._exhausted_outputs(response) + [
                {"role": "user", "content": [{"type": "input_text",
                                              "text": "Tool budget exhausted. Reply now with your transmission only."}]}]
            response = self._create(instructions=instructions, input=input_messages)
            atc_response = (response.output_text or "").strip()

        atc_response = strip_leaked_token(atc_response)
        if atc_response and not atc_response.upper().startswith("NONE") and not looks_like_radio(atc_response):
            # the model occasionally emits junk (seen live: ">tagger"); never put that on the radio
            logger.warning(f"   malformed controller output {atc_response!r}; retrying once")
            response = self._create(instructions=instructions, input=[
                {"role": "user", "content": [{"type": "input_text", "text": f"Pilot transmission: {pilot_transmission}\n"
                                              "Reply with your radio transmission only (plain words)."}]}])
            atc_response = (response.output_text or "").strip()
            if not looks_like_radio(atc_response):
                atc_response = f"{session.callsign}, say again." if session.language != "pt-br" \
                    else f"{session.callsign}, repita."
        if atc_response.upper().startswith("NONE"):
            atc_response = ""        # nothing to say (e.g. self-announcements at an uncontrolled field)
        if atc_response:
            atc_response = self._enforce_position(session, atc_response, instructions,
                                                  f"Pilot transmission: {pilot_transmission}")
        self.last_latency_s = time.time() - t0
        try:
            self.last_usage = {"input_tokens": response.usage.input_tokens,
                               "output_tokens": response.usage.output_tokens}
        except Exception:  # noqa: BLE001
            self.last_usage = {}

        logger.info(f"🎙️ ATC ({self.last_latency_s:.1f}s, tools={self.last_tool_calls}): {atc_response}")
        if atc_response:
            session.add_atc_transmission(atc_response)
            self._extract_clearances(atc_response, session)
            if is_standby(atc_response):
                session.standby_request, session.standby_at = pilot_transmission, datetime.now()
            else:
                session.standby_request = session.standby_at = None
        return atc_response

    INITIATIVE_EN = ("No pilot transmission. Time has passed and the aircraft state above is current ({reason}). "
                     "A controller who is working this aircraft would now either transmit (vectors, descent or climb, "
                     "approach clearance with intercept altitude, sequencing, circuit instruction, landing or takeoff "
                     "clearance, a frequency change, or the answer to a pilot request you told to stand by) or stay quiet. If you would transmit, reply with the transmission "
                     "only. If nothing is needed yet, reply exactly: NONE")
    INITIATIVE_PT = ("Sem transmissao do piloto. O tempo passou e o estado da aeronave acima esta atualizado ({reason}). "
                     "Um controlador trabalhando esta aeronave agora transmitiria (vetores, descida ou subida, autorizacao "
                     "de aproximacao com altitude de interceptacao, sequenciamento, instrucao de circuito, autorizacao de "
                     "pouso ou decolagem, troca de frequencia, ou a resposta a uma solicitacao do piloto que voce mandou aguardar) ou ficaria em silencio. Se voce transmitiria, responda "
                     "apenas com a transmissao. Se nada e necessario ainda, responda exatamente: NONE")

    def controller_initiative(self, session: ATCSession, reason: str = "routine scan") -> Optional[str]:
        """
        Ask the controller whether it wants to say something unprompted. Returns the
        transmission (recorded on the session, clearances parsed) or None.
        """
        t0 = time.time()
        self.last_tool_calls = []
        self._prefetch_weather(session)
        instructions = self._build_system_prompt(session)
        prompt = (self.INITIATIVE_PT if self.language == "pt-br" else self.INITIATIVE_EN).format(reason=reason)
        input_messages: list = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
        response = self._create(instructions=instructions, input=input_messages, tools=TOOLS)
        rounds = 0
        while self._has_tool_calls(response) and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            outputs = []
            for item in response.output:
                if item.type == "function_call":
                    try:
                        args = json.loads(item.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    outputs.append({"type": "function_call_output", "call_id": item.call_id,
                                    "output": self._execute_tool(item.name, args, session)})
            input_messages = list(response.output) + outputs
            chart_msg = self._chart_message()
            if chart_msg:
                input_messages.append(chart_msg)
            response = self._create(instructions=instructions, input=input_messages, tools=TOOLS)
        text = strip_leaked_token((response.output_text or "").strip())
        if not text and self._has_tool_calls(response):
            input_messages = list(response.output) + self._exhausted_outputs(response) + [
                {"role": "user", "content": [{"type": "input_text",
                                              "text": "Tool budget exhausted. Reply with your transmission or NONE."}]}]
            response = self._create(instructions=instructions, input=input_messages)
            text = (response.output_text or "").strip()
        self.last_latency_s = time.time() - t0
        if text and not text.upper().startswith("NONE") and not looks_like_radio(text):
            logger.warning(f"   malformed initiative output {text!r}; staying quiet")
            text = ""
        if text and not text.upper().startswith("NONE"):
            text = self._enforce_position(session, text, instructions, prompt)
            self.last_latency_s = time.time() - t0
        if not text or text.upper().startswith("NONE"):
            logger.info(f"🤫 initiative ({self.last_latency_s:.1f}s): none")
            return None
        logger.info(f"📣 ATC initiative ({self.last_latency_s:.1f}s, tools={self.last_tool_calls}): {text}")
        session.add_atc_transmission(text)
        self._extract_clearances(text, session)
        if not is_standby(text):
            session.standby_request = session.standby_at = None
        return text

    def _enforce_position(self, session: ATCSession, text: str, instructions: str, prompt_text: str) -> str:
        """
        Deterministic position check (position_rules): a reply with a clearance this position may not
        issue is regenerated once with the reason; if the model still gets it wrong, the aircraft is
        handed to the position that owns the clearance instead.
        """
        violations = position_rules.check(session, text)
        if not violations:
            return text
        self.last_tool_calls.append("position_guard")
        logger.warning(f"   position guard: {[v.reason for v in violations]} in {text!r}")
        try:
            response = self._create(instructions=instructions, input=[{"role": "user", "content": [
                {"type": "input_text", "text": f"{prompt_text}\n\nYour draft reply: {text}\n"
                                              f"{position_rules.correction_note(violations)}\n"
                                              "Reply with the corrected transmission only."}]}])
            fixed = strip_leaked_token((response.output_text or "").strip())
        except Exception as e:  # noqa: BLE001
            logger.error(f"   position guard retry failed: {e}")
            fixed = ""
        if fixed and looks_like_radio(fixed) and not position_rules.check(session, fixed):
            logger.info(f"   position guard fixed: {fixed}")
            return fixed
        fb = position_rules.fallback(session, violations)
        if fb:
            logger.info(f"   position guard fallback: {fb}")
            return fb
        return fixed if fixed and looks_like_radio(fixed) else text

    @staticmethod
    def _exhausted_outputs(response) -> list:
        return [{"type": "function_call_output", "call_id": item.call_id,
                 "output": json.dumps({"error": "tool budget exhausted for this transmission"})}
                for item in response.output if item.type == "function_call"]

    @staticmethod
    def _has_tool_calls(response) -> bool:
        return any(item.type == "function_call" for item in response.output)

    # ------------------------------------------------------------------
    # Clearance extraction
    # ------------------------------------------------------------------

    def _extract_clearances(self, response: str, session: ATCSession):
        for pc in parse_clearances(response, session.callsign):
            session.add_clearance(pc.type, response, value=pc.value, runway=pc.runway,
                                  position=pc.position, frequency_khz=pc.frequency_khz)
            logger.info(f"   📋 clearance: {pc}")


# -----------------------------------------------------------------------------
# CLI usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    from airport_data_service import AirportDataService
    from session_manager import ATCPhase, FlightPlan

    airport = sys.argv[1] if len(sys.argv) > 1 else "KJFK"
    session = ATCSession(airport_icao=airport, callsign="N12345", airport_data_service=AirportDataService(),
                         flight_plan=FlightPlan(aircraft_type="C172", flight_rules="IFR", destination="KBOS"))
    session.update_aircraft_state(latitude=40.75, longitude=-73.95, altitude_ft=3000, heading=270,
                                  indicated_airspeed=180, groundspeed=190, vertical_speed=0, squawk=1200,
                                  on_ground=False)
    session.set_phase(ATCPhase.APPROACH)

    service = ATCLLMService()
    pilot_msg = sys.argv[2] if len(sys.argv) > 2 else \
        f"{airport} Approach, November one two three four five, 15 miles northwest, three thousand, request ILS approach"
    print(f"\nPilot: {pilot_msg}\n")
    print(f"ATC: {service.generate_response(pilot_msg, session)}")
    print(f"\nclearances: {[c.to_dict() for c in session.clearances]}")
