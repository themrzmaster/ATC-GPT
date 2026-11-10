"""
Clearance Parser - extract clearances and assigned values from ATC transmissions.

Works on plain text in English (ICAO) and Brazilian Portuguese (DECEA), with numbers
spoken as digits ("three thousand", "dois oito direita") or written ("3000", "28R").

Design notes
- The callsign is removed first: the old regex grabbed "12345" from "N12345" as an
  altitude. Never again.
- Spoken-number groups are converted to digits ("two seven zero" -> 270,
  "um zero direita" -> "10R", "flight level three five zero" -> 35000,
  "dois mil e quinhentos" -> 2500).
- Output is a list of ParsedClearance(type, value, runway, snippet).

This is deliberately a heuristic layer. The LLM service also asks the model for a
structured summary of what it cleared; this parser is the always-on safety net and the
thing unit tests can pin down deterministically.
"""

import re
from dataclasses import dataclass
from typing import Optional

from session_manager import ClearanceType

# ---------------------------------------------------------------------------
# Spoken numbers
# ---------------------------------------------------------------------------

DIGIT_WORDS = {
    # English
    "zero": 0, "one": 1, "two": 2, "three": 3, "tree": 3, "four": 4, "fower": 4,
    "five": 5, "fife": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "niner": 9,
    # Portuguese
    "um": 1, "uno": 1, "dois": 2, "duas": 2, "tres": 3, "três": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "meia": 6, "sete": 7, "oito": 8, "nove": 9,
}
THOUSAND_WORDS = {"thousand": 1000, "mil": 1000}
HUNDRED_WORDS = {"hundred": 100, "cem": 100, "cento": 100}
TENS_PT = {"dez": 10, "vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50,
           "sessenta": 60, "setenta": 70, "oitenta": 80, "noventa": 90,
           "onze": 11, "doze": 12, "treze": 13, "catorze": 14, "quatorze": 14, "quinze": 15,
           "dezesseis": 16, "dezessete": 17, "dezoito": 18, "dezenove": 19,
           "duzentos": 200, "trezentos": 300, "quatrocentos": 400, "quinhentos": 500,
           "seiscentos": 600, "setecentos": 700, "oitocentos": 800, "novecentos": 900}
TENS_EN = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
           "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
           "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
           "ninety": 90}
SIDE_WORDS = {"left": "L", "right": "R", "center": "C", "centre": "C",
              "esquerda": "L", "direita": "R", "central": "C", "centro": "C"}
FILLER = {"and", "e", "feet", "pés", "pes", "ft", "point", "decimal", "ponto", "vírgula", "virgula"}

_WORD = re.compile(r"[A-Za-zÀ-ÿ]+|\d+(?:[.,]\d+)?|[LRC](?![A-Za-z])", re.U)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def spoken_to_number(words: list[str]) -> Optional[int]:
    """
    Convert a run of number words to an integer.
    Handles digit-by-digit ("two seven zero"), thousands/hundreds
    ("three thousand five hundred", "dois mil e quinhentos") and literal digits.
    """
    if not words:
        return None
    total = 0
    current = 0
    digits_run = ""
    saw_scale = False
    for w in words:
        lw = w.lower()
        if lw.isdigit():
            digits_run += lw
            current = current * (10 ** len(lw)) + int(lw) if current else int(lw)
            continue
        if lw in DIGIT_WORDS:
            digits_run += str(DIGIT_WORDS[lw])
            current = current * 10 + DIGIT_WORDS[lw]
            continue
        if lw in TENS_EN or lw in TENS_PT:
            v = TENS_EN.get(lw, TENS_PT.get(lw))
            current += v
            digits_run = ""  # no longer a pure digit run
            continue
        if lw in HUNDRED_WORDS:
            current = (current or 1) * 100
            saw_scale = True
            digits_run = ""
            continue
        if lw in THOUSAND_WORDS:
            total += (current or 1) * 1000
            current = 0
            saw_scale = True
            digits_run = ""
            continue
        if lw in FILLER:
            continue
        return None  # unexpected word: not a number group
    total += current
    if not saw_scale and digits_run:
        return int(digits_run)
    return total


def _number_after(tokens: list[str], start: int, max_words: int = 8) -> tuple[Optional[int], int]:
    """Parse the longest number-word run starting at tokens[start]. Returns (value, next_index)."""
    run = []
    i = start
    while i < len(tokens) and len(run) < max_words:
        lw = tokens[i].lower()
        if lw.isdigit() or lw in DIGIT_WORDS or lw in TENS_EN or lw in TENS_PT \
                or lw in HUNDRED_WORDS or lw in THOUSAND_WORDS or lw in FILLER:
            if lw in FILLER and not run:
                i += 1
                continue
            run.append(tokens[i])
            i += 1
        else:
            break
    while run and run[-1].lower() in FILLER:
        run.pop()
    return spoken_to_number(run), i


def _runway_after(tokens: list[str], start: int) -> Optional[str]:
    """Parse a runway ident: 'two eight right', '28R', '28 R', 'um zero esquerda'."""
    i = start
    while i < len(tokens) and tokens[i].lower() in FILLER:
        i += 1
    if i >= len(tokens):
        return None
    tok = tokens[i]
    m = re.fullmatch(r"(\d{1,2})([LRC])?", tok, re.I)
    if m:
        num = int(m.group(1))
        side = (m.group(2) or "").upper()
        if not side and i + 1 < len(tokens):
            side = SIDE_WORDS.get(tokens[i + 1].lower(), "")
            if not side and re.fullmatch(r"[LRC]", tokens[i + 1], re.I):
                side = tokens[i + 1].upper()
        return f"{num:02d}{side}" if 1 <= num <= 36 else None
    digits = ""
    while i < len(tokens) and tokens[i].lower() in DIGIT_WORDS and len(digits) < 2:
        digits += str(DIGIT_WORDS[tokens[i].lower()])
        i += 1
    if not digits:
        return None
    side = ""
    if i < len(tokens):
        side = SIDE_WORDS.get(tokens[i].lower(), "")
    num = int(digits)
    if not 1 <= num <= 36:
        return None
    return f"{num:02d}{side}"


# ---------------------------------------------------------------------------
# Callsign stripping
# ---------------------------------------------------------------------------

_CALLSIGN_RE = re.compile(r"\b[A-Z]{1,3}-?[A-Z0-9]{2,5}\b")


def strip_callsign(text: str, callsign: Optional[str]) -> str:
    """Remove the callsign (written or spoken) so its digits never look like a clearance value."""
    if not callsign:
        return text
    out = text.replace(callsign, " ")
    out = out.replace(callsign.replace("-", ""), " ")
    # Spoken form: "November one two three four five" -> remove the prefix word + digit words
    spoken = _spoken_callsign_regex(callsign)
    if spoken:
        out = spoken.sub(" ", out)
    return out


_NATO = {"A": "alpha|alfa", "B": "bravo", "C": "charlie", "D": "delta", "E": "echo", "F": "foxtrot|fox",
         "G": "golf", "H": "hotel", "I": "india", "J": "juliett?", "K": "kilo", "L": "lima", "M": "mike",
         "N": "november", "O": "oscar", "P": "papa", "Q": "quebec", "R": "romeo", "S": "sierra",
         "T": "tango", "U": "uniform", "V": "victor", "W": "whiskey", "X": "x-?ray", "Y": "yankee",
         "Z": "zulu"}
_DIGIT_SPOKEN = {"0": "zero", "1": "one|um|uno|wun", "2": "two|dois|duas", "3": "three|tree|tres|três",
                 "4": "four|fower|quatro", "5": "five|fife|cinco", "6": "six|seis|meia", "7": "seven|sete",
                 "8": "eight|oito", "9": "nine|niner|nove"}


def _spoken_callsign_regex(callsign: str) -> Optional[re.Pattern]:
    parts = []
    for ch in callsign.upper().replace("-", ""):
        if ch in _NATO:
            parts.append(f"(?:{_NATO[ch]}|{ch})")
        elif ch in _DIGIT_SPOKEN:
            parts.append(f"(?:{_DIGIT_SPOKEN[ch]}|{ch})")
        else:
            return None
    if not parts:
        return None
    return re.compile(r"\b" + r"[\s,.\-]*".join(parts) + r"\b", re.I)   # STT punctuates: "cinco, X-Ray"


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

@dataclass
class ParsedClearance:
    type: ClearanceType
    value: Optional[int] = None
    runway: Optional[str] = None
    snippet: str = ""
    position: Optional[str] = None   # for FREQUENCY: target position ("tower", "approach", ...)
    frequency_khz: Optional[int] = None  # for FREQUENCY: 132750 for "one three two decimal seven five"

    def __repr__(self):
        extra = f" pos={self.position} freq={self.frequency_khz}" if self.type == ClearanceType.FREQUENCY else ""
        return f"<{self.type.value} value={self.value} rwy={self.runway}{extra}>"


POSITION_WORDS = {
    "delivery": "clearance", "clearance": "clearance", "tráfego": "clearance", "trafego": "clearance",
    "ground": "ground", "solo": "ground",
    "tower": "tower", "torre": "tower",
    "departure": "departure", "departures": "departure", "saída": "departure", "saida": "departure",
    "approach": "approach", "controle": "approach", "radar": "approach",
    "center": "center", "centre": "center", "centro": "center",
}


def _frequency_after(tokens: list[str], start: int, need_decimal: bool = False) -> Optional[int]:
    """Parse '132.75', '132 decimal 75', 'one three two decimal seven five (zero)' -> kHz."""
    i = start
    whole: list[str] = []
    frac: list[str] = []
    target = whole
    seen_sep = False
    while i < len(tokens) and len(whole) + len(frac) < 9:
        lw = tokens[i].lower()
        if lw in ("decimal", "point", "ponto", "vírgula", "virgula", "."):
            if seen_sep:
                break
            seen_sep, target = True, frac
        elif re.fullmatch(r"\d+[.,]\d+", lw):
            a, b = re.split(r"[.,]", lw)
            whole, frac, seen_sep = [a], [b], True
            i += 1
            break
        elif lw.isdigit():
            target.append(lw)
        elif lw in DIGIT_WORDS:
            target.append(str(DIGIT_WORDS[lw]))
        elif lw in FILLER or lw in ("on", "em", "na", "no", "frequency", "frequência", "frequencia"):
            if whole or frac:
                if lw in FILLER and lw not in ("point", "decimal", "ponto"):
                    break
            i += 1
            continue
        else:
            break
        i += 1
    w = "".join(whole)
    f = "".join(frac)
    if len(w) < 3 or (need_decimal and not (seen_sep and f)):
        return None
    mhz_str = f"{w}.{f or '0'}"
    try:
        mhz = float(mhz_str)
    except ValueError:
        return None
    if not 118.0 <= mhz <= 137.0:
        return None
    return int(round(mhz * 1000))


def frequencies_in(text: str, callsign: Optional[str] = None) -> list[int]:
    """
    Every VHF frequency (kHz) mentioned in a transmission: '121.7', 'one two one decimal seven'.
    Only numbers with their own decimal part count ("heading one two zero" is not 120.0).
    """
    tokens = _tokens(strip_callsign(text, callsign))
    found = []
    for i in range(len(tokens)):
        f = _frequency_after(tokens, i, need_decimal=True)
        if f and f not in found:
            found.append(f)
    return found


def parse_handoff(text: str) -> tuple[Optional[str], Optional[int]]:
    """Target position and frequency (kHz) of a 'contact/chame ...' instruction."""
    tokens = _tokens(text)
    lower = [t.lower() for t in tokens]
    position = None
    freq = None
    for i, lw in enumerate(lower):
        if lw in ("contact", "chame", "chamar", "contate", "monitor", "monitore", "escute"):
            for j in range(i + 1, min(i + 6, len(lower))):
                if lower[j] in POSITION_WORDS:
                    position = POSITION_WORDS[lower[j]]
                    break
            for j in range(i + 1, min(i + 12, len(lower))):
                fr = _frequency_after(tokens, j)
                if fr:
                    freq = fr
                    break
            break
    return position, freq


# "negative, runway two seven right does not exist," / "negativo, pista dois sete direita não existe,"
_DENIAL_RE = re.compile(
    r"(?:negative|negativo)?[,\s]*(?:runway|pista|rwy)\s+[^,.;]*?(?:does not exist|doesn't exist|não existe|nao existe|"
    r"is not available|não disponível|nao disponivel)[^,.;]*[,.;]?", re.I)


def _has(text: str, *phrases: str) -> bool:
    return any(p in text for p in phrases)


def _find_after(tokens: list[str], keywords: tuple[str, ...], lookahead: int = 1):
    """Yield indices right after any keyword sequence like ('descend','and','maintain')."""
    lower = [t.lower() for t in tokens]
    for kw in keywords:
        kws = kw.split()
        n = len(kws)
        for i in range(len(lower) - n + 1):
            if lower[i:i + n] == kws:
                yield i + n


def parse_clearances(text: str, callsign: Optional[str] = None) -> list[ParsedClearance]:
    """Return the clearances contained in one ATC transmission."""
    cleaned = strip_callsign(text, callsign)
    cleaned = _DENIAL_RE.sub(" ", cleaned)
    low = cleaned.lower()
    tokens = _tokens(cleaned)
    out: list[ParsedClearance] = []

    def rwy_for(*kws) -> Optional[str]:
        for idx in _find_after(tokens, kws):
            r = _runway_after(tokens, idx)
            if r:
                return r
        return None

    # --- IFR clearance -----------------------------------------------------
    route_words = ("via", "as filed", "conforme plano", "route", "rota", "direct", "direto", "sid", "departure",
                   "saída", "saida")
    if _has(low, "cleared to ") and _has(low, *route_words):
        if not _has(low, "cleared to land", "cleared to cross"):
            out.append(ParsedClearance(ClearanceType.IFR, snippet=text))
    elif re.search(r"\bautorizad[oa]\s+(?:para\s+|ao\s+|a\s+|até\s+|ate\s+)?(?!decola|pouso|pousar|toque|táxi|taxi|"
                   r"alinha|aproxima|ils|rnav|rnp|vor|ndb|visual|procedimento|push|acionamento|cruzar|a\s+decola)", low) \
            and _has(low, *route_words) and _has(low, "transponder", "squawk", "suba", "mantenha", "nível", "nivel",
                                                 "conforme plano", "saída", "saida"):
        # "autorizado SBKP via direto, suba e mantenha..." / "autorizado Santos Dumont, saída OMNI ..."
        out.append(ParsedClearance(ClearanceType.IFR, snippet=text))

    # --- Pushback ---------------------------------------------------------
    if _has(low, "pushback approved", "push back approved", "push approved", "pushback autorizado",
            "push autorizado", "acionamento e pushback", "start approved", "acionamento autorizado",
            "start-up approved", "startup approved"):
        out.append(ParsedClearance(ClearanceType.PUSHBACK, snippet=text))

    # --- Taxi -------------------------------------------------------------
    if _has(low, "taxi to", "taxi via", "taxi runway", "taxi holding point", "táxi para", "taxi para",
            "táxi via", "taxi via", "táxi pista", "taxi pista", "táxi ponto", "taxi ponto") or \
            re.search(r"\b(?:runway|pista)\s+[\w\s]{1,25}?,\s*t[aá]xi\b(?!\s+(?:request|solicit))", low):
        rw = rwy_for("holding point runway", "taxi to runway", "taxi runway", "holding point", "ponto de espera pista",
                     "táxi para pista", "taxi para pista", "táxi pista", "taxi pista", "ponto de espera",
                     "runway", "pista")
        out.append(ParsedClearance(ClearanceType.TAXI, runway=rw, snippet=text))

    # --- Line up ----------------------------------------------------------
    if _has(low, "line up and wait", "line up runway", "lineup", "alinhe e mantenha", "alinhar e manter",
            "alinhe e aguarde", "alinhe pista"):
        rw = rwy_for("runway", "pista")
        out.append(ParsedClearance(ClearanceType.LINEUP, runway=rw, snippet=text))

    # --- Takeoff ----------------------------------------------------------
    if _has(low, "cleared for takeoff", "cleared for take-off", "cleared for take off", "cleared takeoff",
            "cleared for immediate takeoff", "autorizado decolagem", "autorizada decolagem",
            "autorizado a decolagem", "autorizado decolar"):
        rw = rwy_for("takeoff runway", "take-off runway", "decolagem pista", "runway", "pista")
        out.append(ParsedClearance(ClearanceType.TAKEOFF, runway=rw, snippet=text))

    # --- Landing ----------------------------------------------------------
    if _has(low, "cleared to land", "cleared land", "cleared touch and go", "cleared for the option",
            "cleared low approach", "autorizado pouso", "autorizada pouso", "autorizado a pousar",
            "autorizado toque e arremetida", "autorizado pousar"):
        rw = rwy_for("land runway", "pouso pista", "runway", "pista")
        out.append(ParsedClearance(ClearanceType.LANDING, runway=rw, snippet=text))

    # --- Go around --------------------------------------------------------
    if _has(low, "go around", "go-around", "arremeta", "arremeter", "arremetida imediata"):
        out.append(ParsedClearance(ClearanceType.GO_AROUND, snippet=text))

    # --- Approach clearance ----------------------------------------------
    if _has(low, "cleared ils", "cleared rnav", "cleared rnp", "cleared vor", "cleared ndb", "cleared visual",
            "cleared approach", "cleared for the ils", "cleared for the rnav", "cleared for the visual",
            "cleared straight-in", "cleared straight in", "autorizado aproximação", "autorizado aproximacao",
            "autorizado ils", "autorizado rnav", "autorizado rnp", "autorizado vor", "autorizado visual",
            "autorizado procedimento") or \
            re.search(r"\baproxima[çc][ãa]o\s+(?:visual|ils|rnav|rnp|vor|ndb)?[^,.;]{0,40}?\bautorizad[ao]\b", low):
        # PT word order: "aproximação visual pista um três autorizada" (seen live at SBBH)
        rw = rwy_for("approach runway", "runway", "pista", "rwy")
        out.append(ParsedClearance(ClearanceType.APPROACH, runway=rw, snippet=text))

    # --- Altitude ---------------------------------------------------------
    alt_kws = ("climb and maintain", "descend and maintain", "climb to", "descend to", "maintain",
               "climb via", "descend via", "climb flight level", "descend flight level",
               "suba e mantenha", "desça e mantenha", "desca e mantenha", "subir e manter",
               "descer e manter", "suba para", "desça para", "desca para", "mantenha",
               "suba e mantenha nível", "nível de voo", "nivel de voo", "flight level")
    if _has(low, "climb", "descend", "maintain", "suba", "subir", "desça", "desca", "descer", "mantenha",
            "manter", "flight level", "nível", "nivel"):
        best = None
        for idx in _find_after(tokens, alt_kws):
            j = idx
            fl = False
            # "flight level three five zero" / "nível de voo três cinco zero" / "FL350"
            lower_next = [t.lower() for t in tokens[j:j + 3]]
            if lower_next[:2] == ["flight", "level"]:
                fl = True
                j += 2
            elif lower_next[:3] in (["nível", "de", "voo"], ["nivel", "de", "voo"]):
                fl = True
                j += 3
            elif j < len(tokens) and re.fullmatch(r"fl\d{2,3}", tokens[j].lower()):
                best = int(tokens[j][2:]) * 100
                break
            val, _ = _number_after(tokens, j)
            if val is None:
                continue
            if fl or (val < 600 and not _has(low, "feet", "pés", "pes", "ft")):
                # FL or bare 3-digit level -> hundreds of feet
                val = val * 100 if val < 1000 else val
            if 100 <= val <= 60000:
                best = val
                break
        # "maintain" alone is common with headings/speeds - only record when we found an altitude
        if best is not None:
            out.append(ParsedClearance(ClearanceType.ALTITUDE, value=best, snippet=text))

    # --- Heading ----------------------------------------------------------
    hdg_kws = ("heading", "fly heading", "turn left heading", "turn right heading", "proa",
               "curva à esquerda proa", "curva a esquerda proa", "curva à direita proa", "curva a direita proa")
    if _has(low, "heading", "proa"):
        for idx in _find_after(tokens, hdg_kws):
            val, _ = _number_after(tokens, idx, max_words=3)
            if val is not None and 0 <= val <= 360:
                out.append(ParsedClearance(ClearanceType.HEADING, value=val, snippet=text))
                break

    # --- Speed ------------------------------------------------------------
    spd_kws = ("reduce speed to", "increase speed to", "maintain speed", "speed", "reduza velocidade para",
               "reduza para", "aumente velocidade para", "mantenha velocidade", "velocidade")
    if _has(low, "speed", "knots", "velocidade", "nós", "nos"):
        for idx in _find_after(tokens, spd_kws):
            val, _ = _number_after(tokens, idx, max_words=4)
            if val is not None and 60 <= val <= 400:
                out.append(ParsedClearance(ClearanceType.SPEED, value=val, snippet=text))
                break

    # --- Squawk -----------------------------------------------------------
    if _has(low, "squawk", "transponder"):
        for idx in _find_after(tokens, ("squawk", "transponder", "acione transponder", "transponder código",
                                        "transponder codigo")):
            val, _ = _number_after(tokens, idx, max_words=4)
            if val is not None and 0 <= val <= 7777 and all(ch in "01234567" for ch in f"{val:04d}"):
                if not _has(low, "squawk ident", "transponder ident", "squawk vfr", "squawk standby"):
                    out.append(ParsedClearance(ClearanceType.SQUAWK, value=val, snippet=text))
                break

    # --- Frequency change -------------------------------------------------
    if _has(low, "contact ", "chame ", "chamar ", "contate ", "monitor ", "monitore ", "escute "):
        position, freq = parse_handoff(cleaned)
        out.append(ParsedClearance(ClearanceType.FREQUENCY, snippet=text, position=position, frequency_khz=freq))

    # --- Hold -------------------------------------------------------------
    if _has(low, "hold short", "hold position", "holding instructions", "hold at", "mantenha posição",
            "mantenha posicao", "hold for", "hold present position", "hold over", "hold as published",
            "hold north", "hold south", "hold east", "hold west", "órbita", "orbita", "espera sobre",
            "espera na posição", "espera na posicao", "mantenha espera"):
        out.append(ParsedClearance(ClearanceType.HOLD, snippet=text))

    return out


if __name__ == "__main__":
    tests = [
        ("N12345", "N12345, turn left heading two seven zero, descend and maintain three thousand"),
        ("N12345", "November one two three four five, climb and maintain flight level three five zero, squawk four five two one"),
        ("N12345", "November one two three four five, cleared for takeoff runway two eight right, wind zero five zero at four"),
        ("PT-ABC", "PT-ABC, autorizado decolagem pista dois oito direita, vento zero cinco zero, quatro nós"),
        ("PT-ABC", "PT-ABC, táxi para ponto de espera pista um zero esquerda via Charlie, Alfa"),
        ("PT-ABC", "PT-ABC, suba e mantenha dois mil e quinhentos pés, proa zero nove zero, transponder quatro cinco dois um"),
        ("PT-ABC", "PT-ABC, autorizado pouso pista 28L, vento 050 graus 4 nós"),
        ("N12345", "N12345, contact departure on one two seven point one, good day"),
    ]
    for cs, t in tests:
        print(parse_clearances(t, cs), "<-", t)
