"""
Chart Providers - official, publicly downloadable aeronautical charts.

airportwxbot.com still serves chart *metadata* but its PDF endpoint sits behind a
Cloudflare browser challenge (HTTP 403 for any non-browser client, including
OpenAI's file fetcher). This module replaces it with official sources:

- AISWEB (DECEA, Brazil)  - scrapes the public aerodrome page; PDFs are public.
- FAA d-TPP (United States) - parses the public d-TPP metafile XML; PDFs are public.

Both expose the same normalized shape so the LLM tools don't care which one answered:

    {
      "id":       provider-scoped stable id,
      "name":     "ILS Z RWY 28L",
      "category": "AD" | "APP" | "SID" | "STAR" | "MISC",
      "rwys":     ["28L"],          # best-effort, parsed from the name
      "url":      direct PDF url,
      "provider": "aisweb" | "faa",
    }

PDF bytes are downloaded server-side (fetch_pdf) and shipped to the LLM as base64,
so the model provider never has to reach these hosts itself.

Listings are cached in-memory and on disk (.cache/charts/) for 7 days.
"""

import html
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("Charts")

CACHE_DIR = Path(__file__).parent / ".cache" / "charts"
LIST_CACHE_HOURS = 24 * 7
PDF_CACHE_HOURS = 24 * 28

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) VirtualATC/1.0"

Category = str  # "AD" | "APP" | "SID" | "STAR" | "MISC"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RWY_RE = re.compile(r"\b(?:RWY|RWYS|RUNWAY)\s*([0-3]?\d[LRC]?(?:\s*[/&,]\s*[0-3]?\d[LRC]?)*)", re.I)
_RWY_TOKEN = re.compile(r"[0-3]?\d[LRC]?")


def parse_runways(name: str) -> list[str]:
    """Extract runway idents from a chart name like 'ILS Z OR LOC Z RWY 28L' or 'RNAV CGO 4B RWY 28L/28R'."""
    m = _RWY_RE.search(name)
    if not m:
        return []
    return [t.upper() for t in _RWY_TOKEN.findall(m.group(1))]


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _cache_fresh(path: Path, hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) / 3600 < hours


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class ChartProvider:
    name = "base"

    def supports(self, icao: str) -> bool:
        raise NotImplementedError

    def list_charts(self, icao: str) -> list[dict]:
        raise NotImplementedError

    def fetch_pdf(self, chart: dict) -> bytes:
        cache = _cache_path(f"{chart['provider']}_{chart['id']}.pdf")
        if _cache_fresh(cache, PDF_CACHE_HOURS):
            return cache.read_bytes()
        resp = _session().get(chart["url"], timeout=60, allow_redirects=True)
        resp.raise_for_status()
        data = resp.content
        if not data.startswith(b"%PDF"):
            raise ValueError(f"Not a PDF ({resp.headers.get('content-type')}) from {chart['url']}")
        cache.write_bytes(data)
        return data


# ---------------------------------------------------------------------------
# AISWEB (Brazil)
# ---------------------------------------------------------------------------

class AISWebChartProvider(ChartProvider):
    """
    Scrapes https://aisweb.decea.mil.br/?i=aerodromos&codigo=SBGR

    The page has a "Cartas (N)" section with <h4>CATEGORY</h4><ul>...<a href=PDF>NAME</a>
    blocks. Category codes used by DECEA:
        ADC  aerodrome chart            -> AD
        PDC  parking/docking chart      -> AD
        AGMC ground movement chart      -> AD
        AOC  obstacle chart             -> MISC
        VAC  visual approach chart      -> APP
        IAC  instrument approach chart  -> APP
        SID / STAR                      -> SID / STAR
    """

    name = "aisweb"
    PAGE_URL = "https://aisweb.decea.mil.br/?i=aerodromos&codigo={icao}"

    CATEGORY_MAP = {
        "ADC": "AD", "PDC": "AD", "AGMC": "AD",
        "IAC": "APP", "VAC": "APP",
        "SID": "SID", "STAR": "STAR",
        "AOC": "MISC",
    }

    _BLOCK_RE = re.compile(r"<h4>\s*([A-Z ]+?)\s*</h4>\s*<ul[^>]*>(.*?)</ul>", re.S)
    _LINK_RE = re.compile(r'href="(https://aisweb\.decea\.(?:gov|mil)\.br/download/\?arquivo=([^"&]+)[^"]*)"[^>]*>([^<]+)</a>')

    def supports(self, icao: str) -> bool:
        return icao.upper().startswith("SB") or icao.upper().startswith("SD") \
            or icao.upper().startswith("SN") or icao.upper().startswith("SS") \
            or icao.upper().startswith("SW") or icao.upper().startswith("SJ")

    def list_charts(self, icao: str) -> list[dict]:
        icao = icao.upper()
        cache = _cache_path(f"aisweb_{icao}.json")
        if _cache_fresh(cache, LIST_CACHE_HOURS):
            return json.loads(cache.read_text())

        resp = _session().get(self.PAGE_URL.format(icao=icao), timeout=60)
        resp.raise_for_status()
        charts = self.parse_page(resp.text, icao)
        cache.write_text(json.dumps(charts, ensure_ascii=False))
        return charts

    @classmethod
    def parse_page(cls, page_html: str, icao: str) -> list[dict]:
        idx = page_html.find("Cartas (")
        section = page_html[idx:] if idx >= 0 else page_html
        charts: list[dict] = []
        for raw_cat, block in cls._BLOCK_RE.findall(section):
            cat = raw_cat.strip().upper()
            category = cls.CATEGORY_MAP.get(cat)
            if category is None:
                continue  # skip non-chart lists further down the page
            for href, file_id, name in cls._LINK_RE.findall(block):
                name = html.unescape(name).strip()
                charts.append({
                    "id": file_id,
                    "name": name,
                    "category": category,
                    "source_category": cat,
                    "rwys": parse_runways(name),
                    "url": html.unescape(href).replace("aisweb.decea.gov.br", "aisweb.decea.mil.br"),
                    "icao": icao,
                    "provider": cls.name,
                })
        return charts


# ---------------------------------------------------------------------------
# FAA d-TPP (United States)
# ---------------------------------------------------------------------------

class FAAChartProvider(ChartProvider):
    """
    Uses https://aeronav.faa.gov/d-tpp/<cycle>/xml_data/d-TPP_Metafile.xml (~16MB).

    The current cycle is discovered from the directory index (folders named YYCC);
    the newest folder that actually has a metafile is the effective one.

    chart_code mapping: APD -> AD, IAP -> APP, DP -> SID, STR -> STAR, MIN/HOT/LAH/ODP -> MISC
    """

    name = "faa"
    INDEX_URL = "https://aeronav.faa.gov/d-tpp/"
    META_URL = "https://aeronav.faa.gov/d-tpp/{cycle}/xml_data/d-TPP_Metafile.xml"
    PDF_URL = "https://aeronav.faa.gov/d-tpp/{cycle}/{pdf}"

    CATEGORY_MAP = {"APD": "AD", "IAP": "APP", "DP": "SID", "STR": "STAR"}

    def __init__(self):
        self._cycle: Optional[str] = None
        self._index: Optional[dict[str, list[dict]]] = None  # icao -> charts

    def supports(self, icao: str) -> bool:
        icao = icao.upper()
        return icao.startswith("K") or icao.startswith("PA") or icao.startswith("PH") or icao.startswith("TJ")

    # -- cycle discovery ---------------------------------------------------

    def current_cycle(self) -> str:
        if self._cycle:
            return self._cycle
        cache = _cache_path("faa_cycle.txt")
        if _cache_fresh(cache, 24):
            self._cycle = cache.read_text().strip()
            return self._cycle
        s = _session()
        resp = s.get(self.INDEX_URL, timeout=60)
        resp.raise_for_status()
        cycles = sorted(set(re.findall(r'href="/d-tpp/(\d{4})/"', resp.text, re.I)), reverse=True)
        for cyc in cycles:
            head = s.head(self.META_URL.format(cycle=cyc), timeout=30)
            if head.status_code == 200:
                self._cycle = cyc
                cache.write_text(cyc)
                return cyc
        raise RuntimeError("Could not determine current FAA d-TPP cycle")

    # -- metafile ----------------------------------------------------------

    def _load_index(self) -> dict[str, list[dict]]:
        if self._index is not None:
            return self._index
        cycle = self.current_cycle()
        cache = _cache_path(f"faa_index_{cycle}.json")
        if cache.exists():
            self._index = json.loads(cache.read_text())
            return self._index

        xml_cache = _cache_path(f"faa_metafile_{cycle}.xml")
        if not xml_cache.exists():
            logger.info(f"Downloading FAA d-TPP metafile for cycle {cycle} (~16MB)...")
            resp = _session().get(self.META_URL.format(cycle=cycle), timeout=180)
            resp.raise_for_status()
            xml_cache.write_bytes(resp.content)

        self._index = self.parse_metafile(xml_cache.read_bytes(), cycle)
        cache.write_text(json.dumps(self._index))
        return self._index

    @classmethod
    def parse_metafile(cls, xml_bytes: bytes, cycle: str) -> dict[str, list[dict]]:
        root = ET.fromstring(xml_bytes)
        index: dict[str, list[dict]] = {}
        for ap in root.iter("airport_name"):
            icao = ap.attrib.get("icao_ident") or ""
            if not icao:
                continue
            charts = []
            for rec in ap.iter("record"):
                code = rec.findtext("chart_code") or ""
                name = rec.findtext("chart_name") or ""
                pdf = rec.findtext("pdf_name") or ""
                if not pdf:
                    continue
                charts.append({
                    "id": pdf,
                    "name": name,
                    "category": cls.CATEGORY_MAP.get(code, "MISC"),
                    "source_category": code,
                    "rwys": parse_runways(name),
                    "url": cls.PDF_URL.format(cycle=cycle, pdf=pdf),
                    "icao": icao,
                    "provider": cls.name,
                })
            if charts:
                index[icao] = charts
        return index

    def list_charts(self, icao: str) -> list[dict]:
        return list(self._load_index().get(icao.upper(), []))


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class ChartService:
    """Routes an ICAO to the right provider and offers the API the LLM tools use."""

    def __init__(self, providers: Optional[list[ChartProvider]] = None):
        self.providers = providers or [AISWebChartProvider(), FAAChartProvider()]

    def provider_for(self, icao: str) -> Optional[ChartProvider]:
        for p in self.providers:
            if p.supports(icao):
                return p
        return None

    def list_charts(self, icao: str) -> list[dict]:
        p = self.provider_for(icao)
        if not p:
            logger.warning(f"No chart provider for {icao}")
            return []
        try:
            return p.list_charts(icao)
        except Exception as e:
            logger.error(f"Chart listing failed for {icao} via {p.name}: {e}")
            return []

    def get_charts_by_category(self, icao: str, category: str) -> list[dict]:
        category = category.upper()
        return [c for c in self.list_charts(icao) if c["category"] == category]

    def find_chart(self, icao: str, chart_id: str) -> Optional[dict]:
        for c in self.list_charts(icao):
            if c["id"] == chart_id:
                return c
        return None

    def fetch_pdf(self, icao: str, chart_id: str) -> tuple[dict, bytes]:
        chart = self.find_chart(icao, chart_id)
        if not chart:
            raise KeyError(f"Chart {chart_id} not found for {icao}")
        p = self.provider_for(icao)
        return chart, p.fetch_pdf(chart)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    icao = sys.argv[1].upper() if len(sys.argv) > 1 else "SBGR"
    svc = ChartService()
    charts = svc.list_charts(icao)
    print(f"{icao}: {len(charts)} charts via {svc.provider_for(icao).name if svc.provider_for(icao) else 'none'}")
    from collections import Counter
    print(Counter(c["category"] for c in charts))
    for c in charts[:12]:
        print(f"  [{c['category']:4s}] {c['name']:45s} rwys={c['rwys']}")
    if len(sys.argv) > 2:
        chart, pdf = svc.fetch_pdf(icao, charts[int(sys.argv[2])]["id"])
        print(f"downloaded {chart['name']}: {len(pdf)} bytes")
