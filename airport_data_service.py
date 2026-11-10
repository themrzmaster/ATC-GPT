"""
Airport Data Service - airport metadata and runways from OurAirports open data.

Provides, per ICAO:
- name, position (lat/lon), field elevation
- runways (idents, length, headings, threshold positions)
- nearest-airport lookup from a lat/lon (used to auto-tune from X-Plane position)
- great-circle distance helpers (used by trigger engine for handoffs)

Data is downloaded once and cached under .cache/ (same as frequency_service).
"""

import csv
import io
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("AirportData")

AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
RUNWAYS_URL = "https://davidmegginson.github.io/ourairports-data/runways.csv"

CACHE_DIR = Path(__file__).parent / ".cache"
AIRPORTS_CACHE = CACHE_DIR / "airports.csv"
RUNWAYS_CACHE = CACHE_DIR / "runways.csv"
CACHE_MAX_AGE_HOURS = 24 * 30  # monthly refresh is plenty

# Airport types worth tuning to (skip heliports, closed, balloonports, seaplane bases)
ATC_AIRPORT_TYPES = {"large_airport", "medium_airport", "small_airport"}


@dataclass
class Runway:
    ident_low: str          # e.g. "10L"
    ident_high: str         # e.g. "28R"
    length_ft: int
    width_ft: int
    surface: str
    heading_low: Optional[float]   # true heading of low end
    heading_high: Optional[float]
    lat_low: Optional[float] = None
    lon_low: Optional[float] = None
    lat_high: Optional[float] = None
    lon_high: Optional[float] = None
    closed: bool = False

    @property
    def name(self) -> str:
        return f"{self.ident_low}/{self.ident_high}"

    def idents(self) -> list[str]:
        return [self.ident_low, self.ident_high]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "length_ft": self.length_ft,
            "width_ft": self.width_ft,
            "surface": self.surface,
            "heading_low": self.heading_low,
            "heading_high": self.heading_high,
            "closed": self.closed,
        }


@dataclass
class Airport:
    icao: str
    name: str
    type: str
    latitude: float
    longitude: float
    elevation_ft: Optional[int]
    country: str
    municipality: str
    iata: str = ""
    runways: list[Runway] = field(default_factory=list)

    def runway_idents(self) -> list[str]:
        out = []
        for r in self.runways:
            if not r.closed:
                out.extend(r.idents())
        return out

    def has_runway(self, ident: str) -> bool:
        return ident.upper() in self.runway_idents()

    def to_dict(self) -> dict:
        return {
            "icao": self.icao,
            "name": self.name,
            "type": self.type,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation_ft": self.elevation_ft,
            "country": self.country,
            "municipality": self.municipality,
            "iata": self.iata,
            "runways": [r.to_dict() for r in self.runways],
        }

    def summary_line(self) -> str:
        rws = ", ".join(
            f"{r.name} ({r.length_ft}ft{' CLOSED' if r.closed else ''})" for r in self.runways
        ) or "unknown"
        elev = f"{self.elevation_ft}ft" if self.elevation_ft is not None else "unknown"
        return f"{self.name} ({self.icao}), elevation {elev}, runways: {rws}"


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    r_nm = 3440.065
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r_nm * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial true bearing from point 1 to point 2, degrees 0-360."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) / 3600 < CACHE_MAX_AGE_HOURS


def _fetch_cached(url: str, cache: Path) -> str:
    if _cache_fresh(cache):
        return cache.read_text(encoding="utf-8")
    try:
        logger.info(f"Downloading {url}...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text, encoding="utf-8")
        return resp.text
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        if cache.exists():
            logger.info("Using stale cache")
            return cache.read_text(encoding="utf-8")
        raise


def _to_float(v: str) -> Optional[float]:
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def _to_int(v: str) -> Optional[int]:
    f = _to_float(v)
    return int(f) if f is not None else None


class AirportDataService:
    """Airport metadata + runways lookup with lazy loading."""

    def __init__(self, auto_load: bool = True):
        self._airports: dict[str, Airport] = {}
        self._loaded = False
        if auto_load:
            self._ensure_loaded()

    def _ensure_loaded(self):
        if self._loaded:
            return
        airports_csv = _fetch_cached(AIRPORTS_URL, AIRPORTS_CACHE)
        runways_csv = _fetch_cached(RUNWAYS_URL, RUNWAYS_CACHE)
        self._parse(airports_csv, runways_csv)
        self._loaded = True

    def _parse(self, airports_csv: str, runways_csv: str):
        for row in csv.DictReader(io.StringIO(airports_csv)):
            ident = (row.get("ident") or "").upper()
            if not ident or row.get("type") not in ATC_AIRPORT_TYPES:
                continue
            lat, lon = _to_float(row.get("latitude_deg")), _to_float(row.get("longitude_deg"))
            if lat is None or lon is None:
                continue
            self._airports[ident] = Airport(
                icao=ident,
                name=row.get("name", ""),
                type=row.get("type", ""),
                latitude=lat,
                longitude=lon,
                elevation_ft=_to_int(row.get("elevation_ft")),
                country=row.get("iso_country", ""),
                municipality=row.get("municipality", ""),
                iata=row.get("iata_code", ""),
            )

        for row in csv.DictReader(io.StringIO(runways_csv)):
            ident = (row.get("airport_ident") or "").upper()
            ap = self._airports.get(ident)
            if not ap:
                continue
            low, high = row.get("le_ident", ""), row.get("he_ident", "")
            if not low and not high:
                continue
            ap.runways.append(Runway(
                ident_low=low,
                ident_high=high,
                length_ft=_to_int(row.get("length_ft")) or 0,
                width_ft=_to_int(row.get("width_ft")) or 0,
                surface=row.get("surface", ""),
                heading_low=_to_float(row.get("le_heading_degT")),
                heading_high=_to_float(row.get("he_heading_degT")),
                lat_low=_to_float(row.get("le_latitude_deg")),
                lon_low=_to_float(row.get("le_longitude_deg")),
                lat_high=_to_float(row.get("he_latitude_deg")),
                lon_high=_to_float(row.get("he_longitude_deg")),
                closed=row.get("closed") == "1",
            ))
        logger.info(f"Loaded {len(self._airports)} airports")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, icao: str) -> Optional[Airport]:
        self._ensure_loaded()
        return self._airports.get(icao.upper())

    def distance_nm(self, icao: str, lat: float, lon: float) -> Optional[float]:
        ap = self.get(icao)
        if not ap:
            return None
        return haversine_nm(lat, lon, ap.latitude, ap.longitude)

    def bearing_from_airport(self, icao: str, lat: float, lon: float) -> Optional[float]:
        ap = self.get(icao)
        if not ap:
            return None
        return bearing_deg(ap.latitude, ap.longitude, lat, lon)

    def nearest(
        self,
        lat: float,
        lon: float,
        max_nm: float = 50.0,
        min_runway_ft: int = 3000,
        types: Optional[set[str]] = None,
    ) -> list[tuple[Airport, float]]:
        """Airports within max_nm sorted by distance. Filters tiny strips by default."""
        self._ensure_loaded()
        types = types or ATC_AIRPORT_TYPES
        out = []
        # Cheap bounding box pre-filter (1 deg lat ~ 60nm)
        dlat = max_nm / 60.0
        dlon = max_nm / (60.0 * max(math.cos(math.radians(lat)), 0.1))
        for ap in self._airports.values():
            if ap.type not in types:
                continue
            if abs(ap.latitude - lat) > dlat or abs(ap.longitude - lon) > dlon:
                continue
            if not ap.runways or max(r.length_ft for r in ap.runways) < min_runway_ft:
                continue
            d = haversine_nm(lat, lon, ap.latitude, ap.longitude)
            if d <= max_nm:
                out.append((ap, d))
        out.sort(key=lambda t: t[1])
        return out


# Module-level singleton (lazy)
_service: Optional[AirportDataService] = None


def get_airport_data_service() -> AirportDataService:
    global _service
    if _service is None:
        _service = AirportDataService()
    return _service


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    svc = AirportDataService()
    if len(sys.argv) > 1:
        ap = svc.get(sys.argv[1])
        print(ap.summary_line() if ap else "not found")
        if ap:
            for r in ap.runways:
                print("  ", r.to_dict())
    else:
        print("Nearest to SBGR ramp:")
        for ap, d in svc.nearest(-23.4356, -46.4731, max_nm=30)[:5]:
            print(f"  {ap.icao} {ap.name} {d:.1f}nm")
