"""
Weather Service - METAR/TAF from the NOAA Aviation Weather Center (aviationweather.gov).

Public JSON API, no key, worldwide coverage (US and Brazilian stations included), fast.
Replaces airportwxbot.com as the primary source (its METAR endpoint started returning 500
for every airport on 2026-09-28; it is still used as a fallback).

Output uses the same shape the LLM tools already consume from airportwxbot:
    {"metar": {"raw", "wind_direction": {"value"}, "wind_speed": {"value"}, "flight_rules",
               "altimeter": {"repr"}}, "taf": {"raw"}, "active_runways": ["10L", ...]}

Active runways are computed from the wind and the runway headings in the airport database
(the runway end most into the wind; calm/variable -> no preference).
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger("Weather")

AWC_URL = "https://aviationweather.gov/api/data"
CACHE_S = 300


def _altimeter_repr(hpa: Optional[float], icao: str) -> Optional[str]:
    if hpa is None:
        return None
    if icao.upper().startswith(("K", "P", "C")):     # North America reports inches of mercury
        return f"A{round(hpa * 0.0295300 * 100):04d}"
    return f"Q{round(hpa):04d}"


def active_runways(wind_dir, wind_kt, runways) -> list[str]:
    """Runway ends whose heading is closest to the wind direction (all parallels included)."""
    if not runways or not isinstance(wind_dir, (int, float)) or not wind_kt or wind_kt < 3:
        return []
    ends = []
    for r in runways:
        if getattr(r, "closed", False):
            continue
        for ident, hdg in ((r.ident_low, r.heading_low), (r.ident_high, r.heading_high)):
            if ident and hdg is None and ident[:2].isdigit():
                hdg = int(ident[:2]) * 10
            if ident and hdg is not None:
                diff = abs((wind_dir - hdg + 180) % 360 - 180)
                ends.append((diff, ident))
    if not ends:
        return []
    best = min(d for d, _ in ends)
    return sorted({ident for d, ident in ends if d - best < 15})


class WeatherService:
    def __init__(self, timeout: float = 6.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ATC-GPT/1.0 (flight simulator ATC)"
        self._cache: dict[str, tuple[float, Optional[dict]]] = {}

    def _get(self, kind: str, ids: str) -> list:
        r = self.session.get(f"{AWC_URL}/{kind}", params={"ids": ids, "format": "json"}, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content.strip() else []

    def metars(self, icaos: list[str]) -> dict[str, dict]:
        """Raw AWC METAR records for several stations in one request (missing stations are absent)."""
        if not icaos:
            return {}
        return {m["icaoId"]: m for m in self._get("metar", ",".join(i.upper() for i in icaos)) if m.get("icaoId")}

    def get_metar_taf(self, icao: str, airport=None) -> Optional[dict]:
        """Normalised METAR/TAF for icao, or None when the station has no METAR. Cached 5 min."""
        icao = icao.upper()
        hit = self._cache.get(icao)
        if hit and time.time() - hit[0] < CACHE_S:
            return hit[1]
        m = self.metars([icao]).get(icao)
        out = None
        if m:
            out = self._normalise(icao, m, airport)
            try:
                tafs = self._get("taf", icao)
                if tafs:
                    out["taf"] = {"raw": tafs[0].get("rawTAF")}
            except Exception as e:  # noqa: BLE001  TAF is a nice-to-have
                logger.debug(f"TAF {icao}: {e}")
        self._cache[icao] = (time.time(), out)
        return out

    @staticmethod
    def _normalise(icao: str, m: dict, airport=None) -> dict:
        wdir, wspd = m.get("wdir"), m.get("wspd")
        return {
            "metar": {
                "raw": m.get("rawOb"),
                "wind_direction": {"value": wdir},
                "wind_speed": {"value": wspd},
                "flight_rules": m.get("fltCat"),
                "altimeter": {"repr": _altimeter_repr(m.get("altim"), icao)},
            },
            "taf": {"raw": None},
            "active_runways": active_runways(wdir, wspd, airport.runways if airport else None),
            "source": "aviationweather.gov",
        }


_service: Optional[WeatherService] = None


def get_weather_service() -> WeatherService:
    global _service
    if _service is None:
        _service = WeatherService()
    return _service


if __name__ == "__main__":
    import json
    import sys
    from airport_data_service import AirportDataService
    ads = AirportDataService()
    for code in sys.argv[1:] or ["KJFK", "SBGR"]:
        print(code, json.dumps(get_weather_service().get_metar_taf(code, ads.get(code)), indent=1))
