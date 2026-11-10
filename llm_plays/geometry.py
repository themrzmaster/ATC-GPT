"""
Runway geometry for the landing eval: where the aircraft is relative to the runway and to an
ILS-style approach path (3 degree glidepath, 50 ft threshold crossing height).

Frame: origin at the landing threshold, "along" positive in the landing direction (so an aircraft
on final has along < 0), "cross" positive RIGHT of the extended centerline. Flat-earth
approximation (fine within ~20 nm).
"""

import math
from dataclasses import dataclass
from typing import Optional

FT_PER_NM = 6076.12
FT_PER_DEG_LAT = 364_000.0
GS_DOT_DEG = 0.35       # 2 dots = 0.7 deg (ILS glideslope full scale)
LOC_DOT_DEG = 1.25      # 2 dots = 2.5 deg (typical localizer full scale)


@dataclass
class Runway:
    airport: str
    ident: str
    threshold_lat: float
    threshold_lon: float
    heading_true: float
    elevation_ft: float
    length_ft: float
    width_ft: float
    glidepath_deg: float = 3.0
    tch_ft: float = 50.0

    # -- frame --------------------------------------------------------------
    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        """(along_ft, cross_ft) of a position."""
        dn = (lat - self.threshold_lat) * FT_PER_DEG_LAT
        de = (lon - self.threshold_lon) * FT_PER_DEG_LAT * math.cos(math.radians(self.threshold_lat))
        h = math.radians(self.heading_true)
        along = dn * math.cos(h) + de * math.sin(h)
        cross = -dn * math.sin(h) + de * math.cos(h)
        return along, cross

    def from_local(self, along_ft: float, cross_ft: float) -> tuple[float, float]:
        h = math.radians(self.heading_true)
        dn = along_ft * math.cos(h) - cross_ft * math.sin(h)
        de = along_ft * math.sin(h) + cross_ft * math.cos(h)
        lat = self.threshold_lat + dn / FT_PER_DEG_LAT
        lon = self.threshold_lon + de / (FT_PER_DEG_LAT * math.cos(math.radians(self.threshold_lat)))
        return lat, lon

    # -- approach path ---------------------------------------------------------
    def glidepath_height_ft(self, along_ft: float) -> float:
        """Height above threshold elevation on the glidepath at this point (TCH at the threshold)."""
        return self.tch_ft + max(-along_ft, 0.0) * math.tan(math.radians(self.glidepath_deg))

    def deviations(self, lat: float, lon: float, alt_msl_ft: float) -> dict:
        along, cross = self.to_local(lat, lon)
        hat = alt_msl_ft - self.elevation_ft                      # height above threshold
        dist_before = max(-along, 1.0)
        gs_dev_ft = hat - self.glidepath_height_ft(along)
        # angular deviations as an ILS receiver would see them
        gs_angle = math.degrees(math.atan2(hat - self.tch_ft, dist_before + 1000.0)) if along < 0 else self.glidepath_deg
        gs_dots = (gs_angle - self.glidepath_deg) / GS_DOT_DEG
        loc_dist = self.length_ft - along                          # localizer antenna at the far end
        loc_angle = math.degrees(math.atan2(cross, max(loc_dist, 1.0)))
        return {
            "along_ft": along, "cross_ft": cross, "height_above_threshold_ft": hat,
            "distance_to_threshold_nm": max(-along, 0.0) / FT_PER_NM,
            "glidepath_dev_ft": gs_dev_ft, "gs_dots": max(-2.5, min(2.5, gs_dots)),
            "loc_dots": max(-2.5, min(2.5, loc_angle / LOC_DOT_DEG)),
            "on_runway_surface": 0 <= along <= self.length_ft and abs(cross) <= self.width_ft / 2,
        }


def runway_from_db(icao: str, ident: str, airport_data=None) -> Runway:
    """Build a Runway from the OurAirports data already used by the ATC side."""
    if airport_data is None:
        from airport_data_service import AirportDataService
        airport_data = AirportDataService()
    ap = airport_data.get(icao)
    if not ap:
        raise ValueError(f"unknown airport {icao}")
    for r in ap.runways:
        for end, (lat, lon, hdg), (olat, olon) in (
                (r.ident_low, (r.lat_low, r.lon_low, r.heading_low), (r.lat_high, r.lon_high)),
                (r.ident_high, (r.lat_high, r.lon_high, r.heading_high), (r.lat_low, r.lon_low))):
            if end.lstrip("0") != ident.lstrip("0"):
                continue
            if lat is None or lon is None:
                raise ValueError(f"{icao} {ident}: no threshold coordinates in the database")
            if hdg is None and olat is not None:
                hdg = _bearing(lat, lon, olat, olon)
            return Runway(icao, end, lat, lon, float(hdg), float(ap.elevation_ft or 0),
                          float(r.length_ft or 5000), float(r.width_ft or 100))
    raise ValueError(f"{icao} has no runway {ident}")


def _bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def wind_components(wind_from_deg: float, wind_kt: float, runway_heading: float) -> tuple[float, float]:
    """(headwind_kt, crosswind_kt_from_right)."""
    rel = math.radians(wind_from_deg - runway_heading)
    return wind_kt * math.cos(rel), wind_kt * math.sin(rel)


def optional_float(v) -> Optional[float]:
    return None if v is None else float(v)
