"""
Frequency Service - Airport ATC frequency lookup using OurAirports data.

Downloads and caches airport frequencies from OurAirports.com open data.
Provides lookup by airport ICAO and frequency type (TWR, GND, APP, etc.)
"""

import csv
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("FrequencyService")

# OurAirports data URL
FREQUENCIES_URL = "https://davidmegginson.github.io/ourairports-data/airport-frequencies.csv"

# Cache file location
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_FILE = CACHE_DIR / "airport_frequencies.csv"
CACHE_MAX_AGE_HOURS = 24 * 7  # Refresh weekly


@dataclass
class Frequency:
    """A single ATC frequency."""
    airport_icao: str
    type: str  # TWR, GND, APP, DEP, ATIS, CLD, etc.
    description: str
    frequency_mhz: float

    def frequency_hz(self) -> int:
        """Return frequency in Hz (for X-Plane dataref)."""
        return int(self.frequency_mhz * 1_000_000)

    def frequency_formatted(self) -> str:
        """Return formatted frequency string (e.g., '121.900')."""
        return f"{self.frequency_mhz:.3f}"


# Frequency type mapping for common ATC services
FREQUENCY_TYPES = {
    "TWR": ["TWR", "TOWER"],
    "GND": ["GND", "GROUND"],
    "APP": ["APP", "APPROACH"],
    "DEP": ["DEP", "DEPARTURE"],
    "ATIS": ["ATIS"],
    "CLD": ["CLD", "CLNC DEL", "CLEARANCE", "DELIVERY"],
    "CTR": ["CTR", "CENTER", "CENTRE"],
    "UNIC": ["UNIC", "UNICOM"],
    "CTAF": ["CTAF"],
}


class FrequencyService:
    """Service for looking up airport ATC frequencies."""

    def __init__(self, auto_download: bool = True):
        self._frequencies: dict[str, list[Frequency]] = {}
        self._loaded = False

        if auto_download:
            self._ensure_data()

    def _ensure_data(self):
        """Ensure frequency data is loaded, downloading if needed."""
        if self._loaded:
            return

        # Check cache
        if self._cache_valid():
            self._load_from_cache()
        else:
            self._download_and_cache()

        self._loaded = True

    def _cache_valid(self) -> bool:
        """Check if cache file exists and is fresh."""
        if not CACHE_FILE.exists():
            return False

        # Check age
        import time
        age_hours = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        return age_hours < CACHE_MAX_AGE_HOURS

    def _download_and_cache(self):
        """Download frequency data and save to cache."""
        logger.info(f"Downloading airport frequencies from {FREQUENCIES_URL}...")

        try:
            response = requests.get(FREQUENCIES_URL, timeout=30)
            response.raise_for_status()

            # Save to cache
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(response.text)
            logger.info(f"Cached to {CACHE_FILE}")

            # Parse
            self._parse_csv(response.text)

        except Exception as e:
            logger.error(f"Failed to download frequencies: {e}")
            # Try to use stale cache if available
            if CACHE_FILE.exists():
                logger.info("Using stale cache...")
                self._load_from_cache()

    def _load_from_cache(self):
        """Load frequency data from cache file."""
        logger.info(f"Loading frequencies from cache: {CACHE_FILE}")
        self._parse_csv(CACHE_FILE.read_text())

    def _parse_csv(self, csv_text: str):
        """Parse CSV data into frequency objects."""
        reader = csv.DictReader(io.StringIO(csv_text))

        count = 0
        for row in reader:
            icao = row.get("airport_ident", "").upper()
            freq_type = row.get("type", "").upper()
            description = row.get("description", "")
            freq_mhz_str = row.get("frequency_mhz", "")

            if not icao or not freq_mhz_str:
                continue

            try:
                freq_mhz = float(freq_mhz_str)
            except ValueError:
                continue

            freq = Frequency(
                airport_icao=icao,
                type=freq_type,
                description=description,
                frequency_mhz=freq_mhz,
            )

            if icao not in self._frequencies:
                self._frequencies[icao] = []
            self._frequencies[icao].append(freq)
            count += 1

        logger.info(f"Loaded {count} frequencies for {len(self._frequencies)} airports")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_frequencies(self, icao: str) -> list[Frequency]:
        """Get all frequencies for an airport."""
        self._ensure_data()
        return self._frequencies.get(icao.upper(), [])

    def get_frequency(self, icao: str, freq_type: str) -> Optional[Frequency]:
        """
        Get a specific frequency type for an airport.

        Args:
            icao: Airport ICAO code
            freq_type: One of TWR, GND, APP, DEP, ATIS, CLD, CTR, UNIC

        Returns:
            First matching Frequency or None
        """
        self._ensure_data()
        icao = icao.upper()
        freq_type = freq_type.upper()

        # Get type aliases
        type_aliases = FREQUENCY_TYPES.get(freq_type, [freq_type])

        for freq in self._frequencies.get(icao, []):
            if freq.type in type_aliases:
                return freq

        return None

    def get_tower(self, icao: str) -> Optional[Frequency]:
        """Get tower frequency."""
        return self.get_frequency(icao, "TWR")

    def get_ground(self, icao: str) -> Optional[Frequency]:
        """Get ground frequency."""
        return self.get_frequency(icao, "GND")

    def get_approach(self, icao: str) -> Optional[Frequency]:
        """Get approach frequency."""
        return self.get_frequency(icao, "APP")

    def get_departure(self, icao: str) -> Optional[Frequency]:
        """Get departure frequency."""
        return self.get_frequency(icao, "DEP")

    def get_atis(self, icao: str) -> Optional[Frequency]:
        """Get ATIS frequency."""
        return self.get_frequency(icao, "ATIS")

    def get_clearance(self, icao: str) -> Optional[Frequency]:
        """Get clearance delivery frequency."""
        return self.get_frequency(icao, "CLD")

    def get_all_atc(self, icao: str) -> dict[str, Frequency]:
        """
        Get all ATC frequencies for an airport.

        Returns:
            Dict mapping type (TWR, GND, etc.) to Frequency
        """
        result = {}
        for freq_type in ["CLD", "GND", "TWR", "DEP", "APP", "ATIS"]:
            freq = self.get_frequency(icao, freq_type)
            if freq:
                result[freq_type] = freq
        return result

    INFO_TYPES = ("AFIS", "RDO", "A/G", "CTAF", "UNIC", "UNICOM", "MULTICOM")

    def get_info_frequency(self, icao: str) -> Optional[Frequency]:
        """Non-ATC station (AFIS "Radio", CTAF, UNICOM) of a field without a tower, if published."""
        self._ensure_data()
        for t in self.INFO_TYPES:
            for f in self._frequencies.get(icao.upper(), []):
                if f.type.upper() == t:
                    return f
        return None

    def match_frequency(self, icao: str, freq_mhz: float, tolerance: float = 0.01) -> Optional[tuple[str, Frequency]]:
        """
        Find which ATC service matches a given frequency.

        Args:
            icao: Airport ICAO code
            freq_mhz: Frequency in MHz to match
            tolerance: Matching tolerance in MHz

        Returns:
            Tuple of (service_type, Frequency) or None
        """
        self._ensure_data()

        for freq in self._frequencies.get(icao.upper(), []):
            if abs(freq.frequency_mhz - freq_mhz) < tolerance:
                return (freq.type, freq)

        return None

    def frequency_to_service(self, icao: str, freq_hz: int) -> Optional[str]:
        """
        Convert X-Plane frequency (Hz) to ATC service name.

        Args:
            icao: Airport ICAO code
            freq_hz: Frequency in Hz from X-Plane dataref

        Returns:
            Service name (e.g., "Tower", "Ground") or None
        """
        freq_mhz = freq_hz / 1_000_000
        result = self.match_frequency(icao, freq_mhz)

        if result:
            freq_type, freq = result
            # Return human-readable name
            type_names = {
                "TWR": "Tower",
                "GND": "Ground",
                "APP": "Approach",
                "DEP": "Departure",
                "ATIS": "ATIS",
                "CLD": "Clearance",
                "CTR": "Center",
            }
            return type_names.get(freq_type, freq_type)

        return None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    service = FrequencyService()

    if len(sys.argv) < 2:
        print("Usage: python frequency_service.py <ICAO>")
        print("Example: python frequency_service.py SBGR")
        sys.exit(1)

    icao = sys.argv[1].upper()
    freqs = service.get_all_atc(icao)

    if not freqs:
        print(f"No frequencies found for {icao}")
        sys.exit(1)

    print(f"\n📡 ATC Frequencies for {icao}:\n")
    for freq_type, freq in freqs.items():
        print(f"  {freq_type:4s}: {freq.frequency_formatted()} MHz - {freq.description}")

    # Show all frequencies
    print(f"\n📻 All frequencies:\n")
    for freq in service.get_frequencies(icao):
        print(f"  {freq.type:6s}: {freq.frequency_formatted()} MHz - {freq.description}")
