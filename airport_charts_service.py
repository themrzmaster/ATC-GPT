"""
Airport Charts Service - Python client for airportwxbot.com API

Usage:
    from airport_charts_service import AirportChartsService

    service = AirportChartsService()
    charts = service.list_charts("KJFK")
    pdf_bytes = service.get_chart(charts[0]["id"])
"""

import requests
from typing import Optional
from pathlib import Path


class AirportChartsService:
    """Client for Airport Wx Bot charts and weather API."""

    BASE_URL = "https://airportwxbot.com"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()

    # -------------------------------------------------------------------------
    # Charts
    # -------------------------------------------------------------------------

    def list_charts(self, icao: str) -> list[dict]:
        """
        List all available charts for an airport.

        Args:
            icao: Airport ICAO code (e.g., "KJFK", "SBCB")

        Returns:
            List of chart metadata dicts with keys:
                - id: UUID for downloading the chart
                - name: Full chart name
                - cleaned_name: Formatted chart name
                - category: Chart type (AD, APP, SID, STAR, MISC)
                - icao: Airport ICAO
                - code: Country code
                - rwys: List of applicable runways
                - type: Document type (usually "eaip")
        """
        url = f"{self.BASE_URL}/api/planner/charts/{icao.upper()}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_chart(self, chart_id: str) -> bytes:
        """
        Download a chart PDF by its ID.

        Args:
            chart_id: UUID from list_charts response

        Returns:
            PDF file as bytes
        """
        url = f"{self.BASE_URL}/charts/serve_charts/{chart_id}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.content

    def download_chart(self, chart_id: str, output_path: str | Path) -> Path:
        """
        Download a chart PDF and save to file.

        Args:
            chart_id: UUID from list_charts response
            output_path: Where to save the PDF

        Returns:
            Path to saved file
        """
        pdf_bytes = self.get_chart(chart_id)
        output_path = Path(output_path)
        output_path.write_bytes(pdf_bytes)
        return output_path

    def get_charts_by_category(self, icao: str, category: str) -> list[dict]:
        """
        List charts filtered by category.

        Args:
            icao: Airport ICAO code
            category: One of AD, APP, SID, STAR, MISC

        Returns:
            Filtered list of chart metadata
        """
        charts = self.list_charts(icao)
        return [c for c in charts if c.get("category") == category.upper()]

    def get_charts_by_runway(self, icao: str, runway: str) -> list[dict]:
        """
        List charts for a specific runway.

        Args:
            icao: Airport ICAO code
            runway: Runway identifier (e.g., "04L", "28")

        Returns:
            Filtered list of chart metadata
        """
        charts = self.list_charts(icao)
        return [
            c for c in charts
            if runway in c.get("rwys", []) or "*" in c.get("rwys", [])
        ]

    # -------------------------------------------------------------------------
    # Weather
    # -------------------------------------------------------------------------

    def get_metar_taf(self, icao: str, include_active_runways: bool = True) -> dict:
        """
        Get METAR and TAF for an airport.

        Args:
            icao: Airport ICAO code
            include_active_runways: Include active runway info

        Returns:
            Dict with METAR, TAF, and optionally active runway data
        """
        act_rwys = 1 if include_active_runways else 0
        url = f"{self.BASE_URL}/api/planner/metar_taf/{icao.upper()}?act_rwys={act_rwys}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_datis(self) -> dict:
        """
        Get D-ATIS data.

        Returns:
            D-ATIS information
        """
        url = f"{self.BASE_URL}/api/planner/datis"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # NOTAMs
    # -------------------------------------------------------------------------

    def get_notams(self, icao: str) -> dict:
        """
        Get NOTAMs for an airport.

        Args:
            icao: Airport ICAO code

        Returns:
            NOTAM data for the airport
        """
        url = f"{self.BASE_URL}/api/planner/notams/{icao.upper()}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Search & Station Info
    # -------------------------------------------------------------------------

    def search_airports(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search for airports by name or ICAO.

        Args:
            query: Search term
            limit: Max results to return

        Returns:
            List of matching airports
        """
        url = f"{self.BASE_URL}/api/planner/search"
        params = {"query": query, "limit": limit}
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_station(self, station_id: str) -> dict:
        """
        Get station/airport information.

        Args:
            station_id: Station identifier

        Returns:
            Station data
        """
        url = f"{self.BASE_URL}/api/planner/station/{station_id}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Network Data (VATSIM/IVAO)
    # -------------------------------------------------------------------------

    def get_vatsim_pilots(self) -> dict:
        """Get live VATSIM pilot data."""
        url = f"{self.BASE_URL}/api/planner/vatsim/live_data"
        params = {"short": 1, "nopilot": 0}
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_vatsim_atc(self) -> dict:
        """Get live VATSIM ATC data."""
        url = f"{self.BASE_URL}/api/planner/vatsim/live_data"
        params = {"short": 0, "nopilot": 1}
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_ivao_data(self) -> dict:
        """Get live IVAO network data."""
        url = f"{self.BASE_URL}/api/planner/ivao/live_data"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_top_airports(self, limit: int = 10) -> dict:
        """
        Get airports with most network traffic.

        Args:
            limit: Number of airports to return
        """
        url = f"{self.BASE_URL}/api/planner/network/top_airports"
        params = {"limit": limit}
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_chart_url(self, chart_id: str) -> str:
        """
        Get direct URL for a chart PDF.

        Args:
            chart_id: UUID from list_charts response

        Returns:
            Direct download URL
        """
        return f"{self.BASE_URL}/charts/serve_charts/{chart_id}"

    def list_charts_with_urls(self, icao: str) -> list[dict]:
        """
        List all charts with their download URLs included.

        Args:
            icao: Airport ICAO code

        Returns:
            List of chart metadata with 'url' field added
        """
        charts = self.list_charts(icao)
        for chart in charts:
            chart["url"] = self.get_chart_url(chart["id"])
        return charts


# -----------------------------------------------------------------------------
# CLI usage example
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    service = AirportChartsService()

    if len(sys.argv) < 2:
        print("Usage: python airport_charts_service.py <ICAO> [category]")
        print("Example: python airport_charts_service.py KJFK")
        print("Example: python airport_charts_service.py KJFK APP")
        sys.exit(1)

    icao = sys.argv[1].upper()
    category = sys.argv[2].upper() if len(sys.argv) > 2 else None

    print(f"Fetching charts for {icao}...")

    if category:
        charts = service.get_charts_by_category(icao, category)
    else:
        charts = service.list_charts_with_urls(icao)

    print(f"Found {len(charts)} charts:\n")

    for chart in charts:
        print(f"[{chart['category']}] {chart['name']}")
        print(f"  ID: {chart['id']}")
        print(f"  Runways: {', '.join(chart['rwys'])}")
        print(f"  URL: {service.get_chart_url(chart['id'])}")
        print()
