#!/usr/bin/env python3
"""
Debug script to test what the LLM understands from a chart PDF.

Usage:
    python debug_chart.py SBGR APP    # Test approach charts for SBGR
    python debug_chart.py KJFK AD     # Test airport diagram for KJFK
"""

import json
import logging
import os
import sys

from openai import OpenAI

from airport_charts_service import AirportChartsService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ChartDebug")


def test_chart_understanding(icao: str, category: str = "APP"):
    """Test what the LLM can understand from a chart."""

    # Initialize services
    charts_service = AirportChartsService()
    client = OpenAI()

    # Get charts
    logger.info(f"Fetching {category} charts for {icao}...")
    charts = charts_service.get_charts_by_category(icao, category)

    if not charts:
        logger.error(f"No {category} charts found for {icao}")
        return

    logger.info(f"Found {len(charts)} charts:")
    for i, chart in enumerate(charts[:5]):  # Show first 5
        logger.info(f"  [{i}] {chart['name']} (runways: {chart.get('rwys', [])})")

    # Select first chart for testing
    chart = charts[0]
    chart_url = charts_service.get_chart_url(chart["id"])

    logger.info("=" * 60)
    logger.info(f"Testing chart: {chart['name']}")
    logger.info(f"Chart URL: {chart_url}")
    logger.info("=" * 60)

    # Test 1: Ask LLM to describe what it sees
    logger.info("\n📄 TEST 1: Describe the chart")
    logger.info("-" * 40)

    response = client.responses.create(
        model="gpt-4o",  # Use vision-capable model
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_url": chart_url
                    },
                    {
                        "type": "input_text",
                        "text": "Describe what you see in this aviation chart. Include: chart type, airport, runway(s), key waypoints/fixes, altitudes, frequencies, and any important notes."
                    }
                ]
            }
        ]
    )

    logger.info(f"\n🤖 LLM Response:\n{response.output_text}\n")

    # Test 2: Extract specific procedure info
    logger.info("\n📄 TEST 2: Extract procedure details")
    logger.info("-" * 40)

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_url": chart_url
                    },
                    {
                        "type": "input_text",
                        "text": """Extract the following information from this approach chart in JSON format:
{
    "chart_type": "ILS/RNAV/VOR/NDB/etc",
    "airport_icao": "XXXX",
    "runway": "XX",
    "final_approach_course": "XXX degrees",
    "decision_altitude": "XXX ft",
    "missed_approach": "description",
    "initial_approach_fix": "name",
    "final_approach_fix": "name",
    "frequencies": {
        "localizer": "XXX.XX",
        "tower": "XXX.XX",
        "approach": "XXX.XX"
    },
    "key_altitudes": ["list of altitude restrictions"],
    "notes": ["any important notes"]
}

If any field is not visible or not applicable, use null."""
                    }
                ]
            }
        ]
    )

    logger.info(f"\n🤖 LLM Response:\n{response.output_text}\n")

    # Test 3: Simulate ATC giving vectors
    logger.info("\n📄 TEST 3: Simulate ATC vectors")
    logger.info("-" * 40)

    response = client.responses.create(
        model="gpt-4o",
        instructions="You are an approach controller. Use standard ATC phraseology.",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_url": chart_url
                    },
                    {
                        "type": "input_text",
                        "text": f"Based on this approach chart, give me ATC vectors for an aircraft at 3000ft, 10 miles northwest of the airport, to fly this approach. Use proper ATC phraseology."
                    }
                ]
            }
        ]
    )

    logger.info(f"\n🤖 ATC Response:\n{response.output_text}\n")


def list_charts(icao: str):
    """List all available charts for an airport."""
    charts_service = AirportChartsService()
    charts = charts_service.list_charts(icao)

    logger.info(f"\nCharts for {icao} ({len(charts)} total):\n")

    by_category = {}
    for chart in charts:
        cat = chart.get("category", "OTHER")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(chart)

    for cat in sorted(by_category.keys()):
        logger.info(f"  {cat}:")
        for chart in by_category[cat]:
            logger.info(f"    - {chart['name']} (ID: {chart['id'][:8]}...)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug_chart.py <ICAO> [category]  - Test chart understanding")
        print("  python debug_chart.py <ICAO> list        - List all charts")
        print()
        print("Examples:")
        print("  python debug_chart.py SBGR APP     # Test approach charts")
        print("  python debug_chart.py KJFK AD      # Test airport diagram")
        print("  python debug_chart.py SBGR list    # List all charts")
        sys.exit(1)

    icao = sys.argv[1].upper()

    if len(sys.argv) > 2 and sys.argv[2].lower() == "list":
        list_charts(icao)
    else:
        category = sys.argv[2].upper() if len(sys.argv) > 2 else "APP"
        test_chart_understanding(icao, category)
