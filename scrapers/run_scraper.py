"""
Entry point script for running the FastAPI docs scraper.

Usage:
    # Quick test (5 pages)
    python run_scraper.py --test

    # Full scrape
    python run_scraper.py

    # Full scrape with custom output
    python run_scraper.py -o ../data/fastapi_docs/raw_pages.jsonl
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure scrapy can find our project
sys.path.insert(0, str(Path(__file__).parent))

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


def main():
    parser = argparse.ArgumentParser(
        description="FastAPI Docs Scraper - Scrape all FastAPI documentation pages",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (only scrape 5 pages)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path for scraped pages (JSON Lines format)",
    )
    args = parser.parse_args()

    # Get Scrapy settings
    os.environ.setdefault("SCRAPY_SETTINGS_MODULE", "scrapers.settings")
    settings = get_project_settings()

    # Override output file if specified
    if args.output:
        settings.set("FEED_URI", args.output)

    # Ensure output directory exists
    output_path = settings.get("FEED_URI", "")
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Configure and run the crawler
    process = CrawlerProcess(settings)

    process.crawl(
        "fastapi_docs",
        test_mode=args.test,
    )

    print("=" * 60)
    if args.test:
        print("🔬 TEST MODE - Scraping 5 pages")
    else:
        print("🕷️  Starting full documentation scrape")
    print(f"   Output: {settings.get('FEED_URI', 'stdout')}")
    print("=" * 60)

    process.start()


if __name__ == "__main__":
    main()