"""
Scrapy spider for scraping FastAPI documentation.

This spider:
1. Starts from the FastAPI sitemap.xml
2. Extracts all documentation page URLs
3. Scrapes each page's title and main content HTML
4. Follows links to undocumented pages found in the main content area

Usage:
    # Quick test - scrape just 5 pages
    scrapy crawl fastapi_docs -a test_mode=True

    # Full scrape
    scrapy crawl fastapi_docs

    # Save to specific output file
    scrapy crawl fastapi_docs -o ../data/fastapi_docs/raw_pages.jsonl
"""

import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import scrapy
from scrapy.http import Request, Response

from ..items import FastAPIDocPage


class FastAPIDocsSpider(scrapy.Spider):
    """Spider that scrapes FastAPI documentation from fastapi.tiangolo.com."""

    name = "fastapi_docs"
    allowed_domains = ["fastapi.tiangolo.com"]
    start_urls = ["https://fastapi.tiangolo.com/sitemap.xml"]

    # Pages to exclude from scraping
    exclude_patterns = [
        "/blog/",
        "/sponsors/",
        "/resources/",
        "/help/",
        "/about/",
        "/newsletter/",
        "/search/",
    ]

    # URL patterns that indicate documentation pages
    doc_path_prefixes = [
        "/docs",
        "/tutorial",
        "/reference",
        "/advanced",
        "/how-to",
    ]

    def __init__(self, test_mode=False, *args, **kwargs):
        """
        Args:
            test_mode: If True, only scrape a few pages for quick testing.
                       Default: False (full scrape).
        """
        super().__init__(*args, **kwargs)
        self.test_mode = test_mode in (True, "True", "true", "1", "yes")
        if self.test_mode:
            self.logger.info("🔬 TEST MODE: Will only scrape 5 pages for quick verification")
        self.doc_urls = []  # Stores all discovered documentation URLs
        self.pages_scraped = 0

    def parse(self, response: Response):
        """
        Parse the sitemap XML to extract all documentation page URLs.
        """
        self.logger.info(f"📡 Parsing sitemap: {response.url}")

        # Parse XML
        try:
            root = ET.fromstring(response.body)
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse sitemap XML: {e}")
            return

        # XML namespace for sitemaps
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        urls = []
        for loc in root.findall(".//sm:loc", ns):
            url = loc.text.strip()

            # Only include documentation pages
            if not any(prefix in url for prefix in self.doc_path_prefixes):
                continue

            # Skip excluded patterns
            if any(pattern in url for pattern in self.exclude_patterns):
                continue

            urls.append(url)

        # Deduplicate and sort
        self.doc_urls = sorted(set(urls))
        self.logger.info(f"📄 Found {len(self.doc_urls)} documentation pages to scrape")

        # Apply test mode limit
        if self.test_mode and len(self.doc_urls) > 5:
            self.doc_urls = self.doc_urls[:5]
            self.logger.info(f"🔬 Test mode: limited to {len(self.doc_urls)} pages")

        # Scrape each documentation page
        for doc_url in self.doc_urls:
            yield Request(
                url=doc_url,
                callback=self.parse_doc_page,
                errback=self.handle_error,
            )

    def parse_doc_page(self, response: Response):
        """
        Parse a single documentation page, extracting title and HTML content.
        """
        url = response.url
        self.logger.debug(f"📄 Scraping: {url}")

        # Extract the page title from <h1>
        title = response.css("h1::text").get("").strip()
        if not title:
            title = response.css("title::text").get("").strip() or "Untitled"

        # Extract the main content area
        # Try multiple selectors in order of specificity
        content_div = response.css("div.content")
        if not content_div:
            content_div = response.css("article")
        if not content_div:
            content_div = response.css("main")
        if not content_div:
            # Fallback: use body content
            content_div = response.css("body")

        # Get the HTML of the content area
        if content_div:
            html_content = content_div.get()
        else:
            html_content = response.body.decode("utf-8", errors="replace")

        # Infer section from URL
        section = self._infer_section(url)

        # Create the item
        item = FastAPIDocPage(
            url=url,
            title=title,
            html=html_content,
            section=section,
        )

        yield item

    def handle_error(self, failure):
        """Handle request failures gracefully."""
        self.logger.warning(f"⚠️  Failed to scrape {failure.request.url}: {failure.value}")

    def _infer_section(self, url: str) -> str:
        """Infer the documentation section from the URL path."""
        path = url.replace("https://fastapi.tiangolo.com/", "").strip("/")
        top_level = path.split("/")[0] if "/" in path else path
        return top_level if top_level else "general"