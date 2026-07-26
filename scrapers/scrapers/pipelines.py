"""
Pipeline for processing scraped FastAPI documentation pages.
Saves raw HTML pages to JSON Lines format.
"""

import hashlib

from scrapy.exceptions import DropItem


class FastAPIDocsPipeline:
    """
    Pipeline that processes scraped FastAPI documentation pages.
    - Validates page has required fields
    - Cleans HTML content
    - Deduplicates by URL
    """

    seen_urls = set()

    def process_item(self, item, spider):
        """Process a single scraped item."""
        url = item.get("url", "")

        # Skip duplicates
        if url in self.seen_urls:
            spider.logger.debug(f"Skipping duplicate URL: {url}")
            raise DropItem(f"Duplicate URL: {url}")
        self.seen_urls.add(url)

        # Validate required fields
        if not url:
            raise DropItem("Missing URL")

        if not item.get("html"):
            raise DropItem(f"No HTML content for {url}")

        # Infer section from URL path
        if not item.get("section"):
            item["section"] = self._infer_section(url)

        # Generate a unique document ID
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        item["doc_id"] = f"fastapi-{url_hash}"

        # Store page count on spider for progress tracking
        if not hasattr(spider, "pages_scraped"):
            spider.pages_scraped = 0
        spider.pages_scraped += 1

        return item

    def _infer_section(self, url: str) -> str:
        """Infer the documentation section from the URL path."""
        path = url.replace("https://fastapi.tiangolo.com/", "").strip("/")
        if path.startswith("tutorial"):
            return "tutorial"
        elif path.startswith("reference"):
            return "reference"
        elif path.startswith("advanced"):
            return "advanced"
        elif path.startswith("how-to"):
            return "how-to"
        elif path.startswith("project"):
            return "project"
        elif path.startswith("contributing"):
            return "contributing"
        elif path.startswith("alternatives"):
            return "alternatives"
        elif path.startswith("history"):
            return "history"
        elif path.startswith("features"):
            return "features"
        elif path.startswith("deployment"):
            return "deployment"
        elif path.startswith("management"):
            return "management"
        elif path.startswith("community"):
            return "community"
        elif path.startswith("help"):
            return "help"
        elif path.startswith("about"):
            return "about"
        elif path.startswith("resources"):
            return "resources"
        elif path.startswith("sponsors"):
            return "sponsors"
        elif path.startswith("blog"):
            return "blog"
        else:
            return path.split("/")[0] if "/" in path else "general"