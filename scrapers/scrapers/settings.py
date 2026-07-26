"""
Scrapy settings for FastAPI Docs Scraper.
"""

BOT_NAME = "fastapi_docs_scraper"

SPIDER_MODULES = ["scrapers.spiders"]
NEWSPIDER_MODULE = "scrapers.spiders"

# Obey robots.txt
ROBOTSTXT_OBEY = True

# Configure concurrency
CONCURRENT_REQUESTS = 8
DOWNLOAD_DELAY = 0.5
CONCURRENT_REQUESTS_PER_DOMAIN = 8

# Disable cookies (no need for docs)
COOKIES_ENABLED = False

# Enable and configure the FeedExportPipeline
ITEM_PIPELINES = {
    "scrapers.pipelines.FastAPIDocsPipeline": 300,
}

# Output settings
FEED_FORMAT = "jsonlines"
FEED_URI = "../data/fastapi_docs/raw_pages.jsonl"

# Set settings whose default value is deprecated to avoid false warnings
LOG_LEVEL = "INFO"

# AutoThrottle - good citizen scraping
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 10.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0

# Cache HTML for development speed
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 24 hours
HTTPCACHE_DIR = "../.scrapy-httpcache"