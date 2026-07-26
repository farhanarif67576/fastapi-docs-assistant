"""
Scrapy items for FastAPI documentation pages.
"""

import scrapy


class FastAPIDocPage(scrapy.Item):
    """A single FastAPI documentation page."""
    url = scrapy.Field()
    title = scrapy.Field()
    html = scrapy.Field()
    section = scrapy.Field()  # e.g., 'tutorial', 'reference', 'advanced'
    doc_id = scrapy.Field()   # unique document hash