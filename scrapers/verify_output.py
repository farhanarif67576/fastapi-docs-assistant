"""Verify the scraped output."""
import json

with open("../data/fastapi_docs/raw_pages.jsonl") as f:
    pages = [json.loads(line) for line in f if line.strip()]

print(f"Total pages scraped: {len(pages)}")
print(f"Fields per page: {list(pages[0].keys())}")
print(f"\nPage titles:")
for p in pages:
    print(f"  - {p['title']}")
print(f"\nPage URLs:")
for p in pages:
    print(f"  - {p['url']}")
print(f"\nPage sections:")
for p in pages:
    print(f"  - {p['section']}")
print(f"\nHTML sizes (bytes):")
for p in pages:
    print(f"  - {len(p['html'])}")