"""
Inspect the ingestion data and report findings.
"""
import json, os

print("=" * 70)
print("📊 DATA INGESTION REPORT")
print("=" * 70)

# ── 1. Chunks JSON ──────────────────────────────────────────────────────────
chunks_path = "data/fastapi_docs/chunks.json"
if not os.path.exists(chunks_path):
    print("❌ chunks.json NOT FOUND")
    exit(1)

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"\n📄 File: {chunks_path}")
print(f"   Size: {os.path.getsize(chunks_path):,} bytes")
print(f"   Chunks: {len(chunks)}")

# ── 2. Chunk Analysis ───────────────────────────────────────────────────────
total_chars = 0
topics = set()
sections = set()
pages = set()

for c in chunks:
    total_chars += len(c["content"])
    topics.add(c["title"])
    sections.add(c["section"])
    # Extract page from URL
    page = c["url"].replace("https://fastapi.tiangolo.com", "").split("/")[1] if "fastapi.tiangolo.com" in c["url"] else "unknown"
    pages.add(page)

print(f"\n📊 Content Statistics:")
print(f"   Total characters: {total_chars:,}")
print(f"   Average chunk:    {total_chars // len(chunks):,} chars")
print(f"   Unique topics:    {len(topics)}")
print(f"   Unique sections:  {len(sections)}")
print(f"   Pages covered:    {len(pages)}")

# ── 3. URL sources ─────────────────────────────────────────────────────────
print(f"\n🌐 Pages in Knowledge Base:")
for p in sorted(pages):
    count = sum(1 for c in chunks if p in c["url"])
    print(f"   /{p}/ — {count} chunk(s)")

# ── 4. Topics covered ──────────────────────────────────────────────────────
print(f"\n📚 Topics Covered:")
for c in chunks:
    print(f"   • [{c['id']}] {c['title']}")
    print(f"     Section: {c['section']}")
    print(f"     URL: {c['url']}")
    print(f"     Size: {len(c['content']):,} chars")
    print()

# ── 5. Pickle Index ────────────────────────────────────────────────────────
pickle_path = "data/fastapi_docs/text_index.pkl"
if os.path.exists(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    index = data["index"]
    pickle_chunks = data["chunks"]
    print(f"\n📦 Pickle Index: {pickle_path}")
    print(f"   Size: {os.path.getsize(pickle_path):,} bytes")
    print(f"   Documents indexed: {len(pickle_chunks)}")
    print(f"   Text fields: {index.text_fields}")
    print(f"   Keyword fields: {index.keyword_fields}")
else:
    print(f"\n⚠️  Pickle index NOT FOUND at {pickle_path}")
    print("   Run: python -m app.ingest --mode standalone")

# ── 6. What's MISSING ──────────────────────────────────────────────────────
print(f"\n⚠️  COVERAGE GAPS")
print(f"   The official FastAPI docs have ~150+ pages covering:")
print(f"   • Tutorial (30+ pages): first-steps, path-params, query-params, body, etc.")
print(f"   • Advanced (40+ pages): middleware, testing, deployment, security, etc.")
print(f"   • Reference (80+ pages): API reference for every module/class/function")
print(f"")
print(f"   Current chunks.json has ONLY {len(chunks)} chunks from {len(pages)} pages.")
print(f"   A full ingestion would produce ~300-500 chunks from ~150 pages.")
print(f"")
print(f"   To do full ingestion:")
print(f"     python -m app.ingest --mode standalone --regen")
print(f"   (This scrapes live docs from fastapi.tiangolo.com)")

# ── 7. Ingestion Pipeline Summary ──────────────────────────────────────────
print(f"\n🔧 INGESTION PIPELINE FLOW")
print(f"   1. get_doc_urls() — fetches sitemap.xml, filters to /docs/* pages")
print(f"   2. scrape_page(url) — fetches HTML, extracts <h1> + main content div")
print(f"   3. chunk_page(html) — splits by <h2>/<h3> headings (semantic chunking)")
print(f"   4. scrape_all_docs() — loops over all URLs, saves to chunks.json")
print(f"   5. build_and_save_text_index() — builds TF-IDF index, saves to .pkl")
print(f"   6. embed_and_store() — [optional] ONNX embeddings → pgvector")
print(f"")
print(f"   Chunking strategy: Each <h2> section = 1 chunk")
print(f"   Chunk ID format: fastapi-{page-slug}-{section-index}")
print(f"   Fields per chunk: id, title, heading_path, content, url, section")
print(f"   Embedding model: all-MiniLM-L6-v2 (384-dim vectors)")
print(f"   Search index: TF-IDF with field boosting (title:3.0, section:2.5, content:1.0)")

print(f"\n{'='*70}")
print(f"✅ REPORT COMPLETE")
print(f"{'='*70}")