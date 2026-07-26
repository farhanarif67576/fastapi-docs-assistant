"""
Ingestion pipeline: scrapes FastAPI documentation, chunks by heading,
embeds with ONNX, and stores in pgvector (or saves as pickle for standalone).

Usage:
    # Scrapy-based scraping (recommended):
    python -m app.ingest --mode standalone --scrape            # Scrape + chunk + index
    python -m app.ingest --mode standalone --scrape --test      # Quick test (5 pages)
    python -m app.ingest --mode standalone --scrape --regen     # Re-scrape from scratch
    python -m app.ingest --mode full    --scrape --regen        # Full re-index via Scrapy + pgvector
    
    # Legacy requests-based scraping:
    python -m app.ingest --mode standalone                      # Scrape + chunk + index
    python -m app.ingest --mode full                            # Scrape + chunk + embed + store
    python -m app.ingest --mode standalone --regen              # Force re-scrape
"""

import os
import sys
import json
import re
import pickle
import argparse
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple
from tqdm.auto import tqdm

# Add parent directory to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .vector_store import ONNXEmbedder
from . import db


# ── Configuration ──────────────────────────────────────────────────────────────

FASTAPI_DOCS_URL = "https://fastapi.tiangolo.com/"
SITEMAP_URL = "https://fastapi.tiangolo.com/sitemap.xml"
DATA_DIR = os.getenv("DATA_PATH", "data/fastapi_docs")

INDEX_PATH = os.path.join(DATA_DIR, "text_index.pkl")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
SCRAPED_PAGES_PATH = os.path.join(DATA_DIR, "raw_pages.jsonl")

# URLs to skip (non-documentation pages)
EXCLUDE_PATTERNS = [
    "/blog/",
    "/sponsors/",
    "/resources/",
    "/help/",
    "/about/",
    "/newsletter/",
    "/search/",
]


# ── Model Download ─────────────────────────────────────────────────────────────

def ensure_model_downloaded():
    """
    Check if the sentence-transformers model is cached.
    If not, download it with a progress message.
    The model is ~80MB and only needs to be downloaded once.
    """
    from sentence_transformers import SentenceTransformer
    import os as _os
    
    cache_dir = _os.path.expanduser("~/.cache/huggingface/hub")
    model_name = "all-MiniLM-L6-v2"
    
    # Check if model files exist in cache
    model_marker = _os.path.join(
        cache_dir,
        "models--sentence-transformers--all-MiniLM-L6-v2",
    )
    
    if not _os.path.exists(model_marker):
        print("\n📥 Downloading embedding model (all-MiniLM-L6-v2)...")
        print("   This is a one-time download (~80MB)")
        SentenceTransformer(model_name)
        print("✅ Model downloaded and cached\n")
    else:
        print("✅ Embedding model already cached\n")


# ── Legacy Scraping (requests + bs4) ──────────────────────────────────────────

def get_doc_urls() -> List[str]:
    """
    Fetch the FastAPI sitemap and extract all documentation page URLs.
    Uses Python's built-in xml.etree.ElementTree (no extra dependencies needed).
    Filters out non-doc pages (blog, sponsors, etc.).
    """
    print("📡 Fetching sitemap...")
    response = requests.get(SITEMAP_URL)
    response.raise_for_status()
    
    # Parse XML using stdlib (no lxml dependency needed)
    # XML namespace: http://www.sitemaps.org/schemas/sitemap/0.9
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(response.content)
    
    urls = []
    for loc in root.findall(".//sm:loc", ns):
        url = loc.text.strip()
        # Only include documentation pages
        if "/docs" in url or "/tutorial" in url or "/reference" in url:
            # Skip excluded patterns
            if not any(pattern in url for pattern in EXCLUDE_PATTERNS):
                urls.append(url)
    
    # Deduplicate and sort
    urls = sorted(set(urls))
    print(f"   Found {len(urls)} documentation pages to scrape")
    return urls


def scrape_page(url: str) -> Optional[Dict[str, Any]]:
    """
    Scrape a single FastAPI documentation page.
    Returns the page title and raw HTML content.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ⚠ Failed to fetch {url}: {e}")
        return None
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Get the page title
    title_tag = soup.find("h1")
    title = title_tag.text.strip() if title_tag else "Untitled"
    
    # Get the main content area
    content_div = soup.find("div", class_="content")
    if not content_div:
        content_div = soup.find("article")
    if not content_div:
        content_div = soup.find("main")
    if not content_div:
        content_div = soup
    
    return {
        "url": url,
        "title": title,
        "html": str(content_div),
    }


# ── Scrapy-based Scraping ─────────────────────────────────────────────────────

def run_scrapy_scraper(test_mode: bool = False) -> str:
    """
    Run the Scrapy spider to scrape FastAPI documentation pages.
    
    Uses the Scrapy project in the 'scrapers/' directory.
    Provides concurrent scraping, auto-throttle, HTTP caching, 
    robots.txt respect, and robust error handling.
    
    Args:
        test_mode: If True, only scrape 5 pages for quick verification
        
    Returns:
        Path to the output JSONL file with raw page data
    """
    import subprocess
    from pathlib import Path
    
    scrapers_dir = Path(__file__).parent.parent / "scrapers"
    output_path = Path(SCRAPED_PAGES_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "scrapy", "crawl", "fastapi_docs",
        "-o", str(output_path),
    ]
    if test_mode:
        cmd.extend(["-a", "test_mode=True"])
    
    print(f"\n🕷️  Running Scrapy spider")
    print(f"   Output: {output_path}")
    print(f"   Mode: {'TEST (5 pages only)' if test_mode else 'FULL SCRAPE'}")
    
    env = os.environ.copy()
    env.setdefault("SCRAPY_SETTINGS_MODULE", "scrapers.settings")
    
    result = subprocess.run(
        cmd, cwd=str(scrapers_dir), capture_output=False, env=env,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Scrapy spider failed with exit code {result.returncode}")
    
    return str(output_path)


def load_scraped_pages(raw_pages_path: str = None) -> List[Dict[str, Any]]:
    """
    Load scraped pages from a JSON Lines file (Scrapy output format).
    Each line: {url, title, html, section, doc_id}
    
    Returns a list of page dicts compatible with chunk_page().
    """
    if raw_pages_path is None:
        raw_pages_path = SCRAPED_PAGES_PATH
    
    if not os.path.exists(raw_pages_path):
        raise FileNotFoundError(
            f"Raw pages file not found: {raw_pages_path}\n"
            f"Run with --scrape first to generate this file."
        )
    
    pages = []
    with open(raw_pages_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                page = json.loads(line)
                if page.get("url") and page.get("html"):
                    pages.append(page)
            except json.JSONDecodeError as e:
                print(f"  ⚠ Skipping malformed line: {e}")
    
    print(f"📂 Loaded {len(pages)} raw pages from {raw_pages_path}")
    return pages


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_page(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split a documentation page into chunks by h2 headings.
    Each chunk contains: id, title, heading_path, content, url, section
    
    This is semantic chunking — each <h2> section becomes one chunk,
    preserving the heading hierarchy as metadata.
    """
    soup = BeautifulSoup(page["html"], "html.parser")
    url = page["url"]
    page_title = page["title"]
    
    # Extract URL slug for ID generation
    url_slug = url.replace("https://fastapi.tiangolo.com", "").strip("/").replace("/", "-")
    if not url_slug:
        url_slug = "index"
    
    chunks = []
    
    # Find all h2 headings to split the page into sections
    headings = soup.find_all(["h2", "h3"])
    
    if not headings:
        # No sub-headings — treat the entire page as one chunk
        text = soup.get_text(separator=" ", strip=True)
        if len(text) > 50:  # Skip very small fragments
            chunks.append({
                "id": f"fastapi-{url_slug}-0",
                "title": page_title,
                "heading_path": [page_title],
                "content": text,
                "url": url,
                "section": page_title,
            })
        return chunks
    
    # Split content by headings
    current_section = None
    current_heading = None
    current_content = []
    current_heading_path = [page_title]
    
    for element in soup.children:
        if element.name in ["h2", "h3"]:
            # Save previous section
            if current_section is not None and current_content:
                text = " ".join(current_content).strip()
                if len(text) > 50:
                    anchor = re.sub(r"[^a-z0-9-]+", "", current_heading.lower().replace(" ", "-"))
                    section_url = f"{url}#{anchor}" if anchor else url
                    
                    chunks.append({
                        "id": f"fastapi-{url_slug}-{current_section}",
                        "title": current_heading,
                        "heading_path": current_heading_path.copy(),
                        "content": text,
                        "url": section_url,
                        "section": current_heading,
                    })
            
            # Start new section
            current_heading = element.text.strip()
            current_content = []
            current_section = len(chunks)
            
            if element.name == "h2":
                current_heading_path = [page_title, current_heading]
            elif element.name == "h3":
                if len(current_heading_path) < 3:
                    current_heading_path.append(current_heading)
                else:
                    current_heading_path[-1] = current_heading
                    
        elif element.name is not None:
            # Regular content — extract text
            text = element.get_text(separator=" ", strip=True)
            if text:
                current_content.append(text)
    
    # Save the last section
    if current_section is not None and current_content:
        text = " ".join(current_content).strip()
        if len(text) > 50:
            anchor = re.sub(r"[^a-z0-9-]+", "", current_heading.lower().replace(" ", "-"))
            section_url = f"{url}#{anchor}" if anchor else url
            
            chunks.append({
                "id": f"fastapi-{url_slug}-{current_section}",
                "title": current_heading,
                "heading_path": current_heading_path.copy(),
                "content": text,
                "url": section_url,
                "section": current_heading,
            })
    
    # If we ended up with no chunks, create one from the full page text
    if not chunks:
        text = soup.get_text(separator=" ", strip=True)
        if len(text) > 50:
            chunks.append({
                "id": f"fastapi-{url_slug}-0",
                "title": page_title,
                "heading_path": [page_title],
                "content": text,
                "url": url,
                "section": page_title,
            })
    
    return chunks


def chunk_all_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk a list of scraped pages into semantic sections by heading.
    Skips pages that fail to chunk.
    """
    all_chunks = []
    for page in tqdm(pages, desc="Chunking pages"):
        try:
            chunks = chunk_page(page)
            all_chunks.extend(chunks)
        except Exception as e:
            url = page.get("url", "unknown")
            print(f"  ⚠ Failed to chunk {url}: {e}")
    
    return all_chunks


# ── Scrape Pipeline ────────────────────────────────────────────────────────────

def scrape_all_docs(
    output_dir: str = None,
    force_regen: bool = False,
    use_scrapy: bool = False,
    test_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Scrape all FastAPI documentation pages and chunk them.
    
    If chunks.json already exists and force_regen is False, loads from cache.
    If use_scrapy is True, uses the Scrapy spider for robust scraping.
    
    Returns a list of chunk dictionaries.
    """
    if output_dir is None:
        output_dir = DATA_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if cached chunks exist
    if not force_regen and os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        print(f"📂 Loaded {len(all_chunks)} chunks from cache ({CHUNKS_PATH})")
        return all_chunks
    
    if use_scrapy:
        # Scrapy-based: concurrent, cached, robust
        raw_pages_path = run_scrapy_scraper(test_mode=test_mode)
        pages = load_scraped_pages(raw_pages_path)
    else:
        # Legacy: sequential requests + BeautifulSoup
        urls = get_doc_urls()
        if test_mode:
            urls = urls[:5]
            print(f"🔬 Test mode: limited to {len(urls)} pages")
        
        pages = []
        for url in tqdm(urls, desc="Scraping FastAPI docs"):
            page = scrape_page(url)
            if page is not None:
                pages.append(page)
    
    # Chunk all pages
    print(f"\n📄 Chunking {len(pages)} pages...")
    all_chunks = chunk_all_pages(pages)
    
    # Save all chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(all_chunks)} chunks to {CHUNKS_PATH}")
    
    return all_chunks


# ── Text Index ─────────────────────────────────────────────────────────────────

def build_and_save_text_index(chunks: List[Dict[str, Any]]):
    """
    Build the TF-IDF text index from chunks and save to a pickle file.
    This allows the RAG pipeline to work without PostgreSQL.
    """
    from .minsearch import Index as TextIndex
    
    print("\n📚 Building text index...")
    
    # Convert heading_path lists to strings for the text index
    for chunk in chunks:
        if isinstance(chunk.get("heading_path"), list):
            chunk["heading_path"] = " ".join(chunk["heading_path"])
    
    index = TextIndex(
        text_fields=["title", "heading_path", "content", "section"],
        keyword_fields=["id"],
    )
    index.fit(chunks)
    
    # Save index and chunks together
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "chunks": chunks}, f)
    
    print(f"✅ Saved text index with {len(chunks)} documents to {INDEX_PATH}")


# ── Embedding and Storing ─────────────────────────────────────────────────────

def embed_and_store(all_chunks: List[Dict[str, Any]]):
    """
    Embed all chunks using ONNX embedder and store them in pgvector.
    Requires PostgreSQL + pgvector to be running.
    """
    # Ensure model is downloaded first
    ensure_model_downloaded()
    
    embedder = ONNXEmbedder()
    
    # Initialize database
    print("\n🗄️  Initializing database...")
    db.init_db()
    
    # Extract texts to embed
    texts = []
    for chunk in all_chunks:
        # Combine title + heading_path + content for better embeddings
        combined = f"{chunk['title']}: {chunk['content']}"
        texts.append(combined)
    
    # Generate embeddings in batches
    batch_size = 32
    print(f"🔢 Generating embeddings for {len(texts)} chunks...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i + batch_size]
        batch_chunks = all_chunks[i:i + batch_size]
        
        embeddings = embedder.embed(batch_texts)
        
        for j, chunk in enumerate(batch_chunks):
            db.save_chunk(
                chunk_id=chunk["id"],
                title=chunk["title"],
                heading_path=chunk["heading_path"] if isinstance(chunk["heading_path"], list) else [chunk["heading_path"]],
                content=chunk["content"],
                url=chunk["url"],
                section=chunk["section"],
                embedding=embeddings[j],
            )
    
    print(f"✅ Stored {len(all_chunks)} chunks in pgvector")


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    mode: str = "standalone",
    force_regen: bool = False,
    use_scrapy: bool = False,
    test_mode: bool = False,
):
    """
    Run the ingestion pipeline.
    
    Args:
        mode: "standalone" (JSON + pickle, no DB) or "full" (embeds in pgvector)
        force_regen: If True, re-scrape instead of using cache
        use_scrapy: If True, use Scrapy spider (recommended)
        test_mode: If True, only scrape a few pages for quick verification
    """
    print("=" * 60)
    print("🏗️  FastAPI Docs Ingestion Pipeline")
    print(f"   Mode: {mode}")
    print(f"   Force re-scrape: {force_regen}")
    print(f"   Scraper: {'Scrapy' if use_scrapy else 'Requests+BS4'}")
    print(f"   Test mode: {test_mode}")
    print("=" * 60)
    
    # Step 1: Ensure model is downloaded
    if mode == "full":
        ensure_model_downloaded()
    
    # Step 2: Scrape and chunk (or load from cache)
    print("\n📄 Step 1: Scraping and chunking documentation...")
    all_chunks = scrape_all_docs(
        force_regen=force_regen,
        use_scrapy=use_scrapy,
        test_mode=test_mode,
    )
    
    if not all_chunks:
        print("❌ No chunks generated. Check the sitemap URL or network connection.")
        return
    
    # Step 3: Build and save text index
    print("\n📚 Step 2: Building text index...")
    build_and_save_text_index(all_chunks)
    
    # Step 4: Embed and store in pgvector (only in full mode)
    if mode == "full":
        print("\n🔢 Step 3: Embedding and storing in pgvector...")
        try:
            embed_and_store(all_chunks)
        except Exception as e:
            print(f"\n⚠️  Warning: Could not store in pgvector: {e}")
            print("   Text index is still saved. You can run with --mode full later.")
            print("   Make sure PostgreSQL is running: docker compose up -d")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Chunks JSON:   {CHUNKS_PATH}")
    print(f"   Text index:    {INDEX_PATH}")
    if mode == "full":
        print("   Vector store:  pgvector (populated)")
    else:
        print("   Vector store:  not populated (use --mode full)")
    print("=" * 60)


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main():
    """CLI entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="FastAPI Docs Ingestion Pipeline - scrape, chunk, and index FastAPI documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m app.ingest --mode standalone                    # Legacy scrape + chunk + index
  python -m app.ingest --mode standalone --scrape            # Scrapy scrape + chunk + index (recommended)
  python -m app.ingest --mode standalone --scrape --test     # Quick test (5 pages via Scrapy)
  python -m app.ingest --mode standalone --regen             # Force re-scrape (legacy)
  python -m app.ingest --mode full --scrape --regen          # Full re-index via Scrapy + pgvector
        """,
    )
    
    parser.add_argument(
        "--mode",
        choices=["standalone", "full"],
        default="standalone",
        help="'standalone' saves to JSON + pickle (no DB). 'full' also embeds in pgvector.",
    )
    
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force re-scrape from live docs instead of using cached chunks",
    )
    
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Use Scrapy spider for scraping (recommended - concurrent, cached, robust)",
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (only scrape a few pages for quick verification)",
    )
    
    args = parser.parse_args()
    run_pipeline(
        mode=args.mode,
        force_regen=args.regen,
        use_scrapy=args.scrape,
        test_mode=args.test,
    )


if __name__ == "__main__":
    main()