"""
Helper module to load the search index from pickle (standalone mode) or pgvector (full mode).
Used by app.py at startup to initialize the RAG pipeline.
"""

import os
import pickle
import json
from typing import Tuple, Optional, List, Dict, Any

from .minsearch import Index as TextIndex


INDEX_PATH = os.getenv("TEXT_INDEX_PATH", "data/fastapi_docs/text_index.pkl")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/fastapi_docs/chunks.json")


def text_index_exists() -> bool:
    """Check if a pre-built text index pickle exists."""
    return os.path.exists(INDEX_PATH)


def chunks_exist() -> bool:
    """Check if cached chunks JSON exists."""
    return os.path.exists(CHUNKS_PATH)


def load_text_index() -> Tuple[Optional[TextIndex], Optional[List[Dict[str, Any]]]]:
    """
    Load the TF-IDF text index and chunks from the pickle file.
    
    Returns:
        Tuple of (index, chunks) or (None, None) if not found
    """
    if not text_index_exists():
        print(f"⚠️  No text index found at {INDEX_PATH}")
        print("   Run: python -m app.ingest --mode standalone")
        return None, None
    
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    
    index = data["index"]
    chunks = data["chunks"]
    
    print(f"📚 Loaded text index with {len(chunks)} documents from {INDEX_PATH}")
    
    return index, chunks


def load_chunks_from_json() -> Optional[List[Dict[str, Any]]]:
    """
    Load chunks from the cached JSON file (without index).
    Used for rebuilding the index without re-scraping.
    """
    if not chunks_exist():
        print(f"⚠️  No chunks found at {CHUNKS_PATH}")
        return None
    
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"📂 Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    return chunks


def format_chunks_for_index(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format chunks for use in the text index.
    Converts heading_path lists to strings if needed.
    """
    for chunk in chunks:
        if isinstance(chunk.get("heading_path"), list):
            chunk["heading_path"] = " ".join(chunk["heading_path"])
        # Ensure all required fields exist
        chunk.setdefault("title", "")
        chunk.setdefault("heading_path", "")
        chunk.setdefault("content", "")
        chunk.setdefault("section", "")
    return chunks


def get_index_stats() -> dict:
    """
    Get statistics about the current index state.
    Used for health check / diagnostics.
    """
    stats = {
        "text_index_exists": text_index_exists(),
        "chunks_exist": chunks_exist(),
        "text_index_path": INDEX_PATH,
        "chunks_path": CHUNKS_PATH,
    }
    
    if text_index_exists():
        try:
            index, chunks = load_text_index()
            stats["chunk_count"] = len(chunks) if chunks else 0
        except Exception:
            stats["chunk_count"] = 0
    
    return stats