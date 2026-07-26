"""
Script: 02-retrieval-evaluation.py
Purpose: Full retrieval evaluation pipeline with metrics, boost optimization, and alpha tuning.
         Evaluates 3 approaches: text-only, vector-only, and hybrid search.
         Uses Hit Rate and MRR metrics.
         Optimizes boost parameters via random search and alpha via grid search.

Output:
    - data/retrieval-results.csv (final metrics for all approaches)
    - data/best-params.json (optimized boost params + alpha for production)

Usage:
    uv run python notebooks/02-retrieval-evaluation.py

Prerequisites:
    - PostgreSQL with pgvector running (docker compose up -d)
    - Document chunks embedded and stored in pgvector
    - ground-truth-retrieval.csv in data/
    - Or run in text-only mode without pgvector
"""

import os
import json
import csv
import random
import math
import sys
from typing import List, Dict, Any, Callable

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm.auto import tqdm


# ── Configuration ────────────────────────────────────────────────────────────────

CHUNKS_PATH = "data/fastapi_docs/chunks.json"
GROUND_TRUTH_PATH = "data/ground-truth-retrieval.csv"
RESULTS_PATH = "data/retrieval-results.csv"
BEST_PARAMS_PATH = "data/best-params.json"

# Text index field boost ranges for random search
BOOST_RANGES = {
    "title": (0.0, 5.0),
    "heading_path": (0.0, 4.0),
    "content": (0.0, 3.0),
    "section": (0.0, 4.0),
}

# Alpha values to test for hybrid search
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Random search iterations for boost optimization
N_ITERATIONS = 30

# Number of results to retrieve
NUM_RESULTS = 10


# ── Load Data ────────────────────────────────────────────────────────────────────

def load_chunks() -> List[Dict[str, Any]]:
    """Load the FastAPI documentation chunks."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth() -> List[Dict[str, str]]:
    """Load ground truth data (question, id pairs)."""
    df = pd.read_csv(GROUND_TRUTH_PATH)
    return df.to_dict(orient="records")


def build_text_index(chunks: List[Dict[str, Any]]):
    """Build the in-memory TF-IDF text index from chunks."""
    from app.minsearch import Index

    index = Index(
        text_fields=["title", "heading_path", "content", "section"],
        keyword_fields=["id"],
    )
    # Convert heading_path list to string for indexing
    for chunk in chunks:
        if isinstance(chunk.get("heading_path"), list):
            chunk["heading_path"] = " ".join(chunk["heading_path"])

    index.fit(chunks)
    return index, chunks


def build_vector_store(chunks: List[Dict[str, Any]]):
    """
    Build the vector store by embedding chunks and checking pgvector availability.
    Returns True if pgvector is available and populated.
    """
    try:
        from app.vector_store import get_embedder
        from app import db

        # Check if we can connect to PostgreSQL
        conn = db.get_db_connection()
        conn.close()

        # Check if chunks are already in the database
        existing = db.get_all_chunks()
        if existing:
            print(f"   ✅ Found {len(existing)} chunks in pgvector")
            return True

        print("   ⚠ No chunks found in pgvector. Embedding and storing...")
        embedder = get_embedder()
        texts = [f"{c['title']}: {c['content']}" for c in chunks]
        embeddings = embedder.embed(texts)

        for i, chunk in enumerate(chunks):
            db.save_chunk(
                chunk_id=chunk["id"],
                title=chunk["title"],
                heading_path=chunk.get("heading_path", [chunk["title"]]),
                content=chunk["content"],
                url=chunk.get("url", ""),
                section=chunk.get("section", chunk["title"]),
                embedding=embeddings[i],
            )

        print(f"   ✅ Stored {len(chunks)} chunks in pgvector")
        return True

    except Exception as e:
        print(f"\n   ⚠ Vector search unavailable: {e}")
        print("   ℹ️  Make sure PostgreSQL with pgvector is running:")
        print("      docker compose up -d")
        print("      Then run: uv run python -c \"from app.ingest import run_pipeline; run_pipeline()\"")
        return False


# ── Search Functions ─────────────────────────────────────────────────────────────

def create_text_search(text_index, default_boost: dict = None) -> Callable:
    """Create a text-only search function with configurable boost."""
    if default_boost is None:
        default_boost = {
            "title": 3.0,
            "heading_path": 2.0,
            "content": 1.0,
            "section": 2.5,
        }

    def search(query: str, boost: dict = None) -> List[Dict]:
        if boost is None:
            boost = default_boost
        return text_index.search(
            query=query,
            filter_dict={},
            boost_dict=boost,
            num_results=NUM_RESULTS,
        )

    return search


def create_vector_search() -> Callable:
    """Create a vector-only search function."""
    from app.vector_store import get_embedder
    from app import db

    embedder = get_embedder()

    def search(query: str) -> List[Dict]:
        query_embedding = embedder.embed_query(query)
        return db.vector_search(query_embedding, limit=NUM_RESULTS)

    return search


def create_hybrid_search(text_search_fn: Callable, vector_search_fn: Callable, alpha: float = 0.5) -> Callable:
    """Create a hybrid search function with configurable alpha."""
    def normalize_scores(results: List[Dict]) -> List[Dict]:
        if not results:
            return results
        scores = [r.get("_score", 0) for r in results]
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            for r in results:
                r["_score_normalized"] = 1.0
            return results
        for r in results:
            r["_score_normalized"] = (r.get("_score", 0) - min_s) / (max_s - min_s)
        return results

    def search(query: str) -> List[Dict]:
        text_results = text_search_fn(query)
        vector_results = vector_search_fn(query)

        # Normalize scores
        text_results = normalize_scores([{**r} for r in text_results])
        vector_results = normalize_scores([{**r} for r in vector_results])

        # Build lookups
        text_by_id = {r["id"]: r for r in text_results}
        vector_by_id = {r["id"]: r for r in vector_results}

        all_ids = set(list(text_by_id.keys()) + list(vector_by_id.keys()))

        combined = []
        for doc_id in all_ids:
            text_score = text_by_id.get(doc_id, {}).get("_score_normalized", 0)
            vector_score = vector_by_id.get(doc_id, {}).get("_score_normalized", 0)
            hybrid_score = alpha * text_score + (1 - alpha) * vector_score

            doc = text_by_id.get(doc_id) or vector_by_id.get(doc_id)
            if doc:
                result = {**doc}
                result["_text_score"] = text_score
                result["_vector_score"] = vector_score
                result["_score"] = hybrid_score
                combined.append(result)

        combined.sort(key=lambda x: x["_score"], reverse=True)
        return combined[:NUM_RESULTS]

    return search


# ── Evaluation Metrics ───────────────────────────────────────────────────────────

def hit_rate(relevance_total: List[List[bool]]) -> float:
    """
    Hit Rate: Fraction of queries where at least one relevant document was retrieved.
    """
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt += 1
    return cnt / len(relevance_total)


def mrr(relevance_total: List[List[bool]]) -> float:
    """
    Mean Reciprocal Rank: Average of reciprocal ranks of the first relevant document.
    """
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank]:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate(ground_truth: List[Dict], search_function: Callable) -> Dict[str, float]:
    """
    Evaluate a search function against ground truth data.
    
    Returns:
        dict with hit_rate and mrr
    """
    relevance_total = []

    for q in tqdm(ground_truth, desc="  Evaluating"):
        doc_id = q["id"]
        results = search_function(q["question"])
        relevance = [d["id"] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


# ── Boost Optimization (Random Search) ──────────────────────────────────────────

def simple_optimize(
    param_ranges: Dict[str, tuple],
    objective_function: Callable,
    n_iterations: int = N_ITERATIONS,
) -> tuple:
    """
    Random search over boost parameters to maximize Hit Rate.
    
    Returns:
        tuple: (best_params, best_score)
    """
    best_params = None
    best_score = float("-inf")

    for i in range(n_iterations):
        current_params = {}
        for field, (low, high) in param_ranges.items():
            current_params[field] = random.uniform(low, high)

        current_score = objective_function(current_params)

        if current_score > best_score:
            best_score = current_score
            best_params = current_params

        if (i + 1) % 10 == 0:
            print(f"      Iteration {i+1}/{n_iterations}: best Hit Rate = {best_score:.4f}")

    return best_params, best_score


# ── Main Evaluation Pipeline ─────────────────────────────────────────────────────

def run_evaluation():
    """Run the full evaluation pipeline."""
    print("=" * 70)
    print("🔬 RETRIEVAL EVALUATION")
    print("=" * 70)

    # Step 1: Load data
    print("\n📂 Step 1: Loading data...")
    chunks = load_chunks()
    ground_truth = load_ground_truth()
    print(f"   Loaded {len(chunks)} chunks, {len(ground_truth)} ground truth questions")

    # Step 2: Build text index
    print("\n📚 Step 2: Building text index...")
    text_index, indexed_chunks = build_text_index(chunks)
    print(f"   Built TF-IDF index with {len(indexed_chunks)} documents")

    # Step 3: Check vector store availability
    print("\n💾 Step 3: Checking vector store...")
    vector_available = build_vector_store(chunks)

    # Step 4: Split ground truth into validation and test sets
    print("\n✂️  Step 4: Splitting ground truth...")
    random.seed(42)
    shuffled = ground_truth.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.8)
    gt_validation = shuffled[:split_idx]
    gt_test = shuffled[split_idx:]

    print(f"   Validation set: {len(gt_validation)} questions")
    print(f"   Test set:       {len(gt_test)} questions")

    results = []

    # ── Approach A: Text-only search ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🔍 APPROACH A: Text-only Search (TF-IDF)")
    print("=" * 70)

    # A.1: Evaluate with default boosts
    print("\n📊 A.1: Evaluating with default boost parameters...")
    text_search_default = create_text_search(text_index)
    default_results = evaluate(gt_test, text_search_default)
    print(f"   Default: Hit Rate = {default_results['hit_rate']:.4f}, MRR = {default_results['mrr']:.4f}")

    results.append({
        "approach": "text-default",
        "hit_rate": round(default_results["hit_rate"], 4),
        "mrr": round(default_results["mrr"], 4),
        "alpha": 1.0,
        "boost_params": json.dumps({
            "title": 3.0, "heading_path": 2.0, "content": 1.0, "section": 2.5,
        }),
        "note": "Default boost values (no optimization)",
    })

    # A.2: Optimize boost parameters via random search
    print("\n🎯 A.2: Optimizing boost parameters with random search...")

    def objective_text(boost_params):
        search_fn = create_text_search(text_index, default_boost=boost_params)
        val_results = evaluate(gt_validation, search_fn)
        return val_results["hit_rate"]

    best_boosts, best_boost_score = simple_optimize(BOOST_RANGES, objective_text)
    print(f"\n   Best boost params: {best_boosts}")
    print(f"   Best validation Hit Rate: {best_boost_score:.4f}")

    # A.3: Evaluate optimized text search on test set
    print("\n📊 A.3: Evaluating optimized text search on test set...")
    text_search_optimized = create_text_search(text_index, default_boost=best_boosts)
    text_optimized_results = evaluate(gt_test, text_search_optimized)
    print(f"   Optimized: Hit Rate = {text_optimized_results['hit_rate']:.4f}, MRR = {text_optimized_results['mrr']:.4f}")

    results.append({
        "approach": "text-optimized",
        "hit_rate": round(text_optimized_results["hit_rate"], 4),
        "mrr": round(text_optimized_results["mrr"], 4),
        "alpha": 1.0,
        "boost_params": json.dumps({k: round(v, 2) for k, v in best_boosts.items()}),
        "note": "Boost params optimized via random search",
    })

    # ── Approach B & C: Vector and Hybrid (only if available) ──────────────────
    if vector_available:
        # ── Approach B: Vector-only search ────────────────────────────────────
        print("\n" + "=" * 70)
        print("🔍 APPROACH B: Vector-only Search (pgvector cosine similarity)")
        print("=" * 70)

        print("\n📊 B.1: Evaluating vector-only search on test set...")
        vector_search_fn = create_vector_search()

        # Evaluate with alpha=0.0 using hybrid function (which calls vector only)
        def vector_only_search(query):
            return vector_search_fn(query)

        vector_results = evaluate(gt_test, vector_only_search)
        print(f"   Vector-only: Hit Rate = {vector_results['hit_rate']:.4f}, MRR = {vector_results['mrr']:.4f}")

        results.append({
            "approach": "vector-only",
            "hit_rate": round(vector_results["hit_rate"], 4),
            "mrr": round(vector_results["mrr"], 4),
            "alpha": 0.0,
            "boost_params": "{}",
            "note": "pgvector cosine similarity with all-MiniLM-L6-v2 embeddings",
        })

        # ── Approach C: Hybrid search with alpha optimization ──────────────────
        print("\n" + "=" * 70)
        print("🔍 APPROACH C: Hybrid Search (TF-IDF + pgvector)")
        print("=" * 70)

        # Use optimized boosts for text part
        def text_search_with_optimized_boost(query):
            return text_search_optimized(query)

        print("\n🎯 C.1: Optimizing alpha via grid search on validation set...")
        best_alpha = 0.5
        best_alpha_score = 0.0

        print(f"   Testing alpha values: {ALPHA_VALUES}")
        alpha_scores = []

        for alpha in ALPHA_VALUES:
            hybrid_search_fn = create_hybrid_search(
                text_search_with_optimized_boost,
                vector_search_fn,
                alpha=alpha,
            )
            val_results = evaluate(gt_validation, hybrid_search_fn)
            alpha_scores.append((alpha, val_results["hit_rate"]))
            print(f"      alpha={alpha:.1f}: Hit Rate = {val_results['hit_rate']:.4f}")

            if val_results["hit_rate"] > best_alpha_score:
                best_alpha_score = val_results["hit_rate"]
                best_alpha = alpha

        print(f"\n   Best alpha: {best_alpha} (Hit Rate = {best_alpha_score:.4f})")

        # C.2: Evaluate best hybrid on test set
        print("\n📊 C.2: Evaluating best hybrid search on test set...")
        best_hybrid_search = create_hybrid_search(
            text_search_with_optimized_boost,
            vector_search_fn,
            alpha=best_alpha,
        )
        hybrid_results = evaluate(gt_test, best_hybrid_search)
        print(f"   Hybrid (alpha={best_alpha}): Hit Rate = {hybrid_results['hit_rate']:.4f}, MRR = {hybrid_results['mrr']:.4f}")

        results.append({
            "approach": f"hybrid-alpha-{best_alpha}",
            "hit_rate": round(hybrid_results["hit_rate"], 4),
            "mrr": round(hybrid_results["mrr"], 4),
            "alpha": best_alpha,
            "boost_params": json.dumps({k: round(v, 2) for k, v in best_boosts.items()}),
            "note": f"Alpha={best_alpha} via grid search, boosts from random search",
        })

        # Save best params for production
        best_params = {
            "boost": {k: round(v, 2) for k, v in best_boosts.items()},
            "alpha": best_alpha,
            "num_results": NUM_RESULTS,
        }
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n   ✅ Best params saved to {BEST_PARAMS_PATH}")

    else:
        # Vector search unavailable — still produce results for text-only
        print("\n" + "=" * 70)
        print("⚠️  Vector search unavailable — skipping Approaches B and C")
        print("=" * 70)
        print("\n   To enable hybrid evaluation:")
        print("   1. Start PostgreSQL: docker compose up -d")
        print("   2. Run ingestion: uv run python -c \"from app.ingest import run_pipeline; run_pipeline()\"")
        print("   3. Re-run this script")

        # Save text-only params
        best_params = {
            "boost": {k: round(v, 2) for k, v in best_boosts.items()},
            "alpha": 1.0,
            "num_results": NUM_RESULTS,
        }
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n   ✅ Best text-only params saved to {BEST_PARAMS_PATH}")

    # ── Save final results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("💾 Saving results...")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n   Results saved to {RESULTS_PATH}")

    # ── Display summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Approach':<30} {'Hit Rate':<12} {'MRR':<12}")
    print("-" * 54)
    for r in results:
        print(f"{r['approach']:<30} {r['hit_rate']:<12.4f} {r['mrr']:<12.4f}")

    # Determine best approach
    best = max(results, key=lambda r: r["hit_rate"])
    print(f"\n🏆 Best approach: {best['approach']}")
    print(f"   Hit Rate: {best['hit_rate']:.4f}")
    print(f"   MRR: {best['mrr']:.4f}")

    if vector_available and best['approach'].startswith('hybrid'):
        print(f"   ✅ Hybrid search outperforms individual approaches!")
        print(f"   🔧 Use alpha={best['alpha']} and optimized boosts in production")

    print("\n" + "=" * 70)
    print("✅ Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()