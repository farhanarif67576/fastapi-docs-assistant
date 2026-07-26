"""
Script: 04-re-ranking-evaluation.py
Purpose: Evaluate the impact of cross-encoder re-ranking on retrieval quality.
         Compares Hit Rate and MRR with and without re-ranking.

Output: data/re-ranking-results.csv

Usage:
    uv run python notebooks/04-re-ranking-evaluation.py

Prerequisites:
    - PostgreSQL with pgvector running
    - Document chunks embedded and stored
    - ground-truth-retrieval.csv in data/
"""

import os
import sys
import json
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm.auto import tqdm

from app.rag import hybrid_search, minsearch_search, vector_search
from app.re_ranker import get_reranker
from app.vector_store import get_embedder
from app import db

# ── Config ─────────────────────────────────────────────────────────────────────

GROUND_TRUTH_PATH = "data/ground-truth-retrieval.csv"
RESULTS_PATH = "data/re-ranking-results.csv"
BEST_PARAMS_PATH = "data/best-params.json"
NUM_RESULTS = 10


# ── Metrics ────────────────────────────────────────────────────────────────────

def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt += 1
    return cnt / len(relevance_total)


def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank]:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate(ground_truth, search_function, rerank=False, reranker=None):
    relevance_total = []
    for q in tqdm(ground_truth, desc=f"  Evaluating{' + rerank' if rerank else ''}"):
        doc_id = q["id"]
        query = q["question"]
        
        # Get search results
        results = search_function(query)
        
        # Apply re-ranking if enabled
        if rerank and reranker and results:
            results = reranker.rerank(query, results, top_k=NUM_RESULTS)
        
        relevance = [d["id"] == doc_id for d in results]
        relevance_total.append(relevance)
    
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 70)
    print("🔬 RE-RANKING EVALUATION")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH).to_dict(orient="records")
    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)
    
    print(f"   Questions: {len(ground_truth)}")
    print(f"   Best alpha: {best_params['alpha']}")
    
    # Load reranker (this will download the model if not cached)
    print("\n🤖 Loading cross-encoder re-ranker...")
    reranker = get_reranker()
    
    # Define search functions
    def hybrid_search_fn(query):
        return hybrid_search(query, alpha=best_params.get("alpha", 0.4))
    
    results = []
    
    # ── 1. Hybrid Search (no rerank) ──────────────────────────────────────────
    print("\n📊 1. Evaluating hybrid search (NO re-ranking)...")
    hybrid_metrics = evaluate(ground_truth, hybrid_search_fn, rerank=False)
    print(f"   Hit Rate: {hybrid_metrics['hit_rate']:.4f}, MRR: {hybrid_metrics['mrr']:.4f}")
    
    results.append({
        "approach": "hybrid-only",
        "hit_rate": round(hybrid_metrics["hit_rate"], 4),
        "mrr": round(hybrid_metrics["mrr"], 4),
        "description": "Hybrid search (TF-IDF + pgvector, no re-ranking)",
    })
    
    # ── 2. Hybrid Search + Re-ranking ────────────────────────────────────────
    print("\n📊 2. Evaluating hybrid search WITH re-ranking...")
    rerank_metrics = evaluate(ground_truth, hybrid_search_fn, rerank=True, reranker=reranker)
    print(f"   Hit Rate: {rerank_metrics['hit_rate']:.4f}, MRR: {rerank_metrics['mrr']:.4f}")
    
    results.append({
        "approach": "hybrid-plus-rerank",
        "hit_rate": round(rerank_metrics["hit_rate"], 4),
        "mrr": round(rerank_metrics["mrr"], 4),
        "description": "Hybrid search + cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)",
    })
    
    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Approach':<35} {'Hit Rate':<12} {'MRR':<12}")
    print("-" * 59)
    for r in results:
        print(f"{r['approach']:<35} {r['hit_rate']:<12.4f} {r['mrr']:<12.4f}")
    
    improvement_hr = results[1]['hit_rate'] - results[0]['hit_rate']
    improvement_mrr = results[1]['mrr'] - results[0]['mrr']
    
    print(f"\n📈 Improvement from re-ranking:")
    print(f"   Hit Rate: +{improvement_hr:.4f} ({improvement_hr/results[0]['hit_rate']*100:+.1f}%)")
    print(f"   MRR:      +{improvement_mrr:.4f} ({improvement_mrr/results[0]['mrr']*100:+.1f}%)")
    
    # Save results
    print(f"\n💾 Saving to {RESULTS_PATH}...")
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print("✅ Done!")


if __name__ == "__main__":
    run_evaluation()