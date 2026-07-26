"""
Script: 05-query-rewriting-evaluation.py
Purpose: Evaluate the impact of LLM-based query rewriting on retrieval quality.
         Compares Hit Rate and MRR with and without query rewriting.

Output: data/query-rewriting-results.csv

Usage:
    uv run python notebooks/05-query-rewriting-evaluation.py

Prerequisites:
    - PostgreSQL with pgvector running
    - Document chunks embedded and stored
    - ground-truth-retrieval.csv in data/
    - DeepSeek API key in .env
"""

import os
import sys
import json
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

from app.rag import hybrid_search, rewrite_query, search_only
from app.re_ranker import get_reranker
from app.vector_store import get_embedder
from app import db

# ── Config ─────────────────────────────────────────────────────────────────────

GROUND_TRUTH_PATH = "data/ground-truth-retrieval.csv"
RESULTS_PATH = "data/query-rewriting-results.csv"
BEST_PARAMS_PATH = "data/best-params.json"

# Use a sample for evaluation to keep API costs low
SAMPLE_SIZE = 20
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


def evaluate(ground_truth, search_function):
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


# ── Main ───────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 70)
    print("🔬 QUERY REWRITING EVALUATION")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    gt_full = pd.read_csv(GROUND_TRUTH_PATH)
    
    # Take a representative sample
    random.seed(42)
    sample = gt_full.groupby("id").first().reset_index()
    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(n=SAMPLE_SIZE, random_state=42)
    
    ground_truth = sample.to_dict(orient="records")
    print(f"   Sample: {len(ground_truth)} questions")
    
    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)
    print(f"   Best alpha: {best_params['alpha']}")
    
    results = []
    
    # ── 1. Hybrid Search (no rewrite) ────────────────────────────────────────
    print("\n📊 1. Evaluating hybrid search (NO query rewriting)...")
    
    def search_no_rewrite(query):
        return hybrid_search(query, alpha=best_params.get("alpha", 0.4))
    
    no_rewrite_metrics = evaluate(ground_truth, search_no_rewrite)
    print(f"   Hit Rate: {no_rewrite_metrics['hit_rate']:.4f}, MRR: {no_rewrite_metrics['mrr']:.4f}")
    
    results.append({
        "approach": "no-rewriting",
        "hit_rate": round(no_rewrite_metrics["hit_rate"], 4),
        "mrr": round(no_rewrite_metrics["mrr"], 4),
        "description": "Hybrid search with original user query",
    })
    
    # ── 2. Hybrid Search + Query Rewriting ───────────────────────────────────
    print("\n📊 2. Evaluating hybrid search WITH query rewriting...")
    print("   (This calls DeepSeek API to rewrite each query)")
    
    # Pre-compute rewritten queries
    rewritten_queries = {}
    print("\n   Step 2a: Rewriting queries using DeepSeek...")
    for q in tqdm(ground_truth, desc="  Rewriting"):
        original = q["question"]
        rewritten = rewrite_query(original)
        rewritten_queries[original] = rewritten
    
    # Show examples
    print("\n   Example rewrites:")
    for i, q in enumerate(ground_truth[:5]):
        original = q["question"]
        rewritten = rewritten_queries[original]
        print(f"      [{i+1}] '{original}'")
        print(f"           → '{rewritten}'")
    
    # Search with rewritten queries
    def search_with_rewrite(query):
        rewritten = rewritten_queries.get(query, query)
        return hybrid_search(rewritten, alpha=best_params.get("alpha", 0.4))
    
    print("\n   Step 2b: Evaluating with rewritten queries...")
    rewrite_metrics = evaluate(ground_truth, search_with_rewrite)
    print(f"   Hit Rate: {rewrite_metrics['hit_rate']:.4f}, MRR: {rewrite_metrics['mrr']:.4f}")
    
    results.append({
        "approach": "with-rewriting",
        "hit_rate": round(rewrite_metrics["hit_rate"], 4),
        "mrr": round(rewrite_metrics["mrr"], 4),
        "description": "Hybrid search with LLM-rewritten queries",
    })
    
    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Approach':<25} {'Hit Rate':<12} {'MRR':<12}")
    print("-" * 49)
    for r in results:
        print(f"{r['approach']:<25} {r['hit_rate']:<12.4f} {r['mrr']:<12.4f}")
    
    improvement_hr = results[1]['hit_rate'] - results[0]['hit_rate']
    improvement_mrr = results[1]['mrr'] - results[0]['mrr']
    
    print(f"\n📈 Improvement from query rewriting:")
    print(f"   Hit Rate: +{improvement_hr:.4f} ({improvement_hr/results[0]['hit_rate']*100:+.1f}%)")
    print(f"   MRR:      +{improvement_mrr:.4f} ({improvement_mrr/results[0]['mrr']*100:+.1f}%)")
    
    # Save results
    print(f"\n💾 Saving to {RESULTS_PATH}...")
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print("✅ Done!")


if __name__ == "__main__":
    run_evaluation()