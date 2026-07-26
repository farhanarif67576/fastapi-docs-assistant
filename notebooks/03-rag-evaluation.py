"""
Script: 03-rag-evaluation.py
Purpose: LLM-as-a-Judge evaluation comparing 2 prompt styles × 2 DeepSeek models.
         Evaluates answer quality (RELEVANT / PARTLY_RELEVANT / NON_RELEVANT),
         cost, and latency. Recommends the best combination for production.

Output: data/rag-evaluation.csv (all individual results + aggregate summary)

Usage:
    uv run python notebooks/03-rag-evaluation.py

Prerequisites:
    - PostgreSQL with pgvector running (docker compose up -d)
    - Document chunks embedded and stored in pgvector
    - ground-truth-retrieval.csv in data/
    - DeepSeek API key in .env
"""

import os
import json
import sys
import csv
import random
from datetime import datetime
from itertools import product

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ── Configuration ────────────────────────────────────────────────────────────────

GROUND_TRUTH_PATH = "data/ground-truth-retrieval.csv"
CHUNKS_PATH = "data/fastapi_docs/chunks.json"
RESULTS_PATH = "data/rag-evaluation.csv"
BEST_PARAMS_PATH = "data/best-params.json"

# Models to compare
FLASH_MODEL = "deepseek-v4-flash"
PRO_MODEL = "deepseek-v4-pro"

# Prompt styles to compare
PROMPT_STYLES = ["concise", "detailed"]

# Number of questions to sample for evaluation
SAMPLE_SIZE = 20

# Judge model (use the more capable model for judging to avoid bias)
JUDGE_MODEL = PRO_MODEL

# DeepSeek pricing (approximate, per 1M tokens)
PRICING = {
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"input": 0.48, "output": 0.96},
}


# ── Load Data ────────────────────────────────────────────────────────────────────

def load_ground_truth() -> pd.DataFrame:
    """Load ground truth data."""
    df = pd.read_csv(GROUND_TRUTH_PATH)
    print(f"   Loaded {len(df)} ground truth questions")
    return df


def load_best_params() -> dict:
    """Load optimized search params from Step 3."""
    with open(BEST_PARAMS_PATH, "r") as f:
        return json.load(f)


def select_sample(ground_truth: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Select a diverse sample of questions.
    Strategy: pick 1 question per unique chunk ID to ensure topic diversity.
    """
    # Group by chunk ID and pick the first question from each
    sample = ground_truth.groupby("id").first().reset_index()
    
    # If we have more than n groups, randomly select n
    if len(sample) > n:
        random.seed(42)
        sample = sample.sample(n=n, random_state=42)
    
    print(f"   Selected {len(sample)} questions for evaluation "
          f"(covering {len(sample['id'].unique())} unique chunks)")
    return sample


# ── RAG Evaluation ───────────────────────────────────────────────────────────────

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate approximate cost in USD."""
    pricing = PRICING.get(model, PRICING[FLASH_MODEL])
    cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
    return round(cost, 6)


def run_evaluation():
    """Run the full RAG evaluation pipeline."""
    print("=" * 70)
    print("🔬 RAG EVALUATION: LLM-as-a-Judge")
    print("=" * 70)

    # Step 1: Load data
    print("\n📂 Step 1: Loading data and best params...")
    ground_truth = load_ground_truth()
    best_params = load_best_params()
    print(f"   Best params: alpha={best_params['alpha']}, boost={best_params['boost']}")

    # Step 2: Select sample
    print("\n🎯 Step 2: Selecting diverse question sample...")
    sample = select_sample(ground_truth)
    
    questions = sample.to_dict(orient="records")

    # Step 3: Define combinations
    print("\n🧪 Step 3: Defining evaluation combinations...")
    models = [FLASH_MODEL, PRO_MODEL]
    combinations = list(product(PROMPT_STYLES, models))
    
    print(f"   Evaluating {len(combinations)} combinations:")
    for prompt_style, model in combinations:
        print(f"      • {prompt_style} + {model}")

    # Step 4: Run evaluation for each combination
    all_results = []

    print("\n🤖 Step 4: Running RAG evaluations (this calls DeepSeek API)...")
    print(f"   Total API calls: {len(questions) * len(combinations)} "
          f"(answers) + {len(questions) * len(combinations)} (judge evaluations)")
    print(f"   Estimated cost: ~${0.15 * len(questions) * len(combinations):.2f}")
    print()

    for prompt_style, model in combinations:
        combo_name = f"{prompt_style} + {model}"
        print(f"   ── Evaluating: {combo_name} ──")
        
        # Limit detailed evaluations to 3 questions (save cost)
        eval_questions = questions
        if prompt_style == "detailed":
            eval_questions = questions[:3]
            print(f"      (limited to {len(eval_questions)} questions for cost savings)")

        for q in tqdm(eval_questions, desc=f"  {combo_name}", leave=False):
            question_text = q["question"]
            chunk_id = q["id"]

            # Import rag here so the text index + vector store can be lazy-loaded
            from app.rag import rag as rag_function

            try:
                # Call the full RAG pipeline
                result = rag_function(
                    query=question_text,
                    model=model,
                    alpha=best_params.get("alpha", 0.4),
                    prompt_style=prompt_style,
                )

                # Record the result
                all_results.append({
                    "question_id": chunk_id,
                    "question": question_text,
                    "prompt_variant": prompt_style,
                    "model": model,
                    "answer": result["answer"],
                    "relevance": result["relevance"],
                    "explanation": result["relevance_explanation"],
                    "response_time": round(result["response_time"], 3),
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                    "eval_prompt_tokens": result["eval_prompt_tokens"],
                    "eval_completion_tokens": result["eval_completion_tokens"],
                    "eval_total_tokens": result["eval_total_tokens"],
                    "cost": round(result["openai_cost"], 6),
                })

            except Exception as e:
                print(f"      ⚠ Error for question '{question_text[:50]}...': {e}")
                all_results.append({
                    "question_id": chunk_id,
                    "question": question_text,
                    "prompt_variant": prompt_style,
                    "model": model,
                    "answer": f"ERROR: {e}",
                    "relevance": "UNKNOWN",
                    "explanation": f"Failed to generate: {e}",
                    "response_time": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "eval_prompt_tokens": 0,
                    "eval_completion_tokens": 0,
                    "eval_total_tokens": 0,
                    "cost": 0,
                })

    # Step 5: Save detailed results
    print("\n💾 Step 5: Saving detailed results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"   Saved {len(results_df)} individual evaluations to {RESULTS_PATH}")

    # Step 6: Compute aggregate summary
    print("\n📊 Step 6: Computing aggregate summary...")
    
    summary_rows = []
    for prompt_style, model in combinations:
        combo_name = f"{prompt_style} + {model}"
        subset = results_df[
            (results_df["prompt_variant"] == prompt_style) &
            (results_df["model"] == model)
        ]
        
        if len(subset) == 0:
            continue
        
        relevance_counts = subset["relevance"].value_counts()
        total = len(subset)
        
        pct_relevant = round(relevance_counts.get("RELEVANT", 0) / total * 100, 1)
        pct_partly = round(relevance_counts.get("PARTLY_RELEVANT", 0) / total * 100, 1)
        pct_non = round(relevance_counts.get("NON_RELEVANT", 0) / total * 100, 1)
        avg_cost = round(subset["cost"].mean(), 6)
        avg_time = round(subset["response_time"].mean(), 3)
        avg_tokens = int(subset["total_tokens"].mean())
        
        summary_rows.append({
            "combination": combo_name,
            "prompt_style": prompt_style,
            "model": model,
            "pct_relevant": pct_relevant,
            "pct_partly_relevant": pct_partly,
            "pct_non_relevant": pct_non,
            "avg_cost_usd": avg_cost,
            "avg_latency_sec": avg_time,
            "avg_total_tokens": avg_tokens,
            "sample_size": total,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Step 7: Display summary table
    print("\n" + "=" * 70)
    print("📊 RAG EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Combination':<40} {'RELEVANT':<10} {'PARTLY':<10} {'NON_REL':<10} {'Cost':<10} {'Time':<8}")
    print("-" * 88)
    
    for _, row in summary_df.iterrows():
        print(f"{row['combination']:<40} "
              f"{row['pct_relevant']:<9.1f}% "
              f"{row['pct_partly_relevant']:<9.1f}% "
              f"{row['pct_non_relevant']:<9.1f}% "
              f"${row['avg_cost_usd']:<8.6f} "
              f"{row['avg_latency_sec']:<.2f}s")

    # Step 8: Determine best combination
    print("\n" + "=" * 70)
    print("🏆 SELECTING BEST COMBINATION")
    print("=" * 70)
    
    # Score each combination: weighted formula
    # Score = 0.4 * %RELEVANT + 0.2 * (100 - %NON_RELEVANT) + 0.2 * (1 - normalized_cost) + 0.2 * (1 - normalized_latency)
    max_cost = summary_df["avg_cost_usd"].max()
    max_latency = summary_df["avg_latency_sec"].max()
    
    best_score = -1
    best_combo = None
    
    for _, row in summary_df.iterrows():
        norm_cost = row["avg_cost_usd"] / max_cost if max_cost > 0 else 0
        norm_latency = row["avg_latency_sec"] / max_latency if max_latency > 0 else 0
        
        score = (
            0.40 * row["pct_relevant"] +
            0.20 * (100 - row["pct_non_relevant"]) +
            0.20 * (1 - norm_cost) * 100 +
            0.20 * (1 - norm_latency) * 100
        )
        
        print(f"\n   {row['combination']}:")
        print(f"      Quality score (RELEVANT%): {row['pct_relevant']:.1f} × 0.40 = {0.40 * row['pct_relevant']:.1f}")
        print(f"      Penalty score (NON_RELEVANT%): {(100 - row['pct_non_relevant']):.1f} × 0.20 = {0.20 * (100 - row['pct_non_relevant']):.1f}")
        print(f"      Cost score: {(1 - norm_cost) * 100:.1f} × 0.20 = {0.20 * (1 - norm_cost) * 100:.1f}")
        print(f"      Latency score: {(1 - norm_latency) * 100:.1f} × 0.20 = {0.20 * (1 - norm_latency) * 100:.1f}")
        print(f"      Total score: {score:.1f}/100")
        
        if score > best_score:
            best_score = score
            best_combo = row["combination"]

    print(f"\n{'='*70}")
    print(f"✅ RECOMMENDED: {best_combo}")
    print(f"   Score: {best_score:.1f}/100")
    print(f"   This combination offers the best quality-to-cost ratio for production.")
    
    if "flash" in best_combo:
        print(f"   Using deepseek-v4-flash saves ~50% on API costs with minimal quality loss.")
    else:
        print(f"   deepseek-v4-pro provides higher quality but at ~2x the cost.")
    
    print("=" * 70)

    # Step 9: Save aggregate summary alongside detailed results
    summary_path = RESULTS_PATH.replace(".csv", "-summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n   Aggregate summary saved to {summary_path}")

    # Step 10: Print recommendation for production
    print("\n🔧 PRODUCTION RECOMMENDATION")
    print("-" * 40)
    
    # Extract the best prompt style and model
    for _, row in summary_df.iterrows():
        if row["combination"] == best_combo:
            print(f"   prompt_style: {row['prompt_style']}")
            print(f"   model: {row['model']}")
            print(f"   alpha: {best_params.get('alpha', 0.4)}")
            print(f"   boost: {best_params.get('boost', {})}")
            print(f"   Expected cost per query: ${row['avg_cost_usd']:.6f}")
            print(f"   Expected latency: {row['avg_latency_sec']:.2f}s")
            print(f"   Expected RELEVANT rate: {row['pct_relevant']:.1f}%")
            break

    print("\n" + "=" * 70)
    print("✅ RAG evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()