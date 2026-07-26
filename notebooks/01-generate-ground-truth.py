"""
Script: 01-generate-ground-truth.py
Purpose: Generate ground truth data for retrieval evaluation using DeepSeek LLM.
         For each chunk in the FastAPI documentation, we ask DeepSeek to generate
         questions that a developer might ask which would be answered by that chunk.

Output: data/ground-truth-retrieval.csv (question, id pairs)

Usage:
    uv run python notebooks/01-generate-ground-truth.py
"""

import os
import json
import csv
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()

# DeepSeek configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Initialize DeepSeek client (OpenAI-compatible)
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# ── Load chunks ──────────────────────────────────────────────────────────────────

CHUNKS_PATH = "data/fastapi_docs/chunks.json"

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

# ── Generate questions for each chunk ────────────────────────────────────────────

prompt_template = """
You are a developer learning FastAPI. For the documentation section below, generate {num_questions} questions 
that a developer might ask, where the answer can be found in this exact section.

Return the result as a JSON array of objects with 'question' fields only.
Do NOT include the 'id' field - we will add it separately.

Example output format:
["How do I create a path parameter?", "What is the syntax for path parameters?"]

Documentation Section:
Title: {title}
Section: {section}
Content: {content}
""".strip()


def generate_questions(chunk, num_questions=2):
    """Generate questions for a single documentation chunk."""
    prompt = prompt_template.format(
        title=chunk["title"],
        section=chunk["section"],
        content=chunk["content"][:1000],  # Truncate to avoid token limits
        num_questions=num_questions,
    )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )

        output_text = response.choices[0].message.content.strip()

        # Parse the JSON response
        if output_text.startswith("```"):
            output_text = output_text.split("\n", 1)[1]
        if output_text.endswith("```"):
            output_text = output_text.rsplit("\n", 1)[0]
        output_text = output_text.strip()

        questions = json.loads(output_text)
        return questions
    except Exception as e:
        print(f"  ⚠ Failed to generate questions for {chunk['id']}: {e}")
        return []


# Generate questions for all chunks
all_records = []

for chunk in tqdm(chunks, desc="Generating questions"):
    questions = generate_questions(chunk, num_questions=2)
    for q in questions:
        all_records.append({
            "id": chunk["id"],
            "question": q,
        })

# ── Save to CSV ──────────────────────────────────────────────────────────────────

OUTPUT_PATH = "data/ground-truth-retrieval.csv"

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "question"])
    writer.writeheader()
    writer.writerows(all_records)

print(f"\n✅ Generated {len(all_records)} question-answer pairs")
print(f"   Saved to {OUTPUT_PATH}")

# ── Summary statistics ───────────────────────────────────────────────────────────

chunks_with_questions = len(set(r["id"] for r in all_records))
print(f"\n📊 Summary:")
print(f"   Chunks used: {chunks_with_questions} / {len(chunks)}")
print(f"   Total questions: {len(all_records)}")
print(f"   Avg questions per chunk: {len(all_records) / chunks_with_questions:.1f}")