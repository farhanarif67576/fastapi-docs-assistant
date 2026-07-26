"""
Database functions for storing document chunks, conversations, and feedback in PostgreSQL with pgvector.
"""

import os
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from zoneinfo import ZoneInfo


TZ_INFO = os.getenv("TZ", "UTC")
tz = ZoneInfo(TZ_INFO)

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "fastapi_assistant")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")

VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings


def get_db_connection():
    """Get a PostgreSQL database connection."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def init_db():
    """Initialize the database schema with pgvector extension."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Drop existing tables for clean setup
            cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute("DROP TABLE IF EXISTS conversations")
            cur.execute("DROP TABLE IF EXISTS doc_chunks")

            # Document chunks table (vector storage)
            cur.execute(f"""
                CREATE TABLE doc_chunks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    heading_path TEXT[],
                    content TEXT NOT NULL,
                    url TEXT NOT NULL,
                    section TEXT,
                    embedding vector({VECTOR_DIMENSION})
                )
            """)

            # Create IVFFlat index for approximate nearest neighbor search
            cur.execute(f"""
                CREATE INDEX idx_doc_chunks_embedding 
                ON doc_chunks 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            # Conversations table
            cur.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    relevance TEXT NOT NULL,
                    relevance_explanation TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    eval_prompt_tokens INTEGER NOT NULL,
                    eval_completion_tokens INTEGER NOT NULL,
                    eval_total_tokens INTEGER NOT NULL,
                    openai_cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)

            # Feedback table
            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)

        conn.commit()
        print("Database initialized successfully: pgvector + tables created.")
    finally:
        conn.close()


def get_all_chunks():
    """Retrieve all document chunks (without embeddings) for building text index."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                "SELECT id, title, heading_path, content, url, section FROM doc_chunks"
            )
            rows = cur.fetchall()
            chunks = []
            for row in rows:
                chunks.append({
                    "id": row["id"],
                    "title": row["title"],
                    "heading_path": " ".join(row["heading_path"]) if row["heading_path"] else "",
                    "content": row["content"],
                    "url": row["url"],
                    "section": row["section"] or "",
                })
            return chunks
    finally:
        conn.close()


def save_chunk(chunk_id, title, heading_path, content, url, section, embedding):
    """Save a document chunk with its embedding vector."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO doc_chunks (id, title, heading_path, content, url, section, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (id) DO UPDATE
                SET title = EXCLUDED.title,
                    heading_path = EXCLUDED.heading_path,
                    content = EXCLUDED.content,
                    url = EXCLUDED.url,
                    section = EXCLUDED.section,
                    embedding = EXCLUDED.embedding
                """,
                (chunk_id, title, heading_path, content, url, section, embedding.tolist()),
            )
        conn.commit()
    finally:
        conn.close()


def vector_search(query_embedding, limit=10):
    """
    Search for the most similar chunks using cosine similarity.
    Returns list of dicts with similarity scores.
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, heading_path, content, url, section,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM doc_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding.tolist(), query_embedding.tolist(), limit),
            )
            rows = cur.fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "title": row["title"],
                    "heading_path": " ".join(row["heading_path"]) if row["heading_path"] else "",
                    "content": row["content"],
                    "url": row["url"],
                    "section": row["section"] or "",
                    "_score": float(row["similarity"]),
                })
            return results
    finally:
        conn.close()


# ── Conversation and Feedback Functions ──────────────────────────────────────

def save_conversation(conversation_id, question, answer_data, timestamp=None):
    """Save a conversation record to the database."""
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations
                (id, question, answer, model_used, response_time, relevance,
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens,
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens,
                openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    question,
                    answer_data["answer"],
                    answer_data["model_used"],
                    answer_data["response_time"],
                    answer_data["relevance"],
                    answer_data["relevance_explanation"],
                    answer_data["prompt_tokens"],
                    answer_data["completion_tokens"],
                    answer_data["total_tokens"],
                    answer_data["eval_prompt_tokens"],
                    answer_data["eval_completion_tokens"],
                    answer_data["eval_total_tokens"],
                    answer_data["openai_cost"],
                    timestamp,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def save_feedback(conversation_id, feedback, timestamp=None):
    """Save user feedback for a conversation."""
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, %s)",
                (conversation_id, feedback, timestamp),
            )
        conn.commit()
    finally:
        conn.close()