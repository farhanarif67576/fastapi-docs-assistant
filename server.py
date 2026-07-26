"""
FastAPI server for the FastAPI Docs Assistant RAG application.
Self-referential — built with the framework being documented!
"""

import json
import uuid
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag import rag
from app import db


# ── Load Best Params ────────────────────────────────────────────────────────────

BEST_PARAMS_PATH = os.getenv("BEST_PARAMS_PATH", "data/best-params.json")
try:
    with open(BEST_PARAMS_PATH, "r") as f:
        best_params = json.load(f)
    ALPHA = best_params.get("alpha", 0.4)
    print(f"✅ Loaded best params: alpha={ALPHA}")
except FileNotFoundError:
    ALPHA = 0.4
    print(f"⚠️  best-params.json not found at {BEST_PARAMS_PATH}, using default alpha={ALPHA}")

# Production config (from evaluation results)
PRODUCTION_PROMPT_STYLE = "detailed"
PRODUCTION_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


# ── FastAPI App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FastAPI Docs Assistant",
    description="A RAG-powered assistant that answers questions about FastAPI documentation using DeepSeek LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend dev server and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:8080",   # Production frontend (Docker)
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ─────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The FastAPI question to answer")
    model: Optional[str] = Field(None, description="Override the default model")
    prompt_style: Optional[str] = Field(None, description="'concise' or 'detailed'")


class QuestionResponse(BaseModel):
    conversation_id: str
    question: str
    answer: str
    model_used: str
    response_time: float
    relevance: str
    token_usage: dict
    retrieved_chunks: list


class FeedbackRequest(BaseModel):
    conversation_id: str = Field(..., description="The conversation ID from /question response")
    feedback: int = Field(..., ge=-1, le=1, description="1 for thumbs up, -1 for thumbs down")
    # Ensure feedback is exactly 1 or -1, not 0
    @classmethod
    def validate_feedback(cls, v):
        if v not in [1, -1]:
            raise ValueError("feedback must be 1 (up) or -1 (down)")
        return v


class FeedbackResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    model: str
    alpha: float
    prompt_style: str


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model=PRODUCTION_MODEL,
        alpha=ALPHA,
        prompt_style=PRODUCTION_PROMPT_STYLE,
    )


@app.post("/question", response_model=QuestionResponse)
async def handle_question(request: QuestionRequest):
    """
    Ask a question about FastAPI and get an answer grounded in the documentation.
    
    The RAG pipeline:
    1. Hybrid search over FastAPI docs (TF-IDF + pgvector)
    2. Build prompt with retrieved context
    3. Call DeepSeek LLM for answer generation
    4. Evaluate relevance with LLM-as-a-Judge
    5. Log conversation to PostgreSQL
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    conversation_id = str(uuid.uuid4())
    
    # Use provided values or fall back to production defaults
    model = request.model or PRODUCTION_MODEL
    prompt_style = request.prompt_style or PRODUCTION_PROMPT_STYLE

    try:
        # Call the RAG pipeline
        answer_data = rag(
            query=question,
            model=model,
            alpha=ALPHA,
            prompt_style=prompt_style,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

    # Save conversation to database (non-blocking error handling)
    try:
        db.save_conversation(
            conversation_id=conversation_id,
            question=question,
            answer_data=answer_data,
        )
    except Exception as e:
        # Log the error but don't fail the request
        print(f"⚠️  Warning: Could not save conversation to PostgreSQL: {e}")

    # Build the response
    return QuestionResponse(
        conversation_id=conversation_id,
        question=question,
        answer=answer_data["answer"],
        model_used=answer_data["model_used"],
        response_time=round(answer_data["response_time"], 3),
        relevance=answer_data["relevance"],
        token_usage={
            "prompt_tokens": answer_data["prompt_tokens"],
            "completion_tokens": answer_data["completion_tokens"],
            "total_tokens": answer_data["total_tokens"],
        },
        retrieved_chunks=answer_data.get("retrieved_chunks", []),
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def handle_feedback(request: FeedbackRequest):
    """
    Submit feedback (thumbs up/down) for a previous answer.
    
    This helps monitor answer quality and improve the system over time.
    """
    if request.feedback not in [1, -1]:
        raise HTTPException(
            status_code=400,
            detail="feedback must be 1 (thumbs up) or -1 (thumbs down)",
        )

    try:
        db.save_feedback(
            conversation_id=request.conversation_id,
            feedback=request.feedback,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save feedback: {str(e)}",
        )

    return FeedbackResponse(
        message=f"Feedback received: {'👍' if request.feedback == 1 else '👎'}",
    )


# ── Startup / Shutdown Events ───────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Initialize database on startup (doesn't crash if PG is unavailable)."""
    # Step 1: Try to initialize database tables
    print(f"\n{'='*60}")
    print(f"🚀 FastAPI Docs Assistant API")
    print(f"{'='*60}")
    print(f"   Model: {PRODUCTION_MODEL}")
    print(f"   Alpha: {ALPHA}")
    print(f"   Prompt style: {PRODUCTION_PROMPT_STYLE}")
    
    try:
        db.init_db()
        print("   ✅ Database tables initialized")
    except Exception as e:
        print(f"   ⚠️  Database not available: {e}")
        print("   ℹ️  Conversations won't be saved until PostgreSQL is ready")
    
    print(f"   API docs: http://localhost:8000/docs")
    print(f"   ReDoc:    http://localhost:8000/redoc")
    print(f"{'='*60}\n")


# ── Main Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
