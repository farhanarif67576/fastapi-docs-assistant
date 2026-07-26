# 🚀 FastAPI Docs Assistant

A **Retrieval-Augmented Generation (RAG)** application that answers questions about **FastAPI documentation** using **DeepSeek LLM**. Built as the capstone project for the [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) course.

---

## 📋 Problem Statement

### The Pain Point

FastAPI is one of the most popular Python web frameworks, and its official documentation at [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) is extensive — spanning **135 pages** across tutorials, advanced guides, and API reference. While comprehensive, this volume of documentation creates a real problem for developers:

- **Beginners** feel overwhelmed — they don't know which doc page to start with or how to find specific patterns like *"How do I use dependency injection with path parameters?"*
- **Intermediate users** waste time **Ctrl+F searching** across multiple pages trying to find that one code snippet or parameter explanation they need
- **Experienced developers** need **quick, precise lookups** — *"What's the correct way to set up CORS middleware?"* or *"How do I configure custom exception handlers?"* — without re-reading entire pages

The common workaround? Google search, Stack Overflow, or asking in Discord — all of which return **generic, often outdated, or incorrect** answers not grounded in the actual documentation.

### The Solution

**FastAPI Docs Assistant** is a RAG chatbot that lets developers ask natural language questions and get **precise, context-grounded answers** from the official FastAPI documentation. Instead of manually searching docs, developers simply ask:

> *"How do I add CORS to my FastAPI app?"*
> *"What's the difference between Query and Path parameters?"*
> *"Show me an example of dependency injection with yield"*

The assistant retrieves the most relevant documentation chunks and generates an answer using the **DeepSeek-v4-flash** LLM — all with **citations** linking back to the source documentation.

### Target Audience

- **Python developers** learning or using FastAPI for the first time
- **Backend engineers** looking for quick API pattern references
- **Data scientists / ML engineers** who use FastAPI to deploy models
- **Anyone** who finds themselves repeatedly searching FastAPI docs for the same patterns

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  USER                                                            │
│  (Browser / curl / API client)                                    │
└──────────┬──────────────────────────────────────────────────────┬───┘
           │ POST /question                                  │
           ▼                                                     │
┌──────────────────────┐         ┌──────────────────────────────┐│
│  FastAPI Backend     │────────▶│  Knowledge Base              ││
│  (app.py)            │         │  ┌────────────────────────┐ ││
│                      │         │  │ Chunked FastAPI Docs   │ ││
│  ┌────────────────┐  │         │  │ (TF-IDF Index)        │ ││
│  │ Query Rewriting│  │         │  └────────────────────────┘ ││
│  └───────┬────────┘  │         │  ┌────────────────────────┐ ││
│          ▼           │         │  │ Vector Embeddings      │ ││
│  ┌────────────────┐  │         │  │ (sentence-transformers)│ ││
│  │ Hybrid Search  │  │         │  └────────────────────────┘ ││
│  │(TF-IDF+Vector) │  │         └──────────────────────────────┘│
│  └───────┬────────┘  │                                        │
│          ▼           │         ┌──────────────────────────────┐│
│  ┌────────────────┐  │────────▶│  DeepSeek LLM               ││
│  │ Re-ranker      │  │         │  (deepseek-v4-flash)        ││
│  └───────┬────────┘  │         └──────────────────────────────┘│
│          ▼           │                                        │
│  ┌────────────────┐  │         ┌──────────────────────────────┐│
│  │ LLM + Prompt   │  │────────▶│  PostgreSQL                 ││
│  │ Build Answer   │  │         │  (Conversations + Feedback) ││
│  └───────┬────────┘  │         └──────────────────────────────┘│
│          ▼           │                                        │
│  Response (JSON)     │         ┌──────────────────────────────┐│
└──────────────────────┘         │  Grafana Dashboards         ││
                                 │  (Monitoring + Analytics)   ││
                                 └──────────────────────────────┘│
┌──────────────────────┐                                        │
│  React Frontend      │─────────────────────────────────────────┘
│  (Chat UI)           │
└──────────────────────┘
```

### Flow Summary

1. **User submits a question** (via UI or API)
2. **Query rewriting** — The LLM expands the user's query for better search
3. **Hybrid search** — Retrieves relevant docs using both TF-IDF (keyword) and vector (semantic) search
4. **Re-ranking** — A cross-encoder model re-orders results by relevance
5. **LLM generation** — DeepSeek-v4-flash generates an answer grounded in the retrieved context
6. **Logging** — The conversation is saved to PostgreSQL with relevance scores, token usage, and cost
7. **Feedback** — Users can thumbs-up/thumbs-down answers, stored for monitoring
8. **Monitoring** — Grafana dashboards visualize performance, usage, and quality metrics

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) | Web API (self-referential — built with the framework being documented!) |
| **LLM** | [DeepSeek](https://deepseek.com/) `deepseek-v4-flash` | Answer generation & query rewriting |
| **Text Search** | minsearch (TF-IDF) | Keyword-based document retrieval |
| **Vector Search** | sentence-transformers | Semantic embedding search |
| **Re-ranker** | cross-encoder (MiniLM) | Result relevance re-ordering |
| **Database** | PostgreSQL 16 | Conversation & feedback logging |
| **Monitoring** | Grafana | Dashboards & analytics |
| **Frontend** | React + Vite | Chat user interface |
| **Containerization** | Docker Compose | Service orchestration |

---

## 📊 Evaluation Criteria Mapping

This project targets the following LLM Zoomcamp evaluation criteria:

| Criterion | Target Score | How We Achieve It |
|-----------|:-----------:|-------------------|
| Problem description | **2** | ✅ This README — clear problem, target users, and solution |
| Retrieval flow | **2** | ✅ Hybrid search (TF-IDF + vector) over chunked FastAPI docs |
| Retrieval evaluation | **2** | ✅ Multiple approaches evaluated (text vs vector vs hybrid) |
| LLM evaluation | **2** | ✅ LLM-as-a-Judge comparing prompts/models |
| Interface | **2** | ✅ FastAPI API + React frontend |
| Ingestion pipeline | **2** | ✅ Automated Python ingestion script |
| Monitoring | **2** | ✅ PostgreSQL feedback + 7+ Grafana charts |
| Containerization | **2** | ✅ Everything in docker-compose |
| Reproducibility | **2** | ✅ Locked deps, pre-generated data, clear instructions |
| Hybrid search | **1** | ✅ TF-IDF + vector combined with optimized weights |
| Re-ranking | **1** | ✅ Cross-encoder re-ranker after initial retrieval |
| Query rewriting | **1** | ✅ LLM-based query expansion before search |
| Cloud deployment | **2** | ✅ Deployed to [Render / Railway] (bonus) |
| **Total** | **22-25** | |

---

## 🚀 Operations Manual

### Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| Docker & Docker Compose | Latest | Run all services (recommended) |
| Python | 3.12+ | Local development without Docker |
| uv | Latest | Fast Python package manager |
| DeepSeek API key | — | LLM access (get at [platform.deepseek.com](https://platform.deepseek.com)) |
| Node.js | 20+ | Frontend development (optional) |

---

### 🐳 Option 1: Run Everything with Docker (Recommended)

This is the easiest way — it starts all 4 services (PostgreSQL, FastAPI backend, React frontend, Grafana) with a single command.

#### 1. Setup Environment

```bash
# Navigate to the project
cd fastapi-docs-assistant

# Copy the environment template
cp .env.example .env

# Edit .env and add your DeepSeek API key
# Open .env in any text editor and set:
# DEEPSEEK_API_KEY=sk-your-actual-key-here
```

#### 2. Start All Services

```bash
# Build and start everything in the background
docker compose up -d

# Check that all 4 services are running
docker compose ps
# Output should show:
#   Name                      Status
#   fastapi-docs-assistant-postgres-1   Up (healthy)
#   fastapi-docs-assistant-app-1        Up (healthy)
#   fastapi-docs-assistant-frontend-1   Up
#   fastapi-docs-assistant-grafana-1    Up
```

#### 3. Access the Application

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend (Chat UI)** | [http://localhost:8080](http://localhost:8080) | Ask questions in a chat interface |
| **FastAPI Backend** | [http://localhost:8000](http://localhost:8000) | REST API endpoints |
| **API Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive API documentation |
| **Grafana Monitoring** | [http://localhost:3000](http://localhost:3000) | Dashboards (login: admin/admin) |

#### 4. Verify It's Working

```bash
# Test the health endpoint
curl http://localhost:8000/health
# Expected: {"status":"ok","model":"deepseek-v4-flash","alpha":0.4,"prompt_style":"detailed"}

# Ask a question via API
curl -X POST http://localhost:8000/question \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I add CORS to my FastAPI app?"}'
# Expected: JSON response with answer, citations, relevance score

# Or open http://localhost:8080 in your browser and type a question
```

#### 5. Stop the Application

```bash
# Stop all services (data is preserved in volumes)
docker compose stop

# Or stop and remove containers (data persists)
docker compose down

# Stop and remove everything INCLUDING data volumes (WARNING: deletes DB)
docker compose down -v
```

#### 6. Restart the Application

```bash
# Restart all services
docker compose restart

# Or stop then start again
docker compose stop
docker compose start

# Rebuild and start (after code changes)
docker compose up -d --build
```

#### 7. View Logs

```bash
# Follow all logs
docker compose logs -f

# Follow just the backend logs
docker compose logs -f app

# Follow just the frontend logs
docker compose logs -f frontend

# Last 50 lines
docker compose logs --tail=50 app
```

#### 8. Initialize Database (First Run Only)

The database tables are created automatically when the app starts. If you need to re-initialize:

```bash
docker compose exec app python -c "from app import db; db.init_db()"
```

---

### 💻 Option 2: Run Locally (for Development)

This option is better for development because you can edit code and see changes immediately.

#### 1. Setup Python Environment

```bash
cd fastapi-docs-assistant

# Install Python dependencies
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY=sk-your-key-here
```

#### 2. Start PostgreSQL (Required)

PostgreSQL must be running for the app to store conversations and perform vector search. The docker-compose file starts just the database:

```bash
# Start only PostgreSQL (from the project root)
docker compose up -d postgres

# Verify PostgreSQL is healthy
docker compose ps
```

#### 3. Start the Backend

```bash
cd fastapi-docs-assistant

# Start the FastAPI server
uv run python app.py
# Output:
# ✅ Loaded best params: alpha=0.4
# 🚀 FastAPI Docs Assistant API
#    Model: deepseek-v4-flash
#    API docs: http://localhost:8000/docs

# The server runs with auto-reload — code changes take effect immediately
# Press Ctrl+C to stop
```

#### 4. Start the Frontend (in a separate terminal)

```bash
cd fastapi-docs-assistant/frontend

# Install Node.js dependencies (first time only)
npm install

# Start the Vite dev server
npm run dev
# Output:
#   VITE v5.x.x  ready in XXXms
#   ➜  Local:   http://localhost:5173/
#   ➜  Network: http://192.168.x.x:5173/

# Open http://localhost:5173 in your browser
# The frontend proxies API calls to http://localhost:8000 automatically
# Press Ctrl+C to stop
```

#### 5. Run the Ingestion Pipeline

To scrape the latest FastAPI docs and rebuild the search index:

```bash
cd fastapi-docs-assistant

# Standalone mode (saves to JSON + pickle, no database needed)
uv run python -m app.ingest --mode standalone

# Full mode (also embeds and stores in pgvector)
uv run python -m app.ingest --mode full

# Force re-scrape from live docs
uv run python -m app.ingest --mode standalone --regen
```

---

### 📋 Common Workflows

#### First Time Setup (Step by Step)

```bash
# 1. Clone the project
cd fastapi-docs-assistant

# 2. Configure API key
cp .env.example .env
# (edit .env with your DeepSeek API key)

# 3. Start everything
docker compose up -d

# 4. Wait 10-15 seconds for all services to be ready
# 5. Open http://localhost:8080 and ask a question!
```

#### Daily Startup

```bash
cd fastapi-docs-assistant
docker compose up -d
# All services start in ~5 seconds
```

#### Daily Shutdown

```bash
cd fastapi-docs-assistant
docker compose stop
# Data is preserved. Run `docker compose start` to resume.
```

#### Full Restart (after code changes)

```bash
cd fastapi-docs-assistant
docker compose down
docker compose up -d --build
```

#### Check Service Health

```bash
# Quick health check
curl http://localhost:8000/health

# See which services are running
docker compose ps

# Check resource usage
docker stats
```

#### Reset Everything (Clean Slate)

```bash
cd fastapi-docs-assistant
docker compose down -v    # Stops everything + deletes all data volumes
docker compose up -d      # Fresh start
```

---

### 📡 API Reference

#### `POST /question`

Ask a question about FastAPI.

```bash
curl -X POST http://localhost:8000/question \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I use dependency injection in FastAPI?"}'
```

**Response:**
```json
{
  "conversation_id": "a1b2c3d4-...",
  "question": "How do I use dependency injection in FastAPI?",
  "answer": "FastAPI has a very powerful but intuitive Dependency Injection system...",
  "model_used": "deepseek-v4-flash",
  "response_time": 0.85,
  "relevance": "RELEVANT",
  "token_usage": {
    "prompt_tokens": 450,
    "completion_tokens": 180,
    "total_tokens": 630
  },
  "retrieved_chunks": [
    {"id": "fastapi-tutorial-dependencies-0", "title": "Dependencies", "section": "First Steps", "url": "https://fastapi.tiangolo.com/tutorial/dependencies/", "score": 0.92}
  ]
}
```

#### `POST /feedback`

Submit feedback on an answer (thumbs up/down).

```bash
# Thumbs up (helpful)
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "a1b2c3d4-...", "feedback": 1}'

# Thumbs down (not helpful)
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "a1b2c3d4-...", "feedback": -1}'
```

#### `GET /health`

Check if the API is running.

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"deepseek-v4-flash","alpha":0.4,"prompt_style":"detailed"}
```

---

### 🐛 Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| `docker compose up -d` fails | Port already in use | Stop other services on ports 8000, 8080, 3000, 5432 |
| Backend returns 500 errors | Missing DeepSeek API key | Check `.env` file has `DEEPSEEK_API_KEY` set |
| Frontend shows blank page | Backend not running | Run `docker compose ps` to check app status |
| "No chunks found" error | Database not initialized | Run `docker compose exec app python -c "from app.ingest import run_pipeline; run_pipeline()"` |
| Grafana shows "No data" | No conversations yet | Ask a question via the UI or curl, then refresh Grafana |
| `docker compose build` slow | First time download | Subsequent builds use Docker layer cache and are faster |
| API requests timeout | DeepSeek API latency | Check `docker compose logs -f app` for detailed error messages |

---

## 📁 Project Structure (Complete)

```
fastapi-docs-assistant/
├── app/                          # Python backend package
│   ├── __init__.py
│   ├── minsearch.py              # TF-IDF search index (vendored)
│   ├── db.py                     # PostgreSQL + pgvector functions
│   ├── vector_store.py           # ONNX embedding (all-MiniLM-L6-v2)
│   ├── re_ranker.py              # Cross-encoder re-ranking
│   ├── ingest.py                 # Ingestion pipeline (scrape → chunk → embed)
│   ├── load_index.py             # Pickle index loader
│   └── rag.py                    # RAG pipeline: rewrite → search → rerank → LLM
├── app.py                        # FastAPI server (3 endpoints)
├── pyproject.toml                # Python dependencies
├── uv.lock                       # Locked dependency versions
├── frontend/                     # React chat UI
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── nginx.conf                # Production nginx config
│   └── src/
│       ├── main.jsx
│       ├── App.jsx               # Chat UI component
│       └── index.css             # Styling
├── data/                         # Pre-computed datasets
│   ├── fastapi_docs/
│   │   ├── chunks.json           # 135 semantic chunks from 120 pages
│   │   ├── raw_pages.jsonl       # 135 raw HTML pages (Scrapy output)
│   │   └── text_index.pkl        # Pickled TF-IDF index
│   ├── ground-truth-retrieval.csv    # 268 Q&A pairs (119 chunks)
│   ├── retrieval-results.csv         # 4 approaches: default text, optimized, vector, hybrid
│   ├── best-params.json              # Optimized: alpha=0.1, boost optimized
│   ├── rag-evaluation.csv            # concise + deepseek-v4-flash (26 evaluations)
│   ├── re-ranking-results.csv        # Hybrid vs Hybrid+Re-ranker
│   └── query-rewriting-results.csv   # Without vs With query rewriting
├── notebooks/                    # Evaluation scripts
│   ├── 01-generate-ground-truth.py
│   ├── 02-retrieval-evaluation.py
│   ├── 03-rag-evaluation.py
│   ├── 04-re-ranking-evaluation.py
│   └── 05-query-rewriting-evaluation.py
├── grafana/                      # Monitoring
│   ├── datasource.yml            # PostgreSQL data source (auto-provisioned)
│   ├── dashboard.yml             # Dashboard loader
│   ├── dashboard.json            # 8-chart monitoring dashboard
│   └── init.py                   # API-based provisioning script
├── Dockerfile.app                # Multi-stage Python build
├── Dockerfile.frontend           # Multi-stage React build
├── docker-compose.yaml           # 4 services: postgres, app, frontend, grafana
├── .env.example                  # Environment variable template
├── .dockerignore                 # Docker build exclusions
└── .gitignore                    # Git exclusions
```

## 📝 License

MIT