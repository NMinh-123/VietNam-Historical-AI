# VICAL — Vietnam Historical AI

<p align="center">
  <b>AI-powered chatbot specialized in Vietnamese history</b><br/>
  Hybrid RAG · Knowledge Graph · Persona Chat · Streaming SSE
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue" />
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green" />
  <img src="https://img.shields.io/badge/Qdrant-1.17.1-red" />
  <img src="https://img.shields.io/badge/LightRAG-Knowledge%20Graph-purple" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

---

## Overview

**VICAL** is an AI chatbot system focused on Vietnamese history. It combines **Hybrid Retrieval-Augmented Generation**, **Knowledge Graph reasoning**, and **historical persona simulation** to deliver accurate, source-grounded, and explainable answers.

**Users can:**
- Ask questions about Vietnamese history and receive cited, source-grounded answers
- Have live conversations with historical figures such as Ngô Quyền and Trần Hưng Đạo
- Explore Vietnamese dynasty timelines and historical events
- Save and revisit conversation history

---

## System Architecture

```
Client (Browser)
       │
       ▼
  Nginx Reverse Proxy
       │
       ▼
  FastAPI Application (port 8001)
       │
       ├── Authentication Layer
       │      ├── Email / Password (bcrypt)
       │      ├── Google OAuth 2.0
       │      └── Facebook OAuth
       │
       ├── Query Rewriter
       │
       ├── Hybrid Retrieval Pipeline
       │      ├── Dense Search  — Multilingual-E5 Embeddings
       │      ├── Sparse Search — BM25 (fastembed)
       │      ├── Reciprocal Rank Fusion (RRF)
       │      └── Lexical Reranking
       │
       ├── LightRAG Knowledge Graph
       │
       └── LLM (Google Gemini) → Streaming SSE Response
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn, Python 3.11 |
| LLM | Google Gemini (OpenAI-compatible API) |
| Vector Database | Qdrant v1.17.1 |
| Embeddings | multilingual-E5 (dense), BM25 via fastembed (sparse) |
| Knowledge Graph | LightRAG |
| Relational DB | SQLite (development) / PostgreSQL 16 (production) |
| Authentication | authlib, bcrypt, itsdangerous |
| Frontend | Jinja2, Tailwind CSS, Vanilla JavaScript |
| Deployment | Docker, Docker Compose, Nginx Alpine |
| Testing | pytest |
| Rate Limiting | slowapi |

---

## Project Structure

```
VietNam-Historical-AI/
├── app/
│   ├── server/
│   │   ├── routers/           # API route handlers
│   │   ├── auth/              # Authentication & OAuth
│   │   ├── db/                # Database models & queries
│   │   └── persona_data.py    # Historical persona definitions
│   │
│   ├── services/chatbot/
│   │   ├── chatbot/           # RAG pipeline & LLM integration
│   │   ├── persona_chat/      # Persona simulation engine
│   │   └── index_and_retrieve/  # Indexing scripts
│   │
│   ├── data/                  # Documents, vector storage
│   └── static/                # Frontend assets & Jinja2 templates
│
├── data/                      # Seed data, avatar images, timeline DB
├── tests/                     # pytest test suite
├── docker-compose.yml
├── Dockerfile
├── nginx.conf
└── requirements.txt
```

---

## Local Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/NMinh-123/VietNam-Historical-AI.git
cd VietNam-Historical-AI
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Fill in the required variables:

```env
# LLM
GEMINI_KEY=your_api_key
GEMINI_MODEL_NAME=gemini-2.0-flash
OPENAI_COMPAT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# Application
SECRET_KEY=your_secret_key
ENV=dev
REDIRECT_BASE_URL=http://localhost:8001

# OAuth
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
FACEBOOK_APP_ID=
FACEBOOK_APP_SECRET=

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# PostgreSQL (production only)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vical
POSTGRES_USER=vical
POSTGRES_PASSWORD=your_password
```

### 3. Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/app/data/qdrant_db:/qdrant/storage \
  qdrant/qdrant
```

### 6. Build retrieval indexes

```bash
python app/services/chatbot/index_and_retrieve/run_qdrant_index.py
python app/services/chatbot/index_and_retrieve/run_lightrag_index.py
```

### 7. Seed the timeline database

```bash
python data/seed_timeline.py
```

### 8. Run the application

```bash
cd app
uvicorn server.main:app --reload --host 0.0.0.0 --port 8001
```

Open: **http://localhost:8001**

---

## Docker Deployment

```bash
# Start all services (app, postgres, qdrant, nginx)
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop
docker-compose down
```

**Services:**

| Service | Image | Port |
|---|---|---|
| app | Dockerfile (Python 3.11) | 8001 (internal) |
| postgres | postgres:16-alpine | 5432 (internal) |
| qdrant | qdrant/qdrant:v1.17.1 | 6333 HTTP, 6334 gRPC |
| nginx | nginx:alpine | 80 (public) |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ask` | Historical Q&A with cited sources |
| `POST` | `/api/ask/stream` | Streaming SSE response |
| `POST` | `/api/persona-chat/{slug}` | Chat with a historical figure |
| `GET` | `/api/history/{id}/messages` | Retrieve conversation history |
| `POST` | `/api/history/save` | Save a conversation |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html
```

---

## Engineering Highlights

**Hybrid Retrieval** — Dense search (E5 embeddings) + sparse search (BM25) combined via RRF fusion and lexical reranking. Significantly improves retrieval precision over standalone vector search.

**Knowledge Graph** — LightRAG connects historical entities to support multi-hop reasoning and reduce hallucination in long historical contexts.

**Persona Guardrails** — Temporal consistency constraints prevent historical figures from referencing events outside their own era, avoiding anachronisms.

**Streaming SSE** — Responses are streamed token-by-token, reducing perceived latency and improving conversational UX.

---

## Roadmap

- Redis caching layer
- Cross-encoder reranking
- Citation highlighting in the UI
- Multi-agent retrieval orchestration
- Fine-tuned Vietnamese embedding models
- Observability with Prometheus + Grafana

---

## Author

**Hoang Minh**
- GitHub: [NMinh-123](https://github.com/NMinh-123)
- Email: [minh40009@gmail.com](mailto:minh40009@gmail.com)
- Repository: [VietNam-Historical-AI](https://github.com/NMinh-123/VietNam-Historical-AI)

---

## License

This project is licensed under the [MIT License](LICENSE).
