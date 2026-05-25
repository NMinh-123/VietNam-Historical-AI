# VICAL — Vietnam Historical AI

<p align="center">
  <b>AI-powered historical assistant for Vietnamese history</b><br/>
  Hybrid RAG + Knowledge Graph + Persona Chat System
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue" />
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green" />
  <img src="https://img.shields.io/badge/Qdrant-Vector%20Database-red" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

---

## Overview

**VICAL (Vietnam Historical AI)** is an AI chatbot system focused on Vietnamese history.
The project combines **Hybrid Retrieval-Augmented Generation (Hybrid RAG)**, **Knowledge Graph reasoning**, and **historical persona simulation** to provide accurate, contextual, and explainable answers.

Users can:

* Ask questions about Vietnamese history
* Chat with historical figures such as Ngô Quyền or Trần Hưng Đạo
* Receive source-grounded responses with streaming output
* Explore dynasty timelines and historical conversations

This project was built to explore:

* Production-ready RAG architecture
* Retrieval optimization
* Knowledge Graph integration
* Real-time AI streaming systems
* Multi-provider authentication

---

# Demo Features

## Historical Question Answering

* Hybrid retrieval pipeline:

  * Dense Retrieval (E5 multilingual embeddings)
  * Sparse Retrieval (BM25)
  * Reciprocal Rank Fusion (RRF)
  * Lexical reranking
* Source-grounded answers
* Streaming Server-Sent Events (SSE)

## Persona Chat System

* Interactive conversations with historical figures
* Temporal guardrails to prevent timeline inconsistencies
* Persona-specific prompts and speaking styles

## Authentication System

* Email/password login
* Google OAuth
* Facebook OAuth
* Conversation history persistence

## Timeline Exploration

* Vietnamese dynasty timeline
* Historical rulers and events lookup

---

# System Architecture

```text
Client
   │
   ▼
FastAPI Application
   │
   ├── Authentication Layer
   │      ├── Email Authentication
   │      ├── Google OAuth
   │      └── Facebook OAuth
   │
   ├── Query Rewriter
   │
   ├── Hybrid Retrieval Pipeline
   │      ├── Dense Search (E5 Embeddings)
   │      ├── Sparse Search (BM25)
   │      ├── Reciprocal Rank Fusion
   │      └── Lexical Reranking
   │
   ├── LightRAG Knowledge Graph
   │
   └── LLM Response Generation
          └── Streaming SSE Response
```

---

# Tech Stack

| Layer           | Technologies                                     |
| --------------- | ------------------------------------------------ |
| Backend         | FastAPI, Uvicorn                                 |
| LLM Integration | Google Gemini (OpenAI-compatible API)            |
| Vector Database | Qdrant                                           |
| Embeddings      | multilingual-e5, BM25                            |
| Knowledge Graph | LightRAG                                         |
| Database        | SQLite (Development), PostgreSQL 16 (Production) |
| Authentication  | authlib, bcrypt, itsdangerous                    |
| Frontend        | HTML, CSS, JavaScript, Jinja2                    |
| Deployment      | Docker, Docker Compose, Nginx                    |
| Testing         | pytest                                           |

---

# Key Engineering Highlights

## Hybrid Retrieval Strategy

The retrieval pipeline combines:

* Semantic similarity search using multilingual E5 embeddings
* Sparse keyword retrieval using BM25
* Reciprocal Rank Fusion (RRF)
* Lexical reranking for final context refinement

This improves retrieval robustness compared to standalone vector search.

## Knowledge Graph Integration

The system integrates LightRAG to:

* Connect historical entities
* Improve multi-hop reasoning
* Reduce hallucination in long historical contexts

## Streaming AI Responses

Responses are streamed token-by-token using SSE, improving:

* User experience
* Perceived latency
* Real-time interaction quality

## Persona Safety Guardrails

Historical persona chat includes:

* Temporal consistency constraints
* Character-aware prompting
* Context filtering to avoid anachronisms

---

# Project Structure

```text
VietNam-Historical-AI/
│
├── app/
│   ├── server/
│   │   ├── routers/
│   │   ├── auth/
│   │   ├── db/
│   │   └── persona_data.py
│   │
│   ├── services/chatbot/
│   │   ├── chatbot/
│   │   ├── persona_chat/
│   │   └── index_and_retrieve/
│   │
│   ├── data/
│   └── static/
│
├── tests/
├── docker-compose.yml
├── Dockerfile
├── nginx.conf
└── requirements.txt
```

---

# Local Development Setup

## 1. Clone Repository

```bash
git clone https://github.com/NMinh-123/VietNam-Historical-AI.git
cd VietNam-Historical-AI
```

---

## 2. Create Environment Variables

```bash
cp .env.example .env
```

Required environment variables:

```env
# LLM
GEMINI_KEY=your_api_key
GEMINI_MODEL_NAME=gemini-3-flash-preview
OPENAI_COMPAT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# Authentication
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

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vical
POSTGRES_USER=vical
POSTGRES_PASSWORD=your_password
```

---

## 3. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/app/data/qdrant_db:/qdrant/storage \
  qdrant/qdrant
```

---

## 6. Build Retrieval Index

```bash
python app/services/chatbot/index_and_retrieve/run_qdrant_index.py
python app/services/chatbot/index_and_retrieve/run_lightrag_index.py
```

---

## 7. Seed Timeline Database

```bash
python data/seed_timeline.py
```

---

## 8. Run Application

```bash
cd app
uvicorn server.main:app --reload --host 0.0.0.0 --port 8001
```

Application URL:

```text
http://localhost:8001
```

---

# Docker Deployment

## Start Services

```bash
docker-compose up -d
```

## View Logs

```bash
docker-compose logs -f app
```

## Stop Services

```bash
docker-compose down
```

---

# API Endpoints

| Method | Endpoint                     | Description                   |
| ------ | ---------------------------- | ----------------------------- |
| POST   | `/api/ask`                   | Standard chatbot response     |
| POST   | `/api/ask/stream`            | Streaming chatbot response    |
| GET    | `/api/history/{id}/messages` | Retrieve conversation history |
| POST   | `/api/history/save`          | Save conversation             |

---

# Testing

Run all tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=app --cov-report=html
```

---

# Engineering Challenges Solved

## Retrieval Noise Reduction

Implemented:

* Hybrid dense + sparse retrieval
* RRF fusion
* Lexical reranking

to improve retrieval precision and reduce irrelevant context.

## Hallucination Mitigation

Combined:

* Source-grounded retrieval
* Knowledge Graph augmentation
* Persona guardrails

to improve factual consistency.

## Real-time Streaming Architecture

Used SSE streaming to support:

* Progressive token rendering
* Lower perceived latency
* Better conversational UX

---

# Future Improvements

* Redis caching layer
* Cross-encoder reranking
* Citation highlighting UI
* Multi-agent retrieval orchestration
* Fine-tuned Vietnamese embedding models
* Observability with Prometheus + Grafana

---

# Author

**Hoang Minh**

* GitHub: [https://github.com/NMinh-123](https://github.com/NMinh-123)
* Project Repository: [https://github.com/NMinh-123/VietNam-Historical-AI](https://github.com/NMinh-123/VietNam-Historical-AI)
* Email: [minh40009@gmail.com](mailto:minh40009@gmail.com)

---

# License

This project is licensed under the MIT License.
