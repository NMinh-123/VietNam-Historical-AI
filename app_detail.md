# Tài liệu kỹ thuật — VICAL AI
**Cập nhật:** 2026-05-06
**Phiên bản:** 3.0.0

---

## Tổng quan

Vical AI là chatbot lịch sử Việt Nam dùng kiến trúc RAG lai (Hybrid RAG): kết hợp vector search (Qdrant) và knowledge graph (LightRAG) để trả lời câu hỏi có trích dẫn nguồn. Hỗ trợ chat thông thường và nhập vai nhân vật lịch sử (Persona Chat).

**Stack:**
- Backend: FastAPI + Uvicorn, Python 3.11
- DB: SQLite (dev) / PostgreSQL (production via asyncpg)
- Vector DB: Qdrant (hybrid dense+sparse)
- Knowledge Graph: LightRAG
- LLM: Gemini / OpenAI-compatible endpoint (qua `OPENAI_COMPAT_BASE_URL`)
- Embedding: E5 multilingual dense + BM25 sparse (fastembed)
- Auth: email/password (bcrypt) + Google OAuth + Facebook OAuth
- Frontend: Jinja2 templates + Tailwind CSS CDN
- Deployment: Docker Compose (app + postgres + qdrant + nginx)

---

## Cấu trúc dự án

```
app/
├── server/
│   ├── main.py              — FastAPI app, lifespan, middleware, router mount
│   ├── schemas.py           — Pydantic models: AskRequest, AskResponse, SourceItem, PersonaInfo
│   ├── persona_data.py      — PERSONAS dict, ALL_PERSONA_LIST, BOOKS
│   ├── routers/
│   │   ├── pages.py         — HTML page routes (/, /ask, /history, /persona, /timeline, /library)
│   │   ├── chatbot_api.py   — /ask, /api/ask, /api/ask/stream, /personas, /health, /warmup
│   │   └── history_api.py   — /api/history/save, /{id}/messages, /{id}/delete
│   ├── db/
│   │   ├── connection.py    — init_db(), schema, WAL config, SQLite/PG dual backend
│   │   ├── conversations.py — save_turn, list_conversations, get_messages, delete_conversation
│   │   ├── users.py         — CRUD users, bcrypt, OAuth upsert, stats
│   │   └── timeline.py      — get_dynasties() từ timeline.sqlite3 (JOIN query)
│   └── auth/
│       ├── session.py       — itsdangerous serializer, cookie helpers, get_current_user
│       ├── email_auth.py    — /auth/register, /auth/login, /auth/logout
│       ├── oauth_google.py  — Google OAuth 2.0
│       ├── oauth_facebook.py — Facebook OAuth 2.0
│       └── account.py       — /auth/account page, /auth/account/update
│
├── services/chatbot/
│   ├── shared_engine.py     — init_engine() tại startup, get_engine() singleton, get_persona_engine()
│   ├── chatbot/
│   │   └── engine.py        — VietnamHistoryQueryEngine: rewrite → vector+graph → LLM
│   ├── persona_chat/
│   │   ├── persona_config.py — PersonaConfig, ALL_PERSONAS, temporal guardrail
│   │   └── engine.py        — PersonaChatEngine: inject base_engine, persona prompt
│   └── index_and_retrieve/
│       ├── config.py        — paths, model names, env resolution helpers
│       ├── retriever.py     — hybrid search: E5 dense + BM25 sparse, lexical rerank
│       ├── context_builder.py — format context, build source payload
│       ├── history_summarizer.py — build_history_block: inject thẳng hoặc tóm tắt bằng LLM
│       ├── text_utils.py    — lexical scoring, query builder
│       ├── pipeline.py      — ingest pipeline
│       └── qdrant_index.py  — index builder
│
├── data/
│   ├── ocr_data/            — raw PDF/text nguồn sử liệu
│   ├── qdrant_db/           — Qdrant local storage
│   ├── lightrag_storage/    — LightRAG graph storage
│   ├── parent_docs.json     — parent chunks (load vào RAM khi startup)
│   ├── child_docs.json      — child chunks
│   └── timeline.sqlite3     — seed data triều đại + vua
│
data/
└── seed_timeline.py         — script seed dữ liệu timeline

docker-compose.yml           — services: app, postgres, qdrant, nginx
Dockerfile                   — build image cho app service
nginx.conf                   — reverse proxy config
requirements.txt
tests/                       — thư mục tồn tại, chưa có test nào
pytest.ini
```

---

## Luồng xử lý request chính

### `/api/ask/stream` (SSE)

```
Request → get_current_user (cookie) → trial check (session)
       → get_recent_turns_list (DB)
       → build_history_block (inject thẳng < 4 lượt, tóm tắt LLM nếu ≥ 4 lượt)
       → VietnamHistoryQueryEngine.ask_with_sources_stream()
           ├── rewrite_query (bỏ meta-instruction, xử lý causal)
           ├── detect_broad_query → decompose nếu cần
           ├── asyncio.gather(get_vector, get_graph)  [parallel]
           │   ├── get_vector: E5 dense + BM25 sparse → RRF fusion → lexical rerank
           │   └── get_graph: LightRAG knowledge graph search
           ├── build context + sources
           └── stream LLM response token-by-token (SSE)
       → client nhận token → POST /api/history/save (lưu DB)
```

### Persona Chat (`/api/persona-chat/{slug}`)

```
Request → PersonaChatEngine.ask_with_sources()
       → check_temporal_guardrail (giới hạn kiến thức theo năm nhân vật)
       → base_engine.retrieval (dùng chung VietnamHistoryQueryEngine)
       → inject persona system prompt (nhân cách, era, forbidden_topics)
       → LLM → response
```

---

## Database schema

### SQLite / PostgreSQL (dual backend)

```sql
users (id TEXT PK, email UNIQUE, display_name, avatar_url, password_hash, is_active, created_at, updated_at)
oauth_accounts (id PK, user_id FK→users CASCADE, provider, provider_user_id UNIQUE(provider,uid), created_at)
conversations (id TEXT PK, user_id FK→users CASCADE, title, chat_type, persona_slug, message_count, preview, created_at, updated_at)
messages (id PK, conversation_id FK→conversations CASCADE, role, content, sources_json, created_at)
```

**Indexes:** `idx_users_email`, `idx_oauth_provider`, `idx_conversations_user`, `idx_conversations_updated`, `idx_messages_conv`

**Lưu ý:** `messages` có `ON DELETE CASCADE` — xóa conversation tự xóa messages, không cần DELETE messages thủ công.

### timeline.sqlite3 (read-only seed)

```sql
core_dynasty (id, slug, name, start_year, end_year, description, color, order)
core_king    (id, name, reign_start, reign_end, temple_name, description, dynasty_id, order)
```

---

## Auth

| Flow | Endpoint | Ghi chú |
|------|----------|---------|
| Email register | `POST /auth/register` | bcrypt hash, rate limit 5/min |
| Email login | `POST /auth/login` | itsdangerous signed cookie, 30 ngày |
| Logout | `GET/POST /auth/logout` | delete cookie |
| Google OAuth | `GET /auth/google` → callback | PKCE state CSRF |
| Facebook OAuth | `GET /auth/facebook` → callback | PKCE state CSRF |
| Session cookie | `vical_session` | HttpOnly, SameSite=Lax, Secure khi production |

**Trial users:** không đăng nhập được 3 câu hỏi, đếm qua Starlette `SessionMiddleware` (reset được bằng cách xóa cookie — không có rate limit IP).

---

## Cấu hình môi trường (.env)

| Biến | Mô tả | Default |
|------|-------|---------|
| `SECRET_KEY` | Session signing key | random (unsafe) |
| `ENV` | `production` để bật Secure cookie + https_only | `dev` |
| `ALLOWED_ORIGINS` | CORS origins cách nhau dấu phẩy | `http://localhost:8001` |
| `DATABASE_URL` | PostgreSQL DSN (nếu có → dùng PG) | — |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | Thay thế cho DATABASE_URL | localhost/5432/vical/vical |
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `GEMINI_KEY` / `OPENAI_API_KEY` / `SHOPAIKEY_TOKEN` | LLM API key (ưu tiên theo thứ tự) | — |
| `OPENAI_COMPAT_BASE_URL` / `SHOPAIKEY_BASE_URL` | Base URL LLM endpoint | Gemini official |
| `GEMINI_MODEL_NAME` / `SHOPAIKEY_MODEL_NAME` | Tên model | `gemini-3.1-flash-lite` |
| `GEMINI_RPM_LIMIT` | Rate limit request/phút | tự detect theo model prefix |
| `GEMINI_MAX_CONCURRENCY` | Concurrency semaphore | `5` |
| `RETRIEVER_TOP_K` | Số chunks trả về sau rerank | `4` |
| `RETRIEVER_LIMIT` | Số candidates trước rerank | `40` |

---

## Deployment (Docker Compose)

4 services:

| Service | Image | Port |
|---------|-------|------|
| `app` | build từ `Dockerfile` | 8001 (internal) |
| `postgres` | postgres:16-alpine | internal |
| `qdrant` | qdrant/qdrant:latest | 6333, 6334 |
| `nginx` | nginx:alpine | 80 (public) |

`app` depends on `postgres` và `qdrant` healthy. Nginx reverse proxy đến app:8001, serve static files trực tiếp.

---

## Vấn đề kỹ thuật còn tồn tại

### HIGH

1. **`ask_with_sources_stream()` có thể tạo OpenAI client mới mỗi request** (`engine.py`) — cần kiểm tra, nếu còn thì không reuse connection pool, bypass rate limiter của `self.llm`.

2. **`parent_store` load toàn bộ vào RAM** (`engine.py`) — `parent_docs.json` load lúc startup và giữ suốt process. Khi dataset lớn (vài GB) sẽ OOM.

3. **`_warmup_task` accessed từ `main.py`** — `main.py:57` truy cập `engine._warmup_task` (private). Nên expose qua method public `engine.warmup_done()`.

4. **Trial count reset được bằng cách xóa cookie** — Starlette SessionMiddleware, không có IP rate limit. Dễ bypass.

### MEDIUM

5. **`get_persona_engine()` tạo object mới mỗi call** — `PersonaChatEngine` stateless nên không gây lỗi, nhưng không nhất quán intent singleton.

6. **`config.py` tự load .env** (`_load_dotenv()`) — `main.py` đã dùng `python-dotenv`, `session.py` cũng load lại. Ba điểm load dotenv độc lập.

7. **SQLite không có connection pool** — mỗi request tạo connection mới. WAL mode giảm lock contention nhưng concurrent writes vẫn serialize.

8. **`user_id IS NULL` trong conversation queries** — conversations không có owner xem được bởi bất kỳ ai biết `conv_id`. Intentional cho anonymous nhưng chưa documented.

### LOW

9. **Không có CI/CD** — test suite có sẵn nhưng chưa chạy tự động khi push code.

10. **Import bên trong function body** (`persona_chat/engine.py`) — một số import đặt trong `ask_with_sources()` để tránh circular import, nên refactor.

---

## Lịch sử fix đáng chú ý

| Vấn đề | Trạng thái |
|--------|-----------|
| Race condition `get_engine()` (lock không dùng) | ✅ Fix: `init_engine()` gọi 1 lần tại startup, `get_engine()` raise nếu chưa init |
| `https_only=False` hardcoded SessionMiddleware | ✅ Fix: đọc `ENV=production` |
| CORS `allow_origins=["*"]` | ✅ Fix: đọc `ALLOWED_ORIGINS` từ env |
| N+1 query `get_dynasties()` | ✅ Fix: dùng LEFT JOIN duy nhất |
| `color` field bị bỏ sót khi build dynasty dict | ✅ Fix: thêm `"color": r["color"]` |
| God file `db.py` | ✅ Tách thành `db/connection.py`, `db/users.py`, `db/conversations.py`, `db/timeline.py` |
| Auth lộn xộn | ✅ Tách thành `auth/session.py`, `auth/email_auth.py`, `auth/oauth_*.py`, `auth/account.py` |
| Thiếu Pydantic schemas tập trung | ✅ `schemas.py` với `AskRequest`, `AskResponse`, `SourceItem`, `PersonaInfo` |
| Redundant DELETE messages trong `delete_conversation` | ✅ ON DELETE CASCADE xử lý tự động |
| `persona_slug: str = ""` sentinel | ✅ Đổi thành `str | None = None` |
| `chat_type` magic string | ✅ Đổi thành `Literal["ask", "persona"]` |
| Thiếu Dockerfile + nginx | ✅ Có cả hai |
| Thiếu PostgreSQL support | ✅ Dual backend SQLite/PG qua asyncpg |
| Lịch sử hội thoại không có tóm tắt | ✅ `history_summarizer.py`: inject thẳng < 4 lượt, tóm tắt LLM nếu ≥ 4 lượt |
