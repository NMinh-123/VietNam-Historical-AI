# Hướng Dẫn Học Kỹ Thuật Dự Án VICAL

Tài liệu này giúp bạn học **từng kỹ thuật và cú pháp** được dùng trong dự án theo lộ trình từ **RAG/Embedding → các phần liên kết → deployment**.

---

## Lộ Trình Học

```
PHASE 1 — RAG Core (hiểu trước, đây là trái tim dự án)
    Embeddings → Qdrant → LightRAG → RAG Pipeline → LLM → Streaming → Rate Limiting

PHASE 2 — Web & Data (các lớp bao quanh RAG)
    Pydantic → Database → Authentication → FastAPI → Jinja2

PHASE 3 — Quality & Deploy (hoàn thiện)
    Patterns → Testing → Docker + Deployment
```

---

## Mục Lục

**Phase 1 — RAG Core**
1. [Embeddings — E5 + BM25](#1-embeddings)
2. [Qdrant — Vector Search](#2-qdrant)
3. [LightRAG — Knowledge Graph](#3-lightrag)
4. [RAG Pipeline — Hybrid Retrieval](#4-rag-pipeline)
5. [LLM — OpenAI-compatible API + Gemini](#5-llm)
6. [Streaming — Server-Sent Events](#6-streaming)
7. [Rate Limiting + Concurrency](#7-rate-limiting)

**Phase 2 — Web & Data**
8. [Pydantic — Validation](#8-pydantic)
9. [Database — Raw SQL (SQLite + PostgreSQL)](#9-database)
10. [Authentication — Session, OAuth](#10-authentication)
11. [FastAPI — Web Framework](#11-fastapi)
12. [Jinja2 — HTML Templates](#12-jinja2)

**Phase 3 — Quality & Deploy**
13. [Patterns Quan Trọng Trong Dự Án](#13-patterns)
14. [Testing — Pytest + RAGAS](#14-testing)
15. [Docker + Deployment](#15-docker)

---

## 1. Embeddings

**File tham khảo:** `app/data/process_data/e5_embeddings.py`, `app/services/chatbot/index_and_retrieve/`

### 1.1 E5 Multilingual — Dense embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("intfloat/multilingual-e5-small")

def embed_query(question: str) -> np.ndarray:
    # E5 yêu cầu prefix "query: " cho câu hỏi
    return model.encode(f"query: {question}", normalize_embeddings=True)

def embed_passage(text: str) -> np.ndarray:
    # E5 yêu cầu prefix "passage: " cho documents
    return model.encode(f"passage: {text}", normalize_embeddings=True)
```

**Tại sao normalize?** Để dùng cosine similarity = dot product (nhanh hơn).

**Tại sao prefix khác nhau?** E5 được train với cặp (query, passage) — dùng sai prefix sẽ giảm chất lượng retrieval đáng kể.

### 1.2 BM25 — Sparse embeddings (via FastEmbed)

```python
from fastembed import SparseTextEmbedding

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def embed_sparse(text: str) -> dict:
    embeddings = list(sparse_model.embed([text]))
    return {
        "indices": embeddings[0].indices.tolist(),
        "values": embeddings[0].values.tolist(),
    }
```

**Sparse vector** chỉ lưu index của các từ xuất hiện và TF-IDF weight → phù hợp keyword matching.

**Dense vs Sparse:**
- Dense (E5): Hiểu ngữ nghĩa ("vua" ≈ "quân chủ"), nhưng có thể miss exact match
- Sparse (BM25): Exact keyword match ("Bạch Đằng" phải đúng từ), nhưng không hiểu ngữ nghĩa
- Hybrid: Kết hợp cả hai → tốt nhất cả hai mặt

### 1.3 Parent-Child chunking

```
Tài liệu gốc (parent doc, full text)
    │
    ├── Chunk 1 (256 tokens) ──→ Embed + store in Qdrant
    ├── Chunk 2 (256 tokens) ──→ Embed + store in Qdrant
    └── Chunk 3 (256 tokens) ──→ Embed + store in Qdrant

Khi retrieve:
1. Tìm chunk liên quan nhất (dense+sparse)
2. Lookup parent_id → lấy toàn bộ parent text
3. Trả về parent text làm context (nhiều thông tin hơn)
```

**Tại sao?** Chunk nhỏ → embed chính xác hơn. Parent lớn → context cho LLM đầy đủ hơn.

---

## 2. Qdrant

**File tham khảo:** `app/services/chatbot/index_and_retrieve/qdrant_index.py`, `retriever.py`

### 2.1 Tạo collection với hybrid vectors

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    SparseIndexParams
)

client = QdrantClient(path="app/data/qdrant_db")  # Local storage

client.create_collection(
    collection_name="vietnam_history_hybrid",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams()),
    }
)
```

### 2.2 Upsert points (index documents)

```python
from qdrant_client.models import PointStruct, SparseVector

points = []
for chunk in chunks:
    dense_vec = embed_dense(chunk.text)   # numpy array (384,)
    sparse_vec = embed_sparse(chunk.text)  # {"indices": [...], "values": [...]}

    points.append(PointStruct(
        id=chunk.id,
        vector={
            "dense": dense_vec.tolist(),
            "sparse": SparseVector(
                indices=sparse_vec["indices"],
                values=sparse_vec["values"],
            ),
        },
        payload={
            "parent_id": chunk.parent_id,
            "source": chunk.source_file,
            "page": chunk.page_number,
            "title": chunk.document_title,
        }
    ))

client.upsert(collection_name="vietnam_history_hybrid", points=points)
```

### 2.3 Hybrid search với RRF fusion

```python
from qdrant_client.models import (
    Prefetch, Query, FusionQuery, Fusion, SparseVector
)

results = client.query_points(
    collection_name="vietnam_history_hybrid",
    prefetch=[
        # Dense search
        Prefetch(
            query=dense_query_vector,         # List[float]
            using="dense",
            limit=20,
        ),
        # Sparse search
        Prefetch(
            query=SparseVector(indices=sparse_indices, values=sparse_values),
            using="sparse",
            limit=20,
        ),
    ],
    # Fuse kết quả bằng Reciprocal Rank Fusion
    query=Query(fusion=FusionQuery(fusion=Fusion.RRF)),
    limit=10,
    with_payload=True,
)

for point in results.points:
    print(point.id, point.score, point.payload)
```

**RRF (Reciprocal Rank Fusion):** Kết hợp ranking từ dense và sparse. Thay vì cộng điểm (bị ảnh hưởng bởi scale), RRF dùng thứ hạng: `score = 1/(rank_dense + k) + 1/(rank_sparse + k)`.

### 2.4 Kiểm tra IDs đã tồn tại (tránh re-index)

```python
existing = client.retrieve(
    collection_name="vietnam_history_hybrid",
    ids=batch_ids,
    with_payload=False,
    with_vectors=False,
)
existing_ids = {p.id for p in existing}
new_chunks = [c for c in batch if c.id not in existing_ids]
```

---

## 3. LightRAG

**File tham khảo:** `app/services/chatbot/index_and_retrieve/lightrag_index.py`, `runtime.py`

### 3.1 Khởi tạo LightRAG

```python
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

async def embedding_func(texts: list[str]) -> list[list[float]]:
    return e5_model.embed(texts)

rag = LightRAG(
    working_dir="app/data/lightrag_storage",
    llm_model_func=build_gemini_llm_func(),
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=512,
        func=embedding_func,
    ),
)
```

### 3.2 Ingest tài liệu vào knowledge graph

```python
# LightRAG tự extract entities + relations từ text
await rag.ainsert(document_text)
# → Phân tích: "Trần Hưng Đạo lãnh đạo quân Trần đánh bại Mông Cổ năm 1285"
# → Entity: "Trần Hưng Đạo" (person), "Mông Cổ" (organization), "1285" (time)
# → Relation: "lãnh đạo", "đánh bại"
```

### 3.3 Query knowledge graph

```python
result = await rag.aquery(
    query="Trận Bạch Đằng 938 diễn ra như thế nào?",
    param=QueryParam(mode="local"),  # local = entity-focused search
)
# Trả về: text mô tả entities + relations liên quan
```

**Modes:**
- `"local"`: Tìm kiếm theo entities (phù hợp câu hỏi cụ thể)
- `"global"`: Tổng hợp toàn bộ graph (phù hợp câu hỏi tổng quan)
- `"hybrid"`: Kết hợp cả hai

**Vector search vs Knowledge Graph:**
- Vector: Tìm đoạn văn gần nghĩa nhất → tốt cho chi tiết cụ thể
- KG: Traversal entities và relations → tốt cho câu hỏi liên kết nhiều khái niệm ("Trần Hưng Đạo có liên quan đến các trận đánh nào?")

---

## 4. RAG Pipeline

**File tham khảo:** `app/services/chatbot/chatbot/engine.py`

### 4.1 Luồng xử lý toàn bộ (ask_with_sources)

```
Câu hỏi người dùng
    │
    ▼
Query Rewriting (pure Python, 0 LLM calls)
    │  - Bỏ các mẫu meta: "hãy cho biết", "giải thích cho tôi"
    │  - Bỏ causal patterns: "tại sao", "lý do"
    │
    ▼
Topic Shift Detection
    │  - So sánh word overlap với các câu hỏi gần nhất
    │  - Overlap < 12% → câu hỏi mới, không dùng history
    │
    ▼
Broad Query Detection
    │  - Match regex "tất cả", "toàn bộ", "các triều đại"
    │  - Nếu broad → decompose thành 13 sub-queries (mỗi triều đại)
    │
    ▼
Parallel Retrieval (asyncio.gather)
    ├── Vector Search (Qdrant hybrid)
    └── Graph Search (LightRAG)
    │
    ▼
Context Formatting
    │  - Combine vector chunks + graph blocks
    │  - Build source citations
    │
    ▼
LLM Call (Gemini)
    │
    ▼
Answer + Sources
```

### 4.2 Topic shift detection (word overlap)

```python
def _topic_shift(self, current_q: str, recent_qs: list[str]) -> bool:
    if not recent_qs:
        return True
    current_words = set(current_q.lower().split())
    recent_words = set(" ".join(recent_qs).lower().split())
    overlap = len(current_words & recent_words) / max(len(current_words), 1)
    return overlap < 0.12  # Dưới 12% từ trùng → topic mới
```

**Tại sao không dùng LLM?** Tiết kiệm latency và token. Word overlap đủ chính xác cho trường hợp này.

### 4.3 Broad query decomposition

```python
DYNASTIES = ["Ngô", "Đinh", "Tiền Lê", "Lý", "Trần", "Hồ", "Hậu Lê", ...]

async def _decompose_broad_query(self, question: str) -> list[dict]:
    tasks = [self.get_vector(f"{question} triều {dynasty}") for dynasty in DYNASTIES]
    results = await asyncio.gather(*tasks)  # Chạy song song
    return [item for sublist in results for item in sublist]  # Flatten
```

### 4.4 Hybrid scoring sau retrieval

```python
def hybrid_score(rrf_score: float, lexical_score: float, count_bonus: float) -> float:
    FUSED_WEIGHT = 0.7
    LEXICAL_WEIGHT = 0.3
    COUNT_BONUS = 0.2
    return rrf_score * FUSED_WEIGHT + lexical_score * LEXICAL_WEIGHT + count_bonus * COUNT_BONUS
```

**count_bonus:** Parent document xuất hiện ở nhiều chunks → nội dung đó liên quan nhiều → bonus thêm điểm.

---

## 5. LLM

**File tham khảo:** `app/services/chatbot/index_and_retrieve/providers.py`, `chatbot/engine.py`

### 5.1 OpenAI SDK với Gemini endpoint

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

response = await client.chat.completions.create(
    model="gemini-2.0-flash-lite",
    messages=[
        {"role": "system", "content": "Bạn là sử gia người Việt..."},
        {"role": "user", "content": "Trận Bạch Đằng 938 diễn ra như thế nào?"},
    ],
    temperature=0.3,
    max_tokens=2048,
)
answer = response.choices[0].message.content
```

**Tại sao dùng OpenAI SDK cho Gemini?** Gemini cung cấp OpenAI-compatible endpoint → không cần cài thêm `google-generativeai`, reuse code dễ switch provider.

### 5.2 Streaming response từ LLM

```python
stream = await client.chat.completions.create(
    model="gemini-2.0-flash-lite",
    messages=messages,
    stream=True,
)

full_text = ""
async for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    full_text += delta
    yield delta  # Gửi từng token về client (SSE)
```

### 5.3 Prompt template — Cấu trúc dùng trong dự án

```python
SYSTEM_PROMPT = """Bạn là chuyên gia lịch sử Việt Nam...
Chỉ sử dụng thông tin trong ngữ cảnh được cung cấp."""

def build_prompt(question: str, graph_context: str, vector_context: str, history: str):
    return f"""
[THỰC_THỂ VÀ QUAN_HỆ TỪ KNOWLEDGE GRAPH]
{graph_context}

[VĂN_BẢN_GỐC TỪ TÀI LIỆU]
{vector_context}

[LỊCH SỬ HỘI THOẠI]
{history}

[CÂU HỎI]
{question}
"""
```

**Thứ tự trong prompt quan trọng:** LLM thường pay attention nhiều hơn cho phần đầu và cuối. Knowledge graph đặt đầu (overview), câu hỏi đặt cuối (focus).

---

## 6. Streaming

**File tham khảo:** `app/server/routers/chatbot_api.py`

### 6.1 Server-Sent Events (SSE) — Server side

```python
from fastapi.responses import StreamingResponse
import json

async def token_generator(question: str, history: list):
    engine = get_engine()
    async for token in engine.ask_with_sources_stream(question, history):
        # SSE format bắt buộc: "data: {...}\n\n"
        payload = json.dumps({"token": token})
        yield f"data: {payload}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"

@router.post("/api/ask/stream")
async def ask_stream(body: AskRequest):
    return StreamingResponse(
        token_generator(body.question, []),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

**`X-Accel-Buffering: no`:** Báo Nginx không buffer response → tokens về client ngay lập tức.

### 6.2 Client-side SSE (JavaScript)

```javascript
// Dùng fetch + ReadableStream (hỗ trợ POST)
const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question: "..."}),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    for (const line of text.split("\n")) {
        if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));
            if (data.done) return;
            appendToken(data.token);
        }
    }
}
```

### 6.3 Concurrency control với Semaphore

```python
import asyncio

# Giới hạn 4 concurrent LLM requests (tránh quá tải API)
_semaphore = asyncio.Semaphore(4)

async def call_llm_with_limit(messages):
    async with _semaphore:
        return await client.chat.completions.create(...)
```

---

## 7. Rate Limiting

**File tham khảo:** `app/services/chatbot/index_and_retrieve/providers.py`

### 7.1 AsyncRequestRateLimiter (sliding window)

```python
import asyncio

class AsyncRequestRateLimiter:
    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm  # Giây giữa mỗi request
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = asyncio.get_event_loop().time()

limiter = AsyncRequestRateLimiter(rpm=60)

async def call_llm(prompt):
    await limiter.acquire()
    return await client.chat.completions.create(...)
```

### 7.2 Retry với exponential backoff

```python
async def call_with_retry(func, *args, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await func(*args)
        except Exception as e:
            if "model_not_found" in str(e):
                raise  # Non-retryable error → fail ngay
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)  # 1s → 2s → 4s
            await asyncio.sleep(delay)
```

---

## 8. Pydantic

**File tham khảo:** `app/server/schemas.py`

### 8.1 BaseModel — Request/Response schema

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class AskRequest(BaseModel):
    question: str
    persona_slug: Optional[str] = None
    conversation_id: Optional[str] = None
    include_contexts: bool = False

class SourceItem(BaseModel):
    title: str
    file: str
    page: int
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    verification: str = ""
    trial_remaining: Optional[int] = None
```

**Pattern:** `Optional[str] = None` → field không bắt buộc. `Field(default_factory=list)` cho list mặc định (không dùng `default=[]` vì mutable default gây bug).

### 8.2 Validation tự động

FastAPI tự validate request body trước khi vào handler. Nếu sai type → 422 Unprocessable Entity.

```python
# POST /ask với body {"question": 123} → FastAPI tự convert 123 thành "123"
# POST /ask với body {} → lỗi vì "question" là required field
```

---

## 9. Database

**File tham khảo:** `app/server/db/connection.py`, `app/server/db/users.py`, `app/server/db/conversations.py`

### 9.1 Dual backend — SQLite vs PostgreSQL

```python
import os

def _use_postgres() -> bool:
    return os.getenv("ENV", "dev") == "production"
```

### 9.2 SQLite — Cách dùng trong async context

SQLite là blocking I/O nên phải wrap với `asyncio.to_thread`:

```python
import sqlite3
import asyncio

def _get_user_sync(email: str) -> dict:
    conn = sqlite3.connect("db.sqlite3")
    conn.row_factory = sqlite3.Row  # Trả về dict-like thay vì tuple
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None

async def get_user(email: str) -> dict:
    return await asyncio.to_thread(_get_user_sync, email)
```

**Tại sao dùng `?` placeholder?** Tránh SQL injection. Không bao giờ dùng f-string để build SQL.

### 9.3 SQLite — WAL mode (Write-Ahead Logging)

```python
conn = sqlite3.connect("db.sqlite3")
conn.execute("PRAGMA journal_mode=WAL")  # Cho phép đọc đồng thời khi đang ghi
conn.execute("PRAGMA foreign_keys=ON")   # Enforce foreign key constraints
```

### 9.4 asyncpg — PostgreSQL async

```python
import asyncpg

# Tạo connection pool (gọi 1 lần lúc startup)
pool = await asyncpg.create_pool(
    dsn="postgresql://user:pass@localhost/dbname",
    min_size=2,
    max_size=10,
    timeout=30,
)

# Dùng pool trong route handler
async def get_user(email: str) -> dict:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM users WHERE email = $1", email)
        return dict(row) if row else None
```

**Lưu ý:** PostgreSQL dùng `$1`, `$2` (positional). SQLite dùng `?` hoặc `:name`.

### 9.5 Schema SQL — Tạo bảng

```sql
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT DEFAULT '',
    password_hash TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
```

### 9.6 Transaction — Lưu nhiều bảng cùng lúc

```python
# SQLite
def _save_turn_sync(conv_id, question, answer):
    conn = sqlite3.connect("db.sqlite3")
    try:
        conn.execute("BEGIN")
        conn.execute("INSERT INTO messages (...) VALUES (...)", (...))
        conn.execute("UPDATE conversations SET message_count = message_count + 1 WHERE id = ?", (conv_id,))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

# asyncpg
async def save_turn(conn, conv_id, question, answer):
    async with conn.transaction():
        await conn.execute("INSERT INTO messages ...", ...)
        await conn.execute("UPDATE conversations ...", ...)
```

---

## 10. Authentication

**File tham khảo:** `app/server/auth/`

### 10.1 Password hashing với bcrypt

```python
import bcrypt
import asyncio

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

# Vì bcrypt blocking → wrap với asyncio.to_thread
async def hash_password_async(password: str) -> str:
    return await asyncio.to_thread(hash_password, password)
```

### 10.2 Session token với itsdangerous

```python
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

SECRET_KEY = os.getenv("SECRET_KEY")
serializer = URLSafeTimedSerializer(SECRET_KEY, salt="vical-session")

def create_token(user_id: str) -> str:
    return serializer.dumps({"id": user_id})

def verify_token(token: str, max_age: int = 30 * 24 * 3600) -> dict:
    try:
        return serializer.loads(token, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None
```

### 10.3 Set/Clear cookie

```python
from fastapi import Response

def set_session_cookie(response: Response, token: str):
    response.set_cookie(
        key="vical_session",
        value=token,
        httponly=True,        # JS không đọc được (chống XSS)
        samesite="lax",       # Chống CSRF một phần
        secure=IS_PRODUCTION, # Chỉ gửi qua HTTPS trong prod
        max_age=30 * 24 * 3600,
    )

def clear_session_cookie(response: Response):
    response.delete_cookie("vical_session")
```

### 10.4 Google OAuth với authlib

```python
from authlib.integrations.httpx_client import AsyncOAuth2Client

# Bước 1: Redirect user đến Google
@router.get("/auth/google/login")
async def google_login(request: Request):
    client = AsyncOAuth2Client(
        client_id=GOOGLE_CLIENT_ID,
        redirect_uri="http://localhost:8000/auth/google/callback",
    )
    uri, state = client.create_authorization_url(
        "https://accounts.google.com/o/oauth2/auth",
        scope="openid email profile"
    )
    request.session["oauth_state"] = state
    return RedirectResponse(uri)

# Bước 2: Google redirect về callback với ?code=...
@router.get("/auth/google/callback")
async def google_callback(request: Request, code: str, state: str):
    client = AsyncOAuth2Client(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        state=request.session["oauth_state"],
    )
    await client.fetch_token(
        "https://oauth2.googleapis.com/token",
        code=code,
        redirect_uri="http://localhost:8000/auth/google/callback",
    )
    resp = await client.get("https://www.googleapis.com/oauth2/v2/userinfo")
    google_user = resp.json()
    # {"email": "...", "name": "...", "picture": "...", "id": "..."}

    user = await upsert_oauth_user(
        provider="google",
        provider_user_id=google_user["id"],
        email=google_user["email"],
        display_name=google_user["name"],
    )
    ...
```

### 10.5 Rate limiting với slowapi

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@router.post("/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, email: str = Form(...)):
    ...
```

---

## 11. FastAPI

**File tham khảo:** `app/server/main.py`, `app/server/routers/`

### 11.1 Khởi tạo app và lifespan

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code chạy khi server khởi động (startup)
    await init_engine()
    yield
    # Code chạy khi server tắt (shutdown)
    cleanup()

app = FastAPI(lifespan=lifespan)
```

**Tại sao dùng lifespan?** Thay cho `on_event("startup")` (deprecated), dùng để warm-up engine, kết nối DB, v.v.

### 11.2 Router — Tách route ra file riêng

```python
# app/server/routers/chatbot_api.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/ask")
async def ask(request: AskRequest):
    ...

# app/server/main.py
from app.server.routers import chatbot_api
app.include_router(chatbot_api.router)
```

### 11.3 Dependency Injection — `Depends`

```python
from fastapi import Depends, HTTPException, Cookie

async def get_current_user(vical_session: str = Cookie(default=None)):
    if not vical_session:
        raise HTTPException(status_code=401)
    user = verify_token(vical_session)
    return user

@router.get("/history")
async def history(user = Depends(get_current_user)):
    return await get_conversations(user["id"])
```

**Pattern:** Dependency tái sử dụng ở nhiều route. Nếu không có user → tự động raise 401.

### 11.4 Request Body vs Form Data vs Cookie

```python
from fastapi import Form, Cookie, Request

# JSON body (dùng Pydantic model)
@router.post("/ask")
async def ask(body: AskRequest):
    ...

# Form data (HTML form submit)
@router.post("/auth/login")
async def login(email: str = Form(...), password: str = Form(...)):
    ...

# Cookie
@router.get("/me")
async def me(vical_session: str = Cookie(default=None)):
    ...

# Raw request (khi cần IP, headers)
@router.post("/ask")
async def ask(request: Request):
    ip = request.client.host
    ...
```

### 11.5 Response — JSON, HTML, Redirect

```python
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/server/templates")

# JSON response
@router.get("/health")
async def health():
    return {"status": "ok"}  # FastAPI tự serialize thành JSON

# HTML response (Jinja2)
@router.get("/ask")
async def page_ask(request: Request, user=Depends(get_current_user)):
    return templates.TemplateResponse("Ask_question.html", {
        "request": request,
        "user": user,
    })

# Redirect
@router.get("/auth/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("vical_session")
    return response
```

### 11.6 Middleware và Static Files

```python
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
```

---

## 12. Jinja2

**File tham khảo:** `app/server/templates/`

### 12.1 Cơ bản — Truyền biến vào template

```python
# Python (route) — "request" là bắt buộc
return templates.TemplateResponse("Ask_question.html", {
    "request": request,
    "user": current_user,
    "personas": personas_list,
})
```

```html
<!-- HTML template -->
{% if user %}
    <p>Xin chào, {{ user.display_name }}</p>
{% else %}
    <a href="/auth/login">Đăng nhập</a>
{% endif %}

{% for persona in personas %}
    <div class="persona-card">
        <img src="{{ persona.avatar_url }}">
        <h3>{{ persona.name }}</h3>
    </div>
{% endfor %}
```

### 12.2 Template inheritance — base.html

```html
<!-- base.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}VICAL{% endblock %}</title>
</head>
<body>
    {% include "navbar.html" %}
    {% block content %}{% endblock %}
</body>
</html>

<!-- Ask_question.html -->
{% extends "base.html" %}
{% block title %}Hỏi đáp{% endblock %}
{% block content %}
    <div class="chat-container">...</div>
{% endblock %}
```

### 12.3 Filters và escape

```html
{{ user.bio | truncate(100) }}    <!-- Cắt 100 ký tự -->
{{ content | safe }}               <!-- Render HTML (cẩn thận XSS!) -->
{{ date | default("N/A") }}        <!-- Giá trị mặc định nếu None -->
```

---

## 13. Patterns Quan Trọng

### 13.1 Singleton Pattern — Shared Engine

```python
# app/services/chatbot/shared_engine.py
_engine: VietnamHistoryQueryEngine | None = None
_lock = asyncio.Lock()

async def init_engine():
    global _engine
    async with _lock:
        if _engine is None:
            _engine = VietnamHistoryQueryEngine()
            await _engine.initialize()

def get_engine() -> VietnamHistoryQueryEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine
```

**Lý do:** Load embedding model (300MB+) và Qdrant connection chỉ làm 1 lần, tái dùng cho toàn bộ requests.

### 13.2 asyncio.gather — Parallel execution

```python
# Chạy vector search và graph search song song, không tuần tự
vector_results, graph_results = await asyncio.gather(
    self.get_vector(query),
    self.get_graph(query),
)
```

**Khi nào dùng gather?** Khi các tasks độc lập nhau (không cần kết quả của nhau). Nếu task B cần kết quả của task A → phải await tuần tự.

### 13.3 Dataclass — Persona config

```python
from dataclasses import dataclass

@dataclass
class PersonaConfig:
    slug: str
    name: str
    era: str
    knowledge_cutoff_year: int
    system_prompt: str
    blocked_keywords: list[str]
    out_of_bounds_response: str

NGO_QUYEN = PersonaConfig(
    slug="ngo-quyen",
    name="Ngô Quyền",
    era="898-944",
    knowledge_cutoff_year=944,
    system_prompt="Ta là Ngô Quyền, người lãnh đạo trận Bạch Đằng...",
    blocked_keywords=["điện thoại", "internet", "máy tính", "Hồ Chí Minh"],
    out_of_bounds_response="Điều ngươi nhắc đến vượt xa thời đại của ta...",
)
```

### 13.4 Environment variables pattern

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

GEMINI_KEY = os.getenv("GEMINI_KEY")
ENV = os.getenv("ENV", "dev")  # Default "dev" nếu không set

def is_production() -> bool:
    return ENV == "production"
```

---

## 14. Testing

**File tham khảo:** `tests/`

### 14.1 pytest + fixtures

```python
# tests/conftest.py
import pytest
import sqlite3
import tempfile

@pytest.fixture
async def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = sqlite3.connect(f.name)
        conn.executescript("""
            CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT UNIQUE, ...);
            CREATE TABLE conversations (...);
        """)
        yield f.name
        conn.close()
```

### 14.2 Test async functions

```python
import pytest

@pytest.mark.asyncio
async def test_create_user(tmp_db):
    user = await create_user(db_path=tmp_db, email="test@test.com", password="secret123")
    assert user["email"] == "test@test.com"
    assert user["password_hash"] is not None
```

### 14.3 RAGAS evaluation — Đo chất lượng RAG

```python
# tests/eval_ragas.py
from ragas import evaluate
from ragas.metrics import context_relevancy, answer_relevancy, faithfulness
from datasets import Dataset

samples = [
    {
        "question": "Trận Bạch Đằng 938 xảy ra ở đâu?",
        "answer": answer_from_engine,
        "contexts": [chunk.text for chunk in retrieved_chunks],
        "ground_truth": "Sông Bạch Đằng, tỉnh Quảng Ninh",
    }
]

dataset = Dataset.from_list(samples)
score = evaluate(dataset, metrics=[context_relevancy, answer_relevancy, faithfulness])
print(score)
```

**3 metrics chính:**
- `context_relevancy`: Chunks retrieved có liên quan đến câu hỏi không?
- `answer_relevancy`: Câu trả lời có trả lời đúng câu hỏi không?
- `faithfulness`: Câu trả lời có bịa thêm thông tin không có trong context không?

---

## 15. Docker + Deployment

**File tham khảo:** `Dockerfile`, `docker-compose.yml`, `nginx.conf`

### 15.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 15.2 docker-compose.yml

```yaml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./app/data/qdrant_db:/qdrant/storage

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: vical
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: vical
    volumes:
      - pgdata:/var/lib/postgresql/data

  web:
    build: .
    ports:
      - "8000:8000"
    depends_on: [qdrant, db]
    env_file: .env.production

volumes:
  pgdata:
```

### 15.3 Nginx reverse proxy

```nginx
server {
    listen 80;

    location / {
        proxy_pass http://web:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Quan trọng cho SSE streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    # Static files trực tiếp qua nginx (nhanh hơn FastAPI)
    location /static/ {
        alias /app/static/;
        expires 7d;
        gzip on;
    }
}
```

---

## Gợi Ý Học Theo Thứ Tự

**Phase 1 — RAG Core (3-4 tuần)**
1. **Tuần 1:** Embeddings (section 1) → hiểu dense/sparse, chạy thử embed một câu
2. **Tuần 1-2:** Qdrant (section 2) → index vài documents, thử search
3. **Tuần 2:** LightRAG (section 3) → ingest document, query graph
4. **Tuần 3:** RAG Pipeline (section 4) → đọc `engine.py`, trace toàn bộ luồng
5. **Tuần 3:** LLM + Streaming (sections 5, 6) → gọi thử Gemini, stream response
6. **Tuần 4:** Rate Limiting (section 7) → hiểu tại sao cần, cách implement

**Phase 2 — Web & Data (2-3 tuần)**
7. **Tuần 5:** Pydantic + Database (sections 8, 9) → viết CRUD functions
8. **Tuần 5-6:** Auth (section 10) → test login/OAuth flow
9. **Tuần 6:** FastAPI + Jinja2 (sections 11, 12) → chạy server, xem template

**Phase 3 — Deploy (1 tuần)**
10. **Tuần 7:** Patterns (section 13) → đọc lại code với mental model mới
11. **Tuần 7:** Testing + Docker (sections 14, 15) → chạy tests, build container

---

## File Nên Đọc Theo Thứ Tự

| Thứ tự | File | Tại sao |
|--------|------|---------|
| 1 | `app/data/process_data/e5_embeddings.py` | Hiểu cách embed text |
| 2 | `app/services/chatbot/index_and_retrieve/qdrant_index.py` | Cách index vào Qdrant |
| 3 | `app/services/chatbot/index_and_retrieve/retriever.py` | Hybrid search logic |
| 4 | `app/services/chatbot/index_and_retrieve/lightrag_index.py` | Knowledge graph |
| 5 | `app/services/chatbot/chatbot/engine.py` | **Core RAG pipeline** |
| 6 | `app/services/chatbot/shared_engine.py` | Singleton pattern |
| 7 | `app/server/schemas.py` | Data models |
| 8 | `app/server/db/connection.py` | DB setup |
| 9 | `app/server/auth/session.py` | Auth flow |
| 10 | `app/server/main.py` | Toàn bộ app lifecycle |
