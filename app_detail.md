# Đánh giá chi tiết dự án VICAL AI
**Ngày:** 2026-05-01
**Reviewer:** Claude Code
**Score tổng thể:** 7.5/10

---

## Kiến trúc & Thiết kế — 8/10

**Điểm mạnh:**
- RAG tiên tiến: hybrid retrieval (dense E5 + sparse BM25) kết hợp knowledge graph (LightRAG)
- Query rewriting: xóa meta-instructions, decompose câu hỏi tổng quan thành sub-queries theo triều đại
- Parent-child chunking giữ ngữ cảnh tốt hơn flat chunking
- Separation of concerns: server/ (routes) tách services/ (business logic)
- Singleton pattern trong shared_engine.py tránh load model nhiều lần
- Async-first design: toàn bộ pipeline dùng async/await, parallel retrieval với asyncio.gather()

**Điểm yếu:**
- Thiếu abstraction layers: DB logic lẫn trong db.py, không có repository pattern
- LLM provider hard-coded cho Gemini, khó switch sang OpenAI/Anthropic
- Config management lộn xộn: env vars đọc trực tiếp, magic numbers không có constants
- Không có circuit breaker cho external services (Qdrant, Gemini)

---

## Bảo mật — 7/10

**Điểm mạnh:**
- Session: itsdangerous signed tokens, HttpOnly + SameSite=Lax + Secure
- Password: bcrypt hash
- OAuth: đúng flow với state CSRF protection
- Secrets trong .env, không hardcode

**Điểm yếu nghiêm trọng:**
1. **Authorization bug (CRITICAL):** db.py:281 - get_user_stats() đếm TẤT CẢ conversations của mọi user, không filter theo user_id
2. **Thiếu user isolation:** conversations không có user_id foreign key, bất kỳ ai cũng xem/xóa conversation của người khác
3. **Input validation yếu:** không sanitize HTML trong user input (XSS risk), không rate limiting cho /ask endpoint (DoS risk)

---

## Performance & Scalability — 6.5/10

**Điểm mạnh:**
- Async I/O cho endpoints và parallel retrieval
- Rate limiting cho LLM (AsyncRequestRateLimiter, semaphore)
- LightRAG có built-in cache cho LLM calls

**Điểm yếu:**
1. SQLite không scale cho concurrent writes, mỗi request mở connection mới, không có connection pooling
2. Singleton engine giữ models trong RAM suốt process lifetime, không có LRU cache cho embeddings
3. N+1 query problem trong db.py:302 get_dynasties() — nên dùng JOIN thay vì loop
4. Không có response caching — câu hỏi giống nhau vẫn chạy lại toàn bộ pipeline
5. Parent docstore load toàn bộ vào memory (json.load())

---

## Testing & Quality — 4/10

**Vấn đề nghiêm trọng:**
- 0 unit tests, 0 integration tests, 0 E2E tests
- Không có pytest.ini, tests/ directory
- Không có CI/CD: GitHub Actions, pre-commit hooks, linting
- Không có monitoring: error tracking (Sentry), metrics (Prometheus)
- Type hints không đầy đủ

---

## Code Quality — 7.5/10

**Điểm mạnh:**
- Docstrings tiếng Việt rõ ràng, comments giải thích WHY
- Error handling: try-catch với logging, HTTPException đúng status codes, retry logic cho transient errors
- Naming conventions: consistent snake_case, descriptive variable names

**Điểm yếu:**
- God objects: VietnamHistoryQueryEngine làm quá nhiều việc, db.py 320 lines với 20+ functions
- Magic strings: chat_type: str = "ask" nên dùng Enum, persona_slug: str = "" empty string là anti-pattern
- Inconsistent error messages: một số tiếng Việt, một số tiếng Anh, không có error codes

---

## Deployment & Ops — 5/10

**Điểm mạnh:**
- Docker Compose cho Qdrant với healthcheck, volume persistence

**Điểm yếu:**
- Không có Dockerfile cho app, nginx reverse proxy, HTTPS setup
- Không có graceful shutdown
- SQLite files không được backup, Qdrant data không có snapshot schedule
- /health endpoint quá đơn giản, không check Qdrant connection, không check LLM availability

---

## Khuyến nghị ưu tiên

### CRITICAL — Fix ngay

1. **Fix authorization bug:**
   ```sql
   ALTER TABLE conversations ADD COLUMN user_id TEXT REFERENCES users(id);
   -- Filter theo user trong mọi query
   ```

2. **Add rate limiting:**
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   @app.post("/ask")
   @limiter.limit("10/minute")
   async def ask(...):
   ```

3. **Input sanitization:**
   ```python
   import bleach
   question = bleach.clean(body.question)
   ```

### HIGH — Tuần tới

4. **Add tests:** unit tests cho db.py, auth.py; integration tests cho /ask endpoint; target 60% coverage
5. **Switch to PostgreSQL:** SQLite không production-ready cho concurrent writes, dùng SQLAlchemy ORM
6. **Add response caching:** LRU cache cho câu hỏi giống nhau với TTL

### MEDIUM — Tháng tới

7. **Refactor engine:** tách retriever, reranker, generator thành classes riêng; dependency injection cho LLM provider
8. **Add monitoring:** Sentry cho error tracking, Prometheus + Grafana cho metrics
9. **CI/CD pipeline:** GitHub Actions lint → test → build → deploy, pre-commit hooks: black, ruff, mypy

---

## Roadmap dự kiến

- **Sprint 1 (1 tuần):** Fix security bugs + add tests
- **Sprint 2 (2 tuần):** PostgreSQL migration + caching
- **Sprint 3 (1 tháng):** Monitoring + CI/CD
- **Target:** 9/10, sẵn sàng scale lên 1000+ users
