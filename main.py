"""main.py — Điểm khởi chạy, lắp ráp toàn bộ pipeline Vical AI."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ── Python path setup ─────────────────────────────────────────────────────────
# Phải chạy trước tất cả import khác để các module trong app/ có thể tìm thấy nhau
_ROOT_DIR = Path(__file__).resolve().parent
_APP_DIR = _ROOT_DIR / "app"
_SERVER_DIR = _APP_DIR / "server"

for _path in [str(_ROOT_DIR), str(_APP_DIR), str(_SERVER_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ── Load biến môi trường ──────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(_ROOT_DIR / ".env")

# ── FastAPI & middleware ──────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Paths ─────────────────────────────────────────────────────────────────────
_STATIC_DIR = _APP_DIR / "static"
_TEMPLATES_DIR = _APP_DIR / "server" / "templates"
_DB_PATH = _APP_DIR / "server" / "db.sqlite3"
_TIMELINE_PATH = _ROOT_DIR / "data" / "timeline.sqlite3"

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── Auto re-index Qdrant nếu collection mất sau restart ───────────────────────

async def _auto_reindex_qdrant() -> None:
    from qdrant_client import QdrantClient
    from services.chatbot.index_and_retrieve import (
        COLLECTION_NAME,
        QDRANT_HOST,
        QDRANT_PORT,
        RAW_DATA_PATH,
        qdrant_ingest,
    )
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        if client.collection_exists(COLLECTION_NAME):
            _logger.info("Qdrant collection '%s' OK.", COLLECTION_NAME)
            return
        _logger.warning("Collection '%s' không tồn tại. Đang tự động index lại...", COLLECTION_NAME)
        if not RAW_DATA_PATH.exists():
            _logger.error("Thiếu dữ liệu gốc tại '%s'. Hãy chạy: python run_qdrant_index.py", RAW_DATA_PATH)
            return
        await qdrant_ingest(
            test_mode=False,
            recreate_collection=True,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
        )
        _logger.info("Auto re-index Qdrant hoàn tất.")
    except Exception as exc:
        _logger.error("Auto re-index Qdrant thất bại: %s", exc, exc_info=True)


# ── Lifespan: khởi tạo DB + engine ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _logger.info("Khởi tạo database...")
    import db
    await db.init_db(_TIMELINE_PATH, _DB_PATH)
    await db.cleanup_revoked_sessions()

    _logger.info("Khởi tạo RAG engine...")
    from services.chatbot.shared_engine import init_engine
    try:
        engine = init_engine()
        await engine.start()
        try:
            await asyncio.wait_for(engine._warmup_task, timeout=60.0)
            _logger.info("Engine warm-up hoàn tất — sẵn sàng nhận request.")
        except asyncio.TimeoutError:
            _logger.warning("Engine warm-up timeout 60s — sẽ lazy-load khi có request đầu tiên.")
    except Exception as exc:
        _logger.error("Không thể khởi tạo engine: %s", exc)

    asyncio.create_task(_auto_reindex_qdrant())

    # Inject Jinja2 templates vào page/auth routers
    from src.api.routes import set_pages_templates, set_auth_templates
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    set_pages_templates(templates)
    set_auth_templates(templates)

    yield
    _logger.info("Server đang tắt.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Vical AI — Vietnam History Chatbot",
    version="3.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_SECRET = os.getenv("SECRET_KEY") or os.urandom(32).hex()
_IS_PROD = os.getenv("ENV", "dev").lower() == "production"

app.add_middleware(
    SessionMiddleware,
    secret_key=_SECRET,
    same_site="lax",
    https_only=_IS_PROD,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:8001").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# ── Include routers ───────────────────────────────────────────────────────────
from src.api.routes import pages_router, chatbot_router, history_router, auth_router

app.include_router(pages_router)
app.include_router(history_router)
app.include_router(chatbot_router)
app.include_router(auth_router)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        reload=not _IS_PROD,
        log_level="info",
    )
