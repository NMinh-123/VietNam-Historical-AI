"""FastAPI app — khởi tạo, middleware, và include routers."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

_APP_DIR = Path(__file__).resolve().parents[1]
_SERVER_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _APP_DIR / "static"
_TEMPLATES_DIR = _SERVER_DIR / "templates"
_DB_PATH = _SERVER_DIR / "db.sqlite3"
_TIMELINE_PATH = _APP_DIR.parent / "data" / "timeline.sqlite3"

if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import db
from server.routers import pages as pages_router
from server.routers import history_api as history_router
from server.routers import chatbot_api as chatbot_router
import auth as auth_router

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _logger.info("Khởi tạo database...")
    await db.init_db(_TIMELINE_PATH, _DB_PATH)

    _logger.info("Khởi tạo VietnamHistoryQueryEngine...")
    from services.chatbot.shared_engine import init_engine
    try:
        engine = init_engine()
        await engine.start()
        try:
            await asyncio.wait_for(engine._warmup_task, timeout=60.0)
            _logger.info("✓ Engine warm-up hoàn tất, sẵn sàng nhận request.")
        except asyncio.TimeoutError:
            _logger.warning("Engine warm-up timeout 60s — sẽ lazy-load khi có request.")
    except Exception as exc:
        _logger.error("Không thể khởi tạo engine: %s", exc)
        _logger.warning("Server sẽ chạy nhưng /ask endpoint sẽ lazy-load engine lần đầu (chậm).")

    pages_router.set_templates(templates)
    auth_router.set_templates(templates)

    yield
    _logger.info("Server đang tắt.")


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Vical Chatbot API", version="3.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_SESSION_SECRET = os.getenv("SECRET_KEY") or os.urandom(32).hex()
_IS_PROD = os.getenv("ENV", "dev").lower() == "production"
app.add_middleware(SessionMiddleware, secret_key=_SESSION_SECRET, same_site="lax", https_only=_IS_PROD)

_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

app.include_router(pages_router.router)
app.include_router(history_router.router)
app.include_router(chatbot_router.router)
app.include_router(auth_router.router)
