"""Unit tests cho auth/session.py — token signing, cookie, get_current_user."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
SERVER_DIR = APP_DIR / "server"
for p in (str(APP_DIR), str(SERVER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(autouse=True)
def set_secret_key(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-unit-tests-only")
    # Reload module để dùng key mới
    import importlib
    import auth.session as _sess
    importlib.reload(_sess)
    yield


class TestSessionToken:
    def test_create_and_decode_roundtrip(self):
        from auth.session import create_session_token, decode_session_token
        token = create_session_token("user-123")
        assert decode_session_token(token) == "user-123"

    def test_tampered_token_returns_none(self):
        from auth.session import create_session_token, decode_session_token
        token = create_session_token("user-abc")
        tampered = token[:-4] + "xxxx"
        assert decode_session_token(tampered) is None

    def test_expired_token_returns_none(self):
        from auth.session import decode_session_token
        from itsdangerous import URLSafeTimedSerializer
        # Tạo token với timestamp giả ở quá khứ xa
        signer = URLSafeTimedSerializer("test-secret-key-for-unit-tests-only", salt="vical-session")
        with patch("itsdangerous.timed.time") as mock_time:
            mock_time.return_value = 0  # timestamp = epoch 0
            token = signer.dumps({"uid": "user-exp"})
        # max_age=1 giây nhưng token được tạo từ epoch 0 — chắc chắn hết hạn
        assert decode_session_token(token, max_age=1) is None

    def test_different_secrets_are_incompatible(self):
        from itsdangerous import URLSafeTimedSerializer
        s1 = URLSafeTimedSerializer("secret-1", salt="vical-session")
        s2 = URLSafeTimedSerializer("secret-2", salt="vical-session")
        token = s1.dumps({"uid": "user-x"})
        from itsdangerous import BadSignature
        with pytest.raises(BadSignature):
            s2.loads(token, max_age=3600)

    def test_token_contains_no_plaintext_uid(self):
        from auth.session import create_session_token
        token = create_session_token("supersecretuid")
        # Token là signed/serialized, không phải plain text
        assert "supersecretuid" not in token


class TestGetCurrentUser:
    def test_returns_none_when_no_cookie(self, tmp_db):
        from auth.session import get_current_user, SESSION_COOKIE
        request = MagicMock()
        request.cookies = {}
        assert get_current_user(request) is None

    def test_returns_none_for_invalid_token(self, tmp_db):
        from auth.session import get_current_user, SESSION_COOKIE
        request = MagicMock()
        request.cookies = {SESSION_COOKIE: "invalid.token.value"}
        assert get_current_user(request) is None

    def test_returns_user_for_valid_token(self, tmp_db):
        import db as _db
        from auth.session import create_session_token, get_current_user, SESSION_COOKIE
        user = _db.create_user(email="sess@test.com", display_name="Session User")
        token = create_session_token(user["id"])
        request = MagicMock()
        request.cookies = {SESSION_COOKIE: token}
        result = get_current_user(request)
        assert result is not None
        assert result["id"] == user["id"]

    def test_returns_none_for_deleted_user(self, tmp_db):
        """Token valid nhưng user đã xóa khỏi DB."""
        from auth.session import create_session_token, get_current_user, SESSION_COOKIE
        token = create_session_token("deleted-user-id")
        request = MagicMock()
        request.cookies = {SESSION_COOKIE: token}
        assert get_current_user(request) is None


class TestCallbackUrl:
    def test_uses_redirect_base_url_from_env(self, monkeypatch):
        import importlib
        import auth.session as _sess
        monkeypatch.setenv("REDIRECT_BASE_URL", "https://example.com")
        importlib.reload(_sess)
        request = MagicMock()
        url = _sess._callback_url(request, "google")
        assert url == "https://example.com/auth/google/callback"

    def test_falls_back_to_request_host(self, monkeypatch):
        import importlib
        import auth.session as _sess
        monkeypatch.delenv("REDIRECT_BASE_URL", raising=False)
        importlib.reload(_sess)
        request = MagicMock()
        request.headers = {}
        request.url.scheme = "http"
        request.url.netloc = "localhost:8000"
        url = _sess._callback_url(request, "google")
        assert url == "http://localhost:8000/auth/google/callback"
