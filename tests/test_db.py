"""Unit tests cho db/ package — không cần server, không cần Qdrant/LLM."""

from __future__ import annotations

import pytest


# ── Users ─────────────────────────────────────────────────────────────────────

class TestCreateUser:
    async def test_creates_user_and_returns_dict(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="a@test.com", display_name="Alice", password="secret123")
        assert user["email"] == "a@test.com"
        assert user["display_name"] == "Alice"
        assert user["id"]

    async def test_password_is_hashed(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="b@test.com", password="plaintext")
        assert user["password_hash"] != "plaintext"
        assert user["password_hash"].startswith("$2b$")

    async def test_no_password_stores_null(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="c@test.com")
        assert user["password_hash"] is None

    async def test_duplicate_email_raises(self, tmp_db):
        from app import db as _db
        import sqlite3
        await _db.create_user(email="dup@test.com", password="pw123456")
        with pytest.raises(sqlite3.IntegrityError):
            await _db.create_user(email="dup@test.com", password="other123")


class TestVerifyPassword:
    async def test_correct_password_returns_true(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="vp@test.com", password="correct123")
        assert await _db.verify_password(user, "correct123") is True

    async def test_wrong_password_returns_false(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="vp2@test.com", password="correct123")
        assert await _db.verify_password(user, "wrong") is False

    async def test_no_password_hash_returns_false(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="vp3@test.com")
        assert await _db.verify_password(user, "anything") is False


class TestGetUser:
    async def test_get_by_email(self, tmp_db):
        from app import db as _db
        await _db.create_user(email="ge@test.com")
        user = await _db.get_user_by_email("ge@test.com")
        assert user is not None
        assert user["email"] == "ge@test.com"

    async def test_get_by_email_not_found(self, tmp_db):
        from app import db as _db
        assert await _db.get_user_by_email("nobody@test.com") is None

    async def test_get_by_id(self, tmp_db):
        from app import db as _db
        created = await _db.create_user(email="gi@test.com")
        user = await _db.get_user_by_id(created["id"])
        assert user["id"] == created["id"]

    async def test_get_by_id_not_found(self, tmp_db):
        from app import db as _db
        assert await _db.get_user_by_id("nonexistent-id") is None


class TestUpdateUserProfile:
    async def test_updates_display_name(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="upd@test.com", display_name="Old")
        updated = await _db.update_user_profile(user["id"], "New Name")
        assert updated["display_name"] == "New Name"

    async def test_returns_none_for_unknown_user(self, tmp_db):
        from app import db as _db
        result = await _db.update_user_profile("nonexistent", "Name")
        assert result is None


class TestUpsertOAuthAccount:
    async def test_creates_new_user_on_first_login(self, tmp_db):
        from app import db as _db
        user = await _db.upsert_oauth_account(
            provider="google",
            provider_user_id="google-123",
            email="oauth@test.com",
            display_name="OAuth User",
        )
        assert user["email"] == "oauth@test.com"
        assert user["display_name"] == "OAuth User"

    async def test_returns_same_user_on_second_login(self, tmp_db):
        from app import db as _db
        u1 = await _db.upsert_oauth_account(provider="google", provider_user_id="gid-1", email="same@test.com")
        u2 = await _db.upsert_oauth_account(provider="google", provider_user_id="gid-1", email="same@test.com")
        assert u1["id"] == u2["id"]

    async def test_does_not_overwrite_existing_display_name(self, tmp_db):
        from app import db as _db
        await _db.upsert_oauth_account(
            provider="google", provider_user_id="gid-2",
            email="named@test.com", display_name="Original",
        )
        user = await _db.upsert_oauth_account(
            provider="google", provider_user_id="gid-2",
            email="named@test.com", display_name="New Name",
        )
        assert user["display_name"] == "Original"

    async def test_get_oauth_providers(self, tmp_db):
        from app import db as _db
        user = await _db.upsert_oauth_account(provider="google", provider_user_id="gid-3", email="prov@test.com")
        await _db.upsert_oauth_account(provider="facebook", provider_user_id="fb-1", email="prov@test.com")
        providers = await _db.get_oauth_providers(user["id"])
        assert set(providers) == {"google", "facebook"}


class TestGetUserStats:
    async def test_returns_zero_for_new_user(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="stats@test.com")
        stats = await _db.get_user_stats(user["id"])
        assert stats["conversations"] == 0
        assert stats["messages"] == 0

    async def test_counts_only_own_conversations(self, tmp_db):
        from app import db as _db
        u1 = await _db.create_user(email="s1@test.com")
        u2 = await _db.create_user(email="s2@test.com")
        await _db.save_turn(None, "q", "a", [], user_id=u1["id"])
        await _db.save_turn(None, "q", "a", [], user_id=u2["id"])
        stats = await _db.get_user_stats(u1["id"])
        assert stats["conversations"] == 1
        assert stats["messages"] == 1


# ── Conversations ──────────────────────────────────────────────────────────────

class TestSaveTurn:
    async def test_creates_new_conversation(self, tmp_db):
        from app import db as _db
        conv_id = await _db.save_turn(None, "question", "answer", [])
        assert conv_id is not None

    async def test_appends_to_existing_conversation(self, tmp_db):
        from app import db as _db
        conv_id = await _db.save_turn(None, "q1", "a1", [])
        conv_id2 = await _db.save_turn(conv_id, "q2", "a2", [])
        assert conv_id == conv_id2

    async def test_unknown_conv_id_creates_new(self, tmp_db):
        from app import db as _db
        new_id = await _db.save_turn("nonexistent-conv-id", "q", "a", [])
        assert new_id != "nonexistent-conv-id"

    async def test_title_truncated_to_80_chars(self, tmp_db):
        from app import db as _db
        long_q = "x" * 200
        conv_id = await _db.save_turn(None, long_q, "a", [])
        result = await _db.get_messages(conv_id)
        assert len(result["title"]) <= 80

    async def test_user_isolation_prevents_append(self, tmp_db):
        from app import db as _db
        u1 = await _db.create_user(email="iso1@test.com")
        u2 = await _db.create_user(email="iso2@test.com")
        conv_id = await _db.save_turn(None, "q", "a", [], user_id=u1["id"])
        new_id = await _db.save_turn(conv_id, "q2", "a2", [], user_id=u2["id"])
        assert new_id != conv_id


class TestListConversations:
    async def test_returns_only_own_conversations(self, tmp_db):
        from app import db as _db
        u1 = await _db.create_user(email="lc1@test.com")
        u2 = await _db.create_user(email="lc2@test.com")
        await _db.save_turn(None, "q", "a", [], user_id=u1["id"])
        await _db.save_turn(None, "q", "a", [], user_id=u2["id"])
        convs = await _db.list_conversations(user_id=u1["id"])
        assert len(convs) == 1
        assert convs[0]["user_id"] == u1["id"]

    async def test_empty_for_user_with_no_convs(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="lc3@test.com")
        assert await _db.list_conversations(user_id=user["id"]) == []

    async def test_ordered_by_updated_desc(self, tmp_db):
        from app import db as _db
        import time
        user = await _db.create_user(email="lc4@test.com")
        id1 = await _db.save_turn(None, "first", "a", [], user_id=user["id"])
        time.sleep(0.01)
        id2 = await _db.save_turn(None, "second", "a", [], user_id=user["id"])
        convs = await _db.list_conversations(user_id=user["id"])
        assert convs[0]["id"] == id2  # mới nhất lên đầu


class TestGetMessages:
    async def test_returns_messages_for_owner(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="gm@test.com")
        conv_id = await _db.save_turn(None, "hello", "world", [], user_id=user["id"])
        result = await _db.get_messages(conv_id, user_id=user["id"])
        assert result is not None
        assert len(result["messages"]) == 2  # user + assistant

    async def test_returns_none_for_wrong_user(self, tmp_db):
        from app import db as _db
        u1 = await _db.create_user(email="gm1@test.com")
        u2 = await _db.create_user(email="gm2@test.com")
        conv_id = await _db.save_turn(None, "q", "a", [], user_id=u1["id"])
        assert await _db.get_messages(conv_id, user_id=u2["id"]) is None

    async def test_sources_deserialized_from_json(self, tmp_db):
        from app import db as _db
        sources = [{"title": "Sách A", "page": 10}]
        conv_id = await _db.save_turn(None, "q", "a", sources)
        result = await _db.get_messages(conv_id)
        assistant_msg = next(m for m in result["messages"] if m["role"] == "assistant")
        assert assistant_msg["sources"] == sources


class TestDeleteConversation:
    async def test_deletes_own_conversation(self, tmp_db):
        from app import db as _db
        user = await _db.create_user(email="del@test.com")
        conv_id = await _db.save_turn(None, "q", "a", [], user_id=user["id"])
        deleted = await _db.delete_conversation(conv_id, user_id=user["id"])
        assert deleted is True
        assert await _db.get_messages(conv_id, user_id=user["id"]) is None

    async def test_cannot_delete_other_users_conv(self, tmp_db):
        from app import db as _db
        u1 = await _db.create_user(email="del1@test.com")
        u2 = await _db.create_user(email="del2@test.com")
        conv_id = await _db.save_turn(None, "q", "a", [], user_id=u1["id"])
        deleted = await _db.delete_conversation(conv_id, user_id=u2["id"])
        assert deleted is False

    async def test_returns_false_for_nonexistent(self, tmp_db):
        from app import db as _db
        assert await _db.delete_conversation("no-such-id") is False


class TestGetRecentTurns:
    async def test_returns_empty_string_for_empty_conv(self, tmp_db):
        from app import db as _db
        assert await _db.get_recent_turns("nonexistent") == ""

    async def test_returns_empty_for_falsy_conv_id(self, tmp_db):
        from app import db as _db
        assert await _db.get_recent_turns("") == ""
        assert await _db.get_recent_turns(None) == ""

    async def test_formats_turns_correctly(self, tmp_db):
        from app import db as _db
        conv_id = await _db.save_turn(None, "Xin chào", "Chào bạn", [])
        turns = await _db.get_recent_turns(conv_id)
        assert "Người dùng" in turns
        assert "Trợ lý" in turns
        assert "Xin chào" in turns

    async def test_respects_max_turns_limit(self, tmp_db):
        from app import db as _db
        conv_id = await _db.save_turn(None, "q1", "a1", [])
        for i in range(2, 7):
            await _db.save_turn(conv_id, f"q{i}", f"a{i}", [])
        turns = await _db.get_recent_turns(conv_id, max_turns=2)
        lines = [l for l in turns.split("\n") if l.strip()]
        assert len(lines) <= 4


# ── Timeline ───────────────────────────────────────────────────────────────────

class TestGetDynasties:
    async def test_returns_empty_list_when_no_data(self, tmp_db):
        from app import db as _db
        dynasties = _db.get_dynasties()
        assert dynasties == []

    async def test_returns_dynasties_with_kings(self, tmp_db):
        import sqlite3
        from app import db as _db
        from db.connection import _TIMELINE_PATH

        conn = sqlite3.connect(_TIMELINE_PATH)
        conn.execute("INSERT INTO core_dynasty(id,name,'order') VALUES(1,'Nhà Lý',1)")
        conn.execute("INSERT INTO core_king(id,dynasty_id,name,'order') VALUES(1,1,'Lý Thái Tổ',1)")
        conn.commit()
        conn.close()

        dynasties = _db.get_dynasties()
        assert len(dynasties) == 1
        assert dynasties[0]["name"] == "Nhà Lý"
        assert len(dynasties[0]["kings"]) == 1
        assert dynasties[0]["kings"][0]["name"] == "Lý Thái Tổ"
