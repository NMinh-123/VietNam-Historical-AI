"""Integration tests — chạy với server thực (localhost:8001) và Qdrant thực.

Yêu cầu: server đang chạy tại http://localhost:8001
Chạy: pytest tests/test_integration.py -v
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

BASE = "http://localhost:8001"

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE, timeout=120.0) as c:
        yield c


# ── Health / infra ────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["rag_ready"] is True
        assert body["qdrant_ok"] is True

    def test_warmup_ready(self, client):
        r = client.get("/warmup")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["lightrag_ready"] is True
        assert body["qdrant_ok"] is True
        assert body["models_loaded"]["dense"] is True
        assert body["models_loaded"]["sparse"] is True
        from app.core.app_config import get_config
        _vdb = get_config().vectordb
        assert _vdb.collection_name in body["qdrant_collections"]
        assert _vdb.parent_collection_name in body["qdrant_collections"]


# ── Personas ──────────────────────────────────────────────────────────────────


class TestPersonas:
    def test_list_personas_returns_3(self, client):
        r = client.get("/personas")
        assert r.status_code == 200
        personas = r.json()
        assert len(personas) == 3

    def test_persona_slugs_correct(self, client):
        r = client.get("/personas")
        slugs = {p["slug"] for p in r.json()}
        assert slugs == {"ngo-quyen", "tran-hung-dao", "ho-chi-minh"}

    def test_persona_fields_present(self, client):
        r = client.get("/personas")
        for p in r.json():
            assert p.get("display_name")
            assert p.get("era_label")
            assert p.get("bio_short")
            assert p.get("portrait_url")
            assert p.get("accent_color")


# ── HTML pages ────────────────────────────────────────────────────────────────


class TestPages:
    def test_home_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_ask_page_200(self, client):
        r = client.get("/ask")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_timeline_200(self, client):
        r = client.get("/timeline")
        assert r.status_code == 200

    def test_library_200(self, client):
        r = client.get("/library")
        assert r.status_code == 200

    def test_history_200_or_redirect(self, client):
        r = client.get("/history", follow_redirects=False)
        assert r.status_code in (200, 302)

    def test_persona_page_redirect_or_200(self, client):
        r = client.get("/persona", follow_redirects=False)
        assert r.status_code in (200, 302)


# ── Input validation ──────────────────────────────────────────────────────────


class TestInputValidation:
    def test_empty_question_422(self, client):
        r = client.post("/api/ask", json={"question": ""})
        assert r.status_code == 422

    def test_missing_question_422(self, client):
        r = client.post("/api/ask", json={})
        assert r.status_code == 422

    def test_invalid_persona_404(self, client):
        r = client.post("/api/ask", json={"question": "test", "persona_slug": "fake-persona"})
        assert r.status_code == 404
        assert "Slug hợp lệ" in r.json()["detail"]

    def test_trial_status_unauthenticated(self, client):
        r = client.get("/api/trial-status")
        assert r.status_code == 200
        body = r.json()
        assert body["authenticated"] is False
        assert "remaining" in body


# ── RAG pipeline — answer quality ────────────────────────────────────────────


class TestAnswerQuality:
    """Kiểm tra chất lượng câu trả lời từ RAG pipeline."""

    def test_bach_dang_battle_returns_answer(self, client):
        r = client.post("/api/ask", json={"question": "Trận Bạch Đằng năm 938 diễn ra như thế nào?"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["answer"]) > 200
        assert body["sources"]
        assert len(body["sources"]) >= 1

    def test_bach_dang_answer_contains_relevant_keywords(self, client):
        r = client.post("/api/ask", json={"question": "Trận Bạch Đằng năm 938 diễn ra như thế nào?"})
        answer = r.json()["answer"].lower()
        keywords = ["ngô quyền", "bạch đằng", "nam hán", "cọc"]
        matched = [k for k in keywords if k in answer]
        assert len(matched) >= 3, f"Chỉ tìm thấy {matched} trong tổng số {keywords}"

    def test_sources_have_required_fields(self, client):
        r = client.post("/api/ask", json={"question": "Nhà Lý thành lập như thế nào?"})
        body = r.json()
        for src in body["sources"]:
            assert "index" in src
            assert "score" in src
            assert src["score"] >= 0

    def test_verification_field_present(self, client):
        r = client.post("/api/ask", json={"question": "Hồ Chí Minh là ai?"})
        body = r.json()
        assert body.get("verification") is not None
        assert "sử gia" in body["verification"].lower()

    def test_unknown_topic_graceful_fallback(self, client):
        r = client.post("/api/ask", json={"question": "Lịch sử của sao Hỏa?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"]

    def test_broad_query_returns_multiple_sources(self, client):
        r = client.post("/api/ask", json={"question": "Tổng quan các triều đại Việt Nam thế kỷ X đến XIV"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["answer"]) > 500
        assert len(body["sources"]) >= 5


# ── Persona chat quality ──────────────────────────────────────────────────────


class TestPersonaQuality:
    def test_ngo_quyen_answers_in_first_person(self, client):
        r = client.post("/api/ask", json={
            "question": "Chiến lược của ông trong trận Bạch Đằng?",
            "persona_slug": "ngo-quyen"
        })
        assert r.status_code == 200
        body = r.json()
        assert body["persona_slug"] == "ngo-quyen"
        answer = body["answer"].lower()
        # Ngô Quyền dùng "ta" hoặc first-person pronouns
        assert "ta " in answer or "ta\n" in answer or "của ta" in answer

    def test_tran_hung_dao_answers_in_character(self, client):
        r = client.post("/api/ask", json={
            "question": "Ba lần kháng chiến chống Nguyên Mông?",
            "persona_slug": "tran-hung-dao"
        })
        assert r.status_code == 200
        body = r.json()
        answer = body["answer"].lower()
        assert "nguyên" in answer or "mông" in answer or "quân giặc" in answer

    def test_temporal_guardrail_modern_tech_blocked_ngo_quyen(self, client):
        r = client.post("/api/ask", json={
            "question": "Bạn nghĩ gì về điện thoại thông minh?",
            "persona_slug": "ngo-quyen"
        })
        assert r.status_code == 200
        answer = r.json()["answer"]
        # Phải từ chối câu hỏi về công nghệ hiện đại
        assert len(answer) < 500, "Guardrail phải trả lời ngắn, không mô tả công nghệ"
        keywords_refused = ["vượt xa thời đại", "không biết", "chỉ biết", "rời cõi"]
        assert any(k.lower() in answer.lower() for k in keywords_refused), \
            f"Guardrail không từ chối câu hỏi. Answer: {answer[:200]}"

    def test_temporal_guardrail_modern_tech_blocked_tran_hung_dao(self, client):
        r = client.post("/api/ask", json={
            "question": "Máy bay và xe tăng có tác dụng gì trong chiến tranh?",
            "persona_slug": "tran-hung-dao"
        })
        assert r.status_code == 200
        answer = r.json()["answer"].lower()
        # Không nên trả lời chi tiết về máy bay/xe tăng
        assert "máy bay" not in answer or len(answer) < 400

    def test_persona_api_shortcut_route(self, client):
        r = client.post("/api/persona-chat/ho-chi-minh", json={
            "question": "Tuyên ngôn Độc lập 1945?"
        })
        assert r.status_code == 200
        body = r.json()
        assert body["persona_slug"] == "ho-chi-minh"


# ── SSE Streaming ─────────────────────────────────────────────────────────────


class TestStreaming:
    def test_stream_returns_tokens(self, client):
        with client.stream("POST", "/api/ask/stream",
                           json={"question": "Nhà Lý thành lập năm nào?"}) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")

            tokens = []
            done_received = False
            for line in r.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event["type"] == "token":
                        tokens.append(event["text"])
                    elif event["type"] == "done":
                        done_received = True
                        break

            assert len(tokens) > 0, "Không nhận được token nào"
            assert done_received, "Không nhận được sự kiện 'done'"
            full_text = "".join(tokens)
            assert len(full_text) > 100

    def test_stream_done_event_has_sources(self, client):
        sources_from_stream = None
        with client.stream("POST", "/api/ask/stream",
                           json={"question": "Ngô Quyền là ai?"}) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event["type"] == "done":
                        sources_from_stream = event.get("sources", [])
                        break
        assert sources_from_stream is not None
        assert isinstance(sources_from_stream, list)


# ── Query preprocessing ───────────────────────────────────────────────────────


class TestQueryPreprocessing:
    """Test các hàm xử lý query không cần LLM."""

    def test_rewrite_strips_meta_instruction(self):
        from app.core.prompts.prompt_templates import rewrite_query
        cases = [
            ("hãy giải thích về nhà Trần", "nhà Trần"),
            ("tóm tắt lịch sử nhà Lý", "lịch sử nhà Lý"),
            ("tại sao nhà Nguyễn sụp đổ", "nhà Nguyễn sụp đổ"),
            ("vì sao Ngô Quyền thắng", "Ngô Quyền thắng"),
        ]
        for original, expected_contains in cases:
            result = rewrite_query(original)
            assert expected_contains.lower() in result.lower(), \
                f"rewrite_query('{original}') = '{result}', expected to contain '{expected_contains}'"

    def test_broad_query_detection(self):
        from app.core.prompts.prompt_templates import is_broad_query
        assert is_broad_query("tổng quan lịch sử việt nam")
        assert is_broad_query("tóm tắt các triều đại phong kiến")
        assert not is_broad_query("Trần Hưng Đạo sinh năm nào?")
        assert not is_broad_query("Trận Bạch Đằng năm 938")

    def test_decompose_broad_query_covers_dynasties(self):
        from app.core.prompts.prompt_templates import decompose_broad_query, DYNASTIES
        sub_queries = decompose_broad_query("lịch sử các triều đại")
        assert len(sub_queries) == len(DYNASTIES)
        for q in sub_queries:
            assert isinstance(q, str) and len(q) > 5

    def test_topic_shift_detection(self):
        from app.core.prompts.prompt_templates import detect_topic_shift
        # turns phải dùng format {role, content} như get_recent_turns_list trả về
        turns_ngo_quyen = [
            {"role": "user", "content": "Ngô Quyền là ai?"},
            {"role": "assistant", "content": "Người khai quốc..."},
            {"role": "user", "content": "Trận Bạch Đằng 938?"},
            {"role": "assistant", "content": "Diễn ra trên sông..."},
        ]
        assert detect_topic_shift("nhà Trần thế kỷ XIII", turns_ngo_quyen) is True
        assert detect_topic_shift("chiến thuật của Ngô Quyền", turns_ngo_quyen) is False

    def test_build_retrieval_query_uses_window_context(self):
        from app.core.prompts.prompt_templates import build_retrieval_query
        # turns phải dùng format {role, content}
        turns = [
            {"role": "user", "content": "Ngô Quyền là ai?"},
            {"role": "assistant", "content": "Người khai quốc..."},
        ]
        # Câu hỏi followup ngắn → phải gắn context từ turns
        query, shifted = build_retrieval_query("Ông ấy sinh năm nào?", turns)
        assert "ngô quyền" in query.lower() or len(query) > len("Ông ấy sinh năm nào?")

    def test_lexical_score_ranking(self):
        from app.core.utils.helpers import _lexical_score
        # _lexical_score chỉ match từng token đơn (không phải phrases)
        # nên keywords phải là list các từ đơn
        keywords = ["bạch", "đằng", "quyền"]
        high_relevance = "Trận Bạch Đằng năm 938 do Ngô Quyền chỉ huy tại cửa biển"
        low_relevance = "Nhà Lý cai trị trong thế kỷ XI và XII tại Thăng Long"
        assert _lexical_score(keywords, high_relevance) > _lexical_score(keywords, low_relevance)


# ── Response time benchmarks ──────────────────────────────────────────────────


class TestPerformance:
    def test_simple_query_under_30s(self, client):
        start = time.monotonic()
        r = client.post("/api/ask", json={"question": "Ngô Quyền là ai?"})
        elapsed = time.monotonic() - start
        assert r.status_code == 200
        assert elapsed < 30.0, f"Câu hỏi đơn giản mất {elapsed:.1f}s (>30s)"

    def test_health_endpoint_under_1s(self, client):
        start = time.monotonic()
        r = client.get("/health")
        elapsed = time.monotonic() - start
        assert r.status_code == 200
        assert elapsed < 1.0, f"/health mất {elapsed:.2f}s (>1s)"

    def test_personas_endpoint_under_200ms(self, client):
        start = time.monotonic()
        r = client.get("/personas")
        elapsed = time.monotonic() - start
        assert r.status_code == 200
        assert elapsed < 0.2, f"/personas mất {elapsed:.3f}s (>200ms)"
