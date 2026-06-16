"""Kiểm thử các thành phần RAG pipeline của Vical AI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Setup Python paths
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "app"))


# ── Ingestion ─────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_load_pdfs_missing_folder(self, tmp_path):
        from src.ingestion.loader import load_pdfs_from_folder
        docs = load_pdfs_from_folder(str(tmp_path / "nonexistent"))
        assert docs == []

    def test_load_pdfs_empty_folder(self, tmp_path):
        from src.ingestion.loader import load_pdfs_from_folder
        docs = load_pdfs_from_folder(str(tmp_path))
        assert docs == []


# ── Chunking ──────────────────────────────────────────────────────────────────

class TestChunking:
    def test_clean_text_empty(self):
        from src.chunking.chunker import clean_text
        assert clean_text("") == ""
        assert clean_text(None) == ""  # type: ignore[arg-type]

    def test_clean_text_strips_whitespace(self):
        from src.chunking.chunker import clean_text
        assert clean_text("  hello world  ") == "hello world"

    def test_build_parent_child_chunks_empty(self, tmp_path):
        from src.chunking.chunker import build_parent_child_chunks
        child_docs, parent_store, parent_docs = build_parent_child_chunks(
            [], id_registry_path=tmp_path / "registry.json"
        )
        assert child_docs == []
        assert parent_store == {}
        assert parent_docs == []


# ── Embeddings ────────────────────────────────────────────────────────────────

class TestEmbeddings:
    def test_constants(self):
        from src.app_config import get_config
        from src.embeddings.embedder import E5_EMBEDDING_DIM, E5_EMBEDDING_MODEL_NAME
        assert E5_EMBEDDING_DIM == get_config().model.embedding.dim
        assert "e5" in E5_EMBEDDING_MODEL_NAME.lower()

    def test_config_defaults(self):
        from src.app_config import get_config
        from src.embeddings.embedder import E5EmbeddingConfig
        config = E5EmbeddingConfig()
        assert config.embedding_dim == get_config().model.embedding.dim
        assert config.normalize_embeddings is True


# ── Utils / Helpers ───────────────────────────────────────────────────────────

class TestHelpers:
    def test_clean_text(self):
        from src.utils.helpers import clean_text
        assert clean_text("  test  ") == "test"
        assert clean_text("") == ""

    def test_build_query_structure(self):
        from src.utils.helpers import build_query
        result = build_query("lịch sử Việt Nam")
        assert "dense" in result
        assert "sparse" in result
        assert "keywords" in result
        assert isinstance(result["keywords"], list)

    def test_lexical_score_positive(self):
        from src.utils.helpers import _lexical_score
        score = _lexical_score(["việt", "nam"], "lịch sử việt nam rất phong phú")
        assert score > 0.0

    def test_lexical_score_empty_keywords(self):
        from src.utils.helpers import _lexical_score
        score = _lexical_score([], "bất kỳ nội dung nào")
        assert score == 0.0

    def test_build_source_label_with_page(self):
        from src.utils.helpers import build_source_label
        item = {"source": "lichsu.pdf", "page": 4}
        label = build_source_label(item)
        assert "lichsu.pdf" in label
        assert "5" in label  # page + 1

    def test_build_source_label_with_page_label(self):
        from src.utils.helpers import build_source_label
        item = {"source": "lichsu.pdf", "page_label": "42"}
        label = build_source_label(item)
        assert "42" in label

    def test_build_source_label_no_page(self):
        from src.utils.helpers import build_source_label
        item = {"source": "lichsu.pdf"}
        assert build_source_label(item) == "lichsu.pdf"

    def test_format_context_items(self):
        from src.utils.helpers import format_context_items
        items = [{"text": "nội dung 1", "source_label": "sách A"}]
        result = format_context_items(items)
        assert "[nguon=1]" in result
        assert "nội dung 1" in result

    def test_build_source_payload_index(self):
        from src.utils.helpers import build_source_payload
        items = [{"source": "a.pdf", "score": 0.8}, {"source": "b.pdf", "score": 0.5}]
        payload = build_source_payload(items)
        assert len(payload) == 2
        assert payload[0]["index"] == 1
        assert payload[1]["index"] == 2

    def test_split_blocks(self):
        from src.utils.helpers import split_blocks
        text = "block một\n\nblock hai\n\nblock ba"
        blocks = split_blocks(text)
        assert len(blocks) == 3

    def test_coerce_text_string(self):
        from src.utils.helpers import coerce_text
        assert coerce_text("hello") == "hello"

    def test_coerce_text_dict(self):
        from src.utils.helpers import coerce_text
        result = coerce_text({"key": "value"})
        assert "key" in result


# ── LLM Client ────────────────────────────────────────────────────────────────

class TestLLMClient:
    def test_constants(self):
        from src.llm.llm_client import DEFAULT_LLM_MODEL, RETRY_DELAYS, RETRYABLE_STATUS_CODES
        assert DEFAULT_LLM_MODEL
        assert len(RETRY_DELAYS) == 4
        assert 429 in RETRYABLE_STATUS_CODES

    def test_normalize_base_url_strips_trailing(self):
        from src.llm.llm_client import _normalize_base_url
        url = _normalize_base_url("https://api.example.com/v1/")
        assert not url.endswith("/")

    def test_is_shopaikey(self):
        from src.llm.llm_client import _is_shopaikey
        assert _is_shopaikey("https://api.shopaikey.com/v1")
        assert not _is_shopaikey("https://api.openai.com/v1")

    def test_resolve_model_name_returns_string(self):
        from src.llm.llm_client import resolve_model_name
        model = resolve_model_name()
        assert isinstance(model, str)
        assert len(model) > 0


# ── Prompts ───────────────────────────────────────────────────────────────────

class TestPrompts:
    def test_historian_prompt_has_placeholders(self):
        from src.prompts.prompt_templates import HISTORIAN_PROMPT
        assert "{question}" in HISTORIAN_PROMPT
        assert "{entities}" in HISTORIAN_PROMPT
        assert "{vector_context}" in HISTORIAN_PROMPT

    def test_persona_prompt_has_placeholders(self):
        from src.prompts.prompt_templates import PERSONA_PROMPT
        assert "{question}" in PERSONA_PROMPT
        assert "{display_name}" in PERSONA_PROMPT
        assert "{knowledge_cutoff_year}" in PERSONA_PROMPT

    def test_rewrite_query_strips_meta(self):
        from src.prompts.prompt_templates import rewrite_query
        result = rewrite_query("hãy giải thích lịch sử nhà Lý")
        assert "hãy giải thích" not in result.lower()
        assert "nhà lý" in result.lower()

    def test_rewrite_query_keeps_entity(self):
        from src.prompts.prompt_templates import rewrite_query
        result = rewrite_query("Trận Bạch Đằng")
        assert "Bạch Đằng" in result

    def test_is_broad_query(self):
        from src.prompts.prompt_templates import is_broad_query
        assert is_broad_query("tổng quan lịch sử việt nam")
        assert not is_broad_query("Trần Hưng Đạo là ai")

    def test_decompose_broad_query(self):
        from src.prompts.prompt_templates import decompose_broad_query, DYNASTIES
        sub_queries = decompose_broad_query("lịch sử")
        assert len(sub_queries) == len(DYNASTIES)
        assert all("lịch sử" in q for q in sub_queries)

    def test_detect_topic_shift_no_turns(self):
        from src.prompts.prompt_templates import detect_topic_shift
        assert detect_topic_shift("bất kỳ câu hỏi nào", []) is False

    def test_build_retrieval_query_no_turns(self):
        from src.prompts.prompt_templates import build_retrieval_query
        query, shifted = build_retrieval_query("nhà Lý", [])
        assert query == "nhà Lý"
        assert shifted is False


# ── Vector Store (unit) ───────────────────────────────────────────────────────

class TestVectorStore:
    def test_parent_id_to_uuid_stable(self):
        from src.vectordb.vector_store import parent_id_to_uuid
        uid1 = parent_id_to_uuid("parent_000001")
        uid2 = parent_id_to_uuid("parent_000001")
        assert uid1 == uid2

    def test_parent_id_to_uuid_different_inputs(self):
        from src.vectordb.vector_store import parent_id_to_uuid
        uid1 = parent_id_to_uuid("parent_000001")
        uid2 = parent_id_to_uuid("parent_000002")
        assert uid1 != uid2

    def test_collection_names(self):
        from src.vectordb.vector_store import COLLECTION_NAME, PARENT_COLLECTION_NAME
        assert COLLECTION_NAME
        assert PARENT_COLLECTION_NAME
        assert COLLECTION_NAME != PARENT_COLLECTION_NAME
