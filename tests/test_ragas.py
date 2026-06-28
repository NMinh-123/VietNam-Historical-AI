"""RAGAS evaluation — đánh giá chất lượng RAG pipeline của Vical AI.

Yêu cầu: server đang chạy tại http://localhost:8001
Chạy độc lập : python tests/test_ragas.py [--n 10] [--out reports/ragas.json]
Chạy với pytest: pytest tests/test_ragas.py -v -s

Metrics được đánh giá:
  faithfulness       — câu trả lời có bám sát context được truy xuất không?
  answer_relevancy   — câu trả lời có liên quan đến câu hỏi không?
  context_precision  — context truy xuất có chứa thông tin cần thiết không?
  context_recall     — context truy xuất có bao phủ đủ ground truth không?
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

load_dotenv(_ROOT / ".env", override=False)
_EVAL_SET = _HERE / "data" / "ragas_eval_set.json"
_REPORTS_DIR = _ROOT / "tests" / "reports"

BASE_URL = os.getenv("VICAL_BASE_URL", "http://localhost:8000")
TIMEOUT = 500.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)


# ── Gọi API ──────────────────────────────────────────────────────────────────


def _call_api(question: str, client: httpx.Client) -> dict:
    """Gọi /api/ask với include_contexts=True, trả về dict {answer, contexts, sources}."""
    resp = client.post(
        "/api/ask",
        json={"question": question, "include_contexts": True},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ── Xây dựng dataset RAGAS ────────────────────────────────────────────────────


def build_samples(eval_set: list[dict], client: httpx.Client) -> list[dict]:
    """Gọi API cho từng câu hỏi và trả về danh sách raw sample dict."""
    samples = []
    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        _log.info("[%d/%d] %s", i, len(eval_set), q[:80])
        try:
            result = _call_api(q, client)
        except Exception as exc:
            _log.warning("  ✗ Bỏ qua câu hỏi (lỗi API): %s", exc)
            continue

        answer = result.get("answer", "")
        contexts: list[str] = result.get("contexts") or []

        if not answer or not contexts:
            _log.warning("  ✗ Bỏ qua câu hỏi (answer/contexts trống)")
            continue

        samples.append({
            "id": item.get("id", f"Q{i:03d}"),
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "reference": item.get("reference"),
        })
        _log.info("  ✓ %d contexts, answer length=%d", len(contexts), len(answer))

    return samples


# ── Chạy RAGAS evaluate ───────────────────────────────────────────────────────


def run_ragas(samples: list[dict]) -> dict:
    """Chạy RAGAS evaluate và trả về dict scores."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # ── Cấu hình LLM judge ──────────────────────────────────────────────────
    api_key = (
        os.getenv("GEMINI_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("SHOPAIKEY_TOKEN")
        or os.getenv("SHOPAIKEY_API_KEY")
    )
    if not api_key:
        raise EnvironmentError(
            "Thiếu API key. Hãy đặt GEMINI_KEY, OPENAI_API_KEY hoặc SHOPAIKEY_TOKEN."
        )

    def _normalize(url: str) -> str:
        import re
        u = (url or "").strip().rstrip("/")
        if u.endswith("/chat/completions"):
            u = u[: -len("/chat/completions")]
        if u and not re.search(r"/v\d+$", u) and not u.endswith("/openai"):
            u = f"{u}/v1"
        return u

    base_url = _normalize(
        os.getenv("OPENAI_COMPAT_BASE_URL")
        or os.getenv("SHOPAIKEY_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model_name = (
        os.getenv("RAGAS_MODEL")
        or os.getenv("OPENAI_MODEL_NAME")
        or os.getenv("GEMINI_MODEL_NAME")
        or "gemini-2.0-flash-lite"
    )

    _log.info("RAGAS LLM judge: %s @ %s", model_name, base_url)

    lc_llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0)
    ragas_llm = LangchainLLMWrapper(lc_llm)

    # Dùng API embedding thay vì load local model để tiết kiệm RAM (~400MB)
    # answer_relevancy cần embeddings để tính cosine similarity
    from langchain_openai import OpenAIEmbeddings
    lc_emb = OpenAIEmbeddings(
        model=os.getenv("RAGAS_EMBED_MODEL", "text-embedding-3-small"),
        api_key=api_key,
        base_url=base_url,
    )
    ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm

    # ── Tách samples có / không có reference ─────────────────────────────────
    with_ref = [s for s in samples if s.get("reference")]
    without_ref = [s for s in samples if not s.get("reference")]

    all_results: list[dict] = []

    def _score_to_float(v) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Đánh giá 4 metrics cho samples có reference
    if with_ref:
        _log.info("Đánh giá %d samples (4 metrics: faithfulness, answer_relevancy, context_precision, context_recall)...", len(with_ref))
        ragas_dataset = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s["reference"],
            )
            for s in with_ref
        ])
        result = evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        for i, s in enumerate(with_ref):
            row = df.iloc[i]
            all_results.append({
                "id": s["id"],
                "question": s["question"],
                "faithfulness": _score_to_float(row.get("faithfulness")),
                "answer_relevancy": _score_to_float(row.get("answer_relevancy")),
                "context_precision": _score_to_float(row.get("context_precision")),
                "context_recall": _score_to_float(row.get("context_recall")),
            })

    # Đánh giá 2 metrics cho samples không có reference
    if without_ref:
        _log.info("Đánh giá %d samples (2 metrics: faithfulness, answer_relevancy)...", len(without_ref))
        ragas_dataset = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
            )
            for s in without_ref
        ])
        result = evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        for i, s in enumerate(without_ref):
            row = df.iloc[i]
            all_results.append({
                "id": s["id"],
                "question": s["question"],
                "faithfulness": _score_to_float(row.get("faithfulness")),
                "answer_relevancy": _score_to_float(row.get("answer_relevancy")),
                "context_precision": None,
                "context_recall": None,
            })

    return _aggregate(all_results)


def _aggregate(rows: list[dict]) -> dict:
    def _avg(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "summary": {
            "n_samples": len(rows),
            "faithfulness": _avg("faithfulness"),
            "answer_relevancy": _avg("answer_relevancy"),
            "context_precision": _avg("context_precision"),
            "context_recall": _avg("context_recall"),
        },
        "per_question": rows,
    }


# ── Hiển thị kết quả ──────────────────────────────────────────────────────────


def _print_report(report: dict) -> None:
    s = report["summary"]
    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION REPORT — Vical AI")
    print("=" * 60)
    print(f"  Samples evaluated : {s['n_samples']}")
    print(f"  Faithfulness      : {s['faithfulness']:.4f}" if s["faithfulness"] is not None else "  Faithfulness      : N/A")
    print(f"  Answer Relevancy  : {s['answer_relevancy']:.4f}" if s["answer_relevancy"] is not None else "  Answer Relevancy  : N/A")
    print(f"  Context Precision : {s['context_precision']:.4f}" if s["context_precision"] is not None else "  Context Precision : N/A")
    print(f"  Context Recall    : {s['context_recall']:.4f}" if s["context_recall"] is not None else "  Context Recall    : N/A")
    print("=" * 60)
    print("\nPer-question breakdown:")
    for r in report["per_question"]:
        fa = f"{r['faithfulness']:.3f}" if r["faithfulness"] is not None else " N/A"
        ar = f"{r['answer_relevancy']:.3f}" if r["answer_relevancy"] is not None else " N/A"
        cp = f"{r['context_precision']:.3f}" if r["context_precision"] is not None else " N/A"
        cr = f"{r['context_recall']:.3f}" if r["context_recall"] is not None else " N/A"
        print(f"  [{r['id']}] F={fa} AR={ar} CP={cp} CR={cr}  {r['question'][:55]}")
    print()


def _save_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log.info("Kết quả đã lưu: %s", out_path)


# ── Entry point ───────────────────────────────────────────────────────────────


def main(n: int | None = None, out: Path | None = None) -> dict:
    eval_set: list[dict] = json.loads(_EVAL_SET.read_text(encoding="utf-8"))
    if n:
        eval_set = eval_set[:n]

    _log.info("Đang kết nối server %s ...", BASE_URL)
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        # Kiểm tra server healthy
        health = client.get("/health")
        if health.status_code != 200 or not health.json().get("rag_ready"):
            raise RuntimeError(f"Server chưa sẵn sàng: {health.text[:200]}")
        _log.info("Server OK — RAG ready")

        t0 = time.monotonic()
        samples = build_samples(eval_set, client)
        _log.info("Hoàn tất gọi API: %d/%d samples (%.1fs)", len(samples), len(eval_set), time.monotonic() - t0)

    if not samples:
        raise RuntimeError("Không có sample nào để đánh giá.")

    _log.info("Đang chạy RAGAS evaluate ...")
    report = run_ragas(samples)
    report["meta"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "server": BASE_URL,
        "eval_set": str(_EVAL_SET),
        "n_requested": len(eval_set),
    }

    _print_report(report)

    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _REPORTS_DIR / f"ragas_{ts}.json"
    _save_report(report, out)

    return report


# ── Pytest wrapper ────────────────────────────────────────────────────────────


def test_ragas_faithfulness_above_threshold():
    """Pytest: faithfulness trung bình >= 0.5 (ngưỡng cơ bản)."""
    report = main(n=5)
    score = report["summary"]["faithfulness"]
    assert score is not None, "Không có điểm faithfulness"
    assert score >= 0.5, f"Faithfulness quá thấp: {score:.4f} < 0.5"


def test_ragas_answer_relevancy_above_threshold():
    """Pytest: answer_relevancy trung bình >= 0.5."""
    report = main(n=5)
    score = report["summary"]["answer_relevancy"]
    assert score is not None, "Không có điểm answer_relevancy"
    assert score >= 0.5, f"Answer relevancy quá thấp: {score:.4f} < 0.5"


def test_ragas_full_eval():
    """Pytest: chạy đầy đủ 10 câu hỏi, báo cáo kết quả."""
    report = main()
    s = report["summary"]
    _log.info(
        "RAGAS scores — faithfulness=%.3f, answer_relevancy=%.3f, "
        "context_precision=%s, context_recall=%s",
        s["faithfulness"] or 0,
        s["answer_relevancy"] or 0,
        f"{s['context_precision']:.3f}" if s["context_precision"] else "N/A",
        f"{s['context_recall']:.3f}" if s["context_recall"] else "N/A",
    )
    # Không fail hard — chỉ cần pipeline hoàn tất không lỗi
    assert s["n_samples"] > 0


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation cho Vical AI")
    parser.add_argument("--n", type=int, default=None, help="Số câu hỏi đánh giá (mặc định: tất cả)")
    parser.add_argument("--out", type=Path, default=None, help="Đường dẫn file JSON kết quả")
    args = parser.parse_args()

    try:
        main(n=args.n, out=args.out)
    except Exception as exc:
        _log.error("RAGAS evaluation thất bại: %s", exc)
        sys.exit(1)
