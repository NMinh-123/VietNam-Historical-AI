"""
Test runner cho Vical AI — chạy toàn bộ 110 câu hỏi và xuất báo cáo.

Cách dùng:
    # Chạy toàn bộ
    python tests/run_tests.py

    # Chỉ chạy nhóm cụ thể
    python tests/run_tests.py --group nha_tran

    # Chỉ chạy độ khó cụ thể
    python tests/run_tests.py --difficulty hard

    # Chạy với persona
    python tests/run_tests.py --group nha_tran --persona tran-hung-dao

    # Giới hạn số câu hỏi
    python tests/run_tests.py --limit 20

    # Chạy concurrent (mặc định tuần tự)
    python tests/run_tests.py --concurrent 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
QUESTIONS_FILE = Path(__file__).parent / "data" / "questions.json"
REPORT_DIR = Path(__file__).parent / "reports"


# ── Cấu trúc kết quả ──────────────────────────────────────────────────────────

@dataclass
class TestResult:
    q_id: str
    group: str
    difficulty: str
    question: str
    persona: str | None
    status: str          # "pass" | "fail" | "error" | "skip"
    latency_ms: float
    answer: str = ""
    sources_count: int = 0
    missing_keywords: list[str] = field(default_factory=list)
    error_msg: str = ""
    http_status: int = 0


@dataclass
class Report:
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    results: list[TestResult] = field(default_factory=list)


# ── Gửi request ───────────────────────────────────────────────────────────────

async def ask_question(
    client: httpx.AsyncClient,
    question: str,
    persona_slug: str | None = None,
    conversation_id: str | None = None,
) -> tuple[int, dict, float]:
    """Trả về (http_status, response_body, latency_ms)."""
    payload = {
        "question": question,
        "persona_slug": persona_slug,
        "conversation_id": conversation_id,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{BASE_URL}/api/ask", json=payload, timeout=60.0)
        latency = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return resp.status_code, body, latency
    except httpx.TimeoutException:
        latency = (time.perf_counter() - t0) * 1000
        return 0, {"error": "timeout"}, latency
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return 0, {"error": str(e)}, latency


# ── Đánh giá kết quả ──────────────────────────────────────────────────────────

def evaluate(q: dict, http_status: int, body: dict, latency_ms: float) -> TestResult:
    result = TestResult(
        q_id=q["id"],
        group=q["group"],
        difficulty=q["difficulty"],
        question=q["question"],
        persona=q.get("persona"),
        status="pass",
        latency_ms=latency_ms,
        http_status=http_status,
    )

    expect_error = q.get("expect_error", False)
    expect_safe = q.get("expect_safe", False)
    expect_graceful = q.get("expect_graceful", False)

    # Câu hỏi expect lỗi (input không hợp lệ)
    if expect_error:
        if http_status in (400, 422):
            result.status = "pass"
        else:
            result.status = "fail"
            result.error_msg = f"Expected 4xx, got {http_status}"
        return result

    # Request thất bại
    if http_status == 0:
        result.status = "error"
        result.error_msg = body.get("error", "Connection failed")
        return result

    if http_status >= 500:
        result.status = "error"
        result.error_msg = f"HTTP {http_status}: {body.get('detail', '')}"
        return result

    # Câu hỏi expect từ chối lịch sự (ngoài phạm vi)
    if expect_graceful:
        answer = body.get("answer", "").lower()
        if http_status == 200 and len(answer) > 10:
            result.status = "pass"
            result.answer = body.get("answer", "")
        else:
            result.status = "fail"
            result.error_msg = "Không có response graceful"
        return result

    # Câu hỏi expect an toàn (injection)
    if expect_safe:
        answer = body.get("answer", "")
        dangerous = ["<script>", "SECRET_KEY", "DROP TABLE", "password_hash"]
        leaked = [d for d in dangerous if d.lower() in answer.lower()]
        if leaked:
            result.status = "fail"
            result.error_msg = f"Leaked sensitive content: {leaked}"
        else:
            result.status = "pass"
        result.answer = answer
        return result

    # Câu hỏi thường — kiểm tra keywords
    if http_status != 200:
        result.status = "fail"
        result.error_msg = f"HTTP {http_status}"
        return result

    answer = body.get("answer", "")
    result.answer = answer
    result.sources_count = len(body.get("sources", []))

    expected_keywords = q.get("expected_keywords", [])
    if expected_keywords:
        answer_lower = answer.lower()
        missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
        result.missing_keywords = missing
        if len(missing) > len(expected_keywords) * 0.5:
            result.status = "fail"
            result.error_msg = f"Thiếu {len(missing)}/{len(expected_keywords)} keywords: {missing}"

    return result


# ── Chạy một câu hỏi ──────────────────────────────────────────────────────────

async def run_one(client: httpx.AsyncClient, q: dict, semaphore: asyncio.Semaphore) -> TestResult:
    async with semaphore:
        question = q["question"].strip()
        if not question:
            return TestResult(
                q_id=q["id"], group=q["group"], difficulty=q["difficulty"],
                question=q["question"], persona=q.get("persona"),
                status="pass" if q.get("expect_error") else "skip",
                latency_ms=0, http_status=422,
            )

        http_status, body, latency = await ask_question(
            client, question, persona_slug=q.get("persona")
        )
        return evaluate(q, http_status, body, latency)


# ── In báo cáo ────────────────────────────────────────────────────────────────

def print_report(report: Report, verbose: bool = False) -> None:
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  VICAL AI — TEST REPORT  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)
    print(f"  Tổng:     {report.total}")
    print(f"  ✅ Pass:   {report.passed}  ({report.passed/report.total*100:.1f}%)")
    print(f"  ❌ Fail:   {report.failed}  ({report.failed/report.total*100:.1f}%)")
    print(f"  💥 Error:  {report.errors}  ({report.errors/report.total*100:.1f}%)")
    print(f"  ⏭️  Skip:   {report.skipped}")
    print(f"\n  Latency trung bình: {report.avg_latency_ms:.0f}ms")
    print(f"  Latency P95:        {report.p95_latency_ms:.0f}ms")

    # Theo nhóm
    groups: dict[str, dict] = {}
    for r in report.results:
        g = groups.setdefault(r.group, {"pass": 0, "fail": 0, "error": 0})
        g[r.status if r.status in g else "error"] += 1

    print(f"\n  Kết quả theo nhóm:")
    for group, counts in sorted(groups.items()):
        total_g = sum(counts.values())
        print(f"    {group:<20} pass={counts['pass']}/{total_g}  fail={counts['fail']}  error={counts['error']}")

    # Theo độ khó
    print(f"\n  Kết quả theo độ khó:")
    diffs: dict[str, dict] = {}
    for r in report.results:
        d = diffs.setdefault(r.difficulty, {"pass": 0, "fail": 0, "error": 0})
        d[r.status if r.status in d else "error"] += 1
    for diff, counts in sorted(diffs.items()):
        total_d = sum(counts.values())
        rate = counts['pass'] / total_d * 100 if total_d else 0
        print(f"    {diff:<10} {rate:.0f}% pass  ({counts['pass']}/{total_d})")

    # Chi tiết lỗi
    failures = [r for r in report.results if r.status in ("fail", "error")]
    if failures:
        print(f"\n  Chi tiết lỗi ({len(failures)} câu):")
        for r in failures:
            icon = "❌" if r.status == "fail" else "💥"
            print(f"    {icon} [{r.q_id}] {r.question[:60]}...")
            print(f"         → {r.error_msg}")
            if r.missing_keywords:
                print(f"         Thiếu keywords: {r.missing_keywords}")

    # Câu chậm nhất
    if verbose:
        slow = sorted(report.results, key=lambda r: r.latency_ms, reverse=True)[:5]
        print(f"\n  Top 5 câu chậm nhất:")
        for r in slow:
            print(f"    [{r.q_id}] {r.latency_ms:.0f}ms — {r.question[:50]}...")

    print(sep)


def save_json_report(report: Report) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"report_{ts}.json"
    data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "errors": report.errors,
            "pass_rate": f"{report.passed/report.total*100:.1f}%",
            "avg_latency_ms": round(report.avg_latency_ms, 1),
            "p95_latency_ms": round(report.p95_latency_ms, 1),
        },
        "results": [
            {
                "id": r.q_id,
                "group": r.group,
                "difficulty": r.difficulty,
                "status": r.status,
                "latency_ms": round(r.latency_ms, 1),
                "http_status": r.http_status,
                "sources_count": r.sources_count,
                "missing_keywords": r.missing_keywords,
                "error_msg": r.error_msg,
                "answer_preview": r.answer[:200] if r.answer else "",
            }
            for r in report.results
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    # Load câu hỏi — strip comment lines trước khi parse
    raw = QUESTIONS_FILE.read_text(encoding="utf-8")
    lines = [l for l in raw.splitlines() if not l.strip().startswith("//")]
    questions = json.loads("\n".join(lines))

    # Filter
    if args.group:
        questions = [q for q in questions if q["group"] == args.group]
    if args.difficulty:
        questions = [q for q in questions if q["difficulty"] == args.difficulty]
    if args.persona:
        questions = [q for q in questions if q.get("persona") == args.persona]
    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        print("Không có câu hỏi nào thỏa điều kiện filter.")
        sys.exit(1)

    print(f"Chạy {len(questions)} câu hỏi | concurrent={args.concurrent} | server={BASE_URL}")

    semaphore = asyncio.Semaphore(args.concurrent)
    results: list[TestResult] = []

    async with httpx.AsyncClient() as client:
        # Kiểm tra server sống không
        try:
            health = await client.get(f"{BASE_URL}/health", timeout=5.0)
            print(f"Server health: {health.json()}\n")
        except Exception as e:
            print(f"❌ Không kết nối được server: {e}")
            sys.exit(1)

        tasks = [run_one(client, q, semaphore) for q in questions]
        total = len(tasks)

        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            results.append(result)
            icon = {"pass": "✅", "fail": "❌", "error": "💥", "skip": "⏭️"}.get(result.status, "?")
            print(f"  [{i:>3}/{total}] {icon} [{result.q_id}] {result.question[:55]:<55} {result.latency_ms:>6.0f}ms")

    # Tổng hợp
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    latencies.sort()
    p95_idx = int(len(latencies) * 0.95) if latencies else 0

    report = Report(
        total=len(results),
        passed=sum(1 for r in results if r.status == "pass"),
        failed=sum(1 for r in results if r.status == "fail"),
        errors=sum(1 for r in results if r.status == "error"),
        skipped=sum(1 for r in results if r.status == "skip"),
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        p95_latency_ms=latencies[p95_idx] if latencies else 0,
        results=results,
    )

    print_report(report, verbose=args.verbose)

    json_path = save_json_report(report)
    print(f"\n  Báo cáo JSON: {json_path}")

    sys.exit(0 if report.failed == 0 and report.errors == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vical AI Test Runner")
    parser.add_argument("--group", help="Chỉ chạy nhóm câu hỏi cụ thể")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "stress"])
    parser.add_argument("--persona", help="Lọc theo persona slug")
    parser.add_argument("--limit", type=int, help="Giới hạn số câu hỏi")
    parser.add_argument("--concurrent", type=int, default=1, help="Số request đồng thời (default: 1)")
    parser.add_argument("--verbose", action="store_true", help="In thêm top 5 câu chậm nhất")
    args = parser.parse_args()

    asyncio.run(main(args))
