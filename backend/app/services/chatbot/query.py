"""Script CLI để kiểm thử pipeline truy vấn lịch sử Việt Nam.

Chạy trực tiếp:
    python -m services.chatbot.query
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path
from typing import Any

# Đảm bảo backend/app nằm trong sys.path khi chạy trực tiếp
_APP_DIR = Path(__file__).resolve().parents[2]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from services.chatbot.retrieve_and_query import VietnamHistoryQueryEngine  # noqa: E402


def _print_separator(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _print_result(question: str, result: dict[str, Any]) -> None:
    """In kết quả truy vấn ra stdout theo định dạng dễ đọc."""
    _print_separator("═")
    print(f"CÂU HỎI: {question}")
    _print_separator()

    print("\nTRẢ LỜI:")
    for line in result["answer"].splitlines():
        print(textwrap.fill(line, width=90) if len(line) > 90 else line)

    sources = result.get("sources", [])
    if sources:
        print(f"\nNGUỒN ({len(sources)}):")
        for src in sources:
            label = src.get("label") or src.get("file_name") or "Không rõ"
            score = src.get("score", 0.0)
            print(f"  [{src['index']}] {label}  (score={score:.4f})")

    verification = result.get("verification", "")
    if verification:
        print(f"\nXÁC NHẬN: {verification}")

    _print_separator("═")


async def _run_repl(engine: VietnamHistoryQueryEngine) -> None:
    """Vòng lặp REPL tương tác; gõ 'q' hoặc 'exit' để thoát."""
    print("\nCHẾ ĐỘ TƯƠNG TÁC — gõ câu hỏi rồi nhấn Enter (q/exit để thoát)\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break

        if not question:
            continue
        if question.lower() in {"q", "exit", "quit", "thoát"}:
            print("Thoát.")
            break

        result = await engine.ask_with_sources(question)
        _print_result(question, result)
        print()


async def _main() -> None:
    engine = VietnamHistoryQueryEngine()
    await engine.start()  # kick LightRAG warm-up chạy nền
    await _run_repl(engine)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
