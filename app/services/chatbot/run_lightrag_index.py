"""
Nạp dữ liệu vào LightRAG để xây dựng knowledge graph.

Cách chạy (từ thư mục backend/app/services/chatbot/):
    python run_lightrag_index.py                   # full run (đọc từ parent_docs.json)
    python run_lightrag_index.py --test            # test với 10 parent docs đầu
    python run_lightrag_index.py --test --limit 50 # test với 50 parent docs
    python run_lightrag_index.py --reload-pdf      # load PDF lại từ đầu (không dùng cache)
    python run_lightrag_index.py --resume          # resume queue LightRAG bị dở dang

Lưu ý: Chạy run_qdrant_index.py trước để tạo parent_docs.json,
        hoặc dùng --reload-pdf để tự load PDF (chậm hơn).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.chatbot.index_and_retrieve.pipeline import lightrag_ingest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nạp dữ liệu vào LightRAG knowledge graph"
    )
    parser.add_argument("--test", action="store_true",
                        help="Chạy test mode (chỉ lấy --limit parent docs đầu tiên)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Số parent docs tối đa khi chạy test mode (mặc định: 10)")
    parser.add_argument("--reload-pdf", action="store_true",
                        help="Load PDF từ đầu thay vì đọc parent_docs.json đã lưu")
    parser.add_argument("--resume", action="store_true",
                        help="Resume queue LightRAG bị dở dang từ lần chạy trước")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Số doc mỗi batch nạp vào LightRAG")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Số doc xử lý song song tối đa")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    use_saved = not args.reload_pdf

    print("=" * 60)
    print("LIGHTRAG INDEX: PARENT DOCS → KNOWLEDGE GRAPH")
    print("=" * 60)
    print(f"  Chế độ    : {'TEST (giới hạn)' if args.test else 'FULL'}")
    if args.test:
        print(f"  Giới hạn  : {args.limit} parent docs")
    print(f"  Nguồn docs: {'parent_docs.json (cache)' if use_saved else 'Load PDF từ đầu'}")
    print(f"  Resume    : {args.resume}")
    print("=" * 60)

    await lightrag_ingest(
        test_mode=args.test,
        parent_limit=args.limit,
        lightrag_batch_size=args.batch_size,
        lightrag_max_parallel_insert=args.max_parallel,
        resume_existing_queue=args.resume if args.resume else None,
        use_saved_docs=use_saved,
    )


if __name__ == "__main__":
    asyncio.run(main())
