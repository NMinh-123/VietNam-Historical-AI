"""
Nạp dữ liệu vào Qdrant vector database.

Cách chạy (từ thư mục backend/app/services/chatbot/):
    python run_qdrant_index.py                    # full run, recreate collection
    python run_qdrant_index.py --test             # test với 10 parent chunks đầu
    python run_qdrant_index.py --test --limit 50  # test với 50 parent chunks
    python run_qdrant_index.py --no-recreate      # giữ nguyên collection, chỉ upsert thêm
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.chatbot.index_and_retrieve.pipeline import qdrant_ingest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nạp dữ liệu vào Qdrant vector database"
    )
    parser.add_argument("--test", action="store_true",
                        help="Chạy test mode (chỉ lấy --limit parent chunks đầu tiên)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Số parent chunks tối đa khi chạy test mode (mặc định: 10)")
    parser.add_argument("--no-recreate", action="store_true",
                        help="Giữ nguyên Qdrant collection cũ, chỉ upsert thêm")
    parser.add_argument("--qdrant-host", default="localhost",
                        help="Host của Qdrant server (mặc định: localhost)")
    parser.add_argument("--qdrant-port", type=int, default=6333,
                        help="Port của Qdrant server (mặc định: 6333)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Kích thước batch khi upsert vào Qdrant")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    recreate = not args.no_recreate

    print("=" * 60)
    print("QDRANT INDEX: PDF → VECTOR DATABASE")
    print("=" * 60)
    print(f"  Chế độ  : {'TEST (giới hạn)' if args.test else 'FULL'}")
    if args.test:
        print(f"  Giới hạn: {args.limit} parent chunks")
    print(f"  Qdrant  : {args.qdrant_host}:{args.qdrant_port}")
    print(f"  Recreate: {recreate}")
    print("=" * 60)

    await qdrant_ingest(
        test_mode=args.test,
        parent_limit=args.limit,
        recreate_collection=recreate,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_batch_size=args.batch_size,
    )


if __name__ == "__main__":
    asyncio.run(main())
