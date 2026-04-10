"""Dọn các artefact sinh ra bởi pipeline trước khi build lại."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
DATA_DIR = APP_DIR / "data"
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_QDRANT_COLLECTION = "vietnam_history_hybrid"

PARENT_DOCSTORE_PATH = DATA_DIR / "parent_docs.json"
CHILD_DOCSTORE_PATH = DATA_DIR / "child_docs.json"
LIGHTRAG_WORKSPACE = DATA_DIR / "lightrag_storage"
LIGHTRAG_INGEST_MANIFEST_PATH = DATA_DIR / "lightrag_ingest_manifest.json"
QDRANT_DB_PATH = DATA_DIR / "qdrant_db"
CHUNK_ID_REGISTRY_PATH = DATA_DIR / "chunk_id_registry.json"


def _remove_path(path: Path) -> str:
    """Xóa file hoặc thư mục nếu tồn tại, trả về trạng thái để log."""
    if not path.exists():
        return "missing"

    if path.is_dir():
        shutil.rmtree(path)
        return "removed_dir"

    path.unlink()
    return "removed_file"


def _drop_qdrant_collection(
    *,
    host: str,
    port: int,
    collection_name: str,
) -> str:
    """Xóa collection trong Qdrant remote đang chạy qua Docker/HTTP."""
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        return f"qdrant_client_import_error: {exc}"

    try:
        client = QdrantClient(host=host, port=port)
        if not client.collection_exists(collection_name):
            return "missing_remote_collection"
        client.delete_collection(collection_name)
        return "removed_remote_collection"
    except Exception as exc:
        return f"remote_error: {exc}"


def reset_generated_outputs(
    *,
    remove_id_registry: bool = False,
    drop_qdrant_collection: bool = False,
    qdrant_host: str = DEFAULT_QDRANT_HOST,
    qdrant_port: int = DEFAULT_QDRANT_PORT,
    qdrant_collection: str = DEFAULT_QDRANT_COLLECTION,
) -> dict[str, str]:
    """Xóa các file/thư mục sinh ra bởi pipeline ingest."""
    targets = {
        "parent_docs": PARENT_DOCSTORE_PATH,
        "child_docs": CHILD_DOCSTORE_PATH,
        "lightrag_manifest": LIGHTRAG_INGEST_MANIFEST_PATH,
        "lightrag_storage": LIGHTRAG_WORKSPACE,
        "qdrant_db": QDRANT_DB_PATH,
    }

    if remove_id_registry:
        targets["chunk_id_registry"] = CHUNK_ID_REGISTRY_PATH

    results: dict[str, str] = {}
    for label, path in targets.items():
        results[label] = _remove_path(path)

    if drop_qdrant_collection:
        results["qdrant_collection"] = _drop_qdrant_collection(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=qdrant_collection,
        )

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Xóa artefact sinh ra bởi pipeline index/LightRAG.",
    )
    parser.add_argument(
        "--include-id-registry",
        action="store_true",
        help="Xóa luôn chunk_id_registry.json để đánh số parent/child từ đầu.",
    )
    parser.add_argument(
        "--drop-qdrant-collection",
        action="store_true",
        help="Xóa luôn collection trong Qdrant đang chạy qua Docker/HTTP.",
    )
    parser.add_argument(
        "--qdrant-host",
        default=DEFAULT_QDRANT_HOST,
        help="Host của Qdrant remote.",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=DEFAULT_QDRANT_PORT,
        help="Port của Qdrant remote.",
    )
    parser.add_argument(
        "--qdrant-collection",
        default=DEFAULT_QDRANT_COLLECTION,
        help="Tên collection cần xóa trong Qdrant.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    results = reset_generated_outputs(
        remove_id_registry=args.include_id_registry,
        drop_qdrant_collection=args.drop_qdrant_collection,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_collection=args.qdrant_collection,
    )
    for label, status in results.items():
        print(f"{label}: {status}")


if __name__ == "__main__":
    main()
