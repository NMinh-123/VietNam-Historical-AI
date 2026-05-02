"""Định dạng ngữ cảnh và xây dựng metadata nguồn cho pipeline trả lời."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _format_context_items(items: list[dict[str, Any]]) -> str:
    # Định dạng danh sách đoạn văn bản thành chuỗi ngữ cảnh có đánh số nguồn [nguon=i]
    sections = []
    for i, item in enumerate(items, 1):
        source_label = item.get("source_label")
        header = f"[nguon={i}]"
        if source_label:
            header = f"{header} {source_label}"
        sections.append(f"{header}\n{item['text']}")
    return "\n\n---\n\n".join(sections)


def _format_graph_context_items(items: list[dict[str, Any]]) -> str:
    # Định dạng kết quả đồ thị LightRAG thành chuỗi gợi ý có đánh số [goi_y_do_thi=i]
    sections = []
    for i, item in enumerate(items, 1):
        sections.append(f"[goi_y_do_thi={i}]\n{item['text']}")
    return "\n\n---\n\n".join(sections)


def _split_blocks(text: str) -> list[str]:
    # Tách văn bản thành danh sách các đoạn, phân cách bởi dòng trắng
    return [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]


def _coerce_text(value: Any) -> str:
    # Ép kiểu về str: nếu đã là str trả về nguyên, ngược lại serialize JSON đọc được
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _build_source_label(item: dict[str, Any]) -> str:
    # Tạo nhãn nguồn dạng "tên_file, trang X" hoặc chỉ tên file nếu không có số trang
    raw_source = (item.get("source") or "").strip()
    file_name = Path(raw_source).name if raw_source else "Tài liệu không rõ tên"
    page_label = item.get("page_label")
    if page_label:
        return f"{file_name}, trang {page_label}"

    page = item.get("page")
    if isinstance(page, int):
        return f"{file_name}, trang {page + 1}"

    return file_name


def _build_source_payload(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Xây dựng danh sách metadata nguồn để trả về cùng câu trả lời (tên file, trang, điểm...)
    sources = []
    for index, item in enumerate(items, start=1):
        raw_source = (item.get("source") or "").strip()
        file_name = Path(raw_source).name if raw_source else ""
        sources.append(
            {
                "index": index,
                "title": item.get("title") or file_name or "Tài liệu không rõ tên",
                "file_name": file_name,
                "file_path": raw_source,
                "page": item.get("page"),
                "page_label": item.get("page_label"),
                "parent_id": item.get("parent_id"),
                "score": round(float(item.get("score") or 0.0), 4),
                "label": _build_source_label(item),
            }
        )
    return sources


__all__ = [
    "_format_context_items",
    "_format_graph_context_items",
    "_split_blocks",
    "_coerce_text",
    "_build_source_label",
    "_build_source_payload",
]
