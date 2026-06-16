"""Nạp tài liệu PDF thô và tách theo heading chương/phần/mục."""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

# Heading phải đứng đầu dòng, theo sau là số/La Mã; loại false positive câu văn
_HEADING_RE = re.compile(
    r"^\s*("
    r"(?:CHƯƠNG|Chương)\s+[IVXLCDM\d/]+(?:\s|[.:]|$)"
    r"|(?:PHẦN|Phần)\s+(?:THỨ\s+|thứ\s+)?(?:[IVXLCDM]+|\d+)(?:\s|[.:]|$)"
    r"|(?:Phần)\s+thứ\s+\w+"
    r"|(?:MỤC|Mục)\s+[IVXLCDM\d]+(?:\s|[.:]|$)"
    r"|(?:QUYỂN|Quyển)\s+[IVXLCDM\d]+(?:\s|[.:]|$)"
    r")",
    re.UNICODE,
)

_PROSE_SIGNALS = re.compile(r"[,;]|cm|kg|tr\.|sđd|ibid|%", re.IGNORECASE)


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    if not _HEADING_RE.match(line):
        return False
    if _PROSE_SIGNALS.search(line):
        return False
    return True


def _normalize_heading_key(line: str) -> str:
    """Chuẩn hóa heading để so sánh, bỏ qua OCR noise và biến thể font."""
    h = line.lower().strip()
    h = re.sub(r"\s+", " ", h)
    m = re.match(r"(chương|phần|mục|quyển)\s+([^\s.,:]+)", h)
    if not m:
        return h
    keyword, num = m.group(1), m.group(2)
    num = num.replace("/", "i").replace("l", "i").replace("0", "o")
    arabic_to_roman = {
        "1": "i", "11": "ii", "111": "iii",
        "2": "ii", "3": "iii", "4": "iv", "5": "v",
        "6": "vi", "7": "vii", "8": "viii", "9": "ix",
    }
    if num in arabic_to_roman:
        num = arabic_to_roman[num]
    return f"{keyword} {num}"


def _extract_sections(file_path: str) -> list[dict]:
    """Tách PDF thành các section dựa trên heading.

    Chiến lược: heading thật = lần đầu tiên xuất hiện trong sách;
    các lần lặp lại sau là running header và bị loại bỏ.
    """
    reader = PdfReader(file_path)
    title = Path(file_path).stem
    total_pages = len(reader.pages)

    all_lines: list[tuple[str, int]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for line in text.splitlines():
            all_lines.append((line, page_num))

    if not all_lines:
        return []

    # Phát hiện trang bắt đầu mục lục (thường ở 15% cuối sách)
    muc_luc_page = total_pages + 1
    for page_num, page in enumerate(reader.pages, start=1):
        if page_num < total_pages * 0.85:
            continue
        if re.search(r"MỤC\s*LỤC", page.extract_text() or "", re.UNICODE):
            muc_luc_page = page_num
            break

    # Chỉ giữ lần xuất hiện ĐẦU TIÊN của mỗi heading key
    seen_keys: set[str] = set()
    heading_positions: list[tuple[int, str, int]] = []
    for i, (line, page_num) in enumerate(all_lines):
        if not _is_heading(line):
            continue
        if page_num >= muc_luc_page:
            continue
        key = _normalize_heading_key(line.strip())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        heading_positions.append((i, line.strip(), page_num))

    if not heading_positions:
        full_text = "\n".join(line for line, _ in all_lines)
        return [{
            "text": full_text,
            "section_title": title,
            "start_page": all_lines[0][1],
            "end_page": all_lines[-1][1],
            "title": title,
        }]

    sections: list[dict] = []

    first_pos = heading_positions[0][0]
    if first_pos > 0:
        pre_text = "\n".join(line for line, _ in all_lines[:first_pos]).strip()
        if pre_text:
            sections.append({
                "text": pre_text,
                "section_title": title,
                "start_page": all_lines[0][1],
                "end_page": all_lines[first_pos - 1][1],
                "title": title,
            })

    for idx, (pos, heading_text, page_num) in enumerate(heading_positions):
        next_pos = (
            heading_positions[idx + 1][0]
            if idx + 1 < len(heading_positions)
            else len(all_lines)
        )
        section_text = "\n".join(line for line, _ in all_lines[pos:next_pos]).strip()
        if not section_text:
            continue
        end_page = all_lines[next_pos - 1][1] if next_pos <= len(all_lines) else page_num
        sections.append({
            "text": section_text,
            "section_title": heading_text,
            "start_page": page_num,
            "end_page": end_page,
            "title": title,
        })

    # Merge sections có cùng chapter key (xử lý OCR noise)
    merged: list[dict] = []
    for sec in sections:
        key = _normalize_heading_key(sec["section_title"])
        if merged and _normalize_heading_key(merged[-1]["section_title"]) == key:
            merged[-1]["text"] += "\n" + sec["text"]
            merged[-1]["end_page"] = sec["end_page"]
        else:
            merged.append(dict(sec))

    return merged


def load_pdfs_from_folder(folder_path: str) -> list[Document]:
    """Nạp tất cả PDF trong folder, trả về list Document với metadata đầy đủ."""
    documents: list[Document] = []
    if not os.path.exists(folder_path):
        print(f"Không tìm thấy thư mục: {folder_path}")
        return documents

    pdf_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".pdf"))
    print(f"Tìm thấy {len(pdf_files)} file PDF")

    for file in pdf_files:
        file_path = os.path.join(folder_path, file)
        try:
            sections = _extract_sections(file_path)
            for section in sections:
                documents.append(Document(
                    page_content=section["text"],
                    metadata={
                        "source": file,
                        "title": section["title"],
                        "section_title": section["section_title"],
                        "start_page": section["start_page"],
                        "end_page": section["end_page"],
                    },
                ))
            print(f"  Loaded {file}: {len(sections)} sections")
        except Exception as e:
            print(f"  Lỗi đọc {file}: {e}")

    return documents


__all__ = ["load_pdfs_from_folder"]
