from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from hashlib import sha256
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR.parent
DEFAULT_CHUNK_ID_REGISTRY_PATH = DATA_DIR / "chunk_id_registry.json"

# NHÓM 1: CÁC HÀM TIỆN ÍCH (HELPER FUNCTIONS)

def _to_json_safe_metadata(metadata: dict) -> dict:
    """
    Hàm này đảm bảo metadata (siêu dữ liệu như số trang, tên file) an toàn để lưu ra file JSON
    hoặc đẩy lên Vector Database. Các hệ thống Vector DB thường rất "khó tính", 
    chúng sẽ báo lỗi nếu gặp các kiểu dữ liệu lạ không phải str, int, float hay list cơ bản.
    """
    safe_metadata = {}

    for key, value in (metadata or {}).items():
        # Giữ nguyên các kiểu dữ liệu nguyên thủy an toàn
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_metadata[key] = value
        # Nếu là mảng (list/tuple), kiểm tra và ép kiểu từng phần tử bên trong
        elif isinstance(value, (list, tuple)):
            safe_metadata[key] = [
                item if isinstance(item, (str, int, float, bool)) or item is None else str(item)
                for item in value
            ]
        # Ép tất cả các kiểu phức tạp khác (ví dụ: object datetime) về chuỗi (string)
        else:
            safe_metadata[key] = str(value)

    return safe_metadata


def _normalize_whitespace(text: str) -> str:
    """
    Hàm dọn dẹp khoảng trắng thừa.
    Chuyển các chuỗi nhiều dấu cách, tab hoặc xuống dòng liên tiếp thành 1 dấu cách duy nhất.
    """
    return re.sub(r"\s+", " ", (text or "")).strip()


def _load_chunk_id_registry(registry_path: str | os.PathLike | None = None) -> dict:
    """
    Đọc registry ID bền vững cho parent/child chunks.
    Nếu chưa có thì khởi tạo bộ đếm rỗng.
    """
    resolved_path = Path(registry_path or DEFAULT_CHUNK_ID_REGISTRY_PATH)
    default_registry = {
        "parent": {"next_index": 1, "signatures": {}},
        "child": {"next_index": 1, "signatures": {}},
    }

    if not resolved_path.exists():
        return default_registry

    try:
        with open(resolved_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return default_registry

    if not isinstance(data, dict):
        return default_registry

    for chunk_type in ("parent", "child"):
        bucket = data.get(chunk_type)
        if not isinstance(bucket, dict):
            data[chunk_type] = {"next_index": 1, "signatures": {}}
            continue

        if not isinstance(bucket.get("next_index"), int) or bucket["next_index"] < 1:
            bucket["next_index"] = 1
        if not isinstance(bucket.get("signatures"), dict):
            bucket["signatures"] = {}

    return data


def _save_chunk_id_registry(
    registry: dict,
    registry_path: str | os.PathLike | None = None,
) -> None:
    """Lưu registry ID để các lần ingest sau tái sử dụng cùng một mã chunk."""
    resolved_path = Path(registry_path or DEFAULT_CHUNK_ID_REGISTRY_PATH)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=2)


def _build_chunk_signature_key(payload: dict) -> str:
    """Sinh khóa chữ ký gọn từ nội dung và ngữ cảnh chunk."""
    raw_signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return sha256(raw_signature.encode("utf-8")).hexdigest()


def _assign_stable_chunk_id(
    registry: dict,
    chunk_type: str,
    signature_key: str,
) -> str:
    """Cấp ID dạng parent_000001 / child_000001 và giữ ổn định qua nhiều lần chạy."""
    bucket = registry.setdefault(
        chunk_type,
        {"next_index": 1, "signatures": {}},
    )
    signatures = bucket.setdefault("signatures", {})
    existing_id = signatures.get(signature_key)
    if existing_id:
        return existing_id

    next_index = int(bucket.setdefault("next_index", 1))
    assigned_id = f"{chunk_type}_{next_index:06d}"
    signatures[signature_key] = assigned_id
    bucket["next_index"] = next_index + 1
    return assigned_id


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """
    Hàm "dao mổ" cắt văn bản thông minh dựa trên ngữ nghĩa (dấu câu).
    Thay vì cắt ngang một từ, nó sẽ tìm dấu chấm câu để cắt, giúp bảo toàn ý nghĩa trọn vẹn.
    """
    text = _normalize_whitespace(text)
    if not text:
        return []
    
    # Nếu đoạn text đã ngắn hơn mức cho phép, không cần cắt nữa
    if len(text) <= max_chars:
        return [text]

    # BƯỚC 1: Cắt theo câu hoàn chỉnh
    # Regex này tìm các dấu kết thúc câu (. ! ? ; :) CÓ đi kèm khoảng trắng, 
    # VÀ ngay sau đó là một chữ cái viết hoa (bao gồm cả tiếng Việt như À-Ỹ) hoặc chữ số.
    sentences = re.split(r"(?<=[.!?;:])\s+(?=[A-ZÀ-Ỹ0-9])", text)
    
    # BƯỚC 2: Nếu không tìm thấy dấu chấm câu (có thể do lỗi văn bản), thử cắt theo dấu phẩy
    if len(sentences) == 1:
        sentences = re.split(r"(?<=,)\s+", text)
        
    # BƯỚC 3: Nếu vẫn không cắt được (đoạn text quá dài mà không có dấu câu nào),
    # đành phải dùng cách "cắt thô bạo" theo số lượng ký tự (max_chars)
    if len(sentences) == 1:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    # BƯỚC 4: Gom các câu nhỏ lại thành các khối (chunk) có độ dài xấp xỉ max_chars
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        
        # Nếu thêm câu này vào mà vượt quá max_chars, ta đóng gói chunk hiện tại lại
        if current_len + sentence_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
            
        current_chunk.append(sentence)
        # Cộng thêm 1 cho khoảng trắng giữa các câu
        current_len += sentence_len + 1 

    # Đừng quên đẩy phần còn dư vào chunk cuối cùng
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _create_chunks_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Hàm tạo các chunk có phần giao nhau (overlap).
    Overlap giúp AI không bị mất ngữ cảnh (lost in context) khi một sự kiện bị cắt nằm vắt ngang giữa 2 chunk.
    """
    # Đầu tiên, băm text thành các đoạn nhỏ vừa phải (bằng một nửa chunk_size)
    base_chunks = _split_long_text(text, max_chars=chunk_size // 2)
    
    final_chunks = []
    current_text = ""
    
    for chunk in base_chunks:
        if not current_text:
            current_text = chunk
            continue
            
        # Nối thêm chunk tiếp theo
        combined = current_text + " " + chunk
        
        if len(combined) > chunk_size:
            final_chunks.append(current_text)
            # Tạo overlap bằng cách giữ lại một phần của text cũ (cắt từ phía sau)
            # Đảm bảo phần giữ lại không lớn hơn thông số overlap cho phép
            overlap_text = current_text[-overlap:] if overlap > 0 else ""
            current_text = _normalize_whitespace(overlap_text + " " + chunk)
        else:
            current_text = combined
            
    if current_text:
        final_chunks.append(current_text)
        
    return final_chunks
# NHÓM 2: HÀM LÕI XỬ LÝ PARENT-CHILD CHUNKING


def build_parent_child_chunks(
    documents: list, 
    parent_chunk_size: int = 2000, 
    parent_chunk_overlap: int = 400, 
    child_chunk_size: int = 600, 
    child_chunk_overlap: int = 200,
    id_registry_path: str | os.PathLike | None = None,
):
    """
    Trái tim của hệ thống Small-to-Big Retrieval.
    Nhận vào danh sách các Document thô và trả về 3 thành phần:
    1. child_docs: Các mảnh nhỏ dùng để đưa vào Vector DB (để search cho chuẩn).
    2. parent_store: Dictionary chứa các mảnh lớn (để lưu ổ cứng hoặc MongoDB).
    3. parent_docs: Các mảnh lớn dạng Document (để nhồi vào LightRAG vẽ đồ thị).
    """
    child_docs = []
    parent_docs = []
    parent_store = {}
    id_registry = _load_chunk_id_registry(id_registry_path)

    for doc in documents:
        base_metadata = _to_json_safe_metadata(doc.metadata)
        # 1. Tạo các Mảnh Cha (Parent Chunks) lớn để bao quát ngữ cảnh
        p_chunks = _create_chunks_with_overlap(
            doc.page_content, 
            chunk_size=parent_chunk_size, 
            overlap=parent_chunk_overlap
        )

        for parent_chunk_index, p_text in enumerate(p_chunks, start=1):
            parent_id = _assign_stable_chunk_id(
                id_registry,
                "parent",
                _build_chunk_signature_key(
                    {
                        "chunk_type": "parent",
                        "source": base_metadata.get("source"),
                        "page": base_metadata.get("page"),
                        "page_label": base_metadata.get("page_label"),
                        "title": base_metadata.get("title"),
                        "chunk_index": parent_chunk_index,
                        "page_content": _normalize_whitespace(p_text),
                    }
                ),
            )
            
            # Ghi nhận mảnh cha vào bộ nhớ (store)
            parent_store[parent_id] = p_text
            
            # Tạo Document cho Mảnh Cha
            p_metadata = deepcopy(base_metadata)
            p_metadata["doc_id"] = parent_id
            p_metadata["parent_id"] = parent_id
            p_metadata["chunk_type"] = "parent"
            
            # Giả định bạn dùng LangChain Document, ta clone object ra
            p_doc = deepcopy(doc)
            p_doc.page_content = p_text
            p_doc.metadata = p_metadata
            parent_docs.append(p_doc)

            # 2. Băm Mảnh Cha này thành các Mảnh Con (Child Chunks) nhỏ
            c_chunks = _create_chunks_with_overlap(
                p_text, 
                chunk_size=child_chunk_size, 
                overlap=child_chunk_overlap
            )

            for child_chunk_index, c_text in enumerate(c_chunks, start=1):
                child_id = _assign_stable_chunk_id(
                    id_registry,
                    "child",
                    _build_chunk_signature_key(
                        {
                            "chunk_type": "child",
                            "parent_id": parent_id,
                            "source": base_metadata.get("source"),
                            "page": base_metadata.get("page"),
                            "page_label": base_metadata.get("page_label"),
                            "title": base_metadata.get("title"),
                            "chunk_index": child_chunk_index,
                            "page_content": _normalize_whitespace(c_text),
                        }
                    ),
                )
                # Tạo Document cho Mảnh Con (dùng cho Vector DB)
                c_metadata = deepcopy(p_metadata)
                c_metadata["chunk_type"] = "child"
                # SỢI DÂY LIÊN KẾT QUAN TRỌNG NHẤT: Gắn parent_id vào metadata của Mảnh Con
                c_metadata["parent_id"] = parent_id 
                c_metadata["child_id"] = child_id
                c_metadata["doc_id"] = child_id
                
                c_doc = deepcopy(doc)
                c_doc.page_content = c_text
                c_doc.metadata = c_metadata
                child_docs.append(c_doc)

    _save_chunk_id_registry(id_registry, id_registry_path)
    print(f"Đã tạo {len(parent_docs)} parent chunks và {len(child_docs)} child chunks")
    
    return child_docs, parent_store, parent_docs
# NHÓM 3: CÁC HÀM QUẢN LÝ LƯU TRỮ (I/O)

def save_parent_documents(parent_store: dict, output_path: str):
    """
    Lưu bộ nhớ Mảnh Cha (Dictionary) xuống file JSON. 
    Sau này khi VectorDB tìm được Child, ta dựa vào parent_id để mở file này lên tra cứu Mảnh Cha.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(parent_store, file, ensure_ascii=False, indent=2)

    print(f"Đã lưu {len(parent_store)} parent chunks vào {output_path}")


def save_documents(documents: list, output_path: str):
    """
    Lưu một danh sách các objects Document (của LangChain) xuống file JSON.
    Hàm này hữu ích để backup lại toàn bộ dữ liệu metadata và text trước khi nhúng.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = [
        {
            "page_content": document.page_content,
            "metadata": _to_json_safe_metadata(document.metadata),
        }
        for document in documents
    ]

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"Đã lưu {len(payload)} documents vào {output_path}")


def load_parent_documents(input_path: str) -> dict:
    """
    Đọc ngược file JSON chứa Mảnh Cha lên RAM.
    """
    if not os.path.exists(input_path):
        print(f"Cảnh báo: Không tìm thấy file {input_path}")
        return {}

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    return data
