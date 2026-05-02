import re #Regular Expression
import unicodedata

def clean_text(text):
    if not text:
        return ""
    # Chuẩn hoá Unicode về dạng dựng sẵn NFC
    text = unicodedata.normalize('NFC', text)
    # Xử lý ngắt dòng và nối từ bị tách bởi dấu gạch nối cuối dòng
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)   # fix: (?<=w) → (?<=\w)
    # Gom khoảng trắng dư thừa (trừ xuống dòng)
    text = re.sub(r"[^\S\n]+", " ", text)

    cleaned_lines = []
    for raw_line in text.splitlines():
        # Xóa khoảng trắng thừa trong dòng
        line = re.sub(r"\s+", " ", raw_line).strip()

        # Lọc ký tự lỗi — giữ lại:
        #   \w  : chữ/số/gạch dưới
        #   \s  : khoảng trắng
        #   Dấu câu cơ bản: .,;:!?%()/\"'
        #   Dấu ngoặc: []()
        #   Dấu gạch ngang và gạch nối: -–—  (quan trọng trong văn bản lịch sử)
        #   Tiếng Việt: À-ỹ
        line = re.sub(r"[^\w\s.,;:!?%()\[\]/\"'\-–—À-ỹ]", "", line)

        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def clean_documents(documents):
    cleaned_docs = []

    for doc in documents:
        cleaned_text = clean_text(doc.page_content)

        if cleaned_text:  # bỏ trang rỗng
            doc.page_content = cleaned_text
            cleaned_docs.append(doc)

    print(f"Sau khi clean: {len(cleaned_docs)} documents")

    return cleaned_docs