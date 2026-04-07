import re #Regular Expression
import unicodedata

def clean_text(text):
    if not text:
        return ""
    #Chuẩn hoá Uincode về dạng dựng sẵn NFC
    text = unicodedata.normalize('NFC',text)
    #Xử lý ngắt dòng và nối từ 
    text = text.replace("\r\n","\n").replace("\r","\n")
    text = re.sub(r"(?<=w)-\n(?=\w)","",text)
    #Gom khoảng trắng dư thưa
    text = re.sub(r"[^\S\n]+"," ",text)
    
    cleaned_lines = []
    for raw_line in text.splitlines():
        # Xóa khoảng trắng thừa trong dòng
        line = re.sub(r"\s+", " ", raw_line).strip()
        
        # Lọc ký tự lỗi
        line = re.sub(r"[^\w\s.,;:!?%()/\"'À-ỹ-]", "", line)

        if not line:
            # Chỉ thêm dòng trống nếu dòng trước đó không trống (tránh dồn quá nhiều \n)
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        cleaned_lines.append(line)

    # 4. Ghép lại và dọn dẹp các dòng trống dư thừa
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def clean_documents(documents):
    cleaned_docs = []

    for doc in documents:
        cleaned_text = clean_text(doc.page_content)

        if cleaned_text:  # bỏ trang rỗng
            # Tạo copy hoặc gán trực tiếp tùy luồng dữ liệu, metadata vẫn được giữ nguyên
            doc.page_content = cleaned_text
            cleaned_docs.append(doc)

    print(f"Sau khi clean: {len(cleaned_docs)} documents")

    return cleaned_docs