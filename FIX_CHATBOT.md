# Fix luồng hỏi đáp không hoạt động

## Vấn đề phát hiện

1. ✅ Server đang chạy (port 8000)
2. ✅ Qdrant đang chạy (port 6333)
3. ✅ Có 9 PDF files trong `app/data/ocr_data/`
4. ✅ Có `parent_docs.json` (35MB)
5. ❌ **Qdrant collection `vietnam_history_hybrid` có 0 points** → KHÔNG CÓ DỮ LIỆU

## Nguyên nhân

Dữ liệu PDF chưa được index vào Qdrant vector database. Khi gọi `/ask`, engine không tìm được documents nên trả về "Không tìm thấy tài liệu".

## Cách fix

### Bước 1: Index dữ liệu vào Qdrant (BẮT BUỘC)

```bash
cd /Users/hoangminh/Desktop/VietNam\ Historical\ AI/app/services/chatbot

# Test với 10 parent chunks đầu (nhanh, ~2-3 phút)
python run_qdrant_index.py --test

# Hoặc full index (chậm, ~30-60 phút)
python run_qdrant_index.py
```

**Lưu ý:** Script này sẽ:
- Load 9 PDF files từ `app/data/ocr_data/`
- Clean text + split thành parent/child chunks
- Generate E5 embeddings (384 dim) + BM25 sparse vectors
- Upload vào Qdrant collection `vietnam_history_hybrid`

### Bước 2: Index knowledge graph vào LightRAG (TÙY CHỌN)

```bash
# Chạy sau khi Qdrant index xong
python run_lightrag_index.py --test
```

### Bước 3: Restart server để warmup engine

```bash
# Kill server hiện tại
pkill -f "uvicorn main:app"

# Start lại
cd /Users/hoangminh/Desktop/VietNam\ Historical\ AI/app/server
uvicorn main:app --reload
```

Server sẽ log:
```
INFO: Khởi tạo VietnamHistoryQueryEngine...
INFO: ✓ Engine warm-up hoàn tất, sẵn sàng nhận request.
```

### Bước 4: Test endpoint

```bash
# Test warmup
curl http://localhost:8000/warmup | python3 -m json.tool

# Kết quả mong đợi:
# {
#   "status": "ready",
#   "lightrag_ready": true,
#   "qdrant_ok": true,
#   "qdrant_collections": ["vietnam_history_hybrid"]
# }

# Test ask
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Trần Hưng Đạo là ai?"}'
```

## Kiểm tra trạng thái hiện tại

```bash
# Check Qdrant points count
curl -s http://localhost:6333/collections/vietnam_history_hybrid | \
  python3 -c "import sys,json; print('Points:', json.load(sys.stdin)['result']['points_count'])"

# Nếu = 0 → cần chạy index
# Nếu > 0 → đã có dữ liệu
```

## Timeline ước tính

- **Test mode** (10 chunks): ~3 phút
- **Full index** (toàn bộ 9 PDF): ~45 phút
  - Load PDF: 5 phút
  - Generate embeddings: 30 phút
  - Upload Qdrant: 10 phút

## Troubleshooting

### Lỗi: "fastembed crash trên Python 3.14/macOS"

Nếu gặp lỗi này khi chạy index:
```bash
# Downgrade Python về 3.11 hoặc 3.12
pyenv install 3.12.0
pyenv local 3.12.0
pip install -r requirements.txt
```

### Lỗi: "GEMINI_KEY not found"

Kiểm tra `.env` file:
```bash
cat /Users/hoangminh/Desktop/VietNam\ Historical\ AI/.env | grep -E "GEMINI_KEY|OPENAI_API_KEY"
```

Cần có một trong các key:
- `GEMINI_KEY=xxx`
- `OPENAI_API_KEY=xxx`
- `SHOPAIKEY_TOKEN=xxx`

### Lỗi: "Qdrant connection refused"

```bash
# Check Qdrant container
docker ps | grep qdrant

# Restart nếu unhealthy
docker restart vical_qdrant
```

## Sau khi fix

Chatbot sẽ hoạt động với:
- Vector search (E5 + BM25 hybrid)
- Knowledge graph (LightRAG - nếu đã index)
- Query rewriting + broad query decomposition
- Persona chat (Trần Hưng Đạo, Ngô Quyền, Hồ Chí Minh)
