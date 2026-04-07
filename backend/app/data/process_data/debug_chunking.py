import os
import shutil  # Thư viện dùng để xóa toàn bộ một thư mục (kể cả khi bên trong có chứa file)
import sys
import uuid
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

for import_path in (CURRENT_DIR, PARENT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

# NHÓM 1: CƠ CHẾ IMPORT THÔNG MINH (FLEXIBLE IMPORTS)
# Kỹ thuật try-except này giúp file có thể chạy trơn tru trong 2 trường hợp:
# 1. Chạy như một module nằm bên trong package lớn 
# 2. Chạy trực tiếp như một script độc lập trên terminal
try:
    from load_data import load_pdfs_from_folder
    from clean_data import clean_documents
    from e5_embeddings import (
        E5_EMBEDDING_DIM,
        E5_EMBEDDING_MODEL_NAME,
        E5_PASSAGE_PROMPT_NAME,
        E5EmbeddingConfig,
        E5EmbeddingModel,
    )
    from splitter import (
        build_parent_child_chunks,
        save_documents,
        save_parent_documents,
    )
except ImportError:
    from data.process_data.clean_data import clean_documents
    from data.process_data.e5_embeddings import (
        E5_EMBEDDING_DIM,
        E5_EMBEDDING_MODEL_NAME,
        E5_PASSAGE_PROMPT_NAME,
        E5EmbeddingConfig,
        E5EmbeddingModel,
    )
    from data.process_data.load_data import load_pdfs_from_folder
    from data.process_data.splitter import (
        build_parent_child_chunks,
        save_documents,
        save_parent_documents,
    )
# NHÓM 2: CẤU HÌNH ĐƯỜNG DẪN TĨNH (PATH CONFIGURATION)
# Luôn dùng os.path.abspath để lấy đường dẫn tuyệt đối của file hiện tại.
# Điều này đảm bảo dù bạn đứng ở bất kỳ thư mục nào trên terminal gõ lệnh chạy file,
# code vẫn sẽ tìm đúng đến thư mục chứa dữ liệu mà không bị lỗi "File Not Found".
# 3. Trỏ vào thư mục raw_data và các thư mục khác
RAW_DATA_PATH = os.path.join(str(PARENT_DIR), "raw_data")
QDRANT_DB_PATH = os.path.join(str(PARENT_DIR), "qdrant_db")
PARENT_DOCSTORE_PATH = os.path.join(str(PARENT_DIR), "parent_docs.json")
CHILD_DOCSTORE_PATH = os.path.join(str(PARENT_DIR), "child_docs.json")
COLLECTION_NAME = "vietnam_history_hybrid"
# NHÓM 3: HÀM DỌN DẸP MÔI TRƯỜNG (RESET STATE)
def reset_chroma_db():
    """
    Hàm cực kỳ quan trọng trong giai đoạn phát triển (Development).
    Mỗi lần chạy lại pipeline, ta phải xóa sạch Database và các file JSON cũ.
    Nếu không, ChromaDB sẽ liên tục cộng dồn vector mới vào vector cũ, 
    gây ra hiện tượng "nhân bản dữ liệu" (duplicate data) làm lệch kết quả tìm kiếm và tốn dung lượng.
    """
    if os.path.exists(QDRANT_DB_PATH):
        print("Đang xóa Vector Database cũ trong qdrant_db...")
        shutil.rmtree(QDRANT_DB_PATH) # Xóa sạch cả thư mục và các file bên trong

    if os.path.exists(PARENT_DOCSTORE_PATH):
        print("Đang xóa Parent Doc Store cũ...")
        os.remove(PARENT_DOCSTORE_PATH)

    if os.path.exists(CHILD_DOCSTORE_PATH):
        print("Đang xóa Child Doc Store cũ...")
        os.remove(CHILD_DOCSTORE_PATH)


# NHÓM 4: LUỒNG THỰC THI CHÍNH (THE ORCHESTRATOR)
def ingest_historical_data():
    """
    Hàm nhạc trưởng điều phối toàn bộ vòng đời của tài liệu (ETL Pipeline):
    Extract (Lấy) -> Transform (Biến đổi) -> Load (Nạp vào DB)
    """
    print("Bắt đầu chạy Data Ingestion Pipeline...")

    #BƯỚC 0: Dọn dẹp chiến trường trước khi bắt đầu
    reset_chroma_db()

    #BƯỚC 1: EXTRACT (ĐỌC FILE)
    print("\n[1/4] Đang load dữ liệu từ thư mục raw...")
    docs = load_pdfs_from_folder(RAW_DATA_PATH)
    if not docs:
        print("Lỗi: Không tìm thấy file PDF nào trong thư mục 'raw_data'. Dừng quá trình.")
        return

    #BƯỚC 2: TRANSFORM - CLEANING (LÀM SẠCH)
    print("\n[2/4] Đang dọn dẹp khoảng trắng và chuẩn hóa Unicode tiếng Việt...")
    docs = clean_documents(docs)

    #BƯỚC 3: TRANSFORM - SPLITTING (CHIA CẮT PHÂN CẤP)
    print("\n[3/4] Đang chia nhỏ dữ liệu theo mô hình Parent-Child...")
    # Gọi cỗ máy cắt thái từ file splitter.py
    child_chunks, parent_store, parent_docs = build_parent_child_chunks(docs)

    if not child_chunks:
        print("Lỗi: Quá trình phân rã thất bại, không tạo được child chunks.")
        return

    # Lưu lại để làm kho tra cứu ngược (Document Store) khi hệ thống truy vấn
    print("Lưu trữ các mảnh JSON xuống ổ cứng...")
    save_parent_documents(parent_store, PARENT_DOCSTORE_PATH)
    save_documents(child_chunks, CHILD_DOCSTORE_PATH)

    # BƯỚC 4: NHÚNG DỮ LIỆU VỚI MULTILINGUAL-E5-SMALL VÀ BM25
    print("\n[4/4] Khởi tạo multilingual-e5-small & Qdrant...")
    from qdrant_client import QdrantClient, models
    from fastembed import SparseTextEmbedding

    # 4.1 Khởi tạo Model
    print(f"Đang tải {E5_EMBEDDING_MODEL_NAME} (Dense) và BM25 (Sparse)...")
    dense_model = E5EmbeddingModel(
        E5EmbeddingConfig(
            prompt_name=E5_PASSAGE_PROMPT_NAME,
            batch_size=32,
        )
    )
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    
    # lưu vào docker
    client = QdrantClient(url="http://localhost:6333")

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' đã tồn tại, đang xóa để tạo lại schema mới...")
        client.delete_collection(COLLECTION_NAME)

    # 4.2 Tạo collection mới theo schema của multilingual-e5-small
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=E5_EMBEDDING_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )

    # 4.3 Chuẩn bị văn bản. E5 cần prefix passage/query, wrapper sẽ tự thêm prompt phù hợp.
    texts_to_embed = [doc.page_content for doc in child_chunks]
    
    print(f"      Đang chạy mô hình để tạo Hybrid Vector cho {len(texts_to_embed)} mảnh con...")

    # 4.4 Đóng gói và nạp theo lô (Batching)
    print(f"      Đang tiến hành nhúng và nạp dữ liệu theo lô...")
    
    batch_size = min(100, len(texts_to_embed)//10 or 10)  
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        batch_docs = child_chunks[i:i + batch_size]
        
        # Tạo vector cho lô hiện tại
        batch_dense = dense_model.embed(batch_texts)
        batch_sparse = list(sparse_model.embed(batch_texts,parallel=0))
        
        points = []
        for j in range(len(batch_texts)):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": batch_dense[j].tolist(),
                        "sparse": models.SparseVector(
                            indices=batch_sparse[j].indices.tolist(), 
                            values=batch_sparse[j].values.tolist()
                        )
                    },
                    payload={
                        "page_content": batch_docs[j].page_content,
                        **batch_docs[j].metadata
                    }
                )
            )
        
        # Nạp ngay lô này vào Qdrant
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Đã xong: {i + len(batch_texts)}/{len(texts_to_embed)} chunks...")
        
    
    print("\nHOÀN TẤT PIPELINE!")
    print(f"Đã xây dựng thành công.")
    print(f"- Số lượng: {len(parent_docs)} Parent Chunks | {len(child_chunks)} Child Chunks")
# NHÓM 5: ENTRY POINT
if __name__ == "__main__":
    ingest_historical_data()
