# Báo Cáo Buổi 1 — Dense Embedding & E5 Model

**Ngày học:** 2026-05-18
**File học:** `app/data/process_data/e5_embeddings.py`
**Phase:** 1 — RAG Core

---

## 1. Kiến Thức Nền Tảng

### Embedding là gì?

Hàm chuyển đổi text thành vector số học để máy tính so sánh được ý nghĩa.

```
"Ngô Quyền"          → [0.12, -0.34, 0.89, ..., 0.05]   (384 số)
"vị tướng Ngô Quyền" → [0.11, -0.32, 0.88, ..., 0.06]   ← gần nhau
"Phở bò"             → [-0.80, 0.44, -0.12, ..., 0.99]  ← xa nhau
```

Text gần nghĩa → vector gần nhau trong không gian 384 chiều.

### Dense vs Sparse

| | Dense (E5) | Sparse (BM25) |
|---|---|---|
| Biểu diễn | 384 số, tất cả khác 0 | Vài trăm cặp (index, weight) |
| Hiểu ngữ nghĩa | ✅ "vua" ≈ "quân chủ" | ❌ |
| Exact keyword | ❌ có thể miss | ✅ luôn đúng |

### Cách model E5 hoạt động bên trong (4 bước)

```
Text input
    │
    ▼
1. Tokenize  →  cắt thành tokens, mỗi token = 1 số ID
    │
    ▼
2. BERT forward pass  →  hidden state 384-dim cho mỗi token
    │
    ▼
3. Mean pooling  →  trung bình các token → 1 vector (384,)
    │
    ▼
4. Normalize  →  điều chỉnh độ dài vector = 1 (unit vector)
```

---

## 2. Phân Tích Code `e5_embeddings.py`

### Hằng số (dòng 8–17)

```python
E5_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
# HuggingFace model ID — tự download khi khởi tạo (~100MB)

E5_EMBEDDING_DIM = 384
# Output luôn là vector 384 chiều, bất kể input dài hay ngắn

E5_MAX_LENGTH = 512
# Tối đa 512 tokens (~380 từ tiếng Việt) — dài hơn bị cắt → cần chunking

E5_PROMPTS = {
    "query":   "query: ",    # prefix cho câu hỏi của user
    "passage": "passage: ",  # prefix cho document khi indexing
}
```

**Tại sao cần 2 prefix?** E5 được train với cặp `(query, passage)`. Dùng sai prefix → retrieval kém ~10-15%.

---

### `E5EmbeddingConfig` — Dataclass (dòng 20–31)

```python
@dataclass(slots=True)
class E5EmbeddingConfig:
    model_name: str = E5_EMBEDDING_MODEL_NAME   # HuggingFace model ID
    prompt_name: str = E5_PASSAGE_PROMPT_NAME   # "passage" — default cho indexing
    batch_size: int = 32                        # xử lý 32 texts cùng lúc
    normalize_embeddings: bool = True           # chuẩn hóa về unit vector
    device: str | None = None                   # None = auto-detect GPU/CPU

    @property
    def embedding_dim(self) -> int:
        return E5_EMBEDDING_DIM                 # read-only, luôn = 384
```

---

### `E5EmbeddingModel` — Class chính (dòng 33–61)

```python
class E5EmbeddingModel:

    def __init__(self, config: E5EmbeddingConfig | None = None):
        self.config = config or E5EmbeddingConfig()   # dùng default nếu không truyền
        self.model = SentenceTransformer(
            self.config.model_name,
            prompts=E5_PROMPTS,
            default_prompt_name=self.config.prompt_name,
            device=self.config.device,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        encoded = self.model.encode(
            texts,
            prompt_name=self.config.prompt_name,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return np.asarray(encoded, dtype=np.float32)
```

---

## 3. Syntax Python Đã Học

### Class cơ bản

```python
class Foo:
    def __init__(self, x):   # chạy khi Foo(x)
        self.x = x           # lưu vào object

    def bar(self):           # method — luôn có self
        return self.x
```

`self` = bản thân object. Python tự truyền vào, không gọi thủ công.

---

### `@dataclass` và `@dataclass(slots=True)`

```python
# Không có @dataclass:
class Config:
    def __init__(self, name, size):
        self.name = name
        self.size = size

# Có @dataclass — tương đương, ngắn hơn:
@dataclass
class Config:
    name: str
    size: int = 32
```

`slots=True` thêm:
- Truy cập attribute nhanh hơn
- Ngăn tạo attribute không khai báo (bắt typo ngay lập tức)

```python
c = Config("x")
c.siz = 64   # slots=True → AttributeError — phát hiện typo ngay
             # không có slots → yên lặng tạo attribute mới, bug khó tìm
```

---

### `@property`

```python
@property
def embedding_dim(self) -> int:
    return 384
```

- Gọi không có `()`: `obj.embedding_dim` → `384`
- Read-only: `obj.embedding_dim = 9` → `AttributeError`

---

### Type hints

```python
name: str                 # chỉ str
count: int = 0            # int, default = 0
device: str | None = None # str HOẶC None
texts: list[str]          # list các string
-> np.ndarray             # return type
```

Chỉ là ghi chú — Python không tự enforce, nhưng IDE dùng để cảnh báo.

---

### Pattern `or` để gán default

```python
self.config = config or E5EmbeddingConfig()
# Nếu config là None/falsy → dùng E5EmbeddingConfig()
# Nếu config có giá trị   → dùng config đó
```

Falsy values: `None`, `[]`, `{}`, `""`, `0`, `False`.

---

### `if not x` — kiểm tra rỗng

```python
if not texts:   # True nếu texts = [] hoặc None
    return ...
```

---

### Keyword vs Positional arguments

```python
SentenceTransformer(
    "intfloat/e5-small",          # positional — theo vị trí
    prompts=E5_PROMPTS,           # keyword — theo tên
    device=None,                  # keyword — thứ tự không quan trọng
)
```

---

### NumPy cơ bản

```python
np.array([1.0, 2.0])              # tạo array từ list
np.empty((0, 384), dtype=np.float32)  # mảng rỗng 0 hàng × 384 cột
np.asarray(x, dtype=np.float32)   # convert, không copy nếu đã đúng type

arr.shape   # (N, 384) — N texts, 384 chiều
arr[0]      # vector của text đầu tiên, shape (384,)
arr.tolist() # convert về Python list để gửi cho Qdrant
```

`float32` vs `float64`: float32 = 4 bytes, float64 = 8 bytes → dùng float32 tiết kiệm 50% RAM.

---

## 4. Normalize — Tại Sao Quan Trọng

Vector `[3, 4]` → độ dài = √(3² + 4²) = 5

Normalize: `[3/5, 4/5]` = `[0.6, 0.8]` → độ dài = 1

Lợi ích:
```
cosine_similarity(A, B) = (A·B) / (|A| × |B|)

Sau normalize (|A| = |B| = 1):
cosine_similarity(A, B) = A·B  ← chỉ dot product, không cần chia → nhanh hơn
```

---

## 5. Cách Dùng Trong Dự Án (retriever.py)

```python
# Index documents → dùng prompt "passage"
passage_model = E5EmbeddingModel()  # default prompt_name = "passage"

# Embed query của user → dùng prompt "query"
query_model = E5EmbeddingModel(E5EmbeddingConfig(prompt_name="query"))

# Trong retriever.py dòng 39:
dense_vec = dense_model.embed([q["dense"]])[0]
#                              └─────────┘ └─┘
#                              list 1 text   lấy hàng 0 → shape (384,)

dense_vec.tolist()  # convert sang list để gửi lên Qdrant
```

---

## 6. Bảng Tra Cứu Nhanh

| Syntax | Ý nghĩa |
|---|---|
| `class Foo:` | Khai báo class |
| `def __init__(self, x):` | Hàm khởi tạo |
| `self.x = x` | Lưu attribute vào object |
| `@dataclass` | Tự tạo `__init__` từ annotations |
| `@dataclass(slots=True)` | Thêm: nhanh hơn + ngăn typo |
| `@property` | Method dùng không có `()`, read-only |
| `x: str \| None = None` | Type hint — str hoặc None, default None |
| `a or b` | Nếu a falsy → dùng b |
| `if not x:` | Nếu x rỗng/None/0 |
| `np.asarray(x, dtype=np.float32)` | Convert sang NumPy float32 |
| `arr[0]` | Hàng đầu tiên của array |
| `keyword=value` | Keyword argument khi gọi hàm |

---

## 7. Câu Hỏi Ôn Tập

1. Tại sao E5 cần prefix `"query: "` và `"passage: "` khác nhau?
2. `@dataclass(slots=True)` ngăn được loại bug nào?
3. `normalize_embeddings=True` giúp tăng tốc độ search như thế nào?
4. Tại sao dùng `float32` thay vì `float64`?
5. Dòng `dense_vec = dense_model.embed([q["dense"]])[0]` — tại sao wrap trong `[]` rồi lấy `[0]`?
6. `a or b` trong Python trả về gì khi `a = None`? Khi `a = "hello"`?
7. `@property` khác gì so với method thông thường?

---

*Buổi tiếp theo: Sparse Embedding (BM25) + Qdrant — cách index và search.*
