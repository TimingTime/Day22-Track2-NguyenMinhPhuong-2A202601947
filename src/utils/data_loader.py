"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
import hashlib
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    from langchain_community.vectorstores import FAISS

    project_root = Path(__file__).parent.parent.parent
    model_name = str(
        getattr(embeddings, "model", None)
        or getattr(embeddings, "model_name", None)
        or embeddings.__class__.__name__
    )
    fingerprint = hashlib.sha256()
    fingerprint.update(model_name.encode("utf-8"))
    for chunk in chunks:
        fingerprint.update(b"\0")
        fingerprint.update(chunk.encode("utf-8"))
    cache_path = project_root / ".cache" / "faiss" / fingerprint.hexdigest()[:16]

    if (cache_path / "index.faiss").exists() and (cache_path / "index.pkl").exists():
        print(f"♻️  Đang dùng FAISS index đã cache ({len(chunks)} chunks) ...")
        return FAISS.load_local(
            str(cache_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    cache_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(cache_path))
    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
