"""
DIP Diet RAG - Vector Store Ingestion Pipeline
Ingests DIP diet knowledge base documents into ChromaDB.
Uses ChromaDB's built-in ONNX embedding function (no PyTorch required).
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = os.environ.get("data_dir", str(BASE_DIR / "data"))
CHROMA_DB_PATH  = str(BASE_DIR / os.environ.get("CHROMA_DB_PATH", "vector_store/chroma_db"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "dip_diet_knowledge"


# ── Document loader ────────────────────────────────────────────────────────
def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all .txt documents from the data directory."""
    docs = []
    for txt_file in Path(data_dir).glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        print(f"  Loaded: {txt_file.name} ({len(text)} chars)")
        docs.append({"text": text, "source": txt_file.name})
    return docs


# ── Chunker ────────────────────────────────────────────────────────────────
def chunk_documents(docs: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split text into overlapping chunks by sentence-aware boundaries."""
    chunks = []
    for doc in docs:
        paragraphs = doc["text"].split("\n\n")
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append({"content": current, "source": doc["source"]})
                # overlap: keep last `overlap` chars
                current = (current[-overlap:] + "\n\n" + para).strip() if overlap else para.strip()
        if current:
            chunks.append({"content": current, "source": doc["source"]})
    return chunks


# ── Main ingestion ─────────────────────────────────────────────────────────
def ingest():
    """Main ingestion function: load docs → chunk → embed → store in ChromaDB."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb")
        return

    print("\n========================================")
    print("  DIP Diet RAG - Vector Store Builder")
    print("========================================\n")

    # Step 1: Load
    print("Step 1: Loading documents...")
    docs = load_documents()
    if not docs:
        print(f"ERROR: No .txt files found in {DATA_DIR}")
        return
    print(f"  Total documents loaded: {len(docs)}")

    # Step 2: Chunk
    print("Step 2: Chunking documents...")
    chunks = chunk_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk["id"] = f"{chunk['source']}__chunk_{i}"
    print(f"  Total chunks created: {len(chunks)}")

    # Step 3: ChromaDB setup
    print("Step 3: Setting up ChromaDB...")
    try:
        embed_fn = ONNXMiniLM_L6_V2()
        print("  Embedding model: ONNX MiniLM-L6-V2 (built-in)")
    except Exception:
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
            print(f"  Embedding model: SentenceTransformer {EMBEDDING_MODEL}")
        except Exception as e:
            embed_fn = None
            print(f"  Embedding model: default (ONNX unavailable: {e})")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection if present
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    kwargs = {"name": COLLECTION_NAME, "metadata": {"hnsw:space": "cosine"}}
    if embed_fn:
        kwargs["embedding_function"] = embed_fn
    collection = client.get_or_create_collection(**kwargs)
    print(f"  Created collection: {COLLECTION_NAME}")

    # Step 4: Embed & store
    print("Step 4: Embedding and storing chunks...")
    batch_size = 50
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[{"source": c["source"], "content": c["content"]} for c in batch],
        )
        end = min(start + batch_size, len(chunks))
        print(f"  Stored chunks {start} - {end} / {len(chunks)}")

    print(f"\n[OK] Vector store built successfully!")
    print(f"   Location : {CHROMA_DB_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Total chunks: {len(chunks)}")
    print("\nReady for RAG queries.\n")


# ── Retriever factory ──────────────────────────────────────────────────────
def get_retriever(top_k: int = 5):
    """Return a callable retriever for the ChromaDB collection."""
    import chromadb
    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        embed_fn = ONNXMiniLM_L6_V2()
    except Exception:
        embed_fn = None

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    kwargs = {"name": COLLECTION_NAME}
    if embed_fn:
        kwargs["embedding_function"] = embed_fn
    collection = client.get_or_create_collection(**kwargs)

    def retriever(query: str) -> list[dict]:
        results = collection.query(query_texts=[query], n_results=top_k)
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "distance": dist,
            })
        return chunks

    return retriever


if __name__ == "__main__":
    ingest()
