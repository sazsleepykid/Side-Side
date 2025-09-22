import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader

import torch
from transformers import AutoTokenizer, AutoModel

# === GPU Embedding Wrapper ===
class GPUEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            return embeddings.cpu().tolist()

# === Vector Store Setup ===
def setup_vector_store(docs_dir="data/email_docs", index_dir="data/chroma_db", silent_mode=False, progress_callback=None):
    if not silent_mode:
        print(f"[INFO] Setting up vector store from: {docs_dir}")
    
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"[ERROR] Documents directory not found: {docs_dir}")
    
    os.makedirs(index_dir, exist_ok=True)

    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)

    try:
        documents = loader.load()
        if not documents:
            raise ValueError("No documents found to embed.")

        if not silent_mode:
            print(f"[INFO] Loaded {len(documents)} documents.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        if not silent_mode:
            print(f"[INFO] Split into {len(texts)} text chunks.")

        if progress_callback:
            progress_callback(0.3, "Creating embeddings...")

        # 🔥 Use GPU-based embeddings
        embeddings = GPUEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

        if progress_callback:
            progress_callback(0.6, "Building vector store...")

        if not silent_mode:
            print("[INFO] Creating vector store...")

        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=index_dir
        )

        try:
            vector_store.persist()
        except Exception as e:
            if not silent_mode:
                print(f"[WARN] Persist error: {e} — may already be auto-saved.")

        if progress_callback:
            progress_callback(1.0, "Vector store ready!")

        if not silent_mode:
            print(f"[INFO] Vector store saved to {index_dir}")
        
        return vector_store

    except Exception as e:
        if not silent_mode:
            print(f"[ERROR] Failed to build vector store: {e}")
        raise

def load_vector_store(index_dir="data/chroma_db", silent_mode=False):
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Chroma DB not found: {index_dir}. Run `setup_vector_store()` first.")

    try:
        embeddings = GPUEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=index_dir, embedding_function=embeddings)

        if not silent_mode:
            print("[INFO] Vector store loaded from disk.")
        
        return vector_store

    except Exception as e:
        if not silent_mode:
            print(f"[ERROR] Failed to load vector store: {e}")
        raise

def check_vector_store_exists(index_dir="data/chroma_db"):
    if not os.path.exists(index_dir):
        return False

    required_files = ["chroma.sqlite3"]
    return all(os.path.exists(os.path.join(index_dir, f)) for f in required_files)

if __name__ == "__main__":
    try:
        setup_vector_store()
        print("\n✅ Vector store setup complete. Launch your app next!")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("👉 Run `email_reader.py` first to extract emails.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


from db import get_connection

def link_embedding(email_id, chunk_id, vector_store_id="default"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO embeddings (email_id, chunk_id, vector_store_id)
        VALUES (?, ?, ?)
    """, (email_id, chunk_id, vector_store_id))
    conn.commit()
    conn.close()
