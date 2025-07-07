import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def setup_vector_store(docs_dir="data/email_docs", index_dir="data/chroma_db"):
    """Set up the Chroma vector store from email documents"""
    print(f"Setting up vector store from documents in {docs_dir}...")
    
    # Check if directory exists
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Documents directory not found at {docs_dir}")
    
    # Create output directory
    os.makedirs(index_dir, exist_ok=True)
    
    # Load documents
    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    print(f"Split into {len(texts)} text chunks")
    
    # Create embeddings and vector store
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Building vector store...")
    vector_store = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory=index_dir
    )
    
    # Persist the vector store
    vector_store.persist()
    
    print(f"Vector store saved to {index_dir}")
    return vector_store

def load_vector_store(index_dir="data/chroma_db"):
    """Load the Chroma vector store"""
    print(f"Loading vector store from {index_dir}...")
    
    # Check if directory exists
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Chroma index not found at {index_dir}. Run setup_vector_store first.")
    
    # Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=index_dir, embedding_function=embeddings)
    
    print("Vector store loaded successfully!")
    return vector_store

if __name__ == "__main__":
    # This allows running this file directly to set up the vector store
    try:
        setup_vector_store()
        print("\nVector store setup complete. You can now run the app.")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run email_reader.py first to extract and process emails.")