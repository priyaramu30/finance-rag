import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BASE_PERSIST_DIR = "./chroma_db"

def get_persist_directory(provider: str = "openai") -> str:
    """Returns separate vector store directories for different providers."""
    return f"{BASE_PERSIST_DIR}_{provider.lower()}"

def get_embedding_function(provider: str = "openai"):
    """Dynamically returns the selected embedding function."""
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",  # Active Gemini multimodal/text embedding endpoint
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

def process_and_index_pdfs(pdf_paths: list[str], provider: str = "openai"):
    """Processes uploaded PDFs, splits text into chunks, and appends them to ChromaDB."""
    all_docs = []
    
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        filename = os.path.basename(path)
        
        for doc in docs:
            doc.metadata["file_name"] = filename
            doc.metadata["page_label"] = doc.metadata.get("page", 0) + 1
            
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)

    embeddings = get_embedding_function(provider)
    persist_dir = get_persist_directory(provider)

    # Automatically connects to existing collection or creates new one automatically
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Append new document chunks automatically
    vector_store.add_documents(documents=chunks)
    
    return len(pdf_paths), len(chunks)

def get_vector_store(provider: str = "openai"):
    """Loads the existing persisted vector database."""
    embeddings = get_embedding_function(provider)
    persist_dir = get_persist_directory(provider)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
