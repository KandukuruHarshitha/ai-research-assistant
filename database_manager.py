import os
import json
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import Config
from logger import logger

# -------------------- CONFIG --------------------
CHROMA_PATH = Config.CHROMA_PATH
DATA_PATH = Config.DATA_PATH
INDEXED_FILES_LOG = "indexed_files.json"

# -------------------- EMBEDDINGS (shared) --------------------
embeddings = RAGManager.get_embeddings()

# -------------------- FILE TRACKING --------------------
def get_indexed_files() -> set:
    if os.path.exists(INDEXED_FILES_LOG):
        with open(INDEXED_FILES_LOG) as f:
            return set(json.load(f))
    return set()

def save_indexed_files(files: set) -> None:
    with open(INDEXED_FILES_LOG, "w") as f:
        json.dump(sorted(files), f, indent=2)

# -------------------- LOADERS --------------------
def load_documents() -> list:
    """Load all PDFs from DATA_PATH, skipping already-indexed ones."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created '{DATA_PATH}'. Please add PDF files there.")
        return []

    all_pdfs = [
        os.path.join(DATA_PATH, f)
        for f in os.listdir(DATA_PATH)
        if f.endswith(".pdf")
    ]

    if not all_pdfs:
        print("No PDF files found in data directory.")
        return []

    indexed = get_indexed_files()
    new_pdfs = [f for f in all_pdfs if f not in indexed]

    if not new_pdfs:
        print("All files already indexed. Nothing new to process.")
        return []

    print(f"Found {len(new_pdfs)} new file(s) to index.")
    documents = []
    successfully_loaded = []

    for pdf_path in new_pdfs:
        try:
            docs = PyPDFLoader(pdf_path).load()
            if docs:
                documents.extend(docs)
                successfully_loaded.append(pdf_path)
            else:
                print(f"[Warning] No content extracted from: {pdf_path}")
        except Exception as e:
            print(f"[Error] Failed to load {pdf_path}: {e}")

    # Only mark files as indexed after successful load
    save_indexed_files(indexed | set(successfully_loaded))
    return documents

def load_single_document(file_path: str) -> list:
    """Load a single PDF."""
    try:
        docs = PyPDFLoader(file_path).load()
        if not docs:
            print(f"[Warning] No content extracted from: {file_path}")
        return docs
    except Exception as e:
        print(f"[Error] Failed to load {file_path}: {e}")
        return []

# -------------------- SPLITTER --------------------
def split_text(documents: list) -> list:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document pages into {len(chunks)} chunks.")
    return chunks

# -------------------- CHROMA --------------------
def get_vectorstore() -> Chroma:
    """Return existing or new Chroma vectorstore via RAGManager."""
    return RAGManager.get_vectorstore(CHROMA_PATH)

def save_to_chroma(chunks: list, reset: bool = False) -> None:
    """Save chunks to ChromaDB, with optional reset."""
    if not chunks:
        print("No chunks to save.")
        return

    if reset and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared existing database.")

    db = get_vectorstore()
    db.add_documents(chunks)
    print(f"Saved {len(chunks)} chunks to '{CHROMA_PATH}'.")

# -------------------- PROCESS SINGLE FILE --------------------
def process_file(file_path: str) -> None:
    """Process and index a single PDF file."""
    indexed = get_indexed_files()
    if file_path in indexed:
        print(f"[Skip] Already indexed: {file_path}")
        return

    print(f"Processing: {file_path}")
    documents = load_single_document(file_path)
    if not documents:
        return

    chunks = split_text(documents)
    save_to_chroma(chunks, reset=False)
    save_indexed_files(indexed | {file_path})
    print(f"Successfully added '{file_path}' to the database.")

# -------------------- MAIN --------------------
def main():
    documents = load_documents()
    if not documents:
        return
    chunks = split_text(documents)
    save_to_chroma(chunks, reset=False)

if __name__ == "__main__":
    main()