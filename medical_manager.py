"""
medical_manager.py
------------------
Task 3: Build and manage the MedQuAD vector database.

Run this ONCE before starting the app to index the MedQuAD XML files:
    python medical_manager.py

Features:
  - Parallel XML parsing (uses all CPU cores)
  - Batched Chroma inserts (avoids memory spikes)
  - Incremental indexing (skips already-indexed files on re-runs)
  - Separate indexed log from the main app's indexed_files.json
"""

import os
import json
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import Config
from logger import logger
from rag_utils import RAGManager

# -------------------- PATH CONFIG --------------------
MEDQUAD_PATH         = Config.MEDQUAD_PATH
MEDICAL_DB_PATH      = Config.MEDICAL_DB_PATH
MEDICAL_INDEXED_LOG  = "medical_indexed_files.json"
BATCH_SIZE           = 100

# -------------------- EMBEDDINGS (singleton) --------------------
# Using BAAI/bge-small-en-v1.5 — only ~130MB RAM, much lighter than nomic-embed-text-v1.5 (~12GB)
# -------------------- EMBEDDINGS (shared) --------------------
embeddings = RAGManager.get_embeddings()


# -------------------- INDEXED FILE TRACKING --------------------

def get_indexed_files() -> set:
    if os.path.exists(MEDICAL_INDEXED_LOG):
        with open(MEDICAL_INDEXED_LOG, "r") as f:
            return set(json.load(f))
    return set()


def save_indexed_files(files: set) -> None:
    with open(MEDICAL_INDEXED_LOG, "w") as f:
        json.dump(sorted(files), f, indent=2)


# -------------------- XML PARSER --------------------

def parse_medquad_xml(file_path: str) -> list:
    """Parse a single MedQuAD XML file into a list of LangChain Documents."""
    documents = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[Warning] Error parsing {file_path}: {e}")
        return documents

    source_name = os.path.basename(file_path)

    for qa in root.findall(".//QAPair"):
        question = qa.findtext("Question", "").strip()
        answer   = qa.findtext("Answer",   "").strip()
        qtype    = qa.findtext("QuestionType", "").strip()
        focus    = qa.findtext("Focus",    "").strip()

        if not question or not answer:
            continue

        content = (
            f"Medical Question: {question}\n\n"
            f"Medical Answer: {answer}\n\n"
            f"Question Type: {qtype}\n"
            f"Focus: {focus}"
        )

        documents.append(Document(
            page_content=content,
            metadata={
                "source":        source_name,
                "question_type": qtype,
                "focus":         focus,
            }
        ))

    return documents


# -------------------- BATCH INDEXING --------------------

def add_in_batches(vectorstore: Chroma, documents: list) -> None:
    total = len(documents)
    for start in range(0, total, BATCH_SIZE):
        batch = documents[start: start + BATCH_SIZE]
        vectorstore.add_documents(batch)
        done = min(start + BATCH_SIZE, total)
        print(f"  Indexed {done}/{total} documents ...", end="\r")
    print()


# -------------------- BUILD MEDICAL DB --------------------

def build_medical_database() -> None:
    if not os.path.exists(MEDQUAD_PATH):
        print(f"[Error] MedQuAD folder not found at: {MEDQUAD_PATH}")
        print("        Download from: https://github.com/abachaa/MedQuAD")
        return

    # 1. Collect XML files
    all_xml_files = [
        os.path.join(root_dir, fname)
        for root_dir, _, files in os.walk(MEDQUAD_PATH)
        for fname in files
        if fname.endswith(".xml")
    ]

    if not all_xml_files:
        print("[Error] No XML files found in the MedQuAD folder.")
        return

    # 2. Skip already-indexed files
    indexed_files = get_indexed_files()
    new_xml_files = [f for f in all_xml_files if f not in indexed_files]

    print(f"[Info] Total XML files found : {len(all_xml_files)}")
    print(f"[Info] Already indexed       : {len(indexed_files)}")
    print(f"[Info] New files to process  : {len(new_xml_files)}")

    if not new_xml_files:
        print("[Info] Nothing new to index. Medical database is up to date.")
        return

    # 3. Parse in parallel
    num_workers = min(cpu_count(), len(new_xml_files))
    print(f"[Info] Parsing using {num_workers} CPU workers ...")

    with Pool(processes=num_workers) as pool:
        results = pool.map(parse_medquad_xml, new_xml_files)

    all_documents = [doc for docs in results for doc in docs]

    if not all_documents:
        print("[Warning] No valid Q&A pairs found.")
        return

    print(f"[Info] Total Q&A pairs parsed : {len(all_documents)}")

    # 4. Save to Chroma
    vectorstore = RAGManager.get_vectorstore(MEDICAL_DB_PATH)

    print(f"[Info] Indexing into Chroma (batch_size={BATCH_SIZE}) ...")
    add_in_batches(vectorstore, all_documents)

    # 5. Update index log
    save_indexed_files(indexed_files | set(new_xml_files))

    print(f"\n[Done] Medical database updated.")
    print(f"       Q&A pairs indexed : {len(all_documents)}")
    print(f"       DB path           : {MEDICAL_DB_PATH}")


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    build_medical_database()