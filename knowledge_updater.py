import os
import json
import pandas as pd

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader

CHROMA_PATH = "chroma_db"
DATA_PATH = "data"   # make sure your files are here
TRACK_FILE = "indexed_files.json"

embeddings = OllamaEmbeddings(model="nomic-embed-text")


def load_indexed_files():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_indexed_files(files):
    with open(TRACK_FILE, "w") as f:
        json.dump(list(files), f)


def update_knowledge_base():
    indexed_files = load_indexed_files()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    new_documents = []
    new_files = []

    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)

        if file in indexed_files:
            continue

        # TXT files
        if file.endswith(".txt"):
            loader = TextLoader(file_path)
            new_documents.extend(loader.load())
            new_files.append(file)

        # CSV files (robust encoding handling)
        elif file.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin-1")

            for _, row in df.iterrows():
                content = " | ".join(str(v) for v in row.values)
                new_documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": file}
                    )
                )

            new_files.append(file)

    if not new_documents:
        print("No new data found.")
        return

    # 🔴 THIS WAS MISSING
    vectorstore.add_documents(new_documents)

    indexed_files.update(new_files)
    save_indexed_files(indexed_files)

    print(f"Added {len(new_files)} new files to the knowledge base.")


if __name__ == "__main__":
    update_knowledge_base()
