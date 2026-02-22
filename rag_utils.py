import os
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
from logger import logger

class RAGManager:
    """Centralized management for RAG operations: Embeddings, Vectorstores, and Splitters."""
    
    _embeddings = None

    @classmethod
    def get_embeddings(cls):
        if cls._embeddings is None:
            cls._embeddings = FastEmbedEmbeddings(
                model_name=Config.EMBED_MODEL,
                max_length=512,
                threads=2
            )
            logger.info(f"Initialized shared FastEmbedEmbeddings ({Config.EMBED_MODEL})")
        return cls._embeddings

    @staticmethod
    def get_vectorstore(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=RAGManager.get_embeddings()
        )

    @staticmethod
    def get_splitter(chunk_size=800, chunk_overlap=80):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
    @staticmethod
    def get_retriever(persist_directory, k=4):
        if not os.path.exists(persist_directory):
            logger.warning(f"Database directory {persist_directory} does not exist.")
            return None
        vs = RAGManager.get_vectorstore(persist_directory)
        return vs.as_retriever(search_kwargs={"k": k})
