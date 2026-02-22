import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))
    MEDICAL_DB_PATH = os.getenv("MEDICAL_DB_PATH", str(BASE_DIR / "medical_chroma_db"))
    ARXIV_DB_PATH = os.getenv("ARXIV_DB_PATH", str(BASE_DIR / "arxiv_chroma_db"))
    DATA_PATH = os.getenv("DATA_PATH", str(BASE_DIR / "data"))
    MEDQUAD_PATH = os.getenv("MEDQUAD_PATH", str(BASE_DIR / "data" / "medquad"))
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Models
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    
    # App Settings
    APP_NAME = "AI Research Assistant"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # RAG Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    
    # Rate Limiting
    GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
    GROQ_RETRY_DELAY = int(os.getenv("GROQ_RETRY_DELAY", "5")) # seconds
