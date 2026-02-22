import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

# Mocking the retriever to test general logic without needing a live ChromaDB
def test_retrieval_logic_mock():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="The liver is a vital organ.", metadata={"source": "med.pdf"}),
        Document(page_content="Metformin is used for diabetes.", metadata={"source": "med.pdf"})
    ]
    
    query = "What is metformin used for?"
    docs = mock_retriever.invoke(query)
    
    assert len(docs) == 2
    assert "Metformin" in docs[1].page_content
    assert docs[0].metadata["source"] == "med.pdf"

@pytest.mark.skipif(not __import__('os').path.exists("medical_chroma_db"), reason="Medical DB not built")
def test_real_medical_retrieval():
    # Only runs if the user has built the database
    from rag_utils import RAGManager
    from config import Config
    
    retriever = RAGManager.get_retriever(Config.MEDICAL_DB_PATH, k=1)
    
    results = retriever.invoke("What is diabetes?")
    assert len(results) > 0
    assert "diabetes" in results[0].page_content.lower()
