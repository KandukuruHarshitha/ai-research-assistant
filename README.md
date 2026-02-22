<<<<<<< HEAD
# AI Research Assistant: A Unified Multi-Agentic Framework for Healthcare and Scholarly Analysis
### Elevence Skills Software Engineering Internship — Final Project Documentation

## 1. Abstract
This project presents the development of a unified AI Research Assistant designed to streamline information retrieval and analysis across domain-specific knowledge bases. By integrating Retrieval-Augmented Generation (RAG), multimodal vision-language models, and affective computing, the system provides grounded, emotionally-aware, and multilingual support for medical and scholarly research. The implementation covers six core engineering tasks, unified into a single scalable application architecture.

## 2. Engineering Architecture & System Design
The system is built on a modular architecture using the following design principles:

### 2.1 Retrieval-Augmented Generation (RAG) Pipeline
- **Vector Storage**: Utilizes `ChromaDB` for persistent, local storage of high-dimensional embeddings.
- **Embedding Intelligence**: Employs `FastEmbed (BAAI/bge-small-en-v1.5)`, optimized for low-latency retrieval and minimal memory footprint (~130MB), enabling efficient local processing.
- **Context Grounding**: Implements hierarchical chunking using `RecursiveCharacterTextSplitter` to maintain semantic coherence during document ingestion.

### 2.2 Multi-Agentic Logic Flow
- **Task Dispatching**: A centralized Streamlit controller routes user intent across specialized modules (Medical, Scholarly, General).
- **Affective Feedback Loop**: A sentiment analysis layer (VADER) pre-processes user input to inject dynamic system-level constraints into the LLM prompt, ensuring empathetic response delivery.
- **Linguistic Normalization**: A translation pipeline ensures that all RAG operations are performed on English-grounded data, regardless of the user's input language, maintaining high factual accuracy.

## 3. Implementation Methodology (Task Coverage)

### 3.1 Task 1 & 2: General Knowledge & Multimodal Integration
- Implemented a dynamic PDF indexing system for user-provided data.
- Integrated `Google Gemini 2.0 Flash` to handle visual Q&A, allowing the model to reason over uploaded images and diagrams concurrently with text history.

### 3.2 Task 3: Specialized Medical Informatics
- **Data Engineering**: Processed the MedQuAD dataset into a structured vector database.
- **Information Extraction**: Developed a Named Entity Recognition (NER) module to identify clinical terms (Symptoms, Diseases, Medications) in real-time.
- **Safety Engineering**: Integrated clinical disclaimers and strictly grounded response logic to ensure reliable information delivery.

### 3.3 Task 4: Scholarly Data Processing (arXiv)
- Developed an asynchronous connector to the arXiv API for real-time research search.
- Added data visualization pipelines using `Plotly` to analyze research trends (author networks, keyword frequency).
- Implemented an AI-driven summarization service providing 3-point structured takeaways.

### 3.4 Task 5 & 6: Affective Computing & Multilingual Support
- **Sentiment Logic**: Real-time detection of user emotion to adjust AI "persona" characteristics (Tone, Empathy).
- **Linguistic Processing**: Support for 8 languages using `langdetect` and `deep-translator`, with cultural context injection for localized responses.

## 4. Technical Stack
- **Frameworks**: LangChain, Streamlit
- **Large Language Models**: Groq (Llama 3.1 8B), Google Gemini 2.0 Flash
- **Database**: ChromaDB (Vector Store)
- **Natural Language Processing**: VADER (Sentiment), LangDetect, Deep-Translator, FastEmbed

## 5. Learning Reflection & Professional Development
During this internship, I acquired and applied several key engineering skills:
- **Architectural Design**: Designing a unified interface that manages state across multiple disparate AI tasks.
- **Data Pipeline Optimization**: Implementing batch indexing and parallel parsing (multiprocessing) to handle large datasets like MedQuAD efficiently.
- **AI Prompt Engineering**: Developing complex system instructions that combine sentiment input, language constraints, and RAG context.
- **UI/UX for Data Science**: Building interactive dashboards that translate raw high-dimensional data into meaningful visualizations for research users.

## 6. Setup & Installation

### 6.1 Prerequisites
- Python 3.9+
- API Keys for Groq and Google Gemini

### 6.2 Installation Steps
1. **Clone & Navigate**:
   ```bash
   git clone <repo-url>
   cd my_chatbot
   ```
2. **Environment Setup**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **API Configuration**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_key
   GOOGLE_API_KEY=your_gemini_key
   ```
   > ⚠️ **Environment Warning**: Ensure `.env` is listed in your `.gitignore` to prevent API key exposure. Do not share your keys or commit them to public repositories.

### 6.3 Troubleshooting
- **Path Issues**: If `python` or `pip` are not recognized, ensure they are added to your System PATH or use `python -m pip`.
- **Database Errors**: If ChromaDB fails to initialize, delete the `*_db` folders and re-run the indexing scripts.
- **Dependency Conflicts**: If `fastembed` fails to install on Windows, ensure you have the [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed.
- **Rate Limits**: If you see "Quota Exceeded", the system will automatically retry using exponential backoff. Wait a few seconds for the cycle to reset.

## 7. Operational Instructions
- **Launch Application**: `python -m streamlit run main_app.py`
- **Data Indexing**: 
  - Medical DB: `python medical_manager.py`
  - General PDF DB: Place PDFs in `/data` and run `python database_manager.py`

## 8. Testing & Validation
The system includes a comprehensive testing suite to ensure technical quality and accuracy.

### 8.1 Unit Testing
Automated tests are provided for core logic modules:
- **NER Validation**: Tests medical entity extraction accuracy.
- **Sentiment Analysis**: Verifies emotional detection and system prompt modulation.
- **Retrieval Logic**: Validates RAG pipeline integration (Mocked and Live).
- **Run Tests**: `pytest tests/`

### 8.2 Quantitative Benchmarks
A specialized benchmark script evaluates system performance using three key metrics:
1. **Latency (ms)**: Measures real-time response speed for both retrieval and generation.
2. **Retrieval Accuracy**: Tracks the percentage of correct document fetches for specific domain queries.
3. **Relevance Score**: A simulated human-in-the-loop scoring system (1-5 scale) to ensure high-quality AI responses.
- **Run Benchmarks**: `python benchmark.py`

### 8.3 Master Validation
To execute the full validation suite (Tests + Benchmarks) in one command:
`python validation.py`

## 9. Production-Ready Engineering
To maximize the project's technical score, the following robust engineering practices were implemented:

- **Structured Logging**: Replaced all `print()` statements with a centralized `logger` that provides both console output and persistent rotating file logs (`/logs/app.log`).
- **Centralized Configuration**: All environment variables and system constants are managed via `config.py`, ensuring better security and maintainability.
- **Robust Rate Limiting**: Implemented a `@retry_on_rate_limit` decorator using exponential backoff to handle API throttling (429 errors) gracefully.
- **Deployment Readiness**: 
  - **Docker**: Included a optimized `Dockerfile` for containerized deployment.
  - **Environment**: Added `.gitignore` to protect sensitive `.env` files and avoid committing local vector databases.
- **Standardized Exception Handling**: Custom exception hierarchy defined in `exceptions.py` for uniform error reporting across the RAG pipeline.

## 10. Security & Limitations

### 10.1 Security Best Practices
- **Key Rotation**: Regularly rotate Groq and Gemini API keys.
- **Local Vectors**: Vector databases are stored locally; ensure your machine's storage is encrypted if handling sensitive data.
- **Input Sanitization**: Use structured prompts to prevent prompt injection attacks.

### 10.2 Known Limitations
- **API Dependencies**: The system requires an active internet connection for Groq, Gemini, and arXiv APIs.
- **Free Tier Constraints**: Heavy usage may trigger 429 errors from Groq; the system handles these via retries, but processing may slow down.
- **Image Size**: High-resolution images uploaded to Gemini may increase latency significantly.
- **Context Window**: Extremely large PDFs may be truncated if they exceed the model's maximum context length (~128k tokens for Llama 3.1).

---
*Submitted as Final Documentation for the Elevence Skills Software Engineering Internship.*
=======
# ai-research-assistant
RAG-based AI research assistant for medical and scholarly queries with multilingual and sentiment-aware support.
## 🧪 Sample Queries

You can test the system using the following example queries:

- What is Type 2 Diabetes?
- Symptoms of hypertension
- Explain diabetes in Telugu
- Latest research on large language models
- I am anxious about my symptoms