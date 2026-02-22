# AI Research Assistant

RAG-based AI research assistant for medical and scholarly queries with multilingual and sentiment-aware support.

---

## 📌 Overview

AI Research Assistant is a modular Retrieval-Augmented Generation (RAG) system designed to provide:

- 🏥 Medical knowledge support (MedQuAD dataset)
- 📚 Scholarly research integration (arXiv API)
- 🌍 Multilingual query handling
- 💬 Sentiment-aware response modulation
- 📊 Built-in evaluation benchmarks

---

## ✨ Key Features

- 🔍 Retrieval-Augmented Generation (RAG)
- 🧠 Named Entity Recognition (Medical terms)
- 📚 arXiv Research Search & Summarization
- 🌍 Multilingual Query Support (8 languages)
- 💬 Sentiment-Aware Response Adaptation
- 📊 Quantitative Benchmark Evaluation
- 🐳 Docker Deployment Support

---

## 🧪 Sample Queries

You can test the system using the following example queries:

- What is Type 2 Diabetes?
- Symptoms of hypertension
- Explain diabetes in Telugu
- Latest research on large language models
- I am anxious about my symptoms

---

## 🛠 Tech Stack

- **Framework:** Streamlit, LangChain
- **LLM:** Groq (Llama 3.1 8B), Google Gemini
- **Vector Database:** ChromaDB
- **Embeddings:** FastEmbed (BAAI/bge-small-en-v1.5)
- **NLP Tools:** VADER, LangDetect, Deep-Translator
- **Visualization:** Plotly

---

## 📂 Project Structure

```

main_app.py
database_manager.py
medical_manager.py
arxiv_chatbot.py
benchmark.py
validation.py
config.py
exceptions.py
tests/
Dockerfile
.env.example

```

---

## ⚙ Setup & Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/KandukuruHarshitha/ai-research-assistant.git
cd ai-research-assistant
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
MODEL_NAME=llama-3.1-8b-instant
```

⚠️ Never commit `.env`. It is excluded via `.gitignore`.

---

## ▶ Run the Application

```bash
streamlit run main_app.py
```

Open in browser:

[http://localhost:8501](http://localhost:8501)

---

## 🧪 Run Tests

```bash
python -m pytest tests/
```

---

## 📊 Run Benchmark

```bash
python benchmark.py
```

---

## 📊 Benchmark Summary

- **Average Latency:** ~0.6–1.7 seconds
- **Retrieval Accuracy:** ~66–80%
- **Response Relevance Score:** ~4.2/5

These metrics evaluate system responsiveness, retrieval quality, and answer relevance.

---

## 🧪 Run Full Validation

```bash
python validation.py
```

---

## 📁 Datasets

Datasets used in this project are available here:

https://drive.google.com/drive/folders/10AseFraWDnd2sPsHkBg5arbufWq-osp0?usp=drive_link

Vector databases are generated locally and are not included in this repository.

---

## 🔐 Security Practices

- API keys stored via environment variables
- `.env` excluded using `.gitignore`
- Centralized configuration management
- Structured logging
- Rate-limit handling with exponential backoff

---

## ⚠ Known Limitations

- Requires internet connection for Groq, Gemini, and arXiv APIs
- Free-tier API limits may affect performance
- Large PDFs may exceed model context limits
- Benchmark results are validation-level, not research-grade

---

## 🐳 Docker Deployment (Optional)

### Build Image

```bash
docker build -t ai-research-assistant .
```

### Run Container

```bash
docker run -p 8501:8501 --env-file .env ai-research-assistant
```

---

## 🎓 Internship Submission

Submitted as part of the Elevence Skills Software Engineering Internship.

````

