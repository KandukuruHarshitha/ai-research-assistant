"""
main_app.py
-----------
Internship Project — Elevence Skills
All tasks unified in one application.

Task 1 → Dynamic knowledge base (PDF RAG)
Task 2 → Multi-modal chatbot (Gemini image understanding)
Task 3 → Medical Q&A (MedQuAD + RAG + NER)
Task 4 → arXiv Research Chatbot (search, RAG, summarize, visualize)
Task 5 → Sentiment-aware responses (VADER sentiment + emotionally adaptive prompts)
Task 6 → Multilingual support (auto-detect + translate, 8 languages, cultural context)

Run:
    streamlit run main_app.py
"""

# ──────────────────────────────────────────────────────────────
# SYSTEM INITIALIZATION
# ──────────────────────────────────────────────────────────────
from config import Config
from logger import logger
from exceptions import retry_on_rate_limit, AIAssistantError
from rag_utils import RAGManager

# ──────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────
import re
import time
import textwrap
from collections import Counter

import arxiv
import plotly.express as px
import streamlit as st
from PIL import Image

from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gemini_client import gemini_chat
from medical_ner import extract_medical_entities
from sentiment_analyzer import analyze_sentiment
from language_handler import (
    detect_language, translate_to_english, translate_from_english,
    get_language_options, SUPPORTED_LANGUAGES,
)

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant — Elevence Skills",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(160deg, #060612 0%, #0a0a20 40%, #080d1a 100%);
    min-height: 100vh;
}

/* ── Animated gradient header ── */
.main-title {
    background: linear-gradient(90deg, #4facfe 0%, #a78bfa 40%, #f472b6 70%, #4facfe 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}
@keyframes shimmer { to { background-position: 200% center; } }

.sub-title {
    color: #64748b;
    font-size: 0.88rem;
    font-weight: 400;
    letter-spacing: 0.4px;
    margin-bottom: 24px;
}

/* ── Glass cards ── */
.card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(79,172,254,0.15);
    border-radius: 14px;
    padding: 18px 22px;
    margin: 10px 0;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,172,254,0.4), transparent);
}
.card:hover {
    background: rgba(255,255,255,0.055);
    border-color: rgba(79,172,254,0.4);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(79,172,254,0.08);
}
.card-title {
    color: #7dd3fc;
    font-size: 0.97rem;
    font-weight: 600;
    margin-bottom: 6px;
    line-height: 1.4;
}
.card-meta  { color: #64748b; font-size: 0.76rem; margin-bottom: 8px; }
.card-text  { color: #94a3b8; font-size: 0.84rem; line-height: 1.65; }

/* ── Tags / pills ── */
.tag {
    display: inline-block;
    background: rgba(79,172,254,0.1);
    border: 1px solid rgba(79,172,254,0.2);
    color: #7dd3fc;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.70rem;
    font-weight: 500;
    margin: 2px;
    letter-spacing: 0.2px;
}

/* ── Stat boxes ── */
.stat-box {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(79,172,254,0.12);
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
    transition: all 0.25s ease;
}
.stat-box:hover {
    background: rgba(255,255,255,0.055);
    border-color: rgba(79,172,254,0.25);
}
.stat-num   { font-size: 1.9rem; font-weight: 700; color: #7dd3fc; line-height: 1; }
.stat-label { font-size: 0.74rem; color: #64748b; margin-top: 4px; font-weight: 500; letter-spacing: 0.3px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(6,6,18,0.97) !important;
    border-right: 1px solid rgba(79,172,254,0.08) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.025);
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid rgba(79,172,254,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    font-size: 0.84rem;
    font-weight: 500;
    padding: 8px 16px;
    color: #64748b;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(79,172,254,0.2), rgba(167,139,250,0.2)) !important;
    color: #7dd3fc !important;
    border: 1px solid rgba(79,172,254,0.25) !important;
}

/* ── Chat messages ── */
.stChatMessage { background: transparent !important; }
[data-testid="stChatMessageContent"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(79,172,254,0.1) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    backdrop-filter: blur(8px);
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(79,172,254,0.2) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(79,172,254,0.5) !important;
    box-shadow: 0 0 0 3px rgba(79,172,254,0.08) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    transition: all 0.25s ease !important;
    border: 1px solid rgba(79,172,254,0.25) !important;
    background: rgba(79,172,254,0.08) !important;
    color: #7dd3fc !important;
}
.stButton > button:hover {
    background: rgba(79,172,254,0.18) !important;
    border-color: rgba(79,172,254,0.5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(79,172,254,0.15) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4facfe, #a78bfa) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
    box-shadow: 0 6px 24px rgba(79,172,254,0.25) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(79,172,254,0.15) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Warning / info boxes ── */
.stWarning {
    background: rgba(251,191,36,0.06) !important;
    border: 1px solid rgba(251,191,36,0.2) !important;
    border-radius: 10px !important;
}
.stInfo {
    background: rgba(79,172,254,0.06) !important;
    border: 1px solid rgba(79,172,254,0.2) !important;
    border-radius: 10px !important;
}

/* ── Dividers ── */
hr { border-color: rgba(79,172,254,0.08) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(79,172,254,0.2);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(79,172,254,0.4); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
CHROMA_PATH      = Config.CHROMA_PATH
MEDICAL_DB_PATH  = Config.MEDICAL_DB_PATH
ARXIV_DB_PATH    = Config.ARXIV_DB_PATH
LLM_MODEL        = Config.LLM_MODEL

CS_CATEGORIES = {
    "All CS"              : "cs.*",
    "AI / Machine Learning": "cs.AI OR cs.LG",
    "Computer Vision"     : "cs.CV",
    "NLP"                 : "cs.CL",
    "Systems"             : "cs.SY",
    "Robotics"            : "cs.RO",
    "Cryptography"        : "cs.CR",
    "Databases"           : "cs.DB",
    "Software Engineering": "cs.SE",
}

STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","that","this",
    "from","by","as","we","our","their","these","which","it","its","they",
    "paper","propose","proposed","model","method","approach","based","results",
    "show","use","used","using","also","can","new","data","learning","neural",
    "deep","network","networks","large","tasks","task","training","trained",
}

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c8d3e0",
    margin=dict(t=40, b=20, l=20, r=20),
)

# ──────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────
for _k, _v in [
    ("messages",          []),
    ("medical_messages",  []),
    ("arxiv_chat",        []),
    ("arxiv_results",     []),
    ("arxiv_indexed",     {}),
    ("arxiv_last_query",  ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ──────────────────────────────────────────────────────────────
# SINGLETONS
# ──────────────────────────────────────────────────────────────
# Using RAGManager for centralized embeddings
embeddings = RAGManager.get_embeddings()

@st.cache_resource
def get_llm(temperature: float = 0.7):
    return ChatGroq(
        model=LLM_MODEL, 
        temperature=temperature,
        api_key=Config.GROQ_API_KEY
    )

# ──────────────────────────────────────────────────────────────
# RAG CHAIN LOADERS
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_general_chain(model_name: str = LLM_MODEL):
    retriever = RAGManager.get_retriever(CHROMA_PATH, k=4)
    if not retriever:
        return None
    llm      = ChatGroq(model=model_name, temperature=0, api_key=Config.GROQ_API_KEY)
    prompt   = PromptTemplate(
        template="""Use the context below to answer the question.
If answer is not found in context, say you don't know.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    return ({"context": retriever, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser())


@st.cache_resource
def load_medical_chain(model_name: str = LLM_MODEL):
    retriever = RAGManager.get_retriever(MEDICAL_DB_PATH, k=5)
    if not retriever:
        return None, None
    llm       = ChatGroq(model=model_name, temperature=0.1, api_key=Config.GROQ_API_KEY)
    prompt    = PromptTemplate(
        template="""You are a medical information assistant.
Use ONLY the provided medical context.

Rules:
- Do not invent treatments or dosages.
- If unsure, say consult a healthcare professional.
- Be clear and simple.

Medical Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    return ({"context": retriever, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()), retriever


def build_arxiv_rag():
    retriever = RAGManager.get_retriever(ARXIV_DB_PATH, k=5)
    if not retriever or not st.session_state.arxiv_indexed:
        return None
    llm       = ChatGroq(model=LLM_MODEL, temperature=0.3, api_key=Config.GROQ_API_KEY)
    prompt    = PromptTemplate(
        template="""You are an expert research assistant specializing in computer science.
Use the provided research paper context to answer accurately and in depth.
Reference specific papers when relevant.

Context from indexed papers:
{context}

Question: {question}

Provide a detailed, insightful answer:""",
        input_variables=["context", "question"]
    )
    return ({"context": retriever, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser())

# ──────────────────────────────────────────────────────────────
# ARXIV HELPERS
# ──────────────────────────────────────────────────────────────
def search_arxiv(query: str, cat_filter: str, max_results: int = 15):
    full_q  = f"({query})" if ".*" in cat_filter else f"({query}) AND cat:{cat_filter}"
    client  = arxiv.Client()
    search  = arxiv.Search(query=full_q, max_results=max_results,
                           sort_by=arxiv.SortCriterion.Relevance)
    papers  = []
    for r in client.results(search):
        papers.append({
            "id"         : r.entry_id.split("/")[-1],
            "title"      : r.title,
            "authors"    : ", ".join(a.name for a in r.authors[:4])
                           + (" et al." if len(r.authors) > 4 else ""),
            "abstract"   : r.summary.replace("\n", " "),
            "year"       : r.published.year,
            "categories" : ", ".join(r.categories),
            "url"        : r.entry_id,
            "pdf_url"    : r.pdf_url,
        })
    return papers


def index_arxiv_papers(papers: list) -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    vs       = Chroma(persist_directory=ARXIV_DB_PATH, embedding_function=embeddings)
    already  = set(st.session_state.arxiv_indexed.keys())
    docs, ids, added = [], [], 0
    for p in papers:
        if p["id"] in already:
            continue
        text   = f"Title: {p['title']}\nAuthors: {p['authors']}\nAbstract: {p['abstract']}"
        chunks = splitter.create_documents(
            [text], metadatas=[{"arxiv_id": p["id"], "title": p["title"],
                                "year": str(p["year"]), "categories": p["categories"]}]
        )
        for i, c in enumerate(chunks):
            docs.append(c); ids.append(f"{p['id']}_c{i}")
        st.session_state.arxiv_indexed[p["id"]] = p
        added += 1
    if docs:
        vs.add_documents(docs, ids=ids)
    return added


def top_words(papers, n=30):
    all_text = " ".join(p["abstract"] + " " + p["title"] for p in papers)
    words    = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
    return dict(Counter(w for w in words if w not in STOP_WORDS).most_common(n))


def summarize_paper(paper: dict) -> str:
    llm    = ChatGroq(model=LLM_MODEL, temperature=0.4)
    prompt = f"""Summarize this research paper in 3 concise bullet points, then give one key takeaway.

Title: {paper['title']}
Authors: {paper['authors']}
Abstract: {paper['abstract']}

Format:
• [Point 1]
• [Point 2]
• [Point 3]

Key Takeaway: [one sentence]"""
    return llm.invoke([HumanMessage(content=prompt)]).content

# ──────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────
def fig_word_freq(papers):
    wf  = top_words(papers)
    fig = px.bar(x=list(wf.keys()), y=list(wf.values()),
                 title="Top Keywords", color=list(wf.values()),
                 color_continuous_scale="Blues")
    fig.update_layout(**DARK_LAYOUT, coloraxis_showscale=False,
                      xaxis_tickangle=-40, height=340)
    return fig

def fig_year_dist(papers):
    cnt = Counter(p["year"] for p in papers)
    fig = px.bar(x=sorted(cnt.keys()), y=[cnt[y] for y in sorted(cnt.keys())],
                 title="Papers by Year", color_discrete_sequence=["#63b3ed"])
    fig.update_layout(**DARK_LAYOUT, height=300)
    return fig

def fig_category_dist(papers):
    cats = []
    for p in papers:
        cats.extend(c.strip() for c in p["categories"].split(",")[:2])
    cnt = Counter(cats).most_common(10)
    fig = px.pie(names=[c[0] for c in cnt], values=[c[1] for c in cnt],
                 title="Category Distribution", hole=0.45,
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(**DARK_LAYOUT, height=320)
    return fig

def fig_author_freq(papers):
    authors = []
    for p in papers:
        authors.extend(a.strip() for a in p["authors"].replace(" et al.", "").split(",")[:3])
    cnt = Counter(authors).most_common(10)
    fig = px.bar(x=[c[1] for c in cnt], y=[c[0] for c in cnt],
                 orientation="h", title="Top Authors",
                 color_discrete_sequence=["#4299e1"])
    fig.update_layout(**DARK_LAYOUT, height=320, yaxis=dict(autorange="reversed"))
    return fig

# ──────────────────────────────────────────────────────────────
# SAFE INVOKE WRAPPER
# ──────────────────────────────────────────────────────────────
@retry_on_rate_limit(max_retries=Config.GROQ_MAX_RETRIES, initial_delay=Config.GROQ_RETRY_DELAY)
def safe_invoke(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        logger.error(f"Execution error: {e}")
        err = str(e)
        if "RESOURCE_EXHAUSTED" in err or "429" in err:
            return None, "⚠️ Quota exceeded. Retrying via backoff..."
        elif "AuthenticationError" in err or "API_KEY_INVALID" in err:
            return None, "⚠️ Invalid API key. Check Config and restart."
        else:
            return None, f"⚠️ Error: {err}"

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Research Assistant")
    st.markdown("*Elevence Skills Internship*")
    st.markdown("---")
    st.markdown("### 📂 arXiv Domain")
    arxiv_category = st.selectbox("Category", list(CS_CATEGORIES.keys()), index=1,
                                  label_visibility="collapsed")
    arxiv_max_res  = st.slider("Max results", 5, 40, 15)
    st.markdown("---")
    st.markdown("**📑 arXiv indexed:**  " + str(len(st.session_state.arxiv_indexed)))
    st.markdown("**💬 Chat messages:** " + str(len(st.session_state.messages)))

    if st.button("🗑️ Clear arXiv KB", use_container_width=True):
        import shutil
        st.session_state.arxiv_indexed = {}
        st.session_state.arxiv_chat    = []
        if os.path.exists(ARXIV_DB_PATH): shutil.rmtree(ARXIV_DB_PATH)
        st.rerun()

    if st.button("🧹 Clear All Chats", use_container_width=True):
        st.session_state.messages         = []
        st.session_state.medical_messages = []
        st.session_state.arxiv_chat       = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 🌐 Language")
    lang_options  = get_language_options()
    lang_override = st.selectbox(
        "Force output language (or leave Auto-detect)",
        ["🔍 Auto-detect"] + list(lang_options.keys()),
        index=0,
        label_visibility="collapsed",
    )
    if lang_override == "🔍 Auto-detect":
        st.session_state["lang_override"] = None
    else:
        st.session_state["lang_override"] = lang_options[lang_override]
    st.caption("Auto-detects from your message · overrides available for 8 languages")
    st.markdown("---")
    st.caption("Powered by Groq · Gemini · arXiv API · ChromaDB · VADER · LangDetect")

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🤖 AI Research Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Elevence Skills Internship — Tasks 1 · 2 · 3 · 4 · 5 · 6 unified</div>",
            unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 General Chatbot",
    "🩺 Medical Q&A",
    "🔬 arXiv Search",
    "📊 Visualize Concepts",
    "📝 Summarize Papers",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — GENERAL CHATBOT  (Task 1 + Task 2)
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 💬 General Chatbot")
    st.caption("Task 1: PDF RAG · Task 2: Multimodal/Gemini · Task 5: Sentiment · Task 6: Multilingual (8 languages)")

    general_chain = load_general_chain(LLM_MODEL)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    col_img, col_input = st.columns([1, 6])
    with col_img:
        image_file = st.file_uploader("📎", type=["png","jpg","jpeg"],
                                      label_visibility="collapsed")
    with col_input:
        user_input = st.chat_input("Ask anything... (upload an image for visual Q&A)")

    if image_file:
        st.image(image_file, caption="Uploaded image", width=200)

    if user_input:
        # ── Task 5: Sentiment ───────────────────────────────
        sentiment = analyze_sentiment(user_input)
        snt_color = {"positive": "#48bb78", "negative": "#fc8181", "neutral": "#90cdf4"}[sentiment.label]

        # ── Task 6: Language detection & translation ──────────────
        override = st.session_state.get("lang_override", None)
        lang     = detect_language(user_input)
        if override and override != "en":
            lang = detect_language("")          # get blank result
            from language_handler import LanguageResult
            meta   = SUPPORTED_LANGUAGES[override]
            lang   = LanguageResult(
                code=override, name=meta["name"], native_name=meta["native_name"],
                flag=meta["flag"], cultural_note=meta["cultural_note"],
                is_english=(override=="en"), rtl=meta["rtl"], confidence="manual"
            )
        input_en = translate_to_english(user_input, lang.code) if not lang.is_english else user_input

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
            st.markdown(
                f"<span style='font-size:0.72rem;color:{snt_color};'>"
                f"{sentiment.emoji} {sentiment.tone_label} · score {sentiment.compound:+.2f} · {sentiment.intensity}"
                f"</span> &nbsp; "
                f"<span style='font-size:0.72rem;color:#a0aec0;'>"
                f"{lang.flag} {lang.native_name}"
                f"{'&nbsp;·&nbsp;translated' if not lang.is_english else ''}"
                f"</span>",
                unsafe_allow_html=True
            )

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                start = time.time()
                if image_file:
                    img = Image.open(image_file).convert("RGB")
                    answer, err = safe_invoke(gemini_chat, st.session_state.messages, image=img)
                    if err: answer = err
                else:
                    # Build sentiment + language-aware system prompt
                    sys_prompt = (
                        "You are a helpful, emotionally intelligent AI assistant. "
                        f"{sentiment.system_instruction} "
                        f"{lang.system_instruction}"
                    )
                    if general_chain:
                        # RAG processes in English; translate Q in, translate A out
                        en_answer, err = safe_invoke(general_chain.invoke, input_en)
                        if err:
                            answer = err
                        else:
                            answer = translate_from_english(en_answer, lang.code)
                    else:
                        llm = get_llm(0.7)
                        history_msgs = [SystemMessage(content=sys_prompt)]
                        for m in st.session_state.messages:
                            if m["role"] == "user":
                                history_msgs.append(HumanMessage(content=m["content"]))
                            else:
                                history_msgs.append(AIMessage(content=m["content"]))
                        result, err = safe_invoke(llm.invoke, history_msgs)
                        answer = result.content if result else err
                mode_tag = (" · 🖼️ Gemini" if image_file else
                            " · 📄 RAG" if general_chain and not image_file
                            else " · 🤖 Groq")
                st.markdown(answer)
                st.caption(
                    f"⏱ {round(time.time()-start, 2)}s{mode_tag} · "
                    f"{sentiment.emoji} {sentiment.tone_label} · "
                    f"{lang.flag} {lang.native_name}"
                )
        st.session_state.messages.append({"role": "assistant", "content": answer})



# ══════════════════════════════════════════════════════════════
# TAB 2 — MEDICAL Q&A  (Task 3)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🩺 Medical Q&A")
    st.caption("Task 3: MedQuAD RAG · Medical NER · Task 5: Sentiment · Task 6: Multilingual")

    chain, retriever = load_medical_chain(LLM_MODEL)

    st.warning("⚠️ This provides general medical information only. Always consult a qualified healthcare professional.")

    for msg in st.session_state.medical_messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🩺"):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a medical question...")

    if query:
        # Task 5: Sentiment
        sentiment_med = analyze_sentiment(query)
        snt_color_med = {"positive": "#48bb78", "negative": "#fc8181", "neutral": "#90cdf4"}[sentiment_med.label]
        # Task 6: Language
        lang_med  = detect_language(query)
        query_en  = translate_to_english(query, lang_med.code) if not lang_med.is_english else query

        st.session_state.medical_messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)
            st.markdown(
                f"<span style='font-size:0.72rem;color:{snt_color_med};'>"
                f"{sentiment_med.emoji} {sentiment_med.tone_label} · score {sentiment_med.compound:+.2f}"
                f"</span> &nbsp; "
                f"<span style='font-size:0.72rem;color:#a0aec0;'>{lang_med.flag} {lang_med.native_name}</span>",
                unsafe_allow_html=True
            )

        # NER extraction
        entities = extract_medical_entities(query)
        if entities.has_entities():
            ner_parts = []
            for cat, vals in entities.to_dict().items():
                if vals:
                    tags   = "".join(f"<span class='tag'>{v}</span>" for v in vals)
                    ner_parts.append(f"<b>{cat}:</b> {tags}")
            if ner_parts:
                st.markdown("**🔍 Detected Medical Entities:**")
                st.markdown("<div>" + " &nbsp;|&nbsp; ".join(ner_parts) + "</div>",
                            unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Searching medical knowledge base..."):
                start = time.time()
                if chain is None:
                    answer = "Medical database not built yet. Run `python medical_manager.py` to index MedQuAD data."
                else:
                    # Use translated English query for RAG, translate answer back
                    en_ans, err = safe_invoke(chain.invoke, query_en)
                    if err:
                        answer = err
                    else:
                        answer = translate_from_english(en_ans, lang_med.code)
                st.markdown(answer)
                st.caption(
                    f"⏱ {round(time.time()-start, 2)}s"
                    + (" · 📚 MedQuAD RAG" if chain else " · ⚠️ no DB")
                    + f" · {lang_med.flag} {lang_med.native_name}"
                )
        st.session_state.medical_messages.append({"role": "assistant", "content": answer})

    if not chain:
        st.info("💡 Build the medical database: `python medical_manager.py`")

# ══════════════════════════════════════════════════════════════
# TAB 3 — ARXIV SEARCH + CHAT  (Task 4)
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔬 arXiv Research Chatbot")
    st.caption("Task 4: Real-time arXiv search · Index papers · RAG-powered chat")

    st.markdown("#### 🔍 Search Papers")
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        arxiv_query = st.text_input("Search arXiv", placeholder="e.g. large language models, diffusion models",
                                    label_visibility="collapsed")
    with col_btn:
        do_search = st.button("Search", type="primary", use_container_width=True)

    if do_search and arxiv_query:
        cat = CS_CATEGORIES[arxiv_category]
        with st.spinner(f"Searching arXiv for **{arxiv_query}**..."):
            try:
                st.session_state.arxiv_results    = search_arxiv(arxiv_query, cat, arxiv_max_res)
                st.session_state.arxiv_last_query = arxiv_query
            except Exception as e:
                st.error(f"Search failed: {e}")

    results = st.session_state.arxiv_results
    if results:
        st.markdown(f"**{len(results)} papers** for *{st.session_state.arxiv_last_query}*")

        c1, c2, c3, c4 = st.columns(4)
        years = [p["year"] for p in results]
        for col, val, label in zip([c1,c2,c3,c4],
            [len(results), max(years), min(years), len(st.session_state.arxiv_indexed)],
            ["Papers Found","Latest Year","Earliest Year","Indexed"]):
            col.markdown(f"<div class='stat-box'><div class='stat-num'>{val}</div>"
                         f"<div class='stat-label'>{label}</div></div>", unsafe_allow_html=True)
        st.markdown("")

        if st.button("📥 Index All Papers", type="primary"):
            with st.spinner("Indexing papers into knowledge base..."):
                added = index_arxiv_papers(results)
            st.success(f"✅ Indexed {added} new papers!")
            st.rerun()

        st.markdown("---")
        for p in results:
            is_idx = p["id"] in st.session_state.arxiv_indexed
            st.markdown(f"""<div class='card'>
                <div class='card-title'>{p['title']}</div>
                <div class='card-meta'>👥 {p['authors']} &nbsp;|&nbsp; 📅 {p['year']}
                &nbsp;|&nbsp; {'✅ Indexed' if is_idx else '⬜ Not indexed'}</div>
                <div>{''.join(f"<span class='tag'>{c.strip()}</span>" for c in p['categories'].split(',')[:4])}</div>
                <div class='card-text' style='margin-top:8px;'>{textwrap.shorten(p['abstract'], width=300, placeholder='...')}</div>
            </div>""", unsafe_allow_html=True)
            ca, cb = st.columns([1, 5])
            with ca:
                if not is_idx:
                    if st.button("📥 Index", key=f"idx_{p['id']}"):
                        index_arxiv_papers([p]); st.rerun()
                else:
                    st.caption("✅ Indexed")
            with cb:
                st.link_button("📄 PDF", p["pdf_url"])

    st.markdown("---")
    st.markdown("#### 💬 Chat with Indexed Papers")
    if not st.session_state.arxiv_indexed:
        st.info("Index papers above to enable research-grounded chat.")
    else:
        st.caption(f"Knowledge base: {len(st.session_state.arxiv_indexed)} paper(s)")

    for msg in st.session_state.arxiv_chat:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🔬"):
            st.markdown(msg["content"])

    arxiv_q = st.chat_input("Ask about indexed research papers...")
    if arxiv_q:
        st.session_state.arxiv_chat.append({"role": "user", "content": arxiv_q})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(arxiv_q)
        with st.chat_message("assistant", avatar="🔬"):
            with st.spinner("Searching papers..."):
                start = time.time()
                rag  = build_arxiv_rag()
                if rag:
                    answer, err = safe_invoke(rag.invoke, arxiv_q)
                    if err: answer = err
                    mode = "📚 arXiv RAG"
                else:
                    llm = get_llm(0.7)
                    sys = SystemMessage(content="You are an expert CS research assistant.")
                    hist = [sys] + [
                        HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m["content"])
                        for m in st.session_state.arxiv_chat
                    ]
                    result, err = safe_invoke(llm.invoke, hist)
                    answer = result.content if result else err
                    mode   = "🤖 Groq direct"
                st.markdown(answer)
                st.caption(f"⏱ {round(time.time()-start,2)}s · {mode}")
        st.session_state.arxiv_chat.append({"role": "assistant", "content": answer})

# ══════════════════════════════════════════════════════════════
# TAB 4 — VISUALIZE  (Task 4)
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Concept Visualizations")
    st.caption("Task 4: Keyword analysis · Year distribution · Category breakdown · Author network")

    viz_papers = st.session_state.arxiv_results or list(st.session_state.arxiv_indexed.values())

    if not viz_papers:
        st.info("🔍 Search for arXiv papers first (Tab 3) to populate visualizations.")
    else:
        st.markdown(f"Visualizing **{len(viz_papers)} papers**")

        r1c1, r1c2 = st.columns(2)
        with r1c1: st.plotly_chart(fig_word_freq(viz_papers), use_container_width=True)
        with r1c2: st.plotly_chart(fig_year_dist(viz_papers), use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1: st.plotly_chart(fig_category_dist(viz_papers), use_container_width=True)
        with r2c2: st.plotly_chart(fig_author_freq(viz_papers), use_container_width=True)

        # Concept tag cloud
        st.markdown("### 🧠 Key Concept Cloud")
        wf = top_words(viz_papers, n=60)
        max_v = max(wf.values()) if wf else 1
        tags  = " ".join(
            f"<span class='tag' style='font-size:{0.72+v/max_v*0.55:.2f}rem;"
            f"opacity:{0.5+v/max_v*0.5:.2f};'>{w}</span>"
            for w, v in wf.items()
        )
        st.markdown(f"<div style='line-height:2.4;'>{tags}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — SUMMARIZE  (Task 4)
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📝 AI Paper Summarizer")
    st.caption("Task 4: Select any paper → get AI-generated bullet-point summary + key takeaway")

    pool = (list(st.session_state.arxiv_indexed.values()) +
            [p for p in st.session_state.arxiv_results
             if p["id"] not in st.session_state.arxiv_indexed])

    if not pool:
        st.info("🔍 Search and/or index papers in Tab 3 first.")
    else:
        paper_map    = {p["title"][:90]: p for p in pool}
        chosen_title = st.selectbox("Select a paper", list(paper_map.keys()))
        chosen       = paper_map[chosen_title]

        st.markdown(f"**Authors:** {chosen['authors']}  |  **Year:** {chosen['year']}")
        tags_html = "".join(f"<span class='tag'>{c.strip()}</span>"
                            for c in chosen["categories"].split(",")[:4])
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown("")
        st.markdown("**Abstract:**")
        st.markdown(f"> {chosen['abstract']}")

        col_s, col_p = st.columns([1, 5])
        with col_s:
            do_sum = st.button("✨ Summarize", type="primary", use_container_width=True)
        with col_p:
            st.link_button("📄 Full Paper PDF", chosen["pdf_url"])

        if do_sum:
            with st.spinner("Generating AI summary with Groq..."):
                start   = time.time()
                summary, err = safe_invoke(summarize_paper, chosen)
                if err:
                    st.error(err)
                else:
                    st.markdown("---")
                    st.markdown("### 🤖 AI Summary")
                    st.markdown(summary)
                    st.caption(f"⏱ {round(time.time()-start,2)}s · Groq {LLM_MODEL}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pass