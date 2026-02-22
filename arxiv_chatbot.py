"""
arxiv_chatbot.py
----------------
arXiv Research Chatbot — Domain Expert in Computer Science
Features:
  - Real-time arXiv paper search via API
  - RAG-based chat over selected papers (ChromaDB + Groq)
  - AI paper summarization
  - Concept visualizations (word freq, timeline, categories)
Run:
    streamlit run arxiv_chatbot.py
"""

import os
import re
import time
import textwrap
from collections import Counter
from datetime import datetime

import arxiv
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from config import Config
from logger import logger
from exceptions import retry_on_rate_limit
from rag_utils import RAGManager

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
ARXIV_DB_PATH = Config.ARXIV_DB_PATH
LLM_MODEL     = Config.LLM_MODEL
EMBED_MODEL   = Config.EMBED_MODEL

CS_CATEGORIES = {
    "All CS"              : "cs.*",
    "AI / Machine Learning": "cs.AI OR cs.LG",
    "Computer Vision"     : "cs.CV",
    "NLP"                 : "cs.CL",
    "Systems"             : "cs.SY OR cs.OS",
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

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="arXiv Research Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0d0d1a 0%, #0a0a2e 50%, #0d1b2a 100%); }

.paper-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    transition: all 0.3s ease;
}
.paper-card:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(99,179,237,0.5);
    transform: translateY(-2px);
}
.paper-title {
    color: #63b3ed;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.paper-meta {
    color: #8893a7;
    font-size: 0.78rem;
    margin-bottom: 8px;
}
.paper-abstract {
    color: #c8d3e0;
    font-size: 0.85rem;
    line-height: 1.6;
}
.tag {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    color: #63b3ed;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin: 2px;
}
.stat-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.stat-num { font-size: 1.8rem; font-weight: 700; color: #63b3ed; }
.stat-label { font-size: 0.78rem; color: #8893a7; margin-top: 2px; }

section[data-testid="stSidebar"] {
    background: rgba(10,10,30,0.9) !important;
    border-right: 1px solid rgba(99,179,237,0.1);
}
.stChatMessage { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# INIT SESSION STATE
# ──────────────────────────────────────────────────────────────
for key, default in [
    ("search_results", []),
    ("indexed_papers", {}),   # arxiv_id -> paper dict
    ("chat_history", []),
    ("last_query", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────
# EMBEDDINGS (singleton)
# ──────────────────────────────────────────────────────────────
# Using RAGManager for shared embeddings
embeddings = RAGManager.get_embeddings()

@st.cache_resource
def get_llm(temperature: float = 0.7):
    return ChatGroq(
        model=LLM_MODEL, 
        temperature=temperature,
        api_key=Config.GROQ_API_KEY
    )

# ──────────────────────────────────────────────────────────────
# VECTORSTORE HELPERS
# ──────────────────────────────────────────────────────────────
def get_vectorstore():
    return Chroma(persist_directory=ARXIV_DB_PATH, embedding_function=embeddings)

def index_papers(papers: list[dict]) -> int:
    """Chunk and index paper abstracts into ChromaDB. Returns count added."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    docs = []
    ids  = []
    vs   = get_vectorstore()
    already = set(st.session_state.indexed_papers.keys())

    for p in papers:
        pid = p["id"]
        if pid in already:
            continue
        text = f"Title: {p['title']}\n\nAuthors: {p['authors']}\n\nAbstract: {p['abstract']}"
        chunks = splitter.create_documents(
            [text],
            metadatas=[{"arxiv_id": pid, "title": p["title"],
                        "year": str(p["year"]), "categories": p["categories"]}]
        )
        for i, c in enumerate(chunks):
            docs.append(c)
            ids.append(f"{pid}_chunk{i}")
        st.session_state.indexed_papers[pid] = p

    if docs:
        vs.add_documents(docs, ids=ids)
    return len([p for p in papers if p["id"] not in already])

def build_rag_chain():
    if not os.path.exists(ARXIV_DB_PATH) or not st.session_state.indexed_papers:
        return None
    vs        = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    llm       = get_llm(temperature=0.3)
    prompt    = PromptTemplate(
        template="""You are an expert research assistant specializing in computer science.
Use the provided research paper context to answer the question accurately and in depth.
If the context doesn't contain enough information, say so and answer from general knowledge.

Context from papers:
{context}

Question: {question}

Provide a detailed, insightful answer. Reference specific papers when relevant.""",
        input_variables=["context", "question"]
    )
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

# ──────────────────────────────────────────────────────────────
# ARXIV SEARCH
# ──────────────────────────────────────────────────────────────
def search_arxiv(query: str, category_filter: str, max_results: int = 15) -> list[dict]:
    """Search arXiv and return list of paper dicts."""
    full_query = f"({query}) AND cat:{category_filter}" if ".*" not in category_filter else f"({query})"
    client = arxiv.Client()
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for r in client.results(search):
        papers.append({
            "id"         : r.entry_id.split("/")[-1],
            "title"      : r.title,
            "authors"    : ", ".join(a.name for a in r.authors[:4])
                           + (" et al." if len(r.authors) > 4 else ""),
            "abstract"   : r.summary.replace("\n", " "),
            "year"       : r.published.year,
            "month"      : r.published.month,
            "categories" : ", ".join(r.categories),
            "url"        : r.entry_id,
            "pdf_url"    : r.pdf_url,
        })
    return papers

# ──────────────────────────────────────────────────────────────
# NLP HELPERS
# ──────────────────────────────────────────────────────────────
def top_words(papers: list[dict], n: int = 30) -> dict:
    all_text = " ".join(p["abstract"] + " " + p["title"] for p in papers)
    words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
    return dict(Counter(w for w in words if w not in STOP_WORDS).most_common(n))

def summarize_paper(paper: dict) -> str:
    llm    = get_llm(temperature=0.4)
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
DARK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d3e0", margin=dict(t=40, b=20, l=20, r=20))

def fig_word_freq(papers):
    wf = top_words(papers)
    fig = px.bar(x=list(wf.keys()), y=list(wf.values()),
                 title="Top Keywords in Results",
                 color=list(wf.values()), color_continuous_scale="Blues")
    fig.update_layout(**DARK, coloraxis_showscale=False,
                      xaxis_tickangle=-40, height=360)
    return fig

def fig_year_dist(papers):
    years = [p["year"] for p in papers]
    cnt   = Counter(years)
    fig   = px.bar(x=sorted(cnt.keys()), y=[cnt[y] for y in sorted(cnt.keys())],
                   title="Papers by Year", color_discrete_sequence=["#63b3ed"])
    fig.update_layout(**DARK, height=300)
    return fig

def fig_category_dist(papers):
    cats = []
    for p in papers:
        cats.extend(c.strip() for c in p["categories"].split(",")[:2])
    cnt = Counter(cats).most_common(10)
    fig = px.pie(names=[c[0] for c in cnt], values=[c[1] for c in cnt],
                 title="Category Distribution", hole=0.45,
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(**DARK, height=340)
    return fig

def fig_author_network(papers):
    """Simple author frequency bar."""
    authors = []
    for p in papers:
        authors.extend(a.strip() for a in p["authors"].replace(" et al.", "").split(",")[:3])
    cnt = Counter(authors).most_common(12)
    fig = px.bar(x=[c[1] for c in cnt], y=[c[0] for c in cnt],
                 orientation="h", title="Most Frequent Authors",
                 color_discrete_sequence=["#4299e1"])
    fig.update_layout(**DARK, height=340, yaxis=dict(autorange="reversed"))
    return fig

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 arXiv Research Bot")
    st.markdown("---")
    category = st.selectbox("📂 Domain", list(CS_CATEGORIES.keys()), index=1)
    max_res   = st.slider("Max results", 5, 40, 15)
    st.markdown("---")
    st.markdown(f"**📑 Indexed papers:** {len(st.session_state.indexed_papers)}")
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        st.session_state.indexed_papers = {}
        st.session_state.chat_history   = []
        import shutil
        if os.path.exists(ARXIV_DB_PATH):
            shutil.rmtree(ARXIV_DB_PATH)
        st.rerun()
    st.markdown("---")
    st.markdown("**💬 Chat history:** " + str(len(st.session_state.chat_history)) + " messages")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown("---")
    st.caption("Powered by arXiv API · Groq · ChromaDB")

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='background: linear-gradient(90deg,#63b3ed,#9f7aea);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            font-size:2.2rem;margin-bottom:4px;'>
  🔬 arXiv Research Chatbot
</h1>
<p style='color:#8893a7;font-size:0.95rem;margin-bottom:20px;'>
  Domain expert in CS · Search papers · Chat · Summarize · Visualize
</p>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Search Papers",
    "💬 Chat with Papers",
    "📊 Visualize Concepts",
    "📝 Summarize",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SEARCH
# ══════════════════════════════════════════════════════════════
with tab1:
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("Search arXiv", placeholder="e.g., large language models, diffusion models, transformers",
                              label_visibility="collapsed")
    with col_btn:
        search_btn = st.button("Search 🔍", use_container_width=True, type="primary")

    if search_btn and query:
        cat_filter = CS_CATEGORIES[category]
        with st.spinner(f"Searching arXiv for **{query}** in **{category}**..."):
            try:
                results = search_arxiv(query, cat_filter, max_results=max_res)
                st.session_state.search_results = results
                st.session_state.last_query = query
            except Exception as e:
                st.error(f"Search failed: {e}")

    results = st.session_state.search_results
    if results:
        st.markdown(f"**{len(results)} papers found** for *{st.session_state.last_query}*")

        # Stats row
        years = [p["year"] for p in results]
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in zip(
            [c1, c2, c3, c4],
            [len(results), max(years), len(set(y for p in results for y in [p["year"]])), len(st.session_state.indexed_papers)],
            ["Papers Found", "Latest Year", "Year Range", "Indexed"]
        ):
            col.markdown(f"""<div class='stat-box'>
                <div class='stat-num'>{val}</div>
                <div class='stat-label'>{label}</div></div>""", unsafe_allow_html=True)

        st.markdown("")

        # Index all button
        col_ia, col_sp = st.columns([1, 4])
        with col_ia:
            if st.button("📥 Index All Papers", use_container_width=True):
                with st.spinner("Indexing..."):
                    added = index_papers(results)
                st.success(f"✅ Indexed {added} new papers into knowledge base!")

        st.markdown("---")

        # Paper cards
        for p in results:
            is_indexed = p["id"] in st.session_state.indexed_papers
            with st.container():
                st.markdown(f"""<div class='paper-card'>
                    <div class='paper-title'>{p['title']}</div>
                    <div class='paper-meta'>👥 {p['authors']} &nbsp;|&nbsp; 📅 {p['year']} &nbsp;|&nbsp;
                    {'✅ Indexed' if is_indexed else '⬜ Not indexed'}</div>
                    <div>{''.join(f"<span class='tag'>{c.strip()}</span>" for c in p['categories'].split(',')[:4])}</div>
                    <div class='paper-abstract' style='margin-top:8px;'>{textwrap.shorten(p['abstract'],300,'...')}</div>
                </div>""", unsafe_allow_html=True)

                cx, cy, cz = st.columns([1, 1, 4])
                with cx:
                    if not is_indexed:
                        if st.button("📥 Index", key=f"idx_{p['id']}"):
                            index_papers([p])
                            st.rerun()
                    else:
                        st.caption("✅ Indexed")
                with cy:
                    st.link_button("📄 PDF", p["pdf_url"])

# ══════════════════════════════════════════════════════════════
# TAB 2 — CHAT
# ══════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.indexed_papers:
        st.info("💡 **No papers indexed yet.** Go to **Search Papers**, search for a topic, and click **Index All Papers** to build your knowledge base.")
    else:
        st.markdown(f"💡 Chatting over **{len(st.session_state.indexed_papers)} indexed paper(s)**")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🔬"):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask anything about the indexed papers or CS concepts...")

    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_q)

        with st.chat_message("assistant", avatar="🔬"):
            with st.spinner("Thinking..."):
                start = time.time()
                try:
                    rag = build_rag_chain()
                    if rag:
                        answer = rag.invoke(user_q)
                    else:
                        # No indexed papers — direct Groq chat
                        llm = get_llm(0.7)
                        history_msgs = [SystemMessage(content="You are an expert computer science research assistant.")]
                        for m in st.session_state.chat_history:
                            if m["role"] == "user":
                                history_msgs.append(HumanMessage(content=m["content"]))
                            else:
                                history_msgs.append(AIMessage(content=m["content"]))
                        answer = llm.invoke(history_msgs).content
                except Exception as e:
                    answer = f"⚠️ Error: {e}"

                st.markdown(answer)
                st.caption(f"⏱ {round(time.time()-start,2)}s")
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ══════════════════════════════════════════════════════════════
# TAB 3 — VISUALIZE
# ══════════════════════════════════════════════════════════════
with tab3:
    papers_to_viz = st.session_state.search_results or list(st.session_state.indexed_papers.values())

    if not papers_to_viz:
        st.info("🔍 Search for papers first to see visualizations.")
    else:
        st.markdown(f"Visualizing **{len(papers_to_viz)} papers**")

        v1, v2 = st.columns(2)
        with v1:
            st.plotly_chart(fig_word_freq(papers_to_viz), use_container_width=True)
        with v2:
            st.plotly_chart(fig_year_dist(papers_to_viz), use_container_width=True)

        v3, v4 = st.columns(2)
        with v3:
            st.plotly_chart(fig_category_dist(papers_to_viz), use_container_width=True)
        with v4:
            st.plotly_chart(fig_author_network(papers_to_viz), use_container_width=True)

        # Concept extraction
        st.markdown("### 🧠 Key Concepts")
        wf = top_words(papers_to_viz, n=50)
        concept_html = " ".join(
            f"<span class='tag' style='font-size:{0.7+v/max(wf.values())*0.6:.2f}rem;'>{w}</span>"
            for w, v in wf.items()
        )
        st.markdown(f"<div style='line-height:2.2;'>{concept_html}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — SUMMARIZE
# ══════════════════════════════════════════════════════════════
with tab4:
    papers_pool = (
        list(st.session_state.indexed_papers.values()) +
        [p for p in st.session_state.search_results
         if p["id"] not in st.session_state.indexed_papers]
    )

    if not papers_pool:
        st.info("🔍 Search and/or index papers first.")
    else:
        paper_titles = {p["title"][:80]: p for p in papers_pool}
        chosen_title = st.selectbox("Select a paper to summarize", list(paper_titles.keys()))
        chosen_paper = paper_titles[chosen_title]

        col_info, col_btn2 = st.columns([5, 1])
        with col_info:
            st.caption(f"👥 {chosen_paper['authors']} · 📅 {chosen_paper['year']} · {chosen_paper['categories']}")
        with col_btn2:
            summarize_btn = st.button("✨ Summarize", type="primary", use_container_width=True)

        st.markdown("**Abstract:**")
        st.markdown(f"> {chosen_paper['abstract']}")

        if summarize_btn:
            with st.spinner("Generating AI summary..."):
                try:
                    summary = summarize_paper(chosen_paper)
                    st.markdown("---")
                    st.markdown("### 🤖 AI Summary")
                    st.markdown(summary)
                    st.link_button("📄 Read Full Paper", chosen_paper["pdf_url"])
                except Exception as e:
                    st.error(f"⚠️ {e}")
