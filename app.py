"""
app.py
------
Streamlit frontend for the College FAQ RAG Chatbot.
Run: streamlit run app.py
"""

import os
import time
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ingest import ingest_documents, DATA_DIR, VECTORSTORE_DIR

from ingest import ingest_documents, DATA_DIR, VECTORSTORE_DIR
from rag_pipeline import load_rag_chain, get_answer

# ── Load .env (if present) ────────────────────────────────────────────────────
load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PVPIT College FAQ Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #0d47a1 100%);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 4px 20px rgba(26, 35, 126, 0.3);
    }
    .header-banner h1 {
        color: white;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .header-banner p {
        color: rgba(255,255,255,0.8);
        margin: 4px 0 0 0;
        font-size: 0.9rem;
    }
    .header-icon { font-size: 3rem; }

    /* ── Chat messages ── */
    .chat-user {
        background: #e3f2fd;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        color: #0d1b2a;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .chat-bot {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        color: #1a1a2e;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .chat-label-user {
        text-align: right;
        font-size: 0.75rem;
        color: #546e7a;
        margin-bottom: 2px;
    }
    .chat-label-bot {
        font-size: 0.75rem;
        color: #546e7a;
        margin-bottom: 2px;
    }

    /* ── Sidebar ── */
    .sidebar-section {
        background: #f8f9ff;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid #e8eaf6;
    }
    .sidebar-section h4 {
        color: #1a237e;
        margin: 0 0 12px 0;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Status pills ── */
    .pill-green {
        background: #e8f5e9; color: #2e7d32;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
        display: inline-block;
    }
    .pill-red {
        background: #ffebee; color: #c62828;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
        display: inline-block;
    }

    /* ── Quick questions ── */
    .stButton > button {
        border-radius: 10px !important;
        font-size: 0.82rem !important;
        padding: 6px 12px !important;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer { visibility: hidden; }

    /* ── Source expander ── */
    .source-chip {
        background: #ede7f6;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.82rem;
        color: #4527a0;
        margin-top: 6px;
        border-left: 3px solid #7c4dff;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────
def init_session():
    defaults = {
        "messages":      [],        # [{role, content}]
        "chain":         None,
        "memory":        None,
        "chain_ready":   False,
        "ingested":      VECTORSTORE_DIR.exists(),
        "groq_api_key":  os.getenv("GROQ_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 FAQ Chatbot Setup")
    st.markdown("---")

    # ── 1. API Key ──
    with st.container():
        st.markdown("#### 🔑 Groq API Key")
        api_key_input = st.text_input(
            "Enter your Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_...",
            help="Free key from https://console.groq.com"
        )
        if api_key_input:
            st.session_state.groq_api_key = api_key_input
        st.caption("[Get free key →](https://console.groq.com)")

    st.markdown("---")

    # ── 2. Document Upload ──
    with st.container():
        st.markdown("#### 📄 Upload College Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload brochures, handbooks, syllabi etc."
        )

        if uploaded_files:
            if st.button("📥 Save Uploaded Files", use_container_width=True):
                DATA_DIR.mkdir(exist_ok=True)
                for f in uploaded_files:
                    dest = DATA_DIR / f.name
                    with open(dest, "wb") as out:
                        out.write(f.read())
                st.success(f"Saved {len(uploaded_files)} file(s) to data/")

    st.markdown("---")

    # ── 3. Build / Rebuild Index ──
    with st.container():
        st.markdown("#### ⚙️ Vector Store")

        status_html = (
            '<span class="pill-green">✔ Index Ready</span>'
            if st.session_state.ingested
            else '<span class="pill-red">✘ Not Built</span>'
        )
        st.markdown(status_html, unsafe_allow_html=True)
        st.caption("Uses default college_faq.txt if no file uploaded.")

        if st.button("🔨 Build / Rebuild Index", use_container_width=True):
            with st.spinner("Building FAISS index…"):
                try:
                    ingest_documents()
                    st.session_state.ingested      = True
                    st.session_state.chain         = None
                    st.session_state.chain_ready   = False
                    st.success("Index built successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # ── 4. Load Chain ──
    with st.container():
        st.markdown("#### 🤖 Load Chatbot")

        chain_status = (
            '<span class="pill-green">✔ Chatbot Active</span>'
            if st.session_state.chain_ready
            else '<span class="pill-red">✘ Not Loaded</span>'
        )
        st.markdown(chain_status, unsafe_allow_html=True)

        if st.button("🚀 Load Chatbot", use_container_width=True):
            if not st.session_state.groq_api_key:
                st.error("Please enter your Groq API key first.")
            elif not st.session_state.ingested:
                st.error("Please build the FAISS index first.")
            else:
                with st.spinner("Loading RAG chain…"):
                    try:
                        chain, memory = load_rag_chain(st.session_state.groq_api_key)
                        st.session_state.chain       = chain
                        st.session_state.memory      = memory
                        st.session_state.chain_ready = True
                        st.success("Chatbot is ready!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading chain: {e}")

    st.markdown("---")

    # ── 5. Reset Chat ──
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.memory:
            st.session_state.memory.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Made with ❤️ | PVPIT Computer Engineering | LangChain + FAISS + Groq")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="header-banner">
    <div class="header-icon">🎓</div>
    <div>
        <h1>PVPIT College FAQ Chatbot</h1>
        <p>AI-powered assistant for student queries — Admissions · Academics · Placements · Campus Life</p>
    </div>
</div>
""", unsafe_allow_html=True)


# Quick Questions
st.markdown("**💡 Try a quick question:**")
quick_cols = st.columns(4)
quick_questions = [
    "What is the admission process?",
    "What are the placement packages?",
    "What documents are required?",
    "What clubs are available?",
]
for i, (col, q) in enumerate(zip(quick_cols, quick_questions)):
    with col:
        if st.button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state.pending_quick = q

st.markdown("---")

# ── Chat Window ───────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #90a4ae;">
            <div style="font-size: 3rem; margin-bottom: 12px;">💬</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #546e7a;">Ask me anything about PVPIT</div>
            <div style="font-size: 0.9rem; margin-top: 8px;">Admissions · Fees · Exams · Facilities · Placements · Student Life</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-label-user">You</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-label-bot">🎓 FAQ Assistant</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)

                # Show source snippets in expander
                if msg.get("sources"):
                    with st.expander("📎 View Source Chunks", expanded=False):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(
                                f'<div class="source-chip"><b>Chunk {i}:</b> {src[:300]}{"…" if len(src) > 300 else ""}</div>',
                                unsafe_allow_html=True
                            )

# ── Input Box ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

# Handle quick question injection
pending = st.session_state.pop("pending_quick", None)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask your question…",
        value=pending or "",
        placeholder="e.g. What is the fee structure for Compueter Branch?",
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Send ➤", use_container_width=False)


# ── Process Input ─────────────────────────────────────────────────────────────
if submitted and user_input.strip():
    question = user_input.strip()

    # Guard checks
    if not st.session_state.chain_ready:
        st.warning("⚠️ Please build the index and load the chatbot first (see sidebar).")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})

    # Get answer from RAG
    with st.spinner("Thinking…"):
        try:
            result = get_answer(st.session_state.chain, question)
            answer  = result["answer"]
            sources = [doc.page_content for doc in result["source_documents"]]
        except Exception as e:
            answer  = f"⚠️ Error getting response: {e}"
            sources = []

    # Append bot message
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources
    })

    st.rerun()
