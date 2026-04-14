# 🎓 College FAQ Chatbot
### RAG-based AI Chatbot using LangChain + FAISS + Groq + Streamlit

> **Mini Project | B.E. Data Science | PVPIT, Pune**  
> Built with: Python · LangChain · FAISS · HuggingFace Embeddings · Groq (LLaMA 3.1) · Streamlit

---

## 📌 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based chatbot that answers student queries about college-related topics — admissions, fees, academics, placements, campus life, and more.

Instead of relying on a fixed Q&A database, the chatbot **retrieves relevant chunks** from uploaded college documents and uses an **LLM (LLaMA 3.1 via Groq)** to generate accurate, contextual answers.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────┐
│  Streamlit Frontend │  ← app.py
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   RAG Pipeline      │  ← rag_pipeline.py
│  (LangChain Chain)  │
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
FAISS DB    Groq LLM
(Vector     (LLaMA 3.1
 Search)     8B Instant)
    │
    ▼
College Documents (PDF/TXT)
← Ingested by ingest.py
```

---

## 🛠️ Tech Stack

| Component          | Technology                            |
|--------------------|---------------------------------------|
| Frontend           | Streamlit                             |
| RAG Framework      | LangChain                             |
| Vector Store       | FAISS (Facebook AI Similarity Search) |
| Embeddings         | HuggingFace `all-MiniLM-L6-v2`        |
| LLM                | LLaMA 3.1 8B via Groq API (Free)      |
| Document Loading   | LangChain PDF + Text Loaders          |
| Text Splitting     | RecursiveCharacterTextSplitter        |

---

## 📁 Project Structure

```
college_faq_chatbot/
├── app.py               # Streamlit UI
├── rag_pipeline.py      # RAG chain setup (LangChain + Groq + FAISS)
├── ingest.py            # Document loader + FAISS index builder
├── data/
│   └── college_faq.txt  # Default FAQ knowledge base
├── vectorstore/         # Generated FAISS index (auto-created)
├── .env.example         # Environment variable template
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project
```bash
cd college_faq_chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your Groq API Key
- Get a **free** API key at [https://console.groq.com](https://console.groq.com)
- Create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your key: GROQ_API_KEY=gsk_...
```

---

## 🚀 Running the Project

### Step 1: Build the FAISS Index
```bash
python ingest.py
```
This loads `data/college_faq.txt`, chunks it, generates embeddings, and saves the FAISS index.

### Step 2: Launch the Streamlit App
```bash
streamlit run app.py
```

### Step 3: In the Browser
1. Enter your **Groq API Key** in the sidebar
2. Click **"Build / Rebuild Index"** (or skip if already built)
3. Click **"Load Chatbot"**
4. Start asking questions!

---

## 💡 Features

- ✅ **RAG-based** — Answers grounded in actual college documents
- ✅ **Conversational memory** — Remembers last 5 exchanges
- ✅ **Upload custom documents** — Add PDFs or TXT files via sidebar
- ✅ **Source transparency** — View the exact chunks used to answer
- ✅ **Quick question buttons** — One-click sample queries
- ✅ **Free LLM** — Uses Groq's free tier (no OpenAI billing)
- ✅ **Local embeddings** — No API key for embeddings (runs on CPU)

---

## 📊 How RAG Works (for Report/Viva)

1. **Ingestion**: Documents are loaded → split into 500-token chunks → embedded using `all-MiniLM-L6-v2` → stored in FAISS.
2. **Retrieval**: User query is embedded → top-4 similar chunks fetched from FAISS.
3. **Generation**: Chunks + chat history + query sent to LLaMA 3.1 via Groq → answer generated.
4. **Response**: Answer displayed in Streamlit with source chunks shown.

---

## 🔧 Extending the Project

| Enhancement | How |
|---|---|
| Add more documents | Upload via sidebar or drop in `data/` folder |
| Change LLM | Replace `ChatGroq` with `ChatOpenAI` or `ChatGoogleGenerativeAI` |
| Better embeddings | Use `BAAI/bge-large-en` for improved retrieval |
| Deploy online | Push to Streamlit Cloud (free) with `.env` secrets |
| Add reranking | Use `FlashrankRerank` for better chunk selection |

---

## 👤 Team / Credits

- **Student**: Harshad [Roll No.]  
- **Department**: Data Science, T.E.  
- **Institute**: PVPIT, Bavdhan, Pune — 411021  
- **Guide**: [Faculty Name]  
- **Academic Year**: 2024–25

---

# College FAQ Chatbot 🤖

## How to run
pip install -r requirements.txt
streamlit run app.py

## Features
- FAQ chatbot using FAISS
- Offline LLM (no API required)

## 📜 License
This project is for academic purposes at PVPIT, Pune.
