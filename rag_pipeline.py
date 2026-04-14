from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Load RAG
def load_rag_chain(api_key=None):
    embeddings = HuggingFaceEmbeddings()

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever()

    # offline model
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

    return retriever, pipe


# Get answer (manual RAG)
def get_answer(chain, question):
    retriever, pipe = chain

    docs = retriever.get_relevant_documents(question)

    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    result = pipe(prompt)[0]["generated_text"]

    return {
        "answer": result,
        "source_documents": docs
    }
