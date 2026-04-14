from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
def ingest_documents():
    DATA_DIR.mkdir(exist_ok=True)
    VECTORSTORE_DIR.mkdir(exist_ok=True)

    docs = []

    # load txt files
    for file in DATA_DIR.glob("*.txt"):
        loader = TextLoader(str(file))
        docs.extend(loader.load())

    if not docs:
        raise Exception("❌ No documents found in data folder")

    # split documents
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)

    # embeddings
    embeddings = HuggingFaceEmbeddings()

    # create FAISS index
    db = FAISS.from_documents(texts, embeddings)

    # save index
    db.save_local(str(VECTORSTORE_DIR))

    print("✅ FAISS index created successfully")
