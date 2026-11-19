"""RAG pipeline: PDF -> chunks -> embeddings -> ChromaDB -> Gemini answer."""
import os
import uuid

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION = "pdf_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

_embedder = None
_llm = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_llm():
    global _llm
    if _llm is None:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key or key == "your_actual_gemini_api_key":
            raise RuntimeError("Set GOOGLE_API_KEY in your .env file.")
        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, google_api_key=key, temperature=0.2
        )
    return _llm


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(COLLECTION)


def extract_text(pdf_file):
    """Return a list of (page_number, text) from a PDF file object."""
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def split_text(text, chunk_size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def ingest_pdf(pdf_file, filename):
    """Chunk, embed and store a PDF. Returns number of chunks added."""
    pages = extract_text(pdf_file)
    docs, metas = [], []
    for page_no, text in pages:
        for chunk in split_text(text):
            docs.append(chunk)
            metas.append({"source": filename, "page": page_no})
    if not docs:
        return 0
    embeddings = get_embedder().encode(docs).tolist()
    ids = [str(uuid.uuid4()) for _ in docs]
    get_collection().add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    return len(docs)


def clear_database():
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass


def count_chunks():
    return get_collection().count()


def retrieve(question, k=4):
    col = get_collection()
    if col.count() == 0:
        return []
    q_emb = get_embedder().encode([question]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def answer_question(question):
    """Returns (answer, sources) where sources is a list of (file, page)."""
    hits = retrieve(question)
    if not hits:
        return "Please upload a PDF first so I have something to search.", []
    context = "\n\n".join(
        f"[{m['source']} p.{m['page']}]\n{d}" for d, m in hits
    )
    prompt = (
        "You are a helpful assistant answering questions about the user's PDFs. "
        "Use ONLY the context below. If the answer is not in the context, say "
        "you couldn't find it in the document.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    reply = get_llm().invoke(prompt)
    sources = sorted({(m["source"], m["page"]) for _, m in hits})
    return reply.content, sources
