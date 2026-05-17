import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================
# GLOBAL STATE
# ==============================
client = None
retriever = None
vectorstore = None


# ==============================
# LOAD DOCUMENTS
# ==============================
def load_documents():
    documents = []
    pdf_folder = "documents"

    if not os.path.exists(pdf_folder):
        print("WARNING: documents folder not found")
        return documents

    files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not files:
        print("WARNING: No PDF files found")
        return documents

    for file in files:
        path = os.path.join(pdf_folder, file)
        print(f"Loading PDF: {file}")

        try:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    return documents


# ==============================
# INIT RAG SYSTEM
# ==============================
def init_rag():
    global client, retriever, vectorstore

    load_dotenv()

    # ---------------- GROQ CLIENT ----------------
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        client = Groq(api_key=api_key)
    else:
        print("WARNING: GROQ_API_KEY missing")
        client = None

    # ---------------- DOCUMENTS ----------------
    documents = load_documents()

    if len(documents) == 0:
        print("WARNING: No documents loaded, retriever not created")
        retriever = None
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    # ---------------- EMBEDDINGS ----------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------- FAISS INDEX ----------------
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("RAG system initialized successfully")


# ==============================
# QUERY FUNCTION
# ==============================
def ask_question(query: str):
    global retriever, client

    if retriever is None:
        return "RAG system not initialized or no documents found."

    docs = retriever.invoke(query)

    if not docs:
        return "Answer not found in documents."

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a pharmaceutical AI assistant.

Answer ONLY using the provided context.
If not found, say: "Answer not found in documents."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    if client is None:
        return "ERROR: GROQ_API_KEY not configured"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a pharmaceutical AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM ERROR: {str(e)}"