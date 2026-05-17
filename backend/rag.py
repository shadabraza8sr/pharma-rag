import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================
# GLOBAL VARIABLES (lazy init)
# ==============================
client = None
retriever = None


# ==============================
# LOAD PDFS SAFELY
# ==============================
def load_documents():
    documents = []
    pdf_folder = "documents"

    if not os.path.exists(pdf_folder):
        print("WARNING: documents folder not found")
        return documents

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            print(f"Loading PDF: {file}")

            loader = PyPDFLoader(path)
            documents.extend(loader.load())

    print("PDF loading completed")
    return documents


# ==============================
# INIT FUNCTION (CALLED ON STARTUP)
# ==============================
def init_rag():
    global client, retriever

    load_dotenv()

    # --------------------------
    # GROQ CLIENT
    # --------------------------
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        client = Groq(api_key=api_key)
    else:
        print("WARNING: GROQ_API_KEY missing")
        client = None

    # --------------------------
    # LOAD + SPLIT DOCUMENTS
    # --------------------------
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    # --------------------------
    # EMBEDDINGS
    # --------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------
    # VECTOR STORE (FAISS)
    # --------------------------
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print("RAG system initialized successfully")


# ==============================
# QUESTION ANSWERING FUNCTION
# ==============================
def ask_question(query: str):
    global retriever, client

    print(f"\nQuestion: {query}")

    if retriever is None:
        return "RAG system not initialized"

    docs = retriever.invoke(query)

    if not docs:
        return "Answer not found in documents."

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a pharmaceutical AI assistant.

Answer ONLY using the provided context.

If not found, say:
"Answer not found in documents."

================ CONTEXT ================
{context}

================ QUESTION ================
{query}

================ ANSWER ================
"""

    if client is None:
        return "ERROR: GROQ_API_KEY not configured"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a pharmaceutical AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content