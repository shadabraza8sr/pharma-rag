import os

from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# LOAD ENV VARIABLES (LOCAL ONLY)
# ==========================================

load_dotenv()

# ==========================================
# GROQ API KEY (SAFE FOR RENDER)
# ==========================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found. App will run but LLM will fail.")
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# SAFE PDF LOADING FUNCTION
# (IMPORTANT: prevents Render crash)
# ==========================================

def load_documents():
    documents = []
    pdf_folder = "documents"

    if not os.path.exists(pdf_folder):
        print(f"⚠️ Folder '{pdf_folder}' not found")
        return documents

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            print(f"Loading PDF: {file}")

            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    print("PDFs Loaded Successfully")
    return documents


# ==========================================
# LOAD + SPLIT DOCUMENTS
# ==========================================

documents = load_documents()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print("Text Split Into Chunks")


# ==========================================
# EMBEDDINGS MODEL
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings Model Loaded")


# ==========================================
# FAISS VECTOR STORE
# ==========================================

vectorstore = FAISS.from_documents(docs, embeddings)

print("FAISS Vector Store Created")


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# ==========================================
# MAIN QA FUNCTION
# ==========================================

def ask_question(query):
    try:
        print(f"\nQuestion: {query}")

        # Retrieve docs
        relevant_docs = retriever.invoke(query)

        if not relevant_docs:
            return "Answer not found in documents."

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are a pharmaceutical AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
reply exactly:

"Answer not found in documents."

================ CONTEXT ================
{context}

================ QUESTION ================
{query}

================ ANSWER ================
"""

        # If API key missing
        if client is None:
            return "Error: GROQ_API_KEY not configured on server."

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

        answer = response.choices[0].message.content

        print("Answer Generated Successfully")

        return answer

    except Exception as e:
        print("ERROR:", str(e))
        return f"Error: {str(e)}"