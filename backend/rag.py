import os

from dotenv import load_dotenv

from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================
# LOAD ENV VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# CONFIGURE GROQ CLIENT
# ==========================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY not found in .env file"
    )

client = Groq(
    api_key=GROQ_API_KEY
)


# ==========================================
# LOAD PDF DOCUMENTS
# ==========================================

documents = []

pdf_folder = "documents"

if not os.path.exists(pdf_folder):

    raise FileNotFoundError(
        f"Folder '{pdf_folder}' not found"
    )

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(
            pdf_folder,
            file
        )

        print(f"Loading PDF: {file}")

        loader = PyPDFLoader(pdf_path)

        documents.extend(
            loader.load()
        )

print("PDFs Loaded Successfully")


# ==========================================
# SPLIT DOCUMENTS INTO CHUNKS
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(
    documents
)

print("Text Split Into Chunks")


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings Model Loaded")


# ==========================================
# CREATE FAISS VECTOR STORE
# ==========================================

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

print("FAISS Vector Store Created")


# ==========================================
# CREATE RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# ==========================================
# MAIN QUESTION ANSWERING FUNCTION
# ==========================================

def ask_question(query):

    try:

        print(f"\nQuestion: {query}")

        # ==========================================
        # RETRIEVE RELEVANT DOCUMENTS
        # ==========================================

        relevant_docs = retriever.invoke(query)

        if not relevant_docs:

            return "No relevant documents found."

        # ==========================================
        # BUILD CONTEXT
        # ==========================================

        context = "\n\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # ==========================================
        # CREATE PROMPT
        # ==========================================

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

        # ==========================================
        # GROQ API CALL
        # ==========================================

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

        # ==========================================
        # EXTRACT ANSWER
        # ==========================================

        answer = response.choices[0].message.content

        print("\nAnswer Generated Successfully")

        return answer

    except Exception as e:

        print("\nERROR:", str(e))

        return f"Error: {str(e)}"