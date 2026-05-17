from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

from rag import init_rag, ask_question

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Request Schema ----------------
class Query(BaseModel):
    question: str


# ---------------- Health Check Route ----------------
@app.get("/")
def home():
    return {"status": "Backend is running 🚀"}


# ---------------- Startup Event ----------------
@app.on_event("startup")
def startup_event():
    print("Initializing RAG system...")
    init_rag()
    print("RAG system ready ✅")


# ---------------- API Endpoint ----------------
@app.post("/ask")
def ask(data: Query):
    try:
        answer = ask_question(data.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}