from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from rag import init_rag, ask_question

app = FastAPI()

# CORS (safe for frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class Query(BaseModel):
    question: str


# ✅ Load RAG system AFTER server starts (VERY IMPORTANT for Render)
@app.on_event("startup")
def startup_event():
    print("Initializing RAG system...")
    init_rag()
    print("RAG system ready")


# API endpoint
@app.post("/ask")
def ask(data: Query):
    answer = ask_question(data.question)
    return {"answer": answer}