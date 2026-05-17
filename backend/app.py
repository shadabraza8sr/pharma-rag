from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import init_rag, ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str


@app.on_event("startup")
def startup():
    print("Initializing RAG...")
    init_rag()
    print("RAG ready")


@app.post("/ask")
def ask(data: Query):
    return {"answer": ask_question(data.question)}