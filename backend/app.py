import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from rag import init_rag, ask_question

app = FastAPI()

# CORS
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
def startup_event():
    print("Initializing RAG system...")
    init_rag()
    print("RAG system ready")


@app.get("/")
def root():
    return {"status": "Backend running"}


@app.post("/ask")
def ask(data: Query):
    return {"answer": ask_question(data.question)}


# ✅ IMPORTANT: Render port binding
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)