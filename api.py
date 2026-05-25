from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os
import tempfile
import base64
from pathlib import Path

from Agent.Agent import personAgent
from KnowledgeRetriever.Ingestor import KnowledgeIngestor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = None

def initialize_agent():
    global agent
    print("Attempting to initialize Agent...")
    try:
        agent = personAgent("essay_chunk_agentspace", "chunk_index")
        print("Agent initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize AI Agent: {e}")
        agent = None

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str
    thread_id: str

class ThreadResponse(BaseModel):
    thread_id: str

class IngestRequest(BaseModel):
    path: str
    index_name: Optional[str] = "essay_chunk_agentspace"
    file_data: Optional[str] = None  # base64-encoded file from frontend

@app.get("/health")
def health():
    return {"status": "ok", "agent_ready": agent is not None}

@app.post("/thread", response_model=ThreadResponse)
def create_thread():
    if not agent:
        raise HTTPException(status_code=503, detail="AI Agent not ready. Try ingesting data first.")
    thread_id = agent.new_thread_id()
    return {"thread_id": thread_id}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="AI Agent not ready. Try ingesting data first.")
    try:
        response = agent.chat(req.message, req.thread_id)
        return {"response": response, "thread_id": req.thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest_knowledge(req: IngestRequest):
    tmp_path = None
    try:
        if req.file_data:
            suffix = Path(req.path).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(base64.b64decode(req.file_data))
                tmp_path = tmp.name
            full_path = tmp_path
        else:
            full_path = req.path

        print(f"Starting ingestion for: {req.path} → {full_path}")
        ingestor = KnowledgeIngestor(index_name=req.index_name)
        ingestor.ingest(full_path)

        return {
            "status": "success",
            "message": f"Ingested {req.path}",
            "agent_ready": agent is not None
        }
    except Exception as e:
        print(f"Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    print("Starting API server and initializing agent...")
    initialize_agent()
    uvicorn.run(app, host="0.0.0.0", port=8000)