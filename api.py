from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os

# Import from the renamed Agent folder and the new KnowledgeRetriever folder
from Agent.Agent import personAgent
from KnowledgeRetriever.Ingestor import KnowledgeIngestor

app = FastAPI()

# Allow the frontend to connect from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent variable (starts as None)
agent = None

def initialize_agent():
    global agent
    print("Attempting to initialize Agent...")
    try:
        # Initialize the real AI-driven agent
        agent = personAgent("essay_chunk_agentspace", "chunk_index")
        print("Agent initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize AI Agent: {e}")
        # We don't raise here to keep the API alive, but chat will fail later
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
         raise HTTPException(status_code=533, detail="AI Agent not ready. Try ingesting data first.")
    try:
        response = agent.chat(req.message, req.thread_id)
        return {"response": response, "thread_id": req.thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest_knowledge(req: IngestRequest):
    try:
        print(f"Starting ingestion for: {req.path}")
        ingestor = KnowledgeIngestor(index_name=req.index_name)
        ingestor.ingest(req.path)
        print("Ingestion complete. Initializing Agent...")
        
        # As soon as ingestion is done, create the agent
        initialize_agent()
        
        return {
            "status": "success", 
            "message": f"Ingested {req.path}",
            "agent_ready": agent is not None
        }
    except Exception as e:
        print(f"Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start the server WITHOUT the agent initially
    print("Starting API server (Agent will be initialized after ingestion)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
