from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Orchestrator.Agent import personAgent

app = FastAPI()

# allow the frontend to connect from any origin
# tighten this to a specific URL before deploying
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# create the agent once on startup
agent = personAgent("essay_chunk_agentspace", "chunk_index")


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    response: str
    thread_id: str


class ThreadResponse(BaseModel):
    thread_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/thread", response_model=ThreadResponse)
def create_thread():
    thread_id = agent.new_thread_id()
    return {"thread_id": thread_id}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        response = agent.chat(req.message, req.thread_id)
        return {"response": response, "thread_id": req.thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)