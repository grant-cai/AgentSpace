# test_api.py
import requests

BASE_URL = "http://localhost:8000"

# 1. Health check
print("=== Health Check ===")
r = requests.get(f"{BASE_URL}/health")
print(r.json())

# 2. Ingest (skip if already ingested)
print("\n=== Ingest ===")
r = requests.post(f"{BASE_URL}/ingest", json={
    "path": "KnowledgeRetriever/BodyParagraphs.pdf",  # change to your actual path
    "index_name": "essay_chunk_agentspace"
})
print(r.json())

# 3. Create thread
print("\n=== Create Thread ===")
r = requests.post(f"{BASE_URL}/thread")
print(r.json())
thread_id = r.json()["thread_id"]

# 4. Chat
print("\n=== Chat ===")
r = requests.post(f"{BASE_URL}/chat", json={
    "message": "Tell me about yourself",
    "thread_id": thread_id
})
print(r.json())

# 5. Follow-up message (same thread = memory)
print("\n=== Follow-up ===")
r = requests.post(f"{BASE_URL}/chat", json={
    "message": "What is your teaching style?",
    "thread_id": thread_id
})
print(r.json())