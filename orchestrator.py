"""
AgentSpace Orchestrator
Connects the Personality module (interview/FAISS) and Knowledge module (essays/Neo4j)
into a single tutoring pipeline.

Setup:
  1. .env in repo root with NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GOOGLE_API_KEY
  2. FAISS db exists in Personality/faiss_db/
  3. Essays loaded into Neo4j
  4. Run: python orchestrator.py
"""

import os
import sys
import json
import numpy as np
from typing import List
from dotenv import load_dotenv

# Load secrets from .env so they don't end up in the repo
load_dotenv(dotenv_path=".env", override=True)

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Neo4jVector
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


# Copied from Knowledge/Neo4jGraph.ipynb
# If Dylan changes the embedding model, update the model name here too
class GemmaEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return [self._normalize(v).tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        v = self.model.encode([text], convert_to_numpy=True)[0]
        return self._normalize(v).tolist()


def load_personality_prompt(json_path: str) -> str:
    """Builds the system prompt from personality_summary.json."""
    with open(json_path, "r") as f:
        profile = json.load(f)

    prompt = f"""You are {profile['agent_profile']['name']}, a {profile['agent_profile']['role']} with these characteristics:

BACKGROUND:
- Age {profile['agent_profile']['age']}, born in {profile['demographics']['birthplace']} ({profile['demographics']['birth_year']}), raised in {profile['demographics']['raised_in']}
- {profile['education']['undergraduate']['degree']} from {profile['education']['undergraduate']['university']} (minor in {profile['education']['undergraduate']['minor']})
- Currently pursuing {profile['education']['graduate']['degree']} at {profile['education']['graduate']['university']}
- Worked as writing tutor at {profile['work_experience']['tutoring']['employer']} for {profile['work_experience']['tutoring']['duration']}

TUTORING PHILOSOPHY:
- Core Method: {profile['tutoring_philosophy']['core_method']}
- Rationale: {profile['tutoring_philosophy']['rationale']}
- Focus on: {', '.join(profile['tutoring_philosophy']['priorities'])} over grammar
- Use "{profile['tutoring_philosophy']['feedback_method']}"
- Sessions are student-led

SIGNATURE PHRASES:
{chr(10).join('- "' + phrase + '"' for phrase in profile['signature_phrases'])}

TONE & STYLE:
- {profile['personality_traits']['tone']}
- {profile['personality_traits']['humor']}
- Act like a "{profile['personality_traits']['self_description']}"
- {profile['personality_traits']['approach']}

CONSTRAINTS:
- Cannot: {', '.join(profile['constraints']['cannot_do'])}
- Always refer to rubric when needed

When responding to students:
1. Let them explain their concern first
2. Ask guiding questions rather than giving direct answers
3. Encourage them to identify issues themselves
4. Provide frameworks/concepts when needed
5. End with actionable takeaways

You also have access to your own essays. When relevant, you can reference
specific examples from your writing to help students understand concepts.
"""
    return prompt


class Orchestrator:
    """
    Searches both FAISS (interview) and Neo4j (essays) on each query,
    merges the results, and generates a response through Gemini
    using Grant's personality.
    """

    def __init__(
        self,
        personality_db_path: str = "Personality/faiss_db",
        personality_json_path: str = "Personality/personality_summary.json",
        neo4j_index_name: str = "essay_chunk_agentspace",
        personality_k: int = 4,   # matches grant_tutor_rag.py
        knowledge_k: int = 6,    # from RAGAS optimization sweep
    ):
        self.personality_k = personality_k
        self.knowledge_k = knowledge_k
        self.chat_history = []

        # Grant's persona prompt (tells Gemini how to behave)
        print("Loading personality profile...")
        self.system_prompt = load_personality_prompt(personality_json_path)

        # Personality retriever - searches Grant's interview transcript
        # If Aryan changes the embedding model in grant_tutor_rag.py, update it here
        print("Loading Personality retriever (FAISS)...")
        personality_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        personality_vectorstore = FAISS.load_local(
            personality_db_path,
            personality_embeddings,
            allow_dangerous_deserialization=True,
        )
        self.personality_retriever = personality_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.personality_k},
        )

        # Knowledge retriever - searches Grant's actual essays
        # If Dylan changes the embedding model in Neo4jGraph.ipynb, update it here
        print("Loading Knowledge retriever (Neo4j)...")
        knowledge_embeddings = GemmaEmbeddings("google/embeddinggemma-300m")
        self.knowledge_vectorstore = Neo4jVector(
            embedding=knowledge_embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=neo4j_index_name,
            node_label="Chunk",
            text_node_property="text",
            embedding_node_property="embedding",
        )
        self.knowledge_retriever = self.knowledge_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.knowledge_k},
        )

        # LLM for generation (same model/settings as grant_tutor_rag.py)
        print("Setting up Gemini...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.7,
            convert_system_message_to_human=True,
        )

        print("Orchestrator ready!\n")

    def retrieve_personality_context(self, query: str) -> List[str]:
        """Search the interview transcript for how Grant would approach this."""
        docs = self.personality_retriever.invoke(query)
        return [doc.page_content for doc in docs]

    def retrieve_knowledge_context(self, query: str) -> List[str]:
        """Search Grant's essays for relevant examples."""
        docs = self.knowledge_retriever.invoke(query)
        return [doc.page_content for doc in docs]

    def merge_context(self, personality_chunks: List[str], knowledge_chunks: List[str]) -> str:
        """
        Combine interview and essay results into one labeled context block.
        Labels help the LLM distinguish Grant's tutoring advice from his actual writing.
        """
        sections = []

        if personality_chunks:
            sections.append("=== Grant's tutoring approach (from interview) ===")
            for i, chunk in enumerate(personality_chunks, 1):
                sections.append(f"[Interview excerpt {i}]\n{chunk}")

        if knowledge_chunks:
            sections.append("\n=== Grant's essay examples (from his writing) ===")
            for i, chunk in enumerate(knowledge_chunks, 1):
                sections.append(f"[Essay excerpt {i}]\n{chunk}")

        return "\n\n".join(sections)

    def answer(self, query: str) -> str:
        """
        Full pipeline: retrieve from both sources, merge context,
        generate response with Grant's personality, save to history.
        """
        # Retrieve from both databases
        personality_chunks = self.retrieve_personality_context(query)
        knowledge_chunks = self.retrieve_knowledge_context(query)
        merged_context = self.merge_context(personality_chunks, knowledge_chunks)

        # Build message list: system prompt + chat history + current question
        messages = [
            SystemMessage(content=self.system_prompt),
            *self.chat_history,
            HumanMessage(
                content=f"Relevant context:\n{merged_context}\n\nStudent question: {query}"
            ),
        ]

        # Generate and save to history
        response = self.llm.invoke(messages)
        answer_text = response.content

        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=answer_text))

        # Keep last 5 turns to stay within Gemini's context limit
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        return answer_text

    def clear_history(self):
        self.chat_history = []
        print("Conversation history cleared.")


def main():
    missing = []
    for var in ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GOOGLE_API_KEY"]:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(f"ERROR: Missing env variables: {', '.join(missing)}")
        print("Add them to .env in the repo root.")
        sys.exit(1)

    agent = Orchestrator()

    print("=" * 60)
    print("Grant's Writing Tutoring Session")
    print("=" * 60)
    print("\nHi! I'm Grant. What would you like to work on today?")
    print("(Type 'exit' to end, 'clear' to restart conversation)\n")

    while True:
        student_input = input("You: ").strip()

        if student_input.lower() in ["exit", "quit", "bye"]:
            print("\nGrant: Great session! Good luck with your essay!")
            break

        if student_input.lower() == "clear":
            agent.clear_history()
            print("\nGrant: Fresh start! What would you like to work on?")
            continue

        if not student_input:
            continue

        try:
            response = agent.answer(student_input)
            print(f"\nGrant: {response}\n")
        except Exception as e:
            print(f"\nError: {e}")
            print("Let's try that again.\n")


if __name__ == "__main__":
    main()
