from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import sys
import os
import uuid
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

# Import LLM
from langchain_anthropic import ChatAnthropic

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from PersonalityRetriever.personality_wrapper import PersonalityWrapper
from KnowledgeRetriever.Neo4jAdapter import make_neo4j_retriever
from agent_schemas_and_wrapper import create_wrapper_payload

# 1. Define shared state
class AgentState(MessagesState):
    user_query: str
    knowledge_retrieved: Optional[List[Dict[str, Any]]]   # List of dicts with text, source, score
    personality_retrieved: Optional[List[str]]          # List of interview chunks
    personality_profile: Optional[Dict[str, Any]]       # JSON profile
    final_response: Optional[str]                       # Final synthesized text


class personAgent: 
    def __init__(self, personIndex: str, fulltext_index: str):
        # Load environment variables
        load_dotenv(dotenv_path=".env", override=True)
        ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
        NEO4J_URI = os.environ.get("NEO4J_URI")
        NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
        NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

        # Personality wrapper
        self.personality = PersonalityWrapper(
            profile_path="PersonalityRetriever/personality_summary.json",
            faiss_dir="PersonalityRetriever/faiss_db"
        )

        # Knowledge retriever
        self.knowlege_retrieval = make_neo4j_retriever(
            neo4j_uri=NEO4J_URI,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD, 
            index_name=personIndex, 
            fulltext_index_name=fulltext_index
        )
        
        # LLM for synthesis
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0.7,
        )

        # Build and compile the graph
        self.agent = StateGraph(AgentState)
        self._build_graph()


    def load_personality_node(self, state: AgentState) -> Dict[str, Any]:
        """Loads the static JSON profile into state."""
        if state.get("personality_profile"):
            return {}
        profile = self.personality.get_profile()
        return {"personality_profile": profile}

    def knowledge_retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieves factual chunks from Neo4j."""
        query = state["messages"][-1].content
        docs = self.knowlege_retrieval.invoke(query)

         # DEBUG - print retrieved docs
        print(f"\n=== RAG RETRIEVED {len(docs)} DOCS ===")
        for i, doc in enumerate(docs):
            print(f"\n--- Doc {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Score: {doc.metadata.get('rrf_score', 0.0)}")
            print(f"Content: {doc.page_content[:200]}...")  # first 200 chars
        print("=" * 50)
        
        knowledge_chunks = []
        for doc in docs:
            knowledge_chunks.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "score": doc.metadata.get("rrf_score", 0.0)
            })
        return {"knowledge_retrieved": knowledge_chunks}

    def personality_retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieves stylistic interview chunks from FAISS."""
        query = state["messages"][-1].content
        nuances = self.personality.retrieve(query) 
        return {"personality_retrieved": nuances}

    def synthesizer_node(self, state: AgentState) -> Dict[str, Any]:
        """Combines context and invokes LLM to generate the final response."""
        
        # 1. Prepare payload using the shared schema logic
        relevant_framework = "\n".join(state.get("personality_retrieved", []))
        payload = create_wrapper_payload(
            user_query=state["messages"][-1].content,
            parsed_profile=state["personality_profile"],
            retrieved_chunks=state.get("knowledge_retrieved", []),
            relevant_framework=relevant_framework
        )
        
        p_inst = payload['personality_instructions']
        k_cont = payload['knowledge_context']
        
        # 2. Build the Persona-Driven System Prompt
        system_prompt = f"""You are {p_inst['role']}. 

TUTORING PHILOSOPHY & TONE:
- Method: {p_inst['core_teaching_method']}
- Tone: {p_inst['tone_and_style']}
- Constraints: {', '.join(p_inst['strict_constraints'])}
- Signature Phrases: {', '.join(p_inst['phrases_to_use'])}

KNOWLEDGE CONTEXT (Use these specific examples from your own writing to help the student):
{chr(10).join('- ' + fact for fact in k_cont['retrieved_facts'])}

SOURCES:
{', '.join(k_cont['sources'])}

INSTRUCTIONS:
1. Stay strictly in character as {p_inst['role']}.
2. Apply your teaching method: {p_inst['core_teaching_method']}.
3. Use the 'KNOWLEDGE CONTEXT' above to ground your advice in real examples from your work.
4. If the context doesn't directly answer the query, use it to illustrate related concepts.
5. Never violate your constraints: {', '.join(p_inst['strict_constraints'])}.
"""

        # 3. Assemble message history and invoke
        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"][:-1],
            HumanMessage(content=state["messages"][-1].content)
        ]

        response = self.llm.invoke(messages)
        answer_text = response.content
        
        return {
            "messages": [AIMessage(content=answer_text)], 
            "final_response": answer_text
        }


    def _build_graph(self):
        # Define the nodes
        self.agent.add_node("loader", self.load_personality_node)
        self.agent.add_node("knowledge_retriever", self.knowledge_retrieval_node)
        self.agent.add_node("personality_retriever", self.personality_retrieval_node)
        self.agent.add_node("synthesizer", self.synthesizer_node)

        # Define the edges (Sequential execution)
        self.agent.add_edge(START, "loader")
        self.agent.add_edge("loader", "knowledge_retriever")
        self.agent.add_edge("knowledge_retriever", "personality_retriever")
        self.agent.add_edge("personality_retriever", "synthesizer")
        self.agent.add_edge("synthesizer", END)

        # Compile with memory for threading
        memory = MemorySaver()
        self.agent = self.agent.compile(checkpointer=memory)

    def new_thread_id(self) -> str:
        return str(uuid.uuid4())

    def chat(self, user_input: str, thread_id: str) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        return result["final_response"]
