from typing import TypedDict, Optional
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage
import sys
import os
import uuid
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from PersonalityRetriever.personality_wrapper import PersonalityWrapper
from KnowledgeRetriever.Neo4jAdapter import make_neo4j_retriever

# 1. Define your shared state
class AgentState(MessagesState):
    user_query: str
    knowledge_retrieved: Optional[str]   # filled by knowledge retrieval node
    personality_retrieved: Optional[str] # filled by personality retrieval node
    personality_profile: Optional[str]  # filled on initial 
    final_response: Optional[str]      # filled by synthesizer node


class personAgent: 
    def __init__(self, personIndex,fulltext_index):
        # Defining our personality wrapper
        self.personality = PersonalityWrapper(
            profile_path="../PersonalityRetriever/personality_summary.json",
            faiss_dir="../PersonalityRetriever/faiss_db"
        )

        # Defining our knowledge wrapper 
        # Get environment variables for knowledge wrapper 
        load_dotenv(dotenv_path="../.env", override=True)
        NEO4J_URI = os.environ.get("NEO4J_URI")
        NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
        NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

        self.knowlege_retrieval = make_neo4j_retriever(neo4j_uri = NEO4J_URI,
            neo4j_username = NEO4J_USERNAME,
            neo4j_password = NEO4J_PASSWORD, 
            index_name = personIndex, 
            fulltext_index_name = fulltext_index)
        
        # Build and compile the graph
        self.agent = StateGraph(AgentState)
        self._build_graph()


    #function to load in a personality profile
    def load_personality_node(self, state: AgentState) -> AgentState:
        # Only load if not already present to save resources
        if state.get("personality_profile"):
            return {}
        profile = self.personality.get_profile()           # uses module-level wrapper
        return {"personality_profile": profile}

    #function to do knowledge retrieval 
    def knowledge_retrieval_node(self, state: AgentState) -> dict:
        query = state["messages"][-1].content
        # Mocking your RAG call
        result = self.knowlege_retrieval.invoke(query)
        return {"knowledge_retrieved": result}

    #function to do personality retrieval
    def personality_retrieval_node(self, state: AgentState) -> dict:
        query = state["messages"][-1].content
        # Get specific tone or past interaction nuances
        nuance = self.personality.retrieve(query) 
        return {"personality_retrieved": nuance}


    def synthesizer_node(self, state: AgentState) -> dict:
            # Combine everything for the LLM
            context = f"""
            Personality Profile: {state['personality_profile']}
            Specific Nuance: {state['personality_retrieved']}
            Facts: {state['knowledge_retrieved']}
            """

            #haven't actually called an AI msg for now just simulating by concatenating everything 
            ai_msg = AIMessage(content=context)
            
            return {"messages": [ai_msg], "final_response": context}


    def _build_graph(self):
        print("_build_graph started")
        try:
            # Add Nodes
            self.agent.add_node("loader", self.load_personality_node)
            self.agent.add_node("knowledge_retriever", self.knowledge_retrieval_node)
            self.agent.add_node("personality_retriever", self.personality_retrieval_node)
            self.agent.add_node("synthesizer", self.synthesizer_node)

            # Define the "Train" (Edges)
            self.agent.add_edge(START, "loader")
            self.agent.add_edge("loader", "knowledge_retriever")
            self.agent.add_edge("knowledge_retriever", "personality_retriever")
            self.agent.add_edge("personality_retriever", "synthesizer")
            self.agent.add_edge("synthesizer", END)

            # Compile
            memory = MemorySaver()
            self.agent = self.agent.compile(checkpointer=memory)
            print("Graph compiled successfully:", self.agent)  # should not be None
        except Exception as e:
            print("ERROR in _build_graph:", e)
            raise  # re-raise so it doesn't silently swallow the error

    def new_thread_id(self) -> str:
        """Generate a unique thread ID for a new conversation."""
        return str(uuid.uuid4())

    def chat(self, user_input: str, thread_id: str) -> str:
        """Send a message and get a response. Pass the same thread_id to continue a conversation."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        return result["final_response"]