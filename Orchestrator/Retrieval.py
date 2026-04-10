from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from PersonalityRetriever.personality_wrapper import PersonalityWrapper


# 1. Define your shared state
class AgentState(TypedDict):
    user_query: str
    knowledge_retrieved: Optional[str]   # filled by knowledge retrieval node
    personality_retrieved: Optional[str] # filled by personality retrieval node
    personality_profile: Optional[str]  # filled on initial 
    final_response: Optional[str]      # filled by synthesizer node