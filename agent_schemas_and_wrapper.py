import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# 1. Define Generalized Personality Schema
class PersonalityInstructions(BaseModel):
    role: str = Field(
        ..., 
        description="The persona's professional role or function (e.g., 'Writing Tutor', 'Tax Lawyer')"
    )
    tone_and_style: str = Field(
        ..., 
        description="Guidance on the tone and interaction style (e.g., 'casual, friendly', 'formal, direct')"
    )
    core_teaching_method: str = Field(
        ..., 
        description="The primary operational directive or methodology to use (e.g., 'Socratic questioning')"
    )
    strict_constraints: List[str] = Field(
        default_factory=list, 
        description="Absolute rules the persona must follow and what they cannot do (e.g., ['Edit essays directly'])"
    )
    phrases_to_use: List[str] = Field(
        default_factory=list, 
        description="Signature phrases to inject the unique voice of the persona"
    )
    relevant_framework: Optional[str] = Field(
        None, 
        description="If a specific framework applies to the query, include its description here"
    )

# 2. Define Generalized Knowledge Schema
class KnowledgeContext(BaseModel):
    retrieved_facts: List[str] = Field(
        default_factory=list, 
        description="The raw text strings of the highest-scoring chunks relevant to the user's query"
    )
    sources: List[str] = Field(
        default_factory=list, 
        description="The origin metadata of the chunks (e.g., file names or page numbers) for citation"
    )
    highest_confidence_score: float = Field(
        ..., 
        ge=0.0, le=1.0, 
        description="The highest similarity/relevance score from the retrieved chunks"
    )

# 3. Define the Main Wrapper Payload
class WrapperPayload(BaseModel):
    user_query: str = Field(
        ..., 
        description="The exact user input or question"
    )
    personality_instructions: PersonalityInstructions = Field(
        ..., 
        description="The extracted instructions defining how the persona should respond"
    )
    knowledge_context: KnowledgeContext = Field(
        ..., 
        description="The retrieved facts and sources to inform the content of the response"
    )

def create_wrapper_payload(
    user_query: str, 
    parsed_profile: Dict[str, Any], 
    retrieved_chunks: List[Dict[str, Any]],
    relevant_framework: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs the WrapperPayload by dynamically mapping raw profile data and retrieved chunks.
    
    Args:
        user_query: The exact user query.
        parsed_profile: The loaded and parsed personality JSON dict.
        retrieved_chunks: A list of dicts from the Knowledge Retriever, expected to have 
                          'text', 'source', and 'score' keys.
        relevant_framework: Optional extracted framework string relevant to the query.
        
    Returns:
        dict: The serialized Synthesizer payload.
    """
    
    # 1. Map Personality Instructions dynamically
    # Note: These paths assume a structure similar to our standard personality_summary.json
    role = parsed_profile.get('agent_profile', {}).get('role', 'Helpful Assistant')
    
    tone_dict = parsed_profile.get('personality_traits', {})
    tone_and_style = f"{tone_dict.get('tone', 'neutral')}. Self-described as: {tone_dict.get('self_description', 'an assistant')}."
    
    core_method = parsed_profile.get('tutoring_philosophy', {}).get('core_method', 'Directly answer the question.')
    constraints = parsed_profile.get('constraints', {}).get('cannot_do', [])
    phrases = parsed_profile.get('signature_phrases', [])
    
    personality_instructions = PersonalityInstructions(
        role=role,
        tone_and_style=tone_and_style,
        core_teaching_method=core_method,
        strict_constraints=constraints,
        phrases_to_use=phrases,
        relevant_framework=relevant_framework
    )
    
    # 2. Map Knowledge Context dynamically
    facts = []
    sources = []
    highest_score = 0.0
    
    for chunk in retrieved_chunks:
        if 'text' in chunk:
            facts.append(chunk['text'])
        if 'source' in chunk and chunk['source'] not in sources:
            sources.append(chunk['source'])
        if 'score' in chunk and chunk['score'] > highest_score:
            highest_score = float(chunk['score'])
            
    knowledge_context = KnowledgeContext(
        retrieved_facts=facts,
        sources=sources,
        highest_confidence_score=highest_score
    )
    
    # 3. Create the Payload
    payload = WrapperPayload(
        user_query=user_query,
        personality_instructions=personality_instructions,
        knowledge_context=knowledge_context
    )
    
    return payload.model_dump()

# Verification & Testing
if __name__ == "__main__":
    print("Testing generalized wrapper payload generation...\n")
    
    # Mock dynamic data from a parsed personality summary
    mock_profile = {
        "agent_profile": {"role": "Writing Tutor"},
        "personality_traits": {"tone": "casual, friendly", "self_description": "therapist for writing"},
        "tutoring_philosophy": {"core_method": "Socratic questioning - NEVER tell students what to fix directly"},
        "constraints": {"cannot_do": ["Edit essays directly", "Write papers for students"]},
        "signature_phrases": ["How do you feel about that sentence?", "Let's break this down together"]
    }
    
    # Mock dynamic data from the knowledge retriever
    mock_chunks = [
        {"text": "We can define 5 mutationally prone regions (MPRs) in the gene...", "source": "Environ and Mol Mutagen - 2025.pdf", "score": 0.92},
        {"text": "The region from 484 to 507 has all the earmarks of an MPR...", "source": "EM-65-338.pdf", "score": 0.88}
    ]
    
    mock_query = "What is a mutationally prone region and how do I write my thesis about it?"
    mock_framework = "Thesis Structure: Subject + Argument + Focus"
    
    # Generate the payload
    final_payload = create_wrapper_payload(
        user_query=mock_query,
        parsed_profile=mock_profile,
        retrieved_chunks=mock_chunks,
        relevant_framework=mock_framework
    )
    
    # Output the result
    print("Generated Wrapper Payload:")
    print(json.dumps(final_payload, indent=2))
    print("\n✓ Verification successful.")
