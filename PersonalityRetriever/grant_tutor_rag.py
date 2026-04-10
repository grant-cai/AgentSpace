"""
Grant Writing Tutor RAG System
Uses Gemini and FAISS.
"""

import os
from typing import List, Dict
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json

def load_interview_transcript(file_path: str) -> str:
    """Load the markdown interview transcript"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_transcript(transcript: str) -> List[Document]:
    """
    Split transcript by markdown headers to preserve Q&A structure.
    Each chunk will be a Q&A pair.
    """
    headers_to_split = [
        ("##", "part"),
        ("###", "subsection"),
        ("####", "question")
    ]
    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False
    )
    
    chunks = splitter.split_text(transcript)
    return chunks

def create_vector_store(chunks: List[Document], persist_directory: str = "./faiss_db"):
    """
    Create a vector database from the interview chunks.
    Uses HuggingFace embeddings
    Uses FAISS for vector storage
    """
    print("Loading embedding model (first time may take a minute to download)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("Creating embeddings for interview chunks...")
    # Create vector store using FAISS
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    vectorstore.save_local(persist_directory)
    
    print(f"Created vector store with {len(chunks)} chunks")
    return vectorstore

def load_existing_vector_store(persist_directory: str = "./faiss_db"):
    """Load an existing vector store from disk"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def load_agent_profile(json_path: str = "personality_summary.json") -> str:
    """
    Load Grant's personality profile from JSON and convert to prompt text.
    This combines structured data with natural language for the system prompt.
    """
    with open(json_path, 'r') as f:
        profile = json.load(f)
    
    # Build system prompt from JSON data
    prompt = f"""You are {profile['agent_profile']['name']}, a {profile['agent_profile']['role']} with these characteristics:

BACKGROUND:
- Age {profile['agent_profile']['age']}, born in {profile['demographics']['birthplace']} ({profile['demographics']['birth_year']}), raised in {profile['demographics']['raised_in']}
- {profile['education']['undergraduate']['degree']} from {profile['education']['undergraduate']['university']} (minor in {profile['education']['undergraduate']['minor']})
- Currently pursuing {profile['education']['graduate']['degree']} at {profile['education']['graduate']['university']}
- Worked as writing tutor at {profile['work_experience']['tutoring']['employer']} for {profile['work_experience']['tutoring']['duration']}
- {profile['demographics']['political_views']}

CORE VALUES:
- Primary value: {profile['core_values']['primary']}
- Priorities: {', '.join(profile['core_values']['priorities'])}

TUTORING PHILOSOPHY:
- **Core Method**: {profile['tutoring_philosophy']['core_method']}
- Rationale: {profile['tutoring_philosophy']['rationale']}
- Focus on: {', '.join(profile['tutoring_philosophy']['priorities'])} over grammar (unless pattern exists)
- Sessions are {profile['tutoring_philosophy']['student_led'] and 'student-led' or 'tutor-led'}
- Use "{profile['tutoring_philosophy']['feedback_method']}"
- Session focus: {profile['tutoring_philosophy']['session_focus']}

KEY FRAMEWORKS YOU TEACH:
- **Thesis Structure**: {' + '.join(profile['key_frameworks']['thesis_structure']['components'])}
  ({profile['key_frameworks']['thesis_structure']['rule']})
- **Essay Structure**: {profile['key_frameworks']['essay_structure']['model']} model ({profile['key_frameworks']['essay_structure']['description']})
- **Evidence**: {profile['key_frameworks']['evidence']['requirement']}
- **Body Paragraphs**: {profile['key_frameworks']['body_paragraphs']['structure']}

SIGNATURE PHRASES:
{chr(10).join('- "' + phrase + '"' for phrase in profile['signature_phrases'])}

TONE & STYLE:
- {profile['personality_traits']['tone']}
- {profile['personality_traits']['humor']}
- Act like a "{profile['personality_traits']['self_description']}"
- {profile['personality_traits']['approach']}

HANDLING DIFFERENT STUDENT TYPES:
- Lacking confidence: {profile['handling_student_types']['lacking_confidence']}
- Defensive: {profile['handling_student_types']['defensive']}
- Overconfident: {profile['handling_student_types']['overconfident']}
- Passive: {profile['handling_student_types']['passive']}
- Struggling: {profile['handling_student_types']['struggling']}

CONSTRAINTS:
- Cannot: {', '.join(profile['constraints']['cannot_do'])}
- Emergency situations: {profile['constraints']['emergency_situations']}
- Always refer to rubric when needed

VIEWS ON WRITING:
- Academic writing rules: {profile['views_on_writing']['rules']['academic_writing']}
- Creativity: {profile['views_on_writing']['rules']['creativity_within_structure']}
- Common misconceptions: {profile['views_on_writing']['common_misconceptions']}

When responding to students:
1. Let them explain their concern first
2. Ask guiding questions rather than giving direct answers
3. Encourage them to identify issues themselves
4. Provide frameworks/concepts when needed
5. End with actionable takeaways
"""
    return prompt

GRANT_PROFILE = load_agent_profile()

def create_rag_chain(vectorstore):
    """
    Create a RAG chain using modern LCEL (LangChain Expression Language).
    """
    
    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",  # Cheaper and faster
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", GRANT_PROFILE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context from Grant's interview:\n{context}\n\nStudent question: {question}")
    ])
    
    # Create chain using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def prepare_inputs(inputs):
        """Extract question string and pass it to retriever"""
        question = inputs["question"]
        return question
    
    chain = (
        {
            "context": prepare_inputs | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

class GrantTutorAgent:
    """Main agent class - manages conversation history manually"""
    
    def __init__(self, transcript_path: str, rebuild_db: bool = False):
        """
        Initialize Grant's tutoring agent
        
        Args:
            transcript_path: Path to interview_transcript.md
            rebuild_db: If True, rebuild vector store from scratch
        """
        self.persist_dir = "./faiss_db"
        self.chat_history = []
        
        if rebuild_db or not os.path.exists(self.persist_dir):
            print("Building vector database...")
            transcript = load_interview_transcript(transcript_path)
            chunks = chunk_transcript(transcript)
            self.vectorstore = create_vector_store(chunks, self.persist_dir)
        else:
            print("Loading existing vector database...")
            self.vectorstore = load_existing_vector_store(self.persist_dir)
        
        print("Creating RAG chain...")
        self.chain = create_rag_chain(self.vectorstore)
        print("Grant Tutor Agent ready!")
    
    def chat(self, student_question: str) -> str:
        """
        Send a question to Grant and get a response
        
        Args:
            student_question: The student's question
            
        Returns:
            Grant's response as a string
        """
        response = self.chain.invoke({
            "question": student_question,
            "chat_history": self.chat_history
        })
        
        self.chat_history.append(HumanMessage(content=student_question))
        self.chat_history.append(AIMessage(content=response))
        
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.chat_history = []
        print("✓ Conversation history cleared")
    
    def interactive_session(self):
        """Run an interactive tutoring session"""
        print("\n" + "="*60)
        print("Grant's Writing Tutoring Session")
        print("="*60)
        print("\nHi! I'm Grant. What would you like to work on today?")
        print("(Type 'exit' to end, 'clear' to restart conversation)\n")
        
        while True:
            student_input = input("\nYou: ").strip()
            
            if student_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGrant: Great session! Good luck with your essay!")
                break
            
            if student_input.lower() == 'clear':
                self.clear_history()
                print("\nGrant: Okay, fresh start! What would you like to work on?")
                continue
            
            if not student_input:
                continue
            
            try:
                response = self.chat(student_input)
                print(f"\nGrant: {response}")
            except Exception as e:
                print(f"\nError: {e}")
                print("Let's try that again.")

def main():
    """Example usage"""
    
    # Set your Google API key
    # Get from: https://makersuite.google.com/app/apikey
    file_path = 'api_key.txt'

    with open(file_path, 'r') as f:
        api_key = f.read()
    os.environ["GOOGLE_API_KEY"] = api_key
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: Please set GOOGLE_API_KEY environment variable")
        print("Get your key from: https://makersuite.google.com/app/apikey")
        return
    
    agent = GrantTutorAgent(
        transcript_path="interview_transcript.md",
        rebuild_db=True  # Set to False after first run
    )
    
    # Interactive Session
    print("\n" + "="*60)
    print("Interactive Session")
    print("="*60)
    agent.interactive_session()

if __name__ == "__main__":
    main()


# =============================================================================
# INSTALLATION REQUIREMENTS
# =============================================================================
"""
Install required packages:

pip install langchain langchain-google-genai langchain-huggingface langchain-community langchain-text-splitters langchain-core faiss-cpu google-generativeai sentence-transformers

Or one line:
pip install langchain langchain-google-genai langchain-huggingface langchain-community langchain-text-splitters langchain-core faiss-cpu google-generativeai sentence-transformers

API Key needed:
Only Google API key (for Gemini chat): https://makersuite.google.com/app/apikey

Set environment variable:
export GOOGLE_API_KEY='your-google-key-here'

Or in Python:
import os
os.environ["GOOGLE_API_KEY"] = "your-google-key-here"

WHY THIS SETUP?
- Gemini for chat: Cheap, fast, great quality
- HuggingFace for embeddings: FREE, runs locally, no API limits!
- Perfect for development and small-scale deployment
"""

"""
QUICK START:
1. Get Google API key from https://makersuite.google.com/app/apikey
2. Save interview_transcript.md in same directory
3. Set GOOGLE_API_KEY environment variable
4. Run: python grant_tutor_rag.py

TESTING QUESTIONS:
- "My thesis is vague. Can you help me?"
- "How do I use evidence properly?"
- "My essay jumps around. What should I do?"
- "I don't understand what makes a good argument."

The agent will respond with guiding questions, just like Grant would!
"""