import os
import glob
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_environment():
    """Load environment variables from .env file."""
    # Check multiple possible locations for the .env file
    script_dir = os.path.dirname(__file__)
    possible_paths = [
        os.path.join(script_dir, '.env'),            # Knowledge/.env
        os.path.join(script_dir, '..', '.env'),       # AgentSpace/.env
        os.path.abspath('.env')                       # Current Working Directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            load_dotenv(path)
            print(f"DEBUG: Loaded environment from {path}")
            return True
    
    load_dotenv() # Fallback to default behavior
    return False

def create_constraints(graph):
    """Create Neo4j constraints and indexes for optimal performance."""
    # Constraint for Chunk uniqueness and faster lookups
    graph.query("CREATE CONSTRAINT chunk_id_source IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.chunk_id, c.source) IS UNIQUE")
    # Constraint for Document uniqueness
    graph.query("CREATE CONSTRAINT doc_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE")

def ingest_documents(pdf_pattern):
    """Load and chunk PDF documents."""
    docs = []
    files = glob.glob(pdf_pattern)
    print(f"Found {len(files)} documents in {pdf_pattern}")
    
    for file in files:
        loader = PyPDFLoader(file)
        loaded_docs = loader.load()
        for d in loaded_docs:
            d.metadata['source'] = os.path.basename(file)
        docs.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    chunks = splitter.split_documents(docs)
    # Assign chunk_ids
    for idx, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = idx
        
    return chunks

def build_graph_structure(graph, chunks):
    """Create NEXT_CHUNK relationships and Document nodes."""
    chunks_by_source = {}
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        if source not in chunks_by_source:
            chunks_by_source[source] = []
        chunks_by_source[source].append(chunk)

    print("Creating NEXT_CHUNK relationships...")
    relationship_count = 0

    for source, source_chunks in chunks_by_source.items():
        source_chunks.sort(key=lambda x: x.metadata.get('chunk_id', 0))

        batch_data = []
        for i in range(len(source_chunks) - 1):
            batch_data.append({
                "current_id": source_chunks[i].metadata['chunk_id'],
                "next_id": source_chunks[i + 1].metadata['chunk_id']
            })

        query = """
        UNWIND $batch as row
        MATCH (c1:Chunk {chunk_id: row.current_id, source: $source})
        MATCH (c2:Chunk {chunk_id: row.next_id, source: $source})
        MERGE (c1)-[:NEXT_CHUNK]->(c2)
        """
        graph.query(query, params={"batch": batch_data, "source": source})
        relationship_count += len(batch_data)

    print(f"Created {relationship_count} NEXT_CHUNK relationships")
    print("Creating Document nodes...")

    for source, source_chunks in chunks_by_source.items():
        doc_query = """
        MERGE (d:Document {name: $source})
        SET d.chunk_count = $chunk_count
        """
        graph.query(doc_query, params={"source": source, "chunk_count": len(source_chunks)})

        # Link chunks to document
        link_query = """
        MATCH (d:Document {name: $source})
        MATCH (c:Chunk {source: $source})
        MERGE (c)-[:PART_OF]->(d)
        """
        graph.query(link_query, params={"source": source})

    print(f"Created {len(chunks_by_source)} Document nodes")

def get_chunk_context(graph, chunk_id, source, context_size=1):
    """Get neighboring chunks using NEXT_CHUNK relationships"""
    query = f"""
    MATCH (c:Chunk {{chunk_id: $chunk_id, source: $source}})
    WITH c
    OPTIONAL MATCH (before:Chunk)-[:NEXT_CHUNK*1..{int(context_size)}]->(c)
    WITH c, before ORDER BY before.chunk_id ASC
    WITH c, collect(before.text) as before_texts
    OPTIONAL MATCH (c)-[:NEXT_CHUNK*1..{int(context_size)}]->(after:Chunk)
    WITH c, before_texts, after ORDER BY after.chunk_id ASC
    RETURN
        c.text as current_text,
        c.chunk_id as current_id,
        before_texts,
        collect(after.text) as after_texts
    """

    result = graph.query(query, params={"chunk_id": chunk_id, "source": source})
    return result[0] if result else None

def hybrid_search(vectorstore, graph, query_text, k=4, context_size=1):
    """Combine vector search with graph context"""
    results = vectorstore.similarity_search_with_score(query_text, k=k)

    hybrid_results = []
    for doc, score in results:
        metadata = doc.metadata
        chunk_id = metadata.get('chunk_id')
        source = metadata.get('source')
        context = get_chunk_context(graph, chunk_id, source, context_size) if chunk_id is not None else None

        hybrid_results.append({
            'text': doc.page_content,
            'score': score,
            'chunk_id': chunk_id,
            'source': doc.metadata.get('source', 'unknown'),
            'context': context
        })

    return hybrid_results

def main():
    load_environment()
    
    # Get API key from environment variables
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv('GEMMA_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Google API Key not found. Please ensure GOOGLE_API_KEY, GEMMA_KEY, or GEMINI_API_KEY is set in your .env file.")

    # Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    
    # Ingest Data
    # Check for local PDFs or fallback to /content/
    pdf_path = "*.pdf" if glob.glob("*.pdf") else "/content/*.pdf"
    chunks = ingest_documents(pdf_path)
    
    if not chunks:
        print("No documents found to process.")
        return

    # Get Neo4j credentials
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    missing = []
    if not neo4j_uri: missing.append("NEO4J_URI")
    if not neo4j_user: missing.append("NEO4J_USERNAME")
    if not neo4j_password: missing.append("NEO4J_PASSWORD")
    if missing:
        raise ValueError(f"Missing Neo4j credentials in .env: {', '.join(missing)}")

    # Setup Graph Connection first to clear data and set constraints
    print("Initializing Graph Connection...")
    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password
    )
    
    # Clear existing data for a clean "redo"
    print("Clearing existing data...")
    graph.query("MATCH (n:Chunk) DETACH DELETE n")
    graph.query("MATCH (n:Document) DETACH DELETE n")
    
    create_constraints(graph)

    # Setup Vector Store (Nodes)
    print("Initializing Vector Store...")
    vectorstore = Neo4jVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password,
        index_name="essay_chunk_gemini",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding"
    )

    build_graph_structure(graph, chunks)

    # Test Search
    query = "What specific argument does the author make about orchids"
    print(f"\n=== HYBRID SEARCH TEST: '{query}' ===")
    
    results = hybrid_search(vectorstore, graph, query, k=4, context_size=1)
    
    for i, result in enumerate(results, 1):
        print(f"\n---- Hybrid Result {i} ----")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Text: {result['text'][:200]}...")
        if result['context']:
            ctx = result['context']
            print(f"Context: {len(ctx.get('before_texts', []))} before, {len(ctx.get('after_texts', []))} after chunks")

if __name__ == "__main__":
    main()
