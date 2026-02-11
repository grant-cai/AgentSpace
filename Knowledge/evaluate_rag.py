import os
import json
import math
import time
import re
from Aryan_Knowledge_RAG import hybrid_search, load_environment
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

def calculate_metrics(retrieved_ids, expected_ids, k):
    """Calculate Precision@K, Recall@K, Reciprocal Rank, F1, and Hit Rate."""
    # Only look at the top K results
    top_k_retrieved = retrieved_ids[:k]
    
    # Intersection of retrieved and expected
    relevant_retrieved = [rid for rid in top_k_retrieved if rid in expected_ids]
    
    precision = len(relevant_retrieved) / k
    recall = len(relevant_retrieved) / len(expected_ids) if expected_ids else 0
    
    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    hit = 1.0 if len(relevant_retrieved) > 0 else 0.0
    
    # Reciprocal Rank calculation
    rr = 0.0
    for index, rid in enumerate(top_k_retrieved):
        if rid in expected_ids:
            rr = 1 / (index + 1)
            break
            
    return precision, recall, rr, f1, hit

def calculate_ndcg(retrieved_ids, expected_ids, k):
    """Calculate Normalized Discounted Cumulative Gain."""
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in expected_ids:
            # Binary relevance: 1 if in expected, 0 otherwise
            dcg += 1.0 / math.log2(i + 2)
    
    idcg = 0.0
    for i in range(min(len(expected_ids), k)):
        idcg += 1.0 / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def find_expected_ids(graph, source_name, hint):
    """Helper to find chunk_ids containing the ground truth hint."""
    # Clean hint for regex: escape special chars, replace whitespace with \s+
    hint_regex = re.escape(re.sub(r'\s+', ' ', hint.strip()))
    hint_regex = hint_regex.replace(r'\ ', r'\s+')
    
    # Robust query using regex for text matching and flexible source matching
    query = """
    MATCH (c:Chunk)
    WHERE (toLower(c.source) CONTAINS toLower($source) OR toLower($source) CONTAINS toLower(c.source))
      AND toLower(c.text) =~ $regex
    RETURN c.chunk_id as id
    """
    
    results = graph.query(query, params={
        "source": os.path.basename(source_name),
        "regex": f"(?is).*{hint_regex}.*"
    })
    return [r['id'] for r in results]

def get_llm_judge(model_name="gemma-3-12b-it"):
    """Initialize the LLM judge."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv('GEMMA_KEY') or os.getenv('GEMINI_API_KEY')
    return init_chat_model(
        model_name, 
        model_provider="google_genai",
        api_key=api_key
    )

def judge_rag_quality(judge, query, context, answer):
    """Use LLM to judge Faithfulness, Answer Relevance, and Context Relevancy."""
    prompt = ChatPromptTemplate.from_template("""
    You are an expert evaluator for a RAG system. Grade the following based on a score of 0 to 1.
    
    Query: {query}
    Context: {context}
    Answer: {answer}
    
    Provide the output in strict JSON format:
    {{
        "faithfulness": <score_0_to_1>,
        "answer_relevance": <score_0_to_1>,
        "context_relevancy": <score_0_to_1>
    }}
    
    - Faithfulness: Is the answer derived solely from the context?
    - Answer Relevance: Does the answer directly address the query?
    - Context Relevancy: Is the retrieved context actually useful for answering the query?
    """)
    
    try:
        response = judge.invoke(prompt.format(query=query, context=context, answer=answer))
        content = response.content.strip()
        # Robust JSON extraction using regex to handle potential LLM verbosity or markdown
        json_match = re.search(r'\{.*\}', content, re.S)
        if json_match:
            return json.loads(json_match.group())
        return {"faithfulness": 0, "answer_relevance": 0, "context_relevancy": 0}
    except Exception as e:
        print(f"Error during LLM judging: {e}")
        return {"faithfulness": 0, "answer_relevance": 0, "context_relevancy": 0}

def run_evaluation(eval_data, vectorstore, graph, judge, k=3):
    total_precision = 0
    total_recall = 0
    total_rr = 0
    total_ndcg = 0
    total_f1 = 0
    total_hit = 0
    
    # LLM Metrics
    total_faithfulness = 0
    total_ans_relevance = 0
    total_ctx_relevancy = 0
    
    # Performance Metrics
    total_retrieval_time = 0
    total_gen_time = 0
    
    num_queries = len(eval_data)
    processed_queries = 0

    print(f"--- Starting RAG Evaluation (K={k}) ---")

    for entry in eval_data:
        query = entry["query"]
        
        # Dynamically find expected IDs based on the essay content
        expected = find_expected_ids(graph, entry["source"], entry["context_hint"])
        
        if not expected:
            print(f"Warning: Could not find ground truth chunks for query: {query}")
            continue
        
        # Perform the search using your existing pipeline
        start_retrieval = time.time()
        results = hybrid_search(vectorstore, graph, query, k=k, context_size=1)
        total_retrieval_time += (time.time() - start_retrieval)
        
        retrieved_ids = [r['chunk_id'] for r in results]
        full_context = "\n".join([r['text'] for r in results])
        
        # Generate Answer (using judge as generator for eval purposes)
        start_gen = time.time()
        gen_prompt = f"Context: {full_context}\n\nQuestion: {query}\nAnswer:"
        answer = judge.invoke(gen_prompt).content
        total_gen_time += (time.time() - start_gen)
        
        # Calculate IR Metrics
        p, r, rr, f1, hit = calculate_metrics(retrieved_ids, expected, k)
        ndcg = calculate_ndcg(retrieved_ids, expected, k)
        
        total_precision += p
        total_recall += r
        total_rr += rr
        total_ndcg += ndcg
        total_f1 += f1
        total_hit += hit
        
        # Calculate LLM Judge Metrics
        scores = judge_rag_quality(judge, query, full_context, answer)
        total_faithfulness += scores.get('faithfulness', 0)
        total_ans_relevance += scores.get('answer_relevance', 0)
        total_ctx_relevancy += scores.get('context_relevancy', 0)
        
        processed_queries += 1
        
        print(f"Q: {query}")
        print(f"   Retrieved IDs: {retrieved_ids} | Expected: {expected}")
        print(f"   NDCG@{k}: {ndcg:.2f} | Faithfulness: {scores.get('faithfulness'):.2f}")

    if processed_queries == 0: return

    print("\n--- Final Results ---")
    print(f"Mean Precision@{k}: {total_precision / processed_queries:.4f}")
    print(f"Mean Recall@{k}: {total_recall / processed_queries:.4f}")
    print(f"MRR: {total_rr / processed_queries:.4f}")
    print(f"NDCG@{k}: {total_ndcg / processed_queries:.4f}")
    print(f"Mean F1@{k}: {total_f1 / processed_queries:.4f}")
    print(f"Hit Rate@{k}: {total_hit / processed_queries:.4f}")
    print("-" * 20)
    print(f"Mean Faithfulness: {total_faithfulness / processed_queries:.4f}")
    print(f"Mean Answer Relevance: {total_ans_relevance / processed_queries:.4f}")
    print(f"Mean Context Relevancy: {total_ctx_relevancy / processed_queries:.4f}")
    print("-" * 20)
    print(f"Avg Retrieval Latency: {total_retrieval_time / processed_queries:.4f}s")
    print(f"Avg Generation Latency: {total_gen_time / processed_queries:.4f}s")

if __name__ == "__main__":
    load_environment()
    
    # Get API key from environment variables
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv('GEMMA_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Google API Key not found. Please ensure GOOGLE_API_KEY, GEMMA_KEY, or GEMINI_API_KEY is set in your .env file.")

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

    # Initialize connections (same as your main script)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    vectorstore = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password,
        index_name="essay_chunk_gemini",
    )
    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password
    )
    
    # Initialize Judge
    judge = get_llm_judge()

    # Load Ground Truth from JSON
    json_path = os.path.join(os.path.dirname(__file__), 'ground_truth.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            eval_data = json.load(f)
        run_evaluation(eval_data, vectorstore, graph, judge, k=3)
    else:
        print("Error: ground_truth.json not found.")