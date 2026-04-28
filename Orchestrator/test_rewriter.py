# ---
# Test: Does the query rewriter improve retrieval?
# Run this from the Orchestrator/ folder with the agentspace kernel
# ---

# %%
from Agent import personAgent
from langchain_google_genai import ChatGoogleGenerativeAI

agent = personAgent("essay_chunk_agentspace", "chunk_index")

# %%
# Test queries: clean and typo versions
test_queries = [
    {
        "clean": "Tell me the parts of a body paragraph",
        "typo": "tell me the prts of a body paragrph",
    },
    {
        "clean": "What is important in the analysis of a body paragraph",
        "typo": "whats importnt in the anlaysis of a body paragrph",
    },
    {
        "clean": "How do I make my thesis less vague and more specific",
        "typo": "how do i make my thesiss less vague and more specfic",
    },
]

# %%
# Rewriter function (same as in Agent.py)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

def rewrite_query(query):
    prompt = f"""Rewrite the following query to improve search results. Fix any typos or grammatical errors. Keep the original meaning. Do not add extra information. Return only the rewritten query, nothing else. Query: {query}"""
    response = llm.invoke(prompt)
    return response.content

# %%
# Run the comparison
# For each query pair, we test 3 scenarios:
#   1. Clean query (no rewriter needed, best case baseline)
#   2. Typo query WITHOUT rewriter (worst case)
#   3. Typo query WITH rewriter (does it recover?)

results = []

for i, q in enumerate(test_queries):
    print(f"\n{'='*70}")
    print(f"TEST {i+1}")
    print(f"{'='*70}")

    clean = q["clean"]
    typo = q["typo"]
    rewritten = rewrite_query(typo)

    print(f"Clean query:    {clean}")
    print(f"Typo query:     {typo}")
    print(f"Rewritten to:   {rewritten}")

    # Retrieve with clean query (baseline)
    clean_personality = agent.personality.retrieve(clean)
    clean_knowledge = agent.knowlege_retrieval.invoke(clean)
    clean_knowledge_text = [doc.page_content[:100] for doc in clean_knowledge]

    # Retrieve with typo query (no rewriter)
    typo_personality = agent.personality.retrieve(typo)
    typo_knowledge = agent.knowlege_retrieval.invoke(typo)
    typo_knowledge_text = [doc.page_content[:100] for doc in typo_knowledge]

    # Retrieve with rewritten query (rewriter applied)
    rewritten_personality = agent.personality.retrieve(rewritten)
    rewritten_knowledge = agent.knowlege_retrieval.invoke(rewritten)
    rewritten_knowledge_text = [doc.page_content[:100] for doc in rewritten_knowledge]

    result = {
        "clean": clean,
        "typo": typo,
        "rewritten": rewritten,
        "clean_personality": clean_personality,
        "typo_personality": typo_personality,
        "rewritten_personality": rewritten_personality,
        "clean_knowledge": clean_knowledge_text,
        "typo_knowledge": typo_knowledge_text,
        "rewritten_knowledge": rewritten_knowledge_text,
    }
    results.append(result)

    print(f"\n--- Knowledge retrieval (first 100 chars of each chunk) ---")
    print(f"  Clean:     {clean_knowledge_text[:2]}")
    print(f"  Typo:      {typo_knowledge_text[:2]}")
    print(f"  Rewritten: {rewritten_knowledge_text[:2]}")

    print(f"\n--- Personality retrieval (first 80 chars of each chunk) ---")
    print(f"  Clean:     {[c[:80] for c in clean_personality[:2]]}")
    print(f"  Typo:      {[c[:80] for c in typo_personality[:2]]}")
    print(f"  Rewritten: {[c[:80] for c in rewritten_personality[:2]]}")

# %%
# Score: how much overlap between clean results and rewritten results?
# If rewriter works, rewritten results should match clean results closely

def chunk_overlap(list_a, list_b):
    """What percentage of items in list_a also appear in list_b"""
    if not list_a:
        return 0.0
    set_b = set([str(x)[:100] for x in list_b])
    matches = sum(1 for x in list_a if str(x)[:100] in set_b)
    return matches / len(list_a)

print("\n" + "="*70)
print("OVERLAP SCORES")
print("="*70)
print("(1.0 = identical results to clean query, 0.0 = completely different)\n")

for i, r in enumerate(results):
    print(f"Test {i+1}: {r['clean']}")

    # Knowledge overlap
    typo_k_overlap = chunk_overlap(r["clean_knowledge"], r["typo_knowledge"])
    rewritten_k_overlap = chunk_overlap(r["clean_knowledge"], r["rewritten_knowledge"])
    print(f"  Knowledge:   typo={typo_k_overlap:.2f}  rewritten={rewritten_k_overlap:.2f}  {'IMPROVED' if rewritten_k_overlap > typo_k_overlap else 'NO CHANGE' if rewritten_k_overlap == typo_k_overlap else 'WORSE'}")

    # Personality overlap
    typo_p_overlap = chunk_overlap(r["clean_personality"], r["typo_personality"])
    rewritten_p_overlap = chunk_overlap(r["clean_personality"], r["rewritten_personality"])
    print(f"  Personality: typo={typo_p_overlap:.2f}  rewritten={rewritten_p_overlap:.2f}  {'IMPROVED' if rewritten_p_overlap > typo_p_overlap else 'NO CHANGE' if rewritten_p_overlap == typo_p_overlap else 'WORSE'}")
    print()