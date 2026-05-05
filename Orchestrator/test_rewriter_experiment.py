# %%
from Agent import personAgent
from langchain_google_genai import ChatGoogleGenerativeAI

agent = personAgent("essay_chunk_agentspace", "chunk_index")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# %%
queries = [
    ("Tell me the parts of a body paragraph", "tell me the prts of a body paragrph"),
    ("What is important in the analysis of a body paragraph", "whats importnt in the anlaysis of a body paragrph"),
    ("How do I make my thesis less vague and more specific", "how do i make my thesiss less vague and more specfic"),
]

prompts = {
    "none": None,
    "typo_only": "Fix any typos or grammatical errors in the following query. Do not change the meaning or rephrase. Return only the corrected query, nothing else. Query: {query}",
    "general": "Rewrite the following query to improve search results. Fix any typos or grammatical errors. Keep the original meaning. Do not add extra information. Return only the rewritten query, nothing else. Query: {query}",
    "academic": "Rewrite the following query using formal academic language to improve search results against academic documents. Fix any typos. Keep the original meaning. Return only the rewritten query, nothing else. Query: {query}",
    "conversational": "Rewrite the following query in a natural conversational tone, as if someone is asking a tutor for help. Fix any typos. Keep the original meaning. Return only the rewritten query, nothing else. Query: {query}",
    "keyword_expand": "Rewrite the following query to improve search results. Fix any typos. Add 2-3 related keywords that would help find relevant documents. Return only the rewritten query, nothing else. Query: {query}",
}

# %%
def rewrite(query, template):
    if template is None:
        return query
    return llm.invoke(template.format(query=query)).content

def overlap(baseline, test):
    if not baseline:
        return 0.0
    b = set(str(x)[:100] for x in baseline)
    return sum(1 for x in test if str(x)[:100] in b) / len(baseline)

# %%
# generate all rewrites upfront
print("Generating rewrites...")
rewrites = {}
for clean, typo in queries:
    rewrites[typo] = {}
    for name, template in prompts.items():
        r = rewrite(typo, template)
        rewrites[typo][name] = r
        if name != "none":
            print(f"  [{name}] {typo} -> {r}")

# %%
# score each prompt against clean query baseline
scores = {name: {"k": [], "p": []} for name in prompts}

for clean, typo in queries:
    baseline_k = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(clean)]
    baseline_p = agent.personality.retrieve(clean)

    for name in prompts:
        q = rewrites[typo][name]
        test_k = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(q)]
        test_p = agent.personality.retrieve(q)
        scores[name]["k"].append(overlap(baseline_k, test_k))
        scores[name]["p"].append(overlap(baseline_p, test_p))

print(f"\n{'Prompt':<20} {'Knowledge':>10} {'Personality':>12} {'Combined':>10}")
print("-" * 55)

avgs = {}
for name in prompts:
    k = sum(scores[name]["k"]) / len(scores[name]["k"])
    p = sum(scores[name]["p"]) / len(scores[name]["p"])
    c = (k + p) / 2
    avgs[name] = {"k": k, "p": p, "c": c}
    print(f"{name:<20} {k:>10.2f} {p:>12.2f} {c:>10.2f}")

best = max(avgs, key=lambda x: avgs[x]["c"])
best_k = max(avgs, key=lambda x: avgs[x]["k"])
best_p = max(avgs, key=lambda x: avgs[x]["p"])

print(f"\nBest overall: {best}")
print(f"Best for knowledge: {best_k}")
print(f"Best for personality: {best_p}")

# %%
# split rewriter vs single rewriter
# use best knowledge prompt for knowledge, best personality prompt for personality
split = {"k": [], "p": []}
single = {"k": [], "p": []}

for clean, typo in queries:
    baseline_k = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(clean)]
    baseline_p = agent.personality.retrieve(clean)

    # split: different prompt per retriever
    sk = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(rewrites[typo][best_k])]
    sp = agent.personality.retrieve(rewrites[typo][best_p])
    split["k"].append(overlap(baseline_k, sk))
    split["p"].append(overlap(baseline_p, sp))

    # single: one prompt for both
    sq = rewrites[typo][best]
    single_k = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(sq)]
    single_p = agent.personality.retrieve(sq)
    single["k"].append(overlap(baseline_k, single_k))
    single["p"].append(overlap(baseline_p, single_p))

split_avg = (sum(split["k"])/3 + sum(split["p"])/3) / 2
single_avg = (sum(single["k"])/3 + sum(single["p"])/3) / 2
none_avg = avgs["none"]["c"]

print(f"\n{'Approach':<20} {'Combined':>10}")
print("-" * 32)
print(f"{'Split rewriter':<20} {split_avg:>10.2f}")
print(f"{'Single rewriter':<20} {single_avg:>10.2f}")
print(f"{'No rewriter':<20} {none_avg:>10.2f}")

# %%
# per-query detail
for i, (clean, typo) in enumerate(queries):
    print(f"\nQuery {i+1}: {clean}")
    print(f"Typo:    {typo}")
    for name in prompts:
        if name == "none":
            continue
        print(f"  [{name}] -> {rewrites[typo][name]}")
    print(f"\n  {'Prompt':<20} {'Knowledge':>10} {'Personality':>12}")
    print(f"  {'-'*44}")
    for name in prompts:
        print(f"  {name:<20} {scores[name]['k'][i]:>10.2f} {scores[name]['p'][i]:>12.2f}")

# %%
