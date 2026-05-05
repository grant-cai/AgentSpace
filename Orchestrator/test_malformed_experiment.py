# %%
from Agent import personAgent
from langchain_google_genai import ChatGoogleGenerativeAI

agent = personAgent("essay_chunk_agentspace", "chunk_index")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# %%
# 15 queries: clean, typo, and malformed versions
# covers topics from both BodyParagraphs.pdf and Grant's interview

queries = [
    # --- from BodyParagraphs.pdf ---
    ("Tell me the parts of a body paragraph",
     "tell me the prts of a body paragrph",
     "body paragrph parts what are"),

    ("What is important in the analysis of a body paragraph",
     "whats importnt in the anlaysis of a body paragrph",
     "anlaysis body paragph why importnt"),

    ("How should I order my body paragraphs",
     "how shuld i order my body paragrphs",
     "order body paragrahs how"),

    ("What makes a good topic sentence",
     "what makes a good tpoic sentance",
     "tpoic sentance good what"),

    ("How do I use evidence in a body paragraph",
     "how do i use evidnce in a body paragrph",
     "evidnce body paragrph use how"),

    ("Why are transitions important between paragraphs",
     "why are transtions importnt between paragrphs",
     "transtions paragrphs importnt why"),

    ("What is the difference between evidence and analysis",
     "what is the diffrence between evidnce and anlaysis",
     "evidnce anlaysis diffrence"),

    # --- from Grant's interview ---
    ("How do I make my thesis less vague and more specific",
     "how do i make my thesiss less vague and more specfic",
     "thesiss vague how fix specfic"),

    ("What are the three parts of a thesis statement",
     "what are the three prts of a thesiss statment",
     "thesiss statment three prts"),

    ("How does the hourglass essay structure work",
     "how does the hourglss essay struture work",
     "hourglss essay struture how"),

    ("What is the best way to give feedback on writing",
     "what is the best way to give feedbck on writting",
     "feedbck writting best way"),

    ("How do you help a student with a weak thesis",
     "how do you help a studnet with a weak thesiss",
     "studnet weak thesiss help how"),

    ("What should I focus on when revising my essay",
     "what should i focis on when revising my esay",
     "focis revising esay what"),

    ("How do I integrate quotes into my essay",
     "how do i integarte quotes into my esay",
     "integarte quotes esay how"),

    ("What is the sandwich method for feedback",
     "what is the sandwhich method for feedbck",
     "sandwhich method feedbck what"),
]

prompts = {
    "none": None,
    "typo_only": "Fix any typos or grammatical errors in the following query. Do not change the meaning or rephrase. Return only the corrected query, nothing else. Query: {query}",
    "general": "Rewrite the following query to improve search results. Fix any typos or grammatical errors. Keep the original meaning. Do not add extra information. Return only the rewritten query, nothing else. Query: {query}",
    "academic": "Rewrite the following query using formal academic language to improve search results against academic documents. Fix any typos. Keep the original meaning. Return only the rewritten query, nothing else. Query: {query}",
    "conversational": "Rewrite the following query in a natural conversational tone, as if someone is asking a tutor for help. Fix any typos. Keep the original meaning. Return only the rewritten query, nothing else. Query: {query}",
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
# generate all rewrites for both typo and malformed queries
print("Generating rewrites for typo queries...")
typo_rewrites = {}
for clean, typo, malformed in queries:
    typo_rewrites[typo] = {}
    for name, template in prompts.items():
        typo_rewrites[typo][name] = rewrite(typo, template)

print("Generating rewrites for malformed queries...")
malformed_rewrites = {}
for clean, typo, malformed in queries:
    malformed_rewrites[malformed] = {}
    for name, template in prompts.items():
        malformed_rewrites[malformed][name] = rewrite(malformed, template)

print("Done.\n")

# %%
# score all prompts on typo queries
print("=" * 65)
print("TYPO QUERIES (misspelled but grammatically correct)")
print("=" * 65)

typo_scores = {name: {"k": [], "p": []} for name in prompts}

for clean, typo, malformed in queries:
    baseline_k = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(clean)]
    baseline_p = agent.personality.retrieve(clean)
    for name in prompts:
        q = typo_rewrites[typo][name]
        test_k = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(q)]
        test_p = agent.personality.retrieve(q)
        typo_scores[name]["k"].append(overlap(baseline_k, test_k))
        typo_scores[name]["p"].append(overlap(baseline_p, test_p))

n = len(queries)
print(f"\n{'Prompt':<20} {'Knowledge':>10} {'Personality':>12} {'Combined':>10}")
print("-" * 55)
typo_avgs = {}
for name in prompts:
    k = sum(typo_scores[name]["k"]) / n
    p = sum(typo_scores[name]["p"]) / n
    c = (k + p) / 2
    typo_avgs[name] = {"k": k, "p": p, "c": c}
    print(f"{name:<20} {k:>10.2f} {p:>12.2f} {c:>10.2f}")

best_typo = max(typo_avgs, key=lambda x: typo_avgs[x]["c"])
print(f"\nBest for typo queries: {best_typo} ({typo_avgs[best_typo]['c']:.2f})")

# %%
# score all prompts on malformed queries
print("\n" + "=" * 65)
print("MALFORMED QUERIES (broken grammar + misspelled)")
print("=" * 65)

mal_scores = {name: {"k": [], "p": []} for name in prompts}

for clean, typo, malformed in queries:
    baseline_k = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(clean)]
    baseline_p = agent.personality.retrieve(clean)
    for name in prompts:
        q = malformed_rewrites[malformed][name]
        test_k = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(q)]
        test_p = agent.personality.retrieve(q)
        mal_scores[name]["k"].append(overlap(baseline_k, test_k))
        mal_scores[name]["p"].append(overlap(baseline_p, test_p))

print(f"\n{'Prompt':<20} {'Knowledge':>10} {'Personality':>12} {'Combined':>10}")
print("-" * 55)
mal_avgs = {}
for name in prompts:
    k = sum(mal_scores[name]["k"]) / n
    p = sum(mal_scores[name]["p"]) / n
    c = (k + p) / 2
    mal_avgs[name] = {"k": k, "p": p, "c": c}
    print(f"{name:<20} {k:>10.2f} {p:>12.2f} {c:>10.2f}")

best_mal = max(mal_avgs, key=lambda x: mal_avgs[x]["c"])
print(f"\nBest for malformed queries: {best_mal} ({mal_avgs[best_mal]['c']:.2f})")

# %%
# split rewriter test
best_typo_k = max(typo_avgs, key=lambda x: typo_avgs[x]["k"])
best_typo_p = max(typo_avgs, key=lambda x: typo_avgs[x]["p"])

print("\n" + "=" * 65)
print("SPLIT vs SINGLE REWRITER (on typo queries)")
print(f"Knowledge prompt: {best_typo_k}")
print(f"Personality prompt: {best_typo_p}")
print("=" * 65)

split = {"k": [], "p": []}
single = {"k": [], "p": []}

for clean, typo, malformed in queries:
    baseline_k = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(clean)]
    baseline_p = agent.personality.retrieve(clean)

    sk = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(typo_rewrites[typo][best_typo_k])]
    sp = agent.personality.retrieve(typo_rewrites[typo][best_typo_p])
    split["k"].append(overlap(baseline_k, sk))
    split["p"].append(overlap(baseline_p, sp))

    sq = typo_rewrites[typo][best_typo]
    single_k = [doc.page_content[:100] for doc in agent.knowlege_retrieval.invoke(sq)]
    single_p = agent.personality.retrieve(sq)
    single["k"].append(overlap(baseline_k, single_k))
    single["p"].append(overlap(baseline_p, single_p))

split_avg = (sum(split["k"])/n + sum(split["p"])/n) / 2
single_avg = (sum(single["k"])/n + sum(single["p"])/n) / 2

print(f"\n{'Approach':<20} {'Combined':>10}")
print("-" * 32)
print(f"{'Split rewriter':<20} {split_avg:>10.2f}")
print(f"{'Single rewriter':<20} {single_avg:>10.2f}")
print(f"{'No rewriter':<20} {typo_avgs['none']['c']:>10.2f}")

# %%
# summary
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"\nTypo queries:")
for name in prompts:
    print(f"  {name:<20} {typo_avgs[name]['c']:.2f}")
print(f"\nMalformed queries:")
for name in prompts:
    print(f"  {name:<20} {mal_avgs[name]['c']:.2f}")
print(f"\nBest for typos:     {best_typo}")
print(f"Best for malformed: {best_mal}")

if best_typo == best_mal:
    print(f"\nSame prompt wins both. Use '{best_typo}' for the rewriter.")
else:
    print(f"\nDifferent winners. Typo queries need '{best_typo}', malformed queries need '{best_mal}'.")
    print(f"Consider using '{best_mal}' since it handles the harder case.")

# %%
