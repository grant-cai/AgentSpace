"""
Tests for PersonalityWrapper
"""

import os
from PersonalityRetriever.personality_wrapper import PersonalityWrapper

DIR = os.path.dirname(os.path.abspath(__file__))

wrapper = PersonalityWrapper(
    profile_path=os.path.join(DIR, "personality_summary.json"),
    faiss_dir=os.path.join(DIR, "faiss_db"),
)


# ---------------------------------------------------------------------------
# get_profile() tests
# ---------------------------------------------------------------------------

def test_returns_dict():
    assert isinstance(wrapper.get_profile(), dict)

def test_top_level_keys():
    profile = wrapper.get_profile()
    for key in ["agent_profile", "tutoring_philosophy", "key_frameworks", "personality_traits", "handling_student_types", "constraints"]:
        assert key in profile, f"Missing key: {key}"

def test_agent_profile_fields():
    ap = wrapper.get_profile()["agent_profile"]
    assert ap["name"] == "Grant"
    assert ap["role"] == "Writing Tutor"

def test_consistent():
    assert wrapper.get_profile() == wrapper.get_profile()


# ---------------------------------------------------------------------------
# retrieve() tests
# ---------------------------------------------------------------------------

def print_retrieval(query, results):
    print(f"    Query   : {query}")
    for i, chunk in enumerate(results, 1):
        print(f"    Chunk {i} : {chunk[:200]}")

def test_returns_list():
    query = "thesis structure"
    results = wrapper.retrieve(query)
    print_retrieval(query, results)
    assert isinstance(results, list)

def test_default_k():
    query = "how do you give feedback"
    results = wrapper.retrieve(query)
    print_retrieval(query, results)
    assert len(results) == 4

def test_k_param():
    for k in [1, 2, 6]:
        results = wrapper.retrieve("writing tutor", k=k)
        assert len(results) == k, f"Expected {k} results, got {len(results)}"

def test_returns_strings():
    query = "student confidence"
    results = wrapper.retrieve(query)
    print_retrieval(query, results)
    assert all(isinstance(r, str) for r in results)

def test_non_empty():
    query = "essay structure"
    results = wrapper.retrieve(query)
    print_retrieval(query, results)
    assert all(len(r) > 0 for r in results)

def test_relevance_thesis():
    query = "how do you help a student with their thesis"
    results = wrapper.retrieve(query, k=2)
    print_retrieval(query, results)
    combined = " ".join(results).lower()
    assert any(term in combined for term in ["thesis", "argument", "claim", "subject"]), (
        f"Expected thesis-related terms, got:\n{combined[:300]}"
    )

def test_relevance_feedback():
    query = "how do you give feedback to students"
    results = wrapper.retrieve(query, k=2)
    print_retrieval(query, results)
    combined = " ".join(results).lower()
    assert any(term in combined for term in ["feedback", "positive", "sandwich", "constructive", "encourage"]), (
        f"Expected feedback-related terms, got:\n{combined[:300]}"
    )

def test_different_queries():
    query_a = "thesis and argument structure"
    query_b = "handling a student who is defensive or overconfident"
    results_a = wrapper.retrieve(query_a)
    results_b = wrapper.retrieve(query_b)
    print_retrieval(query_a, results_a)
    print_retrieval(query_b, results_b)
    assert results_a != results_b, "Expected different queries to return different chunks"



def run_test(name, fn):
    print(f"\n  > {name}")
    try:
        fn()
        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")

def main():
    print("=" * 60)
    print("get_profile()")
    print("=" * 60)
    run_test("returns dict", test_returns_dict)
    run_test("top-level keys present", test_top_level_keys)
    run_test("agent_profile name and role", test_agent_profile_fields)
    run_test("consistent across calls", test_consistent)

    print("\n" + "=" * 60)
    print("retrieve()")
    print("=" * 60)
    run_test("returns list", test_returns_list)
    run_test("default k=4 returns 4 chunks", test_default_k)
    run_test("k parameter respected", test_k_param)
    run_test("returns strings", test_returns_strings)
    run_test("chunks are non-empty", test_non_empty)
    run_test("relevance: thesis query", test_relevance_thesis)
    run_test("relevance: feedback query", test_relevance_feedback)
    run_test("different queries return different results", test_different_queries)


if __name__ == "__main__":
    main()
