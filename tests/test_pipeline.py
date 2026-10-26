from ara.agents import CITE, Claim, critique, lexical_support
from ara.pipeline import agentic, baseline_rag
from ara.retrieval import BM25Index, load_corpus

index = BM25Index(load_corpus("data/corpus"))

def test_bm25_finds_gold_doc():
    top = index.search("DoS detection accuracy classifier", 3)
    assert top[0][0].doc == "synthetic_dos"

def test_fabricated_number_is_unsupported():
    passage = "The classifier detected DoS floods with 97 percent accuracy."
    assert lexical_support("The classifier detected DoS floods with 97 percent accuracy.", passage) == 1.0
    assert lexical_support("The classifier detected DoS floods with 99 percent accuracy.", passage) == 0.0

def test_critic_flags_unbacked_citation():
    claims = [Claim("x y z", "a#0", verified=True)]
    assert critique("- x y z [a#0]", claims) == []
    assert critique("- x y z [b#1]", claims)

def test_agentic_answers_with_valid_citations():
    rep = agentic("How can spoofed measurement reports attack predictive handover and what defenses exist?", index)
    assert rep.claims and not rep.issues
    assert all(c in index.by_id for c in CITE.findall(rep.answer))

def test_agentic_abstains_when_no_evidence():
    rep = agentic("What is the capital of France?", index)
    assert not rep.claims and "Insufficient" in rep.answer

def test_wilson_interval_contains_rate():
    from ara.stats import wilson
    lo, hi = wilson(10, 100)
    assert lo < 0.10 < hi and 0 <= lo and hi <= 1
