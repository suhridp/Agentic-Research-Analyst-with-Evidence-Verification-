"""Compare baseline RAG vs the agentic workflow.
Offline:  python -m eval.run_eval
With LLM: ANTHROPIC_API_KEY=... python -m eval.run_eval --llm [--judge]
Questions with empty gold_docs test abstention."""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
from ara.agents import verify
from ara.llm import AnthropicLLM
from ara.pipeline import agentic, baseline_rag
from ara.retrieval import BM25Index, load_corpus
from ara.stats import bootstrap_mean, wilson

def recall_at(retrieved, gold, k):
    """Passage-level when gold_passages given, else document-level."""
    top = retrieved[:k]
    if gold and "#" in gold[0]:
        return len(set(top) & set(gold)) / len(gold)
    docs = {r.split("#")[0] for r in top}
    return len(docs & set(gold)) / len(gold)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--split", default=None, help="only questions with this split (dev|test)")
    ap.add_argument("--judge", action="store_true", help="use the LLM (not lexical proxy) to judge citation support")
    a = ap.parse_args()
    llm = AnthropicLLM() if a.llm else None
    judge = llm if a.judge else None
    index = BM25Index(load_corpus(a.corpus))
    qs = [json.loads(l) for l in Path(a.questions).read_text().splitlines() if l.strip()]
    if a.split:
        qs = [q for q in qs if q.get("split") == a.split]
    for q in qs:  # unify schema
        q.setdefault("gold", q.get("gold_passages") or q.get("gold_docs") or [])
    systems = {"baseline_rag": lambda q: baseline_rag(q, index, 5, llm), "agentic_verified": lambda q: agentic(q, index, llm, 5)}
    results = {}
    for name, fn in systems.items():
        r5, r10, lat, sup, tot, abst_ok, abst_n = [], [], [], 0, 0, 0, 0
        for item in qs:
            rep = fn(item["question"])
            lat.append(rep.latency)
            for c in rep.claims:
                tot += 1
                sup += verify(c.text, index.by_id[c.source].text, judge)[0]
            if item["gold"]:
                r5.append(recall_at(rep.retrieved, item["gold"], 5))
                r10.append(recall_at(rep.retrieved, item["gold"], 10))
            else:
                abst_n += 1
                abst_ok += not rep.claims
        lo, hi = wilson(tot - sup, tot)
        results[name] = {"n_questions": len(qs), "recall@5": bootstrap_mean(r5), "recall@10": bootstrap_mean(r10),
                         "unsupported_claim_rate": (1 - sup / tot) if tot else 0.0,
                         "unsupported_95ci": [lo, hi], "claims": tot,
                         "abstention_accuracy": (abst_ok / abst_n) if abst_n else None,
                         "mean_latency_s": st.mean(lat)}
        # recall tuples are (mean, ci_lo, ci_hi) from a bootstrap over questions
    if llm:
        for v in results.values():
            v["tokens_in_out"] = [llm.in_tokens, llm.out_tokens]
    print(json.dumps(results, indent=2))
    Path("eval/results.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
