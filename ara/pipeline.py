from __future__ import annotations
import time
from dataclasses import dataclass, field
from .agents import CITE, Claim, critique, extract, plan, synthesize, verify
from .retrieval import BM25Index

@dataclass
class Report:
    question: str
    answer: str
    claims: list[Claim]
    retrieved: list[str]
    rejected: int = 0
    issues: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    latency: float = 0.0

def baseline_rag(q: str, index: BM25Index, k: int = 5, llm=None) -> Report:
    """Single-pass RAG: retrieve top-k, answer once, no verification."""
    t0 = time.time()
    passages = [p for p, _ in index.search(q, k)]
    if llm:
        ctx = "\n".join(f"[{p.id}] {p.text}" for p in passages)
        ans = llm.complete("Answer using only the passages. End each sentence with a citation like [doc#0].",
                           f"Question: {q}\nPassages:\n{ctx}")
        claims = [Claim(CITE.sub("", ln).strip(" -"), cid) for ln in ans.splitlines() for cid in CITE.findall(ln)]
    else:
        claims = extract(q, passages)
        ans = synthesize(q, claims)
    return Report(q, ans, claims, [p.id for p in passages], trace=["baseline"], latency=time.time() - t0)

def agentic(q: str, index: BM25Index, llm=None, k: int = 5, max_rounds: int = 2) -> Report:
    """plan -> retrieve -> extract -> verify -> synthesize -> critique (with retry and abstention)."""
    t0, trace, retrieved, verified, seen, rejected = time.time(), [], {}, [], set(), 0
    subs = plan(q, llm)
    trace.append(f"planner: {len(subs)} sub-question(s)")
    for rnd in range(max_rounds):
        for s in subs:
            passages = [p for p, _ in index.search(s, k * (rnd + 1))]
            for p in passages:
                retrieved.setdefault(p.id, p)
            for c in extract(s, passages, llm):
                if (c.text, c.source) in seen:
                    continue
                seen.add((c.text, c.source))
                c.verified, c.score = verify(c.text, retrieved[c.source].text, llm)
                if c.verified:
                    verified.append(c)
                else:
                    rejected += 1
        trace.append(f"round {rnd + 1}: {len(verified)} verified, {rejected} rejected")
        if verified:
            break
    answer = synthesize(q, verified, llm)
    issues = critique(answer, verified) if verified else []
    if issues:
        trace.append(f"critic: {len(issues)} issue(s); falling back to verified-claim list")
        answer = "\n".join(f"- {c.text} [{c.source}]" for c in verified)
        issues = critique(answer, verified)
    if not verified:
        trace.append("abstained: no verified evidence")
    return Report(q, answer, verified, list(retrieved), rejected, issues, trace, time.time() - t0)
