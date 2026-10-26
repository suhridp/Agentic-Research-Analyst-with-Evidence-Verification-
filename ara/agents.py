"""Six agents: planner, extractor, verifier, synthesizer, critic (+ retrieval in pipeline).
Each uses an LLM when given one, else a deterministic heuristic so everything runs offline."""
from __future__ import annotations
import re
from dataclasses import dataclass
from .llm import parse_json
from .retrieval import Passage, sentences, tokens

@dataclass
class Claim:
    text: str
    source: str
    verified: bool | None = None
    score: float = 0.0

NUM = re.compile(r"\d+(?:\.\d+)?")
CITE = re.compile(r"\[([^\[\]#\s]+#\d+)\]")

SYS_PLAN = "Break the question into 3-5 focused research sub-questions. Return ONLY a JSON list of strings."
SYS_EXTRACT = ('Extract claims that answer the question, using ONLY the passages. Return ONLY a JSON list of '
               '{"claim": str, "source": "<passage id>"}. Keep numbers exact. Never add facts not in a passage.')
SYS_VERIFY = 'Does the passage fully support the claim, including every number? Return ONLY {"supported": true|false}.'
SYS_SYNTH = ("Write a concise structured report answering the question using ONLY the verified claims. "
             "Every sentence must end with its citation like [doc#0]. Note disagreements between sources.")

def plan(question: str, llm=None) -> list[str]:
    if llm:
        out = parse_json(llm.complete(SYS_PLAN, question))
        if isinstance(out, list) and out:
            return [str(x) for x in out][:5]
    parts = [p.strip() for p in re.split(r"\band\b|;|,", question.rstrip("?")) if len(tokens(p)) >= 2]
    return [question] + parts if len(parts) > 1 else [question]

def extract(subq: str, passages: list[Passage], llm=None) -> list[Claim]:
    if llm:
        ctx = "\n".join(f"[{p.id}] {p.text}" for p in passages)
        out = parse_json(llm.complete(SYS_EXTRACT, f"Question: {subq}\nPassages:\n{ctx}"))
        valid = {p.id for p in passages}
        if isinstance(out, list):
            return [Claim(str(c["claim"]), str(c["source"])) for c in out
                    if isinstance(c, dict) and "claim" in c and c.get("source") in valid]
        return []
    q, claims = set(tokens(subq)), []
    for p in passages:
        best = max(sentences(p.text), key=lambda s: len(q & set(tokens(s))), default="")
        if len(q & set(tokens(best))) >= 2:
            claims.append(Claim(best, p.id))
    return claims

def lexical_support(claim: str, passage: str) -> float:
    """Proxy for entailment: content-word coverage, zeroed if any number is not in the passage."""
    ct, pt = set(tokens(claim)), set(tokens(passage))
    if not set(NUM.findall(claim)) <= set(NUM.findall(passage)):
        return 0.0
    return len(ct & pt) / max(len(ct), 1)

def verify(claim: str, passage_text: str, llm=None, threshold: float = 0.8) -> tuple[bool, float]:
    if llm:
        out = parse_json(llm.complete(SYS_VERIFY, f"Passage: {passage_text}\nClaim: {claim}", 50))
        if isinstance(out, dict) and "supported" in out:
            ok = bool(out["supported"])
            return ok, 1.0 if ok else 0.0
    s = lexical_support(claim, passage_text)
    return s >= threshold, s

def synthesize(question: str, claims: list[Claim], llm=None) -> str:
    if not claims:
        return "Insufficient verified evidence to answer this question."
    if llm:
        ctx = "\n".join(f"- {c.text} [{c.source}]" for c in claims)
        return llm.complete(SYS_SYNTH, f"Question: {question}\nVerified claims:\n{ctx}")
    return "\n".join(f"- {c.text} [{c.source}]" for c in claims)

def critique(report: str, claims: list[Claim]) -> list[str]:
    ok = {c.source for c in claims if c.verified}
    issues = []
    for line in report.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        ids = CITE.findall(line)
        if not ids:
            issues.append(f"uncited: {line[:60]}")
        issues += [f"citation not backed by a verified claim: {i}" for i in ids if i not in ok]
    return issues
