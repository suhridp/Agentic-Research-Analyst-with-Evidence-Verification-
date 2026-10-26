"""Dependency-free BM25 retrieval over sentence-window passages."""
from __future__ import annotations
import math, re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOP = set("the a an of and or to in on for with is are was were be by as at from that this it its we our can which what how do does".split())

def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]

def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

@dataclass(frozen=True)
class Passage:
    doc: str
    idx: int
    text: str
    @property
    def id(self) -> str:
        return f"{self.doc}#{self.idx}"

def load_corpus(folder: str | Path, window: int = 2) -> list[Passage]:
    out: list[Passage] = []
    files = sorted(Path(folder).glob("*.txt")) + sorted(Path(folder).glob("*.md"))
    for p in files:
        body = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines() if not l.startswith("#"))
        sents = sentences(body)
        for i in range(0, len(sents), window):
            out.append(Passage(p.stem, i // window, " ".join(sents[i:i + window])))
    return out

class BM25Index:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages, self.k1, self.b = passages, k1, b
        self.tf = [Counter(tokens(p.text)) for p in passages]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(w for t in self.tf for w in t)
        n = len(passages)
        self.idf = {w: math.log(1 + (n - d + .5) / (d + .5)) for w, d in df.items()}
        self.by_id = {p.id: p for p in passages}

    def search(self, query: str, k: int = 5) -> list[tuple[Passage, float]]:
        q, scored = tokens(query), []
        for i, t in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = t.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.passages[i], s) for s, i in scored[:k]]
