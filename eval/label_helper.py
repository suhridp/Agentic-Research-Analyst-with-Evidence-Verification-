"""Speed up gold labeling. For each question in a draft file, write the top BM25 passages to a
review file; YOU read them and record the truly supporting passage IDs (and the reference answer).
Usage: python -m eval.label_helper --draft data/questions_draft.jsonl --out data/label_review.md
Draft line: {"question": "...", "type": "factual|numeric|comparison|gap|unanswerable"}
Final line (written by you): add "gold_passages": ["doc#3"], "reference": "...", "split": "dev|test"."""
import argparse, json
from pathlib import Path
from ara.retrieval import BM25Index, load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="data/corpus"); ap.add_argument("--draft", required=True)
ap.add_argument("--out", default="data/label_review.md"); ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()
idx = BM25Index(load_corpus(a.corpus))
lines = []
for n, l in enumerate(Path(a.draft).read_text().splitlines(), 1):
    if not l.strip():
        continue
    q = json.loads(l)
    lines += [f"## Q{n} ({q.get('type','?')}): {q['question']}", ""]
    for p, s in idx.search(q["question"], a.k):
        lines.append(f"- **{p.id}** (score {s:.1f}): {p.text}")
    lines += ["", "Gold passages: ____   Reference answer: ____", ""]
Path(a.out).write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {a.out}")
