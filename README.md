# Agentic Research Analyst with Evidence Verification

A research assistant that decomposes a question, retrieves from a document collection, extracts claims,
**verifies each claim against its cited passage**, and abstains when evidence is missing. Includes an
evaluation harness comparing it against a single-pass RAG baseline.

## Pipeline
`planner -> retrieval (BM25) -> extractor -> verifier -> synthesizer -> critic`, with a retry round
that widens retrieval and an abstention path. Unsupported claims are dropped before the report is written;
the critic rejects any citation not backed by a verified claim.

## Run
```bash
python -m pytest -q                 # 5 tests, offline
python -m eval.run_eval             # offline heuristics, writes eval/results.json
export ANTHROPIC_API_KEY=...        # LLM mode: planner/extractor/verifier/synthesizer use Claude
python -m eval.run_eval --llm --judge
streamlit run app.py                # optional UI
```
Offline mode uses deterministic heuristics (so the repo runs anywhere). The meaningful comparison
is **LLM mode**, where the baseline can hallucinate and the verifier earns its keep.

## Honest limitations
- `data/corpus` is **synthetic sample data** for testing only. Replace with real papers (.txt/.md).
- Without `--judge`, citation support is a lexical proxy (word coverage + number matching); paraphrases can be under-counted. Use `--judge` and spot-check by hand.
- 5 sample questions is a smoke test, not a benchmark.

## Roadmap to a resume-grade project
1. Build the ~100-question eval set with gold passages from real papers (Recall@5/10, citation correctness, unsupported-claim rate, completeness, latency, cost).
2. Add dense retrieval (Sentence Transformers + FAISS) fused with BM25, plus a reranker; add as a third system.
3. Port the pipeline to LangGraph and expose it via FastAPI.
4. Report results per system, plus failure-case analysis and cost per report.

## Building the real corpus and eval set (Phase 1)
1. Put open-access PDFs in `data/papers/` and record `file,title,url,license` in `data/papers/SOURCES.csv`.
2. `python scripts/ingest.py` converts them to `data/corpus/*.txt` (needs `pdftotext`).
3. Write draft questions to `data/questions_draft.jsonl` (types: factual, numeric, comparison, gap, unanswerable).
4. `python -m eval.label_helper --draft data/questions_draft.jsonl` writes `data/label_review.md`; read the candidate passages and record true gold passage IDs and reference answers into `data/questions.jsonl` with `"split": "dev"|"test"`.
5. `python -m eval.run_eval --split dev` (use `--split test` once, at the end).
