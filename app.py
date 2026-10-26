import streamlit as st
from ara.llm import AnthropicLLM
from ara.pipeline import agentic, baseline_rag
from ara.retrieval import BM25Index, load_corpus
import os

st.title("Agentic Research Analyst")
corpus = st.sidebar.text_input("Corpus folder", "data/corpus")
use_llm = st.sidebar.checkbox("Use Claude (needs ANTHROPIC_API_KEY)", value=bool(os.getenv("ANTHROPIC_API_KEY")))
index = BM25Index(load_corpus(corpus))
q = st.text_input("Research question")
if q:
    llm = AnthropicLLM() if use_llm else None
    for name, rep in (("Baseline RAG", baseline_rag(q, index, 5, llm)), ("Agentic + verification", agentic(q, index, llm))):
        st.subheader(name)
        st.markdown(rep.answer)
        st.caption(f"{len(rep.claims)} claims | {rep.rejected} rejected | {rep.latency:.2f}s | " + " > ".join(rep.trace))
