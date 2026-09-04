"""Streamlit UI for inspecting retrieval quality before wiring up generation.

Type a question and see which chunks the retriever pulls from the combined
BRSR / GHG Protocol index, along with their similarity scores and metadata.
Switch between pure vector search, pure BM25, and a hybrid RRF fusion of
the two.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retriever import retrieve, bm25_retrieve, hybrid_retrieve, hybrid_rerank  # noqa: E402
from generator import generate_answer  # noqa: E402

st.set_page_config(page_title="Carbon Flat RAG - Retrieval Check", layout="wide")
st.title("Carbon Flat RAG — Retrieval Check")
st.caption(
    "Search the combined BRSR 2023 / GHG Protocol index and inspect the raw "
    "chunks a query retrieves, before they get passed to a generator."
)

query = st.text_input(
    "Query",
    placeholder="e.g. What GHG emissions does BRSR Principle 6 EI6 require?",
)

col1, col2, col3 = st.columns([1, 1, 2])
mode = col1.radio("Mode", ["hybrid", "vector", "bm25", "hybrid+rerank"], horizontal=True)
if mode == "hybrid+rerank":
    fetch_k = col2.slider("Candidate pool", min_value=10, max_value=50, value=30)
    top_k = col2.slider("Final results", min_value=1, max_value=10, value=5)
else:
    top_k = col2.slider("Top K", min_value=1, max_value=50, value=5)
doc_filter = col3.radio(
    "Filter by document", ["All", "brsr_2023", "ghg_protocol"], horizontal=True
)

_RETRIEVERS = {"vector": retrieve, "bm25": bm25_retrieve, "hybrid": hybrid_retrieve}

if query:
    doc_id = None if doc_filter == "All" else doc_filter
    if mode == "hybrid+rerank":
        hits = hybrid_rerank(query, fetch_k=fetch_k, top_n=top_k, doc_id=doc_id)
    else:
        hits = _RETRIEVERS[mode](query, top_k=top_k, doc_id=doc_id)

    if not hits:
        st.warning("No results.")

    if hits:
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, hits)
        st.markdown("### Answer")
        st.markdown(answer)
        st.markdown("### Retrieved chunks")

    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        scores = []
        if "rerank_score" in hit:
            scores.append(f"rerank `{hit['rerank_score']:.4f}`")
        if "rrf_score" in hit:
            scores.append(f"rrf `{hit['rrf_score']:.4f}`")
        if "distance" in hit:
            scores.append(f"distance `{hit['distance']:.4f}`")
        if "bm25_score" in hit:
            scores.append(f"bm25 `{hit['bm25_score']:.4f}`")
        with st.container(border=True):
            st.markdown(
                f"**#{i}**  ·  `{meta['doc_id']}`  ·  page {meta['page']}  ·  "
                f"type `{meta['type']}`  ·  " + "  ·  ".join(scores)
            )
            st.caption(meta["section"])
            st.text(hit["text"])
