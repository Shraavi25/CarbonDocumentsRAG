"""Query the Chroma collection built by embedder.py for the most relevant chunks.

Three retrieval modes are available:
- `retrieve` -- pure dense vector search (cosine similarity over embeddings).
- `bm25_retrieve` -- pure sparse lexical search (BM25 over chunk text), which
  catches exact terms/codes (e.g. "Scope 1", "NF3", "Principle 6 Essential
  Indicator 6") that the small sentence-transformers model can rank poorly.
- `hybrid_retrieve` -- fuses the two rankings with Reciprocal Rank Fusion
  (RRF), so a chunk that ranks highly in either method surfaces near the top.
  When no `doc_id` filter is given, global RRF is computed first; if that
  leaves a document with zero chunks in top_k -- which can happen for a
  cross-document query whose vocabulary leans toward one document (e.g.
  "GHG Protocol definition of Scope 1 and Scope 2") -- that document's best
  chunk is swapped in for the weakest slot of an over-represented document,
  guaranteeing both documents are represented.

BM25 is built once (lazily, from the same chunk list and ids used by
embedder.py) and cached for the life of the process.
"""

import re
from collections import Counter

from rank_bm25 import BM25Okapi

from chunker import chunk_documents
from embedder import COLLECTION_NAME, embed_texts, get_client
from loader import DOCUMENTS
from reranker import rerank

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_bm25 = None
_bm25_chunks = None
_bm25_ids = None


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _build_bm25():
    global _bm25, _bm25_chunks, _bm25_ids
    chunks = chunk_documents()
    ids = [f"{c['metadata']['doc_id']}_p{c['metadata']['page']}_{i}" for i, c in enumerate(chunks)]
    _bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    _bm25_chunks = chunks
    _bm25_ids = ids


def get_bm25():
    if _bm25 is None:
        _build_bm25()
    return _bm25, _bm25_chunks, _bm25_ids


def get_collection():
    return get_client().get_collection(COLLECTION_NAME)


def retrieve(query, top_k=5, doc_id=None):
    """Embed `query` and return the top_k nearest chunks, each with its
    id, text, metadata, and cosine distance (lower = more similar)."""
    collection = get_collection()
    query_embedding = embed_texts([query])[0]

    where = {"doc_id": doc_id} if doc_id else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    hits = []
    for id_, text, metadata, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"id": id_, "text": text, "metadata": metadata, "distance": distance})
    return hits


def bm25_retrieve(query, top_k=5, doc_id=None):
    """Return the top_k chunks by BM25 lexical score, each with its
    id, text, metadata, and bm25_score (higher = more similar)."""
    bm25, chunks, ids = get_bm25()
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    if doc_id:
        ranked = [i for i in ranked if chunks[i]["metadata"]["doc_id"] == doc_id]

    hits = []
    for i in ranked[:top_k]:
        hits.append({
            "id": ids[i],
            "text": chunks[i]["text"],
            "metadata": chunks[i]["metadata"],
            "bm25_score": float(scores[i]),
        })
    return hits


def _rrf_fuse(vector_hits, bm25_hits, rrf_k=60):
    """Combine two ranked hit lists via Reciprocal Rank Fusion. A chunk's RRF
    score is the sum, over the methods that retrieved it, of
    1 / (rrf_k + rank), where rank is 0-based."""
    rrf_scores = {}
    items = {}
    for rank, hit in enumerate(vector_hits):
        rrf_scores[hit["id"]] = rrf_scores.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
        items.setdefault(hit["id"], hit)
    for rank, hit in enumerate(bm25_hits):
        rrf_scores[hit["id"]] = rrf_scores.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
        items.setdefault(hit["id"], hit)
    return rrf_scores, items


def hybrid_retrieve(query, top_k=5, doc_id=None, fetch_k=None, rrf_k=60):
    """Fuse vector and BM25 rankings via Reciprocal Rank Fusion (RRF).

    Each method independently retrieves `fetch_k` candidates (default
    max(top_k, 50)) and a chunk's RRF score is the sum, over the methods
    that retrieved it, of 1 / (rrf_k + rank), where rank is 0-based.

    If `doc_id` is given, RRF is computed once over that document's
    candidates. If not, RRF is computed globally first (so single-document
    queries aren't diluted with irrelevant chunks from the other document).
    Then, if any document ends up with *zero* chunks in the top_k -- because
    the query's vocabulary leans toward the other document and its content
    is ranked just outside top_k -- that document's single best-scoring
    chunk is swapped in for the weakest slot held by an over-represented
    document. This guarantees both documents are represented for
    cross-document questions, without forcing per-document parity on
    single-document questions.
    """
    fetch_k = fetch_k or max(top_k, 50)

    if doc_id:
        vector_hits = retrieve(query, top_k=fetch_k, doc_id=doc_id)
        bm25_hits = bm25_retrieve(query, top_k=fetch_k, doc_id=doc_id)
        rrf_scores, items = _rrf_fuse(vector_hits, bm25_hits, rrf_k=rrf_k)
        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [{**items[id_], "rrf_score": rrf_scores[id_]} for id_ in ranked_ids]

    vector_hits = retrieve(query, top_k=fetch_k)
    bm25_hits = bm25_retrieve(query, top_k=fetch_k)
    rrf_scores, items = _rrf_fuse(vector_hits, bm25_hits, rrf_k=rrf_k)
    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    covered = {items[id_]["metadata"]["doc_id"] for id_ in ranked_ids}
    for d in DOCUMENTS:
        if d in covered:
            continue
        d_vector_hits = retrieve(query, top_k=fetch_k, doc_id=d)
        d_bm25_hits = bm25_retrieve(query, top_k=fetch_k, doc_id=d)
        d_scores, d_items = _rrf_fuse(d_vector_hits, d_bm25_hits, rrf_k=rrf_k)
        if not d_scores:
            continue
        best_id = max(d_scores, key=d_scores.get)
        rrf_scores[best_id] = d_scores[best_id]
        items[best_id] = d_items[best_id]

        counts = Counter(items[id_]["metadata"]["doc_id"] for id_ in ranked_ids)
        weakest = next(
            (id_ for id_ in reversed(ranked_ids) if counts[items[id_]["metadata"]["doc_id"]] > 1),
            ranked_ids[-1] if len(ranked_ids) >= top_k else None,
        )
        if weakest is not None:
            ranked_ids.remove(weakest)
        ranked_ids.append(best_id)

    ranked_ids = sorted(ranked_ids, key=rrf_scores.get, reverse=True)
    return [{**items[id_], "rrf_score": rrf_scores[id_]} for id_ in ranked_ids]


def hybrid_rerank(query, fetch_k=30, top_n=5, doc_id=None, rrf_k=60):
    """Hybrid retrieval followed by cross-encoder reranking."""

    fetch_k = max(1, int(fetch_k))
    top_n = max(1, int(top_n))

    candidates = hybrid_retrieve(
        query,
        top_k=fetch_k,
        doc_id=doc_id,
        rrf_k=rrf_k,
    )

    if not candidates:
        return []

    # Never ask reranker for more results than candidates available
    top_n = min(top_n, len(candidates))

    # Keep only valid chunks
    candidates = [
        candidate
        for candidate in candidates
        if candidate.get("id") is not None
        and candidate.get("text")
    ]

    if not candidates:
        return []

    top_n = min(top_n, len(candidates))

    return rerank(
        query,
        candidates,
        top_n=top_n,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the carbon_flat_rag index.")
    parser.add_argument("query", nargs="*", help="query text")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid", "rerank"], default="hybrid")
    parser.add_argument("--fetch-k", type=int, default=30, help="candidate pool size for --mode rerank")
    parser.add_argument("--doc-id", choices=["brsr_2023", "ghg_protocol"], default=None)
    args = parser.parse_args()

    query = " ".join(args.query) or "What GHG emissions does BRSR Principle 6 EI6 require?"
    print(f"query: {query!r}  |  mode={args.mode}  top_k={args.top_k}  doc_id={args.doc_id}\n")

    if args.mode == "vector":
        hits = retrieve(query, top_k=args.top_k, doc_id=args.doc_id)
    elif args.mode == "bm25":
        hits = bm25_retrieve(query, top_k=args.top_k, doc_id=args.doc_id)
    elif args.mode == "rerank":
        hits = hybrid_rerank(query, fetch_k=args.fetch_k, top_n=args.top_k, doc_id=args.doc_id)
    else:
        hits = hybrid_retrieve(query, top_k=args.top_k, doc_id=args.doc_id)

    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        scores = []
        if "rerank_score" in hit:
            scores.append(f"rerank={hit['rerank_score']:.4f}")
        if "rrf_score" in hit:
            scores.append(f"rrf={hit['rrf_score']:.4f}")
        if "distance" in hit:
            scores.append(f"distance={hit['distance']:.4f}")
        if "bm25_score" in hit:
            scores.append(f"bm25={hit['bm25_score']:.4f}")
        print(f"#{i:<2} [{meta['doc_id']} p{meta['page']} {meta['type']}] {' '.join(scores)}")
        print(f"     section: {meta['section']}")
        print(f"     {hit['text'][:200]}")
        print()
