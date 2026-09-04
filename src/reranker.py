"""Re-rank retrieved chunks with a cross-encoder for higher-precision ordering.

Unlike the bi-encoder used for vector search (which embeds the query and each
chunk independently), a cross-encoder scores a (query, chunk) pair jointly --
much more accurate, but too slow to run over the whole corpus. It's meant to
re-score a small candidate pool (e.g. the top 20-30 from hybrid_retrieve)
down to the few chunks actually passed to the generator.
"""

from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank(query, candidates, top_n=5):
    """Score each candidate jointly with the query and return the top_n,
    each augmented with a `rerank_score` (higher = more relevant)."""
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = get_cross_encoder().predict(pairs)

    rescored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    return sorted(rescored, key=lambda c: c["rerank_score"], reverse=True)[:top_n]
