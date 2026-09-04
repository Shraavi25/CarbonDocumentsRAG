"""Turn cleaned BRSR / GHG Protocol records into retrieval chunks.

The two documents need different strategies:
- GHG Protocol is long-form prose -- chunk by paragraph, merging short
  paragraphs forward and splitting oversized ones on sentence boundaries
  with a small overlap.
- BRSR is mostly structural headings/numbered items (prose) plus tables
  that have already been linearized into self-describing sentences (one
  sentence per disclosure row, tagged with the question it answers) --
  each table becomes its own chunk, and the prose is chunked the same way
  as GHG but without the "merge short paragraphs" step, since a short BRSR
  heading block is still a meaningful unit on its own.

Both paths share `_chunk_prose`/`_split_oversized`; only the size knobs and
the extra per-table pass differ. All chunks end up in one flat list with a
`metadata` dict (doc_id, page, section, type) for a single combined index.
"""

import re

from loader import load_document, DOCUMENTS
from cleaner import clean_documents

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# GHG Protocol: real paragraphs of running prose, so merge tiny ones
# (e.g. a lone heading) into a neighbour and cap chunk size generously.
GHG_MIN_CHARS = 200
GHG_MAX_CHARS = 1200
GHG_OVERLAP_CHARS = 150

# BRSR: prose blocks are already short (headings/numbered items), and
# table chunks are already one-sentence-per-row -- only the rare oversized
# table (e.g. the 22-row water table) needs splitting.
BRSR_MAX_CHARS = 1500
BRSR_OVERLAP_CHARS = 100


def _split_into_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _split_oversized(text, max_chars, overlap_chars):
    """Split text on sentence boundaries into pieces <= max_chars, carrying
    the tail of each piece into the next as overlap for context continuity."""
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_RE.split(text)
    pieces, current = [], ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            tail = current[-overlap_chars:]
            current = f"{tail} {sentence}".strip()
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _chunk_prose(text, max_chars, overlap_chars, min_chars=0):
    """Split text into paragraphs, optionally merging consecutive short ones
    forward so no chunk falls below min_chars, then split any chunk that's
    still over max_chars on sentence boundaries."""
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    merged, current = [], ""
    for p in paragraphs:
        current = f"{current}\n\n{p}".strip() if current else p
        if len(current) >= min_chars:
            merged.append(current)
            current = ""
    if current:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{current}"
        else:
            merged.append(current)

    chunks = []
    for piece in merged:
        chunks.extend(_split_oversized(piece, max_chars, overlap_chars))
    return chunks


def _make_chunk(record, text, chunk_type):
    return {
        "text": text,
        "metadata": {
            "doc_id": record["doc_id"],
            "page": record["page"],
            "section": record["section"],
            "type": chunk_type,
        },
    }


def chunk_ghg(records):
    chunks = []
    for r in records:
        for text in _chunk_prose(r["text"], GHG_MAX_CHARS, GHG_OVERLAP_CHARS, min_chars=GHG_MIN_CHARS):
            chunks.append(_make_chunk(r, text, "prose"))
    return chunks


def chunk_brsr(records):
    chunks = []
    for r in records:
        for text in _chunk_prose(r["text"], BRSR_MAX_CHARS, BRSR_OVERLAP_CHARS):
            chunks.append(_make_chunk(r, text, "prose"))
        for t in r["tables"]:
            for text in _split_oversized(t["text"], BRSR_MAX_CHARS, BRSR_OVERLAP_CHARS):
                chunks.append(_make_chunk(r, text, t["type"]))
    return chunks


_CHUNKERS = {"ghg_protocol": chunk_ghg, "brsr_2023": chunk_brsr}


def chunk_documents():
    chunks = []
    for doc_id in DOCUMENTS:
        records = clean_documents(load_document(doc_id))
        chunks.extend(_CHUNKERS[doc_id](records))
    return chunks


if __name__ == "__main__":
    chunks = chunk_documents()
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["metadata"]["doc_id"], []).append(c)

    for doc_id, doc_chunks in by_doc.items():
        sizes = [len(c["text"]) for c in doc_chunks]
        print(f"{doc_id}: {len(doc_chunks)} chunks, "
              f"avg {sum(sizes) // len(sizes)} chars, max {max(sizes)} chars")

    print()
    sample = next(c for c in chunks if c["metadata"]["doc_id"] == "brsr_2023" and c["metadata"]["type"] == "schema")
    print("sample BRSR table chunk:")
    print(sample["metadata"])
    print(sample["text"][:300])

    print()
    sample = next(c for c in chunks if c["metadata"]["doc_id"] == "ghg_protocol")
    print("sample GHG prose chunk:")
    print(sample["metadata"])
    print(sample["text"][:300])
