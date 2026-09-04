"""Assemble a prompt from retrieved chunks + the user's question and call Gemini.

This is the final step of the RAG pipeline: take the chunks returned by
retriever.py, format each with its citation metadata (doc_id, page, section),
and send SYSTEM_PROMPT + context + question to Gemini.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a sustainability analyst. Answer the user's question using ONLY the
context chunks provided below -- do not use any outside knowledge.

Each chunk is preceded by a header in this exact format:
[doc_id: <doc_id>, page: <page>, section: "<section>", type: <type>]

Do not put these headers inline in your answer. Instead, number the distinct
chunks you use as [1], [2], [3], ... in the order you first cite them, and
mark each claim with its matching bracketed number immediately after it
(e.g. "...Scope 1 and Scope 2 emissions [1]."). If multiple chunks are used,
mark each one where its information is used.

At the very end of your answer, add a line "Sources:" followed by each
number on its own line paired with its full header, e.g.:
Sources:
[1] [doc_id: brsr_2023, page: 24, section: "PRINCIPLE 6: Businesses should respect and make efforts to protect", type: schema]

If a cited chunk has type: schema (a table) and the question asks about its
contents, enumerate every row/sub-item in that table -- do not summarize or
drop rows. Preserve exact figures, units, and labels as given; do not round
or rephrase numbers.

If the provided context does not contain enough information to answer the
question:
- Say "I don't have enough information in the provided context to answer
  this."
- If any provided chunk is topically related (even if it doesn't fully
  answer the question), suggest it as a starting point, citing it the same
  way (a bracketed number plus its entry in Sources).
- Do not cite or reference anything that is not part of the provided
  context.

Keep answers concise and factual."""

_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client


def build_context(chunks):
    blocks = []
    for c in chunks:
        m = c["metadata"]
        header = (
            f'[doc_id: {m["doc_id"]}, page: {m["page"]}, '
            f'section: "{m["section"]}", type: {m["type"]}]'
        )
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question, chunks, model=GEMINI_MODEL):
    context = build_context(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text


if __name__ == "__main__":
    import sys

    from retriever import hybrid_rerank

    question = " ".join(sys.argv[1:]) or (
        "What greenhouse gas emissions does BRSR Principle 6 Essential "
        "Indicator 6 require companies to disclose?"
    )
    chunks = hybrid_rerank(question, fetch_k=30, top_n=5)

    print(f"question: {question!r}\n")
    print(generate_answer(question, chunks))
