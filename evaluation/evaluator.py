"""End-to-end RAG evaluator: retrieve → generate → LLM-judge score.

For each question in test_questions.json, runs hybrid_rerank to get chunks,
generates an answer with Gemini, then calls a Gemini judge to score on four
dimensions: context_precision, faithfulness, completeness, correctness.

Results are written to evaluation/results/<timestamp>.json and a summary
table is printed to stdout.

Usage:
    # Run all 26 questions
    python evaluation/evaluator.py

    # Run specific question IDs
    python evaluation/evaluator.py --ids 21 22 23 24 25 26

    # Run only sensemaking questions
    python evaluation/evaluator.py --type sensemaking

    # Adjust retrieval pool and final results count
    python evaluation/evaluator.py --fetch-k 30 --top-n 5
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from google.genai import types

from generator import GEMINI_MODEL, generate_answer, get_client
from retriever import hybrid_rerank

QUESTIONS_FILE = Path(__file__).resolve().parent / "test_questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Maps test_questions.json "source" to Chroma doc_id filter.
_SOURCE_TO_DOC_ID = {"brsr": "brsr_2023", "ghg_protocol": "ghg_protocol", "both": None}

JUDGE_PROMPT = """\
You are a RAG evaluation judge. Given a question, retrieved chunks, a generated \
answer, and a ground truth answer, score the generated answer on four dimensions.

QUESTION:
{question}

RETRIEVED CHUNKS (K={k}):
{chunks}

GENERATED ANSWER:
{generated_answer}

GROUND TRUTH:
{ground_truth}

Score each dimension and give one sentence of reasoning. Return ONLY valid JSON \
in exactly this structure — no markdown, no extra text:
{{
  "context_precision": {{
    "score": <integer 0 to {k}>,
    "reason": "<how many chunks were genuinely useful and why>"
  }},
  "faithfulness": {{
    "score": <1 or 0>,
    "reason": "<what claim, if any, goes beyond the retrieved chunks>"
  }},
  "completeness": {{
    "score": <integer 1 to 5>,
    "reason": "<which key points from ground truth were covered or missed>"
  }},
  "correctness": {{
    "score": <integer 1 to 5>,
    "reason": "<where the answer agrees or disagrees with ground truth>"
  }}
}}

Scoring guides:
- context_precision : count of chunks (0-{k}) that directly contributed facts used in the answer
- faithfulness      : 1 = every claim is traceable to the retrieved chunks; 0 = at least one claim is invented
- completeness      : 5=all key points in ground truth covered; 4=most covered; 3=main point only; 2=partial; 1=missed most
- correctness       : 5=fully correct; 4=mostly correct with minor gaps; 3=partially correct; 2=mostly wrong; 1=wrong or contradicts ground truth\
"""


def _format_chunks_for_judge(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        header = f"[{i}] {m['doc_id']} p{m['page']} | {m['section'][:70]}"
        body = c["text"][:500]
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _call_judge(question, chunks, generated_answer, ground_truth, retries=3):
    k = len(chunks)
    prompt = JUDGE_PROMPT.format(
        question=question,
        k=k,
        chunks=_format_chunks_for_judge(chunks),
        generated_answer=generated_answer,
        ground_truth=ground_truth,
    )
    last_err = None
    for attempt in range(retries):
        try:
            response = get_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            return _extract_json(response.text)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))  # 5s, 10s back-off
    raise last_err


def evaluate_one(q, fetch_k=30, top_n=5):
    """Run the full pipeline for a single question and return a scored result dict."""
    question = q["question"]
    ground_truth = q["ground_truth"]
    doc_id = _SOURCE_TO_DOC_ID.get(q["source"])

    chunks = hybrid_rerank(question, fetch_k=fetch_k, top_n=top_n, doc_id=doc_id)
    generated_answer = generate_answer(question, chunks)

    try:
        scores = _call_judge(question, chunks, generated_answer, ground_truth)
    except Exception as e:
        scores = {"error": str(e)}

    return {
        "id": q["id"],
        "question": question,
        "source": q["source"],
        "type": q["type"],
        "generated_answer": generated_answer,
        "chunks_retrieved": len(chunks),
        "scores": scores,
    }


def _summarize(results):
    valid = [r for r in results if "error" not in r.get("scores", {})]
    if not valid:
        print("No valid results to summarize.")
        return

    def avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    faith = [r["scores"]["faithfulness"]["score"] for r in valid]
    cp_norm = [
        r["scores"]["context_precision"]["score"] / max(r["chunks_retrieved"], 1)
        for r in valid
    ]
    compl = [r["scores"]["completeness"]["score"] for r in valid]
    corr = [r["scores"]["correctness"]["score"] for r in valid]

    print("\n" + "=" * 64)
    print(f"OVERALL  (n={len(valid)})")
    print("=" * 64)
    print(f"  Faithfulness (% pass)       : {avg(faith)*100:5.1f}%")
    print(f"  Context Precision (avg N/K) : {avg(cp_norm):5.2f}")
    print(f"  Completeness  (avg /5)      : {avg(compl):5.2f}")
    print(f"  Correctness   (avg /5)      : {avg(corr):5.2f}")

    types_seen = sorted(set(r["type"] for r in valid))
    print("\nBY QUESTION TYPE")
    print("-" * 64)
    print(f"{'Type':<22} {'N':>3}  {'Faith%':>7}  {'CtxP':>6}  {'Compl':>6}  {'Corr':>6}")
    print("-" * 64)
    for t in types_seen:
        grp = [r for r in valid if r["type"] == t]
        f_g = avg([r["scores"]["faithfulness"]["score"] for r in grp]) * 100
        cp_g = avg([
            r["scores"]["context_precision"]["score"] / max(r["chunks_retrieved"], 1)
            for r in grp
        ])
        co_g = avg([r["scores"]["completeness"]["score"] for r in grp])
        cr_g = avg([r["scores"]["correctness"]["score"] for r in grp])
        print(f"{t:<22} {len(grp):>3}  {f_g:>6.0f}%  {cp_g:>6.2f}  {co_g:>6.2f}  {cr_g:>6.2f}")


def run_evaluation(ids=None, types=None, fetch_k=30, top_n=5):
    questions = json.loads(QUESTIONS_FILE.read_text())
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    if types:
        questions = [q for q in questions if q["type"] in types]
    if not questions:
        print("No questions matched the filters.")
        return []

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"eval_{timestamp}.json"

    print(f"Evaluating {len(questions)} question(s) | hybrid+rerank "
          f"fetch_k={fetch_k} top_n={top_n}")
    print("-" * 64)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Q{q['id']:02d} ({q['type']:<14}) "
              f"{q['question'][:55]}...")
        try:
            r = evaluate_one(q, fetch_k=fetch_k, top_n=top_n)
            s = r["scores"]
            if "error" not in s:
                print(f"           cp={s['context_precision']['score']}/{r['chunks_retrieved']}  "
                      f"faith={s['faithfulness']['score']}  "
                      f"compl={s['completeness']['score']}/5  "
                      f"corr={s['correctness']['score']}/5")
            else:
                print(f"           judge error: {s['error']}")
        except Exception as e:
            print(f"           pipeline error: {e}")
            r = {
                "id": q["id"], "question": q["question"],
                "source": q["source"], "type": q["type"], "error": str(e),
            }
        results.append(r)
        if i < len(questions):
            time.sleep(1)

    output = {
        "timestamp": timestamp,
        "config": {"fetch_k": fetch_k, "top_n": top_n, "retrieval_mode": "hybrid+rerank"},
        "results": results,
    }
    results_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nFull results saved → {results_path}")

    _summarize(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the carbon_flat_rag pipeline.")
    parser.add_argument("--ids", nargs="*", type=int, help="specific question IDs to run")
    parser.add_argument("--type", dest="types", nargs="*",
                        help="filter by type (factual, sensemaking, cross_document, ...)")
    parser.add_argument("--fetch-k", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    run_evaluation(ids=args.ids, types=args.types, fetch_k=args.fetch_k, top_n=args.top_n)