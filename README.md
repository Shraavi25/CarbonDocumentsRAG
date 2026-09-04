# Carbon Flat RAG

A flat (single-collection) Retrieval-Augmented Generation system built over sustainability compliance reference documents:

- **BRSR 2023** (Business Responsibility and Sustainability Report)
- **GHG Protocol Corporate Standard**

Given a query about BRSR or GHG Protocol requirements, the system retrieves the most relevant source passages and generates a grounded answer — rather than relying on an LLM's raw knowledge of these standards.

## Why "flat" RAG

This implementation uses a single flat vector collection (as opposed to a hierarchical/graph-based retrieval structure) to test how far a simpler retrieval architecture goes on regulatory/compliance Q&A before reaching for more complex approaches.

## Pipeline

PDF source docs
     │
     ▼
  loader.py        → raw text/tables extracted from PDFs
     │
     ▼
  cleaner.py        → text normalization
     │
     ▼
  chunker.py         → chunking strategy
     │
     ▼
  embedder.py        → embedding model + vector store writes
     │
     ▼
  db/chroma_store/   → persisted Chroma vector DB
     │
     ▼
  retriever.py       → query → relevant chunks
     │
     ▼
  reranker.py         → reorders retrieved chunks by relevance
     │
     ▼
  generator.py        → prompt assembly + LLM call
     │
     ▼
     Answer
```

## Structure

```
carbon_flat_rag/
├── data/                        # Source PDFs (BRSR 2023, GHG Protocol Corporate Standard)
├── cleaned_preview/              # Preview output of cleaned/normalized text
├── db/
│   └── chroma_store/             # Persisted Chroma vector DB
├── evaluation/
│   ├── evaluator.py               # Eval harness (RAGAS)
│   └── test_questions.json        # 25-question golden evaluation dataset
├── src/
│   ├── loader.py                  # PDF → raw text/tables
│   ├── cleaner.py                 # Text normalization
│   ├── chunker.py                 # Chunking strategy
│   ├── embedder.py                # Embedding model + vector store writes
│   ├── retriever.py                # Query → relevant chunks
│   ├── reranker.py                 # Reorders retrieved chunks by relevance
│   └── generator.py                # Prompt assembly + LLM call
├── .claude/                      # Claude Code project config
├── app.py                        # Entry point
├── chunks.json                   # Generated chunk store
├── .env                          # API keys (not committed)
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

1. Navigate to the project folder:
   ```bash
   cd carbon_flat_rag
   ```

2. Activate the virtual environment (already present in this project as `venv/`):
   ```bash
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
   If you need to recreate it: `python3 -m venv venv` (or `conda create -n carbon-flat-rag python=3.12` as an alternative — use a distinct env name from other projects).

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables in `.env` — this project calls an LLM and an embedding model, so it requires the relevant API key(s) (check `generator.py` and `embedder.py` for the exact variable names expected, e.g. `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`).

## Building the knowledge base

Before querying, the source PDFs need to be ingested and embedded:

1. Place source PDFs (BRSR 2023, GHG Protocol Corporate Standard) into `data/`.
2. Run the ingestion pipeline (loader → cleaner → chunker → embedder) to populate `db/chroma_store/` and generate `chunks.json`.

## Running the app

```bash
python app.py
```
*(or `streamlit run app.py` if this is a Streamlit interface — check the top of `app.py` to confirm)*

## Evaluation

Retrieval and generation quality is evaluated with **RAGAS** against a 25-question golden dataset:

```bash
python evaluation/evaluator.py
```

Results are scored against `evaluation/test_questions.json`.

## Methodology

- Source standards: BRSR 2023, GHG Protocol Corporate Standard
- Single flat vector collection (Chroma) — no graph/hierarchical retrieval structure
- Retrieval augmented with a reranking step before generation
- Evaluated quantitatively via RAGAS metrics against a hand-authored golden question set
