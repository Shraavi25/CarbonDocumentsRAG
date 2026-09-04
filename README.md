# Carbon Flat RAG

A flat (single-collection) Retrieval-Augmented Generation system over:

- BRSR 2023 (Business Responsibility and Sustainability Report)
- GHG Protocol Corporate Standard

## Structure

```
carbon_flat_rag/
├── data/                 # source PDFs
├── src/
│   ├── loader.py         # PDF -> raw text/tables
│   ├── cleaner.py         # text normalization
│   ├── chunker.py         # chunking strategy
│   ├── embedder.py         # embedding model + vector store writes
│   ├── retriever.py         # query -> relevant chunks
│   └── generator.py         # prompt assembly + LLM call
├── db/chroma_store/       # persisted Chroma DB
├── evaluation/
│   ├── evaluator.py        # eval harness
│   └── test_questions.json # eval question set
├── app.py                  # entry point
├── requirements.txt
└── .env
```
