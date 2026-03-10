# Documind

Production RAG system for private document intelligence — hybrid retrieval, parent-child chunking, streaming answers with citations, and automated RAGAS evaluation CI/CD gate.

---

## Project Structure

```
documind/
│
├── api/                        # FastAPI app
│   ├── main.py                 # App entrypoint
│   ├── models.py               # Request/response schemas
│   └── routes/
│       ├── health.py           # Health check endpoint
│       ├── ingest.py           # Document ingestion endpoint
│       └── query.py            # Query endpoint
│
├── ingestion/                  # Document processing pipeline
│   ├── parser.py               # Parse raw files into elements
│   ├── chunker.py              # Split elements into chunks
│   ├── embedder.py             # Generate embeddings
│   └── indexer.py              # Index chunks into vector store
│
├── retrieval/                  # Retrieval pipeline
│   ├── dense_retriever.py      # Vector similarity search
│   ├── sparse_retriever.py     # BM25 keyword search
│   ├── rrf.py                  # Reciprocal rank fusion
│   ├── reranker.py             # Cross-encoder reranking
│   └── parent_fetcher.py       # Parent-child chunk fetcher
│
├── generation/                 # Answer generation
│   ├── llm_client.py           # LLM API client
│   ├── prompt_templates.py     # Prompt templates
│   ├── streamer.py             # Streaming response handler
│   └── memory.py               # Conversation memory
│
├── cache/
│   └── semantic_cache.py       # Redis semantic cache
│
├── evaluation/                 # RAGAS evaluation
│   ├── ragas_evaluator.py      # Evaluation logic
│   └── run_eval.py             # Evaluation runner
│
├── tests/
│   ├── fixtures/               # Test files (PDFs, etc.)
│   └── test_ingestion/
│       └── test_parser.py
│
├── config.py                   # Pydantic settings
├── pyproject.toml              # Dependencies and project metadata
├── Dockerfile
├── docker-compose.yml
└── .env.example
```
