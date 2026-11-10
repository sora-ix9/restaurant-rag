# Restaurant RAG API

An **Agentic Hybrid Restaurant Intelligence System** built with FastAPI, Milvus, and OpenAI. It answers restaurant-domain questions with Retrieval-Augmented Generation over a Milvus vector store, with a pluggable LLM provider (GPT-4o or Qwen) and a multi-format ingestion pipeline.

## Architecture

```
app/
├── main.py                    # FastAPI app factory and entry point
├── config.py                  # pydantic-settings, env-overridable
├── routers/
│   └── api.py                 # Route definitions
├── models/
│   ├── requests.py            # QueryRequest
│   └── responses.py           # QueryResponse, SourceDocument
└── services/
    ├── loaders.py             # Per-format chunk loaders (.md/.csv/.json/.pdf)
    ├── milvus_service.py      # Ingestion + dense retrieval
    ├── openai_service.py      # Embeddings + GPT-4o generation
    ├── qwen_service.py        # Qwen generation via DashScope
    ├── prompts.py             # Shared prompt templates
    └── rag_service.py         # Pipeline orchestration
data/                          # Source corpora (git-ignored)
docs/                          # Design docs and research notes
notebook/                      # Prototyping and EDA notebooks
scripts/                       # Ingestion CLI
tests/                         # Test suite
```

### Request flow

```
POST /query
   └─ rag_service.answer_query()
        ├─ milvus_service.search()      embed query → Milvus top-K
        ├─ build_context()              join retrieved chunks
        └─ generate_answer()            GPT-4o or Qwen, grounded in context
```

Retrieval and generation are separate concerns: `search()` calls only the embedding API, never an LLM.

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| API framework | FastAPI | 0.135.2 |
| ASGI server | Uvicorn | 0.42.0 |
| Vector database | PyMilvus | 2.6.9 |
| LLM / Embeddings | OpenAI | 2.20.0 |
| Settings | pydantic-settings | 2.7.1 |
| HTTP client (Qwen) | httpx | 0.28.1 |
| Tokenizer | tiktoken | 0.13.0 |
| PDF extraction | pypdf | — |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check / root |
| `POST` | `/query` | RAG query endpoint |

### `POST /query`

```jsonc
// Request
{
  "question": "ร้านไหนก๋วยเตี๋ยวอร่อย",
  "top_k": 5,              // optional, defaults to settings.top_k
  "provider": "openai"     // optional: "openai" | "qwen"
}

// Response
{
  "answer": "...",
  "sources": [
    { "text": "...", "source": "w_review_train.csv#row42", "score": 0.81 }
  ]
}
```

Interactive docs at `http://localhost:8000/docs` (Swagger UI) and `/redoc` when the server is running.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for the Milvus stack)
- An OpenAI API key

### Installation

```bash
pip install -r requirements.txt
```

### Start Milvus

```bash
docker-compose up -d
```

Brings up Milvus standalone with etcd and MinIO.

### Configuration

Settings live in [app/config.py](app/config.py) and are overridable via `env/.env`:

```bash
OPENAI_API_KEY=sk-...

# Optional — only needed when llm_provider=qwen
DASHSCOPE_API_KEY=sk-...
LLM_PROVIDER=openai

# Milvus target
MILVUS_URI=http://localhost:19530
MILVUS_DATABASE=restaurant_db
MILVUS_COLLECTION=wongnai_review_coll
```

> The read path uses `MILVUS_DATABASE` / `MILVUS_COLLECTION`. If you ingest into a different database, update these or `/query` will not find the data.

### Run

```bash
python -m app.main
```

Serves on `http://0.0.0.0:8000` with hot-reload.

## Ingestion

Load a corpus into Milvus with [scripts/ingest.py](scripts/ingest.py).

### Supported formats

| Extension | Chunk unit | `source` format |
|---|---|---|
| `.md` | Section (split on `# `) | `product_faq.md` |
| `.csv` | Row | `catalogue.csv#row42` |
| `.json` / `.jsonl` | Record | `reviews.jsonl#87` |
| `.pdf` | Page | `policy.pdf#page3` |

CSV delimiters are auto-detected (`,` and `;`), with a counting fallback for files that defeat `csv.Sniffer`.

### Flags

| Flag | Purpose |
|---|---|
| `--pattern` | Glob for the corpus |
| `--database` | Target Milvus database, created if absent |
| `--collection` | Target collection |
| `--keep-existing` | Append instead of dropping first |
| `--limit N` | Embed at most N chunks (the whole corpus is still parsed) |
| `--columns` | CSV: column names; supplying these treats the file as headerless |
| `--text-columns` | CSV: columns to embed (default: all but `--meta-columns`) |
| `--meta-columns` | CSV: columns stored as metadata but never embedded |

### Examples

```bash
# Defaults from config
python -m scripts.ingest

# Markdown corpus into a named collection
python -m scripts.ingest --pattern "./data/faq/*.md" --collection faq_docs

# Headerless semicolon CSV; rating stored but kept out of the vector
python -m scripts.ingest \
  --pattern "./data/reviews.csv" \
  --database restaurant_db --collection reviews \
  --columns review,rating --meta-columns rating
```

> `--limit N` exists for cheap trial runs — embedding is billed per token, so validating a new corpus on a few hundred rows before committing to the full set is worth the extra flag.

### Metadata vs. embedded text

`--meta-columns` splits a row: named columns are stored on the Milvus record but excluded from the embedded text. A numeric rating in the vector makes two reviews look similar for sharing a score rather than a topic.

Metadata is currently **stored but not yet queryable** — `search()` does not request it. Filtered search lands in Phase 2; storing it now avoids re-embedding the corpus later.

## Testing

```bash
pytest tests/ -q
```

Loader tests run against temp files and need neither Milvus nor network access.

## References

- [Milvus Documentation](https://milvus.io/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Wongnai Corpus](https://github.com/wongnai/wongnai-corpus)
