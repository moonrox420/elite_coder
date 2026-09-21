# Elite Coding Assistant

A self-hosted coding assistant: DistilBERT dual-head intent/language
classification → hybrid RAG (dense + BM25 + cross-encoder rerank) → Ollama
generation, with tree-sitter chunking, conversation memory, active learning,
and benchmarks.

## Requirements

- **Python 3.10+** with `pip install -r requirements.txt`
- **Ollama** running locally with:
  - `gemma4:cloud` (generation)
  - `nomic-embed-text` (embeddings)
- **PostgreSQL 17/18** with the **pgvector** extension (vector store)

## Database setup (one command)

The vector store runs on PostgreSQL + pgvector (ChromaDB was replaced by it).

```bash
python db_setup.py                        # creates elite_rag + vector extension + HNSW index
python run.py --db-setup                  # same, via the orchestrator
```

`db_setup.py` defaults to `postgres@localhost:5432` and creates a database
named `elite_rag`. Override with `--user/--port/--host/--db`. If your superuser
needs a password, set `PGPASSWORD` before running.

It writes the connection string into `.env`, which the app loads automatically
(real environment variables win over `.env`). Manually:

```bash
# PowerShell
$env:RAG_DATABASE_URL = "postgresql://postgres@localhost:5432/elite_rag"
```

On this machine (Postgres 18 on port 5432 with local trust auth) no password
is needed for the `postgres` user. If neither `RAG_DATABASE_URL` nor `.env`
is set, the app falls back to `postgresql://localhost:5432/elite_rag` (your OS
username). `run.py --start` also runs a fail-fast health check and aborts with
a clear message if the database or the `vector` extension is unreachable.

**Note:** the embedding-column dimension is fixed at table creation (768 for
`nomic-embed-text`). Changing embedding models later requires recreating the
table — a dimension mismatch raises a clear error rather than corrupting data.

## Setup and run

```bash
pip install -r requirements.txt
python run.py --setup        # creates dirs + installs Python deps
python run.py --train        # fine-tune the intent/language classifier
python run.py --start        # interactive assistant REPL
python run.py --benchmark    # latency/throughput/retrieval metrics
python run.py --review       # active-learning labeling queue
```

For `--start` you must have a trained model in `./multi-task-final`
(`run.py --start` trains it automatically if missing).

## Migrating from ChromaDB

If you previously ran the Chroma-based version, its vector store lived in
`./rag_db`. Copy it into Postgres (idempotent upsert):

```bash
python -m pip install chromadb        # temp, only needed to read the old store
python migrate_chroma_to_pg.py        # reads ./rag_db, writes to Postgres
```

Then delete the old `./rag_db` directory (no longer used).

## Tests

```bash
python -m pytest test_pgstore.py -v
```

Runs against the local Postgres (`RAG_DATABASE_URL`/`.env`, same resolution
as the app) using throwaway `test_*` tables, so real data is never touched.

## Index benchmark

```bash
python benchmark_index.py          # clones/seeds on throwaway tables
```

Compares seq scan (exact) vs HNSW vs IVFFlat on the same query set, reporting
latency and recall@10 (needs pgvector >= 0.5 for recall). Use it to decide
HNSW parameters (`m`, `ef_construction`) once real data exists.

## Project layout

| File | Purpose |
| --- | --- |
| `run.py` | Orchestrator: setup / train / start / benchmark / review / db-setup |
| `assistant.py` | REPL entry point; intent+language classification, RAG, memory |
| `retrieval.py` | Hybrid retriever (dense + BM25 + rerank), backed by pgstore |
| `pgstore.py` | PostgreSQL + pgvector collection (Chroma-compatible API) |
| `db_setup.py` | One-command DB bootstrap (used by `run.py --db-setup`) |
| `migrate_chroma_to_pg.py` | One-off ChromaDB → Postgres migration |
| `benchmark_index.py` | HNSW vs IVFFlat vs exact index benchmark |
| `test_pgstore.py` | PgCollection pytest suite (throwaway tables) |
| `chunker.py` | Tree-sitter aware code chunking |
| `memory.py` | Conversation memory with Ollama summarization |
| `active_learning.py` | Uncertainty logging + labeling queue |
| `benchmarks.py` | Retrieval/latency benchmarks |
| `training/train_multitask.py` | Multi-task classifier fine-tuning |# elite_coder
