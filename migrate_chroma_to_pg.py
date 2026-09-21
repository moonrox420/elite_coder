#!/usr/bin/env python3
"""
One-off migration: copy the local ChromaDB vector store (./rag_db) into the
PostgreSQL + pgvector collection.

    python migrate_chroma_to_pg.py            # uses RAG_DATABASE_URL or defaults
    python migrate_chroma_to_pg.py --rag-dir /old/path --table code

Requirements:
- The destination database must exist with the `vector` extension (run `python db_setup.py` first).
- `chromadb` must be importable for reading the old store (it was removed
from requirements.txt after the migration to Postgres), e.g.
`pip install chromadb` if it is no longer installed.

The migration is idempotent: rows are upserted on primary-key conflict, so
re-running it is safe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pgstore import PgCollection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rag-dir",
        default="./rag_db",
        help="Path of the old ChromaDB persistent directory (default: ./rag_db)",
    )
    parser.add_argument(
        "--chroma-collection",
        default="code",
        help="Chroma collection name (default: code)",
    )
    parser.add_argument(
        "--table",
        default="code",
        help="Destination Postgres table (default: code)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Postgres DSN (default: RAG_DATABASE_URL env or localhost elite_rag)",
    )
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    if not (rag_dir.exists() and rag_dir.is_dir()):
        print(f"[Migrate] No Chroma store found at '{rag_dir}'. Nothing to do.")
        sys.exit(0)

    try:
        import chromadb  # type: ignore[import-not-found]
        from chromadb.config import Settings  # type: ignore[import-not-found]
    except ImportError:
        print(
            "[Migrate] chromadb is not installed. The old store cannot be read.\n"
            "  Install it temporarily:  python -m pip install chromadb\n"
            "  (It was removed from requirements.txt after the Postgres migration.)"
        )
        sys.exit(1)

    print(
        f"[Migrate] Reading Chroma collection '{args.chroma_collection}' from {rag_dir} ..."
    )
    client = chromadb.PersistentClient(
        path=str(rag_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(args.chroma_collection)
    data = collection.get(include=["documents", "metadatas", "embeddings"])

    ids: list[str] = data.get("ids") or []
    docs: list[str] = data.get("documents") or []
    metas: list = data.get("metadatas") or []
    embs: list = data.get("embeddings") or []

    if not ids:
        print("[Migrate] Chroma collection is empty. Nothing to migrate.")
        sys.exit(0)

    embed_dim: int = len(embs[0]) if embs else 768
    print(
        f"[Migrate] Found {len(ids)} documents (embeddings dim={embed_dim}). "
        "Writing to Postgres ..."
    )
    pg = PgCollection(table=args.table, dsn=args.dsn, embed_dim=embed_dim)

    metas_safe: list[dict] = [dict(m) if isinstance(m, dict) else {} for m in metas]
    pg.upsert(
        ids=ids,
        embeddings=embs,
        documents=docs,
        metadatas=metas_safe,
    )
    total: int = pg.count()
    pg.close()

    print(f"[Migrate] Done. {total} rows now in Postgres table '{args.table}'.")


if __name__ == "__main__":
    main()
