#!/usr/bin/env python3
"""
Tests for the PostgreSQL + pgvector collection, against the local Postgres
instance (same RAG_DATABASE_URL / .env resolution as the app).

Each run creates a unique throwaway table (test_<pid>) and drops it at the
end, so real app data is never touched.

Run:  python -m pytest test_pgstore.py -v
Requires: psycopg, pgvector, pytest, and a local Postgres with `vector`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

from pgstore import PgCollection

DIM: int = 8  # small dim keeps tests fast; column grows if needed

_pytestmark = pytest.mark.usefixtures("collection")


@pytest.fixture()
def collection() -> Iterator[PgCollection]:
    table = f"test_{uuid.uuid4().hex[:10]}"
    pg = PgCollection(table=table, embed_dim=DIM)
    yield pg
    try:
        pg._conn.cursor().execute(f'DROP TABLE IF EXISTS "{table}"')
        pg._conn.commit()
    finally:
        pg.close()


def vec(value: float) -> list[float]:
    return [value] * DIM


def test_upsert_and_count(collection: PgCollection) -> None:
    assert collection.count() == 0
    collection.upsert(
        ids=["a", "b"],
        embeddings=[vec(0.1), vec(0.9)],
        documents=["first doc", "second doc"],
        metadatas=[{"language": "python"}, {"language": "javascript"}],
    )
    assert collection.count() == 2

    # Upsert on existing id replaces, not duplicates.
    collection.upsert(
        ids=["a"],
        embeddings=[vec(0.5)],
        documents=["first doc v2"],
        metadatas=[{"language": "python", "rev": 2}],
    )
    assert collection.count() == 2


def test_get_documents(collection: PgCollection) -> None:
    collection.upsert(
        ids=["a", "b"],
        embeddings=[vec(0.1), vec(0.9)],
        documents=["doc a", "doc b"],
        metadatas=[{"language": "python"}, {"language": "rust"}],
    )
    data = collection.get(include=["documents"])
    assert data["ids"] == ["a", "b"]
    assert data["documents"] == ["doc a", "doc b"]


def test_get_with_where(collection: PgCollection) -> None:
    collection.upsert(
        ids=["a", "b"],
        embeddings=[vec(0.1), vec(0.9)],
        documents=["doc a", "doc b"],
        metadatas=[{"language": "python"}, {"language": "rust"}],
    )
    data = collection.get(where={"language": "rust"})
    assert data["ids"] == ["b"]


def test_query_nearest_by_cosine(collection: PgCollection) -> None:
    collection.upsert(
        ids=["near", "far"],
        embeddings=[vec(0.9), vec(0.1)],
        documents=["close to query", "far from query"],
        metadatas=[{"language": "python"}, {"language": "python"}],
    )
    res = collection.query(
        query_embeddings=[vec(1.0)],
        n_results=2,
        include=["documents", "metadatas"],
    )
    assert res["ids"][0] == ["near", "far"]  # nearest first
    assert res["documents"][0][0] == "close to query"


def test_query_language_filter(collection: PgCollection) -> None:
    collection.upsert(
        ids=["py", "js", "go"],
        embeddings=[vec(0.9), vec(0.9), vec(0.2)],
        documents=["py doc", "js doc", "go doc"],
        metadatas=[
            {"language": "python"},
            {"language": "javascript"},
            {"language": "go"},
        ],
    )
    res = collection.query(
        query_embeddings=[vec(1.0)],
        n_results=5,
        where={"language": "python"},
    )
    assert res["ids"][0] == ["py"]


def test_hnsw_index_created(collection: PgCollection) -> None:
    with collection._conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = %s AND indexdef LIKE '%%hnsw%%'",
            (collection.table,),
        )
        rows = cur.fetchall()
    assert rows, "expected an HNSW index on the embedding column"
    assert rows[0][0].endswith("_embedding_hnsw")


def test_dimension_mismatch_raises(collection: PgCollection) -> None:
    """pgvector columns are fixed-dimension: a mismatched embedding is an error."""
    collection.upsert(
        ids=["d1"],
        embeddings=[vec(1.0)],
        documents=["doc"],
        metadatas=[{}],
    )
    assert collection.count() == 1
    with pytest.raises(ValueError, match="does not match column dimension"):
        collection.upsert(
            ids=["d2"],
            embeddings=[[1.0] * (DIM + 4)],
            documents=["bigger doc"],
            metadatas=[{}],
        )
    assert collection.count() == 1  # nothing half-inserted


def test_dimension_mismatch_on_empty_upsert_batch(collection: PgCollection) -> None:
    """The dimension check runs before any row is written in a batch."""
    with pytest.raises(ValueError, match="does not match column dimension"):
        collection.upsert(
            ids=["x", "y"],
            embeddings=[vec(1.0), [1.0] * (DIM + 4)],
            documents=["a", "b"],
            metadatas=[{}, {}],
        )
    assert collection.count() == 0
