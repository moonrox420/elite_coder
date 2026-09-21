#!/usr/bin/env python3
"""PostgreSQL + pgvector backed document collection.

A drop-in replacement for the ChromaDB collection interface used by
EliteRetriever, backed by a Postgres table with a ``vector`` column.
Uses psycopg3 and the ``pgvector`` extension.

Connection is configured via the ``RAG_DATABASE_URL`` environment variable
(or the ``dsn`` constructor argument).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql

logger: logging.Logger = logging.getLogger("pgstore")


def _load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """
    Minimal .env loader (no third-party dependency). Sets only keys that are
    not already present in the environment; real env vars win over .env.
    """
    env_file = Path(path) if path else Path(__file__).resolve().parent / ".env"
    if not env_file.is_file():
        return
    try:
        with env_file.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as e:
        logger.warning("[pgstore] Could not read %s: %s", env_file, e)


_load_dotenv()

DEFAULT_DSN: str = os.environ.get(
    "RAG_DATABASE_URL", "postgresql://localhost:5432/elite_rag"
)
TABLE_RE: re.Pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_table(table: str) -> str:
    if not TABLE_RE.match(table):
        raise ValueError(f"Invalid collection table name: {table!r}")
    return table.lower()


def _read_dim(conn: psycopg.Connection, table: str) -> int | None:
    """Parse the embedding column dimension via pg_type format_type."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_catalog.format_type(atttypid, atttypmod) "
                "FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = %s",
                (table, "embedding"),
            )
            row = cur.fetchone()
        if row is None:
            return None
        m = re.search(r"vector\((\d+)\)", row[0])
        return int(m.group(1)) if m else None
    except psycopg.Error as e:
        logger.warning("[pgstore] Could not read embedding dim: %s", e)
        return None


class PgCollection:
    """Maintains a ``code``-style documents table in Postgres.

    Implements the subset of the Chroma ``Collection`` API used by
    EliteRetriever: ``count``, ``get``, ``upsert``, and ``query``.
    """

    def __init__(
        self,
        table: str = "code",
        dsn: str | None = None,
        embed_dim: int | None = None,
    ) -> None:
        """Init.

        Args:
            table (str): Name of the Postgres table holding documents.
            dsn (str): SQLAlchemy/nullable libpq connection string.
            embed_dim (int): Number of embedding dimensions used when the
                table must be created (defaults to 768 when omitted).
        """
        self.table: str = _normalize_table(table)
        self.dsn: str = dsn or DEFAULT_DSN
        self._conn: psycopg.Connection = psycopg.connect(self.dsn, autocommit=True)
        register_vector(self._conn)
        self._enable_extension()
        self._ensure_table(embed_dim)
        self._ensure_hnsw_index()

    # ------------------------------------------------------------------ setup
    def _enable_extension(self) -> None:
        try:
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.Error as e:
            logger.warning("[pgstore] Could not enable 'vector' extension: %s", e)

    def _table_exists(self) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (self.table,))
            row = cur.fetchone()
            return row is not None and row[0] is not None

    def _ensure_table(self, embed_dim: int | None) -> None:
        if self._table_exists():
            return
        dim: int = int(embed_dim or 768)
        # pgvector requires a literal dimension (bound params are rejected).
        # `dim` is an internal int and `table` is regex-validated, so inlining is safe.
        ddl: str = (
            f"CREATE TABLE {self.table} ("
            " id TEXT PRIMARY KEY,"
            f" embedding vector({dim}),"
            " document TEXT,"
            " metadata JSONB NOT NULL DEFAULT '{}'::jsonb"
            ")"
        )
        with self._conn.cursor() as cur:
            cur.execute(ddl)  # type: ignore[arg-type]
        logger.info("[pgstore] Created table '%s' with vector(%s)", self.table, dim)
        self._ensure_hnsw_index()

    def _ensure_hnsw_index(self) -> None:
        """Create the cosine HNSW index if the table exists and it is missing."""
        if not self._table_exists():
            return
        index_name: str = f"{self.table}_embedding_hnsw"
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} "
                    "ON {table} USING hnsw (embedding vector_cosine_ops)"
                ).format(
                    index=sql.Identifier(index_name),
                    table=sql.Identifier(self.table),
                )
            )

    def _check_dim(self, emb: list[float] | Any) -> None:
        """
        Validate an embedding's dimension against the vector column.

        pgvector fixes a column's dimension at creation and cannot resize a
        column that already contains rows, so a mismatch is a hard error rather
        than an automatic grow.
        """
        dim = len(emb) if hasattr(emb, "__len__") else 0
        if dim <= 0:
            return
        current: int | None = _read_dim(self._conn, self.table)
        if current is not None and dim != current:
            raise ValueError(
                f"Embedding dimension {dim} does not match column dimension "
                f"{current} for table '{self.table}'. The column dimension is "
                "fixed by the embedding model at creation; check the embed "
                "model or re-create the table with embed_dim."
            )

    # ------------------------------------------------------------- chroma api
    def count(self) -> int:
        """Number of documents in the collection."""
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(self.table))
            )
            row = cur.fetchone()
            return int(row[0] or 0) if row is not None else 0

    def get(
        self,
        ids: list[str] | None = None,
        include: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return rows as {ids, documents, metadatas}, mirroring Chroma."""
        include = include or ["documents", "metadatas"]
        cols: list[str] = []
        if "documents" in include:
            cols.append("document")
        if "metadatas" in include:
            cols.append("metadata")
        select: str = "id"
        if cols:
            select += ", " + ", ".join(cols)

        clauses: list[str] = []
        params: list[Any] = []
        if ids:
            clauses.append("id = ANY(%s)")
            params.append(ids)
        if where:
            for key, val in where.items():
                clauses.append(" metadata->>%s = %s")
                params.extend([key, val])

        sql: str = f"SELECT {select} FROM {self.table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)  # type: ignore[arg-type]
            rows = cur.fetchall()

        result: dict[str, Any] = {"ids": [r[0] for r in rows]}
        col_order: list[str] = ["document", "metadata"] if cols else []
        for colname in col_order:
            idx: int = 1 + col_order.index(colname)
            if colname == "document" and "document" in cols:
                result["documents"] = [r[idx] for r in rows]
            elif colname == "metadata" and "metadata" in cols:
                result["metadatas"] = [r[idx] for r in rows]
        return result

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or replace documents on primary-key conflict."""
        for emb in embeddings:  # every row must match the fixed column dim
            self._check_dim(emb)
        with self._conn.cursor() as cur:
            stmt = sql.SQL(
                "INSERT INTO {} (id, embedding, document, metadata) "
                "VALUES (%s, %s, %s, %s::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET "
                " embedding = EXCLUDED.embedding,"
                " document = EXCLUDED.document,"
                " metadata = EXCLUDED.metadata"
            ).format(sql.Identifier(self.table))
            for iid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                cur.execute(
                    stmt,
                    (iid, Vector(emb), doc, json.dumps(meta)),
                )

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Nearest-neighbour search ordered by cosine distance, Chroma-style."""
        include = include or ["documents", "metadatas"]
        select: str = "id"
        if "documents" in include:
            select += ", document"
        if "metadatas" in include:
            select += ", metadata"
        select += ", 1 - (embedding <=> %s) AS score"

        clauses: list[str] = []
        params: list[Any] = [Vector(query_embeddings[0])]
        if where:
            for key, val in where.items():
                clauses.append(" metadata->>%s = %s")
                params.extend([key, val])

        sql: str = f"SELECT {select} FROM {self.table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY score DESC LIMIT %s"
        params.append(int(n_results))

        with self._conn.cursor() as cur:
            cur.execute(sql, params)  # type: ignore[arg-type]
            rows = cur.fetchall()

        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]] if "documents" in include else [[]],
            "metadatas": [[r[2] for r in rows]] if "metadatas" in include else [[]],
        }

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()
