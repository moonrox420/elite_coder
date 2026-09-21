#!/usr/bin/env python3
"""
Benchmark the pgvector index strategy on throwaway tables.

For k-nearest-neighbour queries it compares:
  - seq scan (no index, exact ground truth)
  - HNSW    (current production index)
  - IVFFlat (alternative)

Everything runs on temporary `bench_*` tables cloned from the real `code`
table (or seeded synthetically when the source is empty), so production data
is never modified.

Usage:
    python benchmark_index.py                     # default 5000 synthetic rows if empty
    python benchmark_index.py --seed 20000 --q 100
    python benchmark_index.py --no-seed           # refuse synthetic rows
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
import uuid

import psycopg
from psycopg import sql

from pgstore import DEFAULT_DSN

logging.basicConfig(level=logging.WARNING)
COSINE_OP: str = "vector_cosine_ops"


def random_unit_vector(dim: int) -> list[float]:
    emb = [random.uniform(-1, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in emb)) or 1.0
    return [x / norm for x in emb]


def ensure_bench_base(
    conn: psycopg.Connection, source: str, dim: int, seed: int, allow_seed: bool
) -> str:
    """Fresh bench_base: clone of `source` if it has rows, else synthetic."""
    table = f"bench_base_{uuid.uuid4().hex[:6]}"
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(source)))
        row = cur.fetchone()
        n = int(row[0]) if row is not None else 0

        if n > 0:
            print(f"[bench] Cloning {n} real rows from '{source}' ...")
            cur.execute(
                sql.SQL("SELECT embedding, document, metadata FROM {}").format(
                    sql.Identifier(source)
                )
            )
            rows = cur.fetchall()
            cur.execute(
                sql.SQL(
                    "CREATE TABLE {} (id TEXT PRIMARY KEY, "
                    "embedding vector({}), document TEXT, metadata JSONB)"
                ).format(sql.Identifier(table), sql.Literal(dim))
            )
            for i, (emb, doc, meta) in enumerate(rows):
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (id, embedding, document, metadata) "
                        "VALUES (%s, %s, %s, %s::jsonb)"
                    ).format(sql.Identifier(table)),
                    (f"seed-{i}", emb, doc, meta),
                )
        elif allow_seed:
            print(f"[bench] '{source}' is empty; seeding {seed} synthetic rows ...")
            cur.execute(
                sql.SQL(
                    "CREATE TABLE {} (id TEXT PRIMARY KEY, "
                    "embedding vector({}), document TEXT, metadata JSONB)"
                ).format(sql.Identifier(table), sql.Literal(dim))
            )
            for i in range(seed):
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (id, embedding, document, metadata) "
                        "VALUES (%s, %s, %s, %s::jsonb)"
                    ).format(sql.Identifier(table)),
                    (f"seed-{i}", random_unit_vector(dim), f"synthetic {i}", "{}"),
                )
        else:
            print("[bench] Source empty and --no-seed set; nothing to do.")
            return ""
    conn.commit()
    return table


def clone_with_index(
    conn: psycopg.Connection, base: str, kind: str, n_rows: int
) -> str:
    """Unindexed clone of base, then attach the requested index kind."""
    table = f"bench_{kind}_{uuid.uuid4().hex[:6]}"
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING ALL)").format(
                sql.Identifier(table), sql.Identifier(base)
            )
        )
        cur.execute(
            sql.SQL(
                "INSERT INTO {} (id, embedding, document, metadata) "
                "SELECT id, embedding, document, metadata FROM {}"
            ).format(sql.Identifier(table), sql.Identifier(base))
        )
        if kind == "hnsw":
            cur.execute(
                sql.SQL(
                    "CREATE INDEX ON {} USING hnsw (embedding {}) "
                    "WITH (m = 16, ef_construction = 64)"
                ).format(sql.Identifier(table), sql.Identifier(COSINE_OP))
            )
        elif kind == "ivfflat":
            lists = max(1, min(2000, int(math.sqrt(n_rows))))
            cur.execute(
                sql.SQL(
                    "CREATE INDEX ON {} USING ivfflat (embedding {}) WITH (lists = {})"
                ).format(
                    sql.Identifier(table),
                    sql.Identifier(COSINE_OP),
                    sql.Literal(lists),
                )
            )
    conn.commit()
    return table


def run_queries(
    conn: psycopg.Connection,
    table: str,
    queries: list[list[float]],
    k: int,
) -> tuple[float, list[list[str]]]:
    """Run the same query set against one table; return (avg_ms, id lists)."""
    times: list[float] = []
    ids: list[list[str]] = []
    with conn.cursor() as cur:
        for qv in queries:
            t0 = time.perf_counter()
            cur.execute(
                sql.SQL(
                    "SELECT id FROM {} ORDER BY embedding <=> %s::vector LIMIT %s"
                ).format(sql.Identifier(table)),
                (qv, k),
            )
            rows = cur.fetchall()
            times.append((time.perf_counter() - t0) * 1000.0)
            ids.append([r[0] for r in rows])
    return sum(times) / len(times), ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="code", help="Source collection table")
    parser.add_argument(
        "--seed", type=int, default=5000, help="Synthetic rows if empty"
    )
    parser.add_argument(
        "--no-seed", action="store_true", help="Never seed synthetic data"
    )
    parser.add_argument("--q", type=int, default=50, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Neighbors per query")
    args = parser.parse_args()

    dim: int = 768
    conn = psycopg.connect(DEFAULT_DSN, connect_timeout=10)
    temps: list[str] = []

    try:
        base = ensure_bench_base(conn, args.table, dim, args.seed, not args.no_seed)
        if not base:
            return
        temps.append(base)

        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(base)))
            row = cur.fetchone()
            n_rows = int(row[0]) if row is not None else 0

        print(f"[bench] {n_rows} rows | {args.q} queries x k={args.k}\n")

        seq_t = clone_with_index(conn, base, "seq", n_rows)
        temps.append(seq_t)
        hnsw_t = clone_with_index(conn, base, "hnsw", n_rows)
        temps.append(hnsw_t)
        ivf_t = clone_with_index(conn, base, "ivfflat", n_rows)
        temps.append(ivf_t)

        # One shared query set so recall comparisons are meaningful.
        queries = [random_unit_vector(dim) for _ in range(args.q)]
        seq_ms, seq_ids = run_queries(conn, seq_t, queries, args.k)
        hnsw_ms, hnsw_ids = run_queries(conn, hnsw_t, queries, args.k)
        ivf_ms, ivf_ids = run_queries(conn, ivf_t, queries, args.k)

        def recall(found: list[list[str]]) -> float:
            return sum(
                len(set(f).intersection(s)) / args.k for f, s in zip(found, seq_ids)
            ) / len(seq_ids)

        print(f"  {'strategy':14s} {'avg ms/query':>14s}   recall@{args.k}")
        print(f"  {'seq (exact)':14s} {seq_ms:>14.2f}   100.0%")
        print(f"  {'HNSW':14s} {hnsw_ms:>14.2f}   {recall(hnsw_ids):>5.1%}")
        print(f"  {'IVFFlat':14s} {ivf_ms:>14.2f}   {recall(ivf_ids):>5.1%}")

        faster_hnsw = seq_ms / hnsw_ms if hnsw_ms else float("inf")
        print(
            f"\n[bench] HNSW is {faster_hnsw:.1f}x faster than exact "
            f"at {recall(hnsw_ids):.0%} recall. "
            "Tune with m/ef_construction on the HNSW index for real data."
        )
    finally:
        for t in temps:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(t))
                )
        conn.commit()
        conn.close()


if __name__ == "__main__":
    main()
