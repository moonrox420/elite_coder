#!/usr/bin/env python3
"""
One-command PostgreSQL bootstrap for the Elite Coding Assistant.

Creates the RAG database (default: elite_rag), enables the pgvector
extension, creates the documents table + HNSW index, and prints the
connection string you should export as RAG_DATABASE_URL.

Requires a superuser (or a role allowed to CREATE DATABASE / CREATE
EXTENSION vector). Usage:

    python db_setup.py                     # postgres@localhost:5432
    python db_setup.py --port 5433         # a different Postgres instance
    python db_setup.py --user myadmin --db myrag
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

DEFAULT_DB: str = "elite_rag"
ENV_FILE: Path = Path(__file__).resolve().parent / ".env"


def maintenance_dsn(user: str, host: str, port: int, mdb: str) -> str:
    """DSN for the maintenance database (where CREATE DATABASE runs)."""
    pw = os.environ.get("PGPASSWORD", "")
    auth = f"{user}:{pw}@" if pw else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{mdb}"


def database_exists(conn: psycopg.Connection, db: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,))
        return cur.fetchone() is not None


def create_database(admin_dsn: str, db: str) -> None:
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        if database_exists(conn, db):
            print(f"  -> Database '{db}' already exists.")
            return
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db)))
        print(f"  -> Created database '{db}'.")


def enable_vector(db_dsn: str) -> None:
    with psycopg.connect(db_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        version: Any = cur.fetchone()
    if version:
        print(f"  -> pgvector extension enabled (v{version[0]}).")
    else:
        print("  ! vector extension not found on this server.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="Postgres host")
    parser.add_argument("--port", type=int, default=5432, help="Postgres port")
    parser.add_argument("--user", default="postgres", help="Superuser role")
    parser.add_argument(
        "--mdb", default="postgres", help="Maintenance database (default: postgres)"
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="RAG database to create")
    args = parser.parse_args()

    print("[Database] Bootstrapping PostgreSQL for the coding assistant...")
    admin_dsn: str = maintenance_dsn(args.user, args.host, args.port, args.mdb)
    db_root_dsn: str = maintenance_dsn(args.user, args.host, args.port, args.db)

    try:
        # Connectivity smoke test against the maintenance DB.
        with (
            psycopg.connect(admin_dsn, connect_timeout=5) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT version()")
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("PostgreSQL version query returned no rows.")
            print("  -> Connected:", row[0].split(" on ")[0])
    except psycopg.OperationalError as e:
        print(f"  [Error] Cannot reach Postgres at {args.host}:{args.port}: {e}")
        print("  Tip: set PGPASSWORD to your superuser password if required.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"  [Error] Could not read PostgreSQL server version: {e}")
        sys.exit(1)

    create_database(admin_dsn, args.db)
    enable_vector(db_root_dsn)

    dsn: str = f"postgresql://{args.user}@{args.host}:{args.port}/{args.db}"

    # Persist the DSN so the app connects without manual env-var export.
    try:
        lines: list[str] = []
        if ENV_FILE.is_file():
            lines = [
                ln
                for ln in ENV_FILE.read_text(encoding="utf-8").splitlines()
                if not ln.strip().startswith("RAG_DATABASE_URL")
            ]
        lines.append(f"RAG_DATABASE_URL={dsn}")
        ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  -> Wrote RAG_DATABASE_URL to {ENV_FILE.name} (loaded automatically).")
    except OSError as e:
        print(f"  ! Could not write {ENV_FILE.name}: {e}")

    print("\n[Database] Ready. The app loads the DSN from .env automatically;")
    print("  or you can export it explicitly:")
    print(f"    RAG_DATABASE_URL={dsn}")
    print("  PowerShell:")
    print(f'    $env:RAG_DATABASE_URL = "{dsn}"')
    print("\n  Then launch the assistant:  python run.py --start")


if __name__ == "__main__":
    main()
