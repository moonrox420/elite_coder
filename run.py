#!/usr/bin/env python3
"""
Elite Coding Assistant Orchestrator.
System entry point to automate environment setups, model training, active learning, and benchmark execution.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import psycopg

# Windows consoles often default to cp1252, which cannot encode the "→" used in
# output below. Reconfigure to UTF-8 (never raising) so the CLI works anywhere.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REQUIRED_DIRECTORIES: list[Path] = [
    Path("./multi-task-final"),
]

DEPENDENCIES: list[str] = [
    "torch",
    "transformers",
    "datasets",
    "scikit-learn",
    "numpy",
    "ollama",
    "psycopg[binary]",
    "pgvector",
    "rank_bm25",
    "sentence-transformers",
    "tree-sitter",
    "tree-sitter-python",
    "tree-sitter-javascript",
    "tree-sitter-typescript",
    "tree-sitter-go",
    "tree-sitter-rust",
    "tree-sitter-java",
    "tenacity",
]


def check_environment() -> bool:
    """
    Verifies system directories and warns of structural anomalies.
    """
    print("[Orchestrator] Verifying workspace environment...")
    for directory in REQUIRED_DIRECTORIES:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"  → Created directory: '{directory}'")
            except OSError as e:
                print(f"  [Error] Failed to initialize directory '{directory}': {e}")
                return False
        else:
            print(f"  → Verified directory: '{directory}'")
    return True


def install_dependencies() -> None:
    """
    Validates and updates environment dependencies safely inside the runtime.
    """
    print("[Orchestrator] Validating dependencies installation...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            capture_output=True,
        )
        print("  → Core packaging structures updated.")

        # Incremental installation to prevent dependency lock freezes
        for dep in DEPENDENCIES:
            print(f"  → Checking package: {dep}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                check=True,
                capture_output=True,
            )
        print("[Orchestrator] Environment dependencies compiled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"  [Error] Dependency installer execution failure: {e.stderr.decode()}")
        sys.exit(1)


def check_database() -> bool:
    """
    Fail-fast health check: verifies Postgres is reachable and pgvector is
    enabled before launching the assistant. Returns True when healthy.
    """
    try:
        from pgstore import DEFAULT_DSN  # keeps DSN/.env resolution in one place
    except ImportError:
        print(
            "[Error] pgstore.py is required but not importable. "
            "Run: python -m pip install -r requirements.txt"
        )
        return False

    print("[Orchestrator] Verifying PostgreSQL vector store...")
    try:
        with (
            psycopg.connect(DEFAULT_DSN, connect_timeout=5) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
            has_vector: int = int(cur.fetchone()[0] or 0)
    except psycopg.OperationalError as e:
        print(f"  [Error] Cannot reach PostgreSQL: {e}")
        print(
            "  Run 'python run.py --db-setup' to create the database, or check RAG_DATABASE_URL/.env."
        )
        return False

    if not has_vector:
        print("  [Error] pgvector extension is not enabled in the database.")
        print("  Run 'python run.py --db-setup' to enable it.")
        return False

    print("  -> PostgreSQL + pgvector OK")
    return True


def execute_submodule(script_name: str, args_list: list[str] | None = None) -> None:
    """
    Subprocess launcher utilizing correct python path overrides to execute optimized scripts.
    """
    script_path = Path(script_name)
    if not script_path.exists():
        print(
            f"[Error] Target script module '{script_name}' is missing in active workspace."
        )
        sys.exit(1)

    print(f"\n[Orchestrator] Launching module: '{script_name}'")
    print("-" * 60)

    cmd: list[str] = [sys.executable, str(script_path)]
    if args_list:
        cmd.extend(args_list)

    try:
        # Stream live stdout/stderr of executing subprocess straight to parent console
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"\n[Error] Module execution step returned a non-zero termination status: {e.returncode}"
        )
    except KeyboardInterrupt:
        print(
            f"\n[Orchestrator] Submodule '{script_name}' execution terminated by user."
        )
    print("-" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elite Coding Assistant Orchestrator console wrapper."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--setup",
        action="store_true",
        help="Validate directories and install/upgrade all required workspace pip packages.",
    )
    group.add_argument(
        "--train",
        action="store_true",
        help="Fine-tune and compile the dual-head intent/language classifier.",
    )
    group.add_argument(
        "--start",
        action="store_true",
        help="Launch the interactive RAG-powered Coding Assistant terminal.",
    )
    group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance latency, throughput, and retrieval validation metrics.",
    )
    group.add_argument(
        "--review",
        action="store_true",
        help="Open the interactive CLI active learning labeling queue.",
    )
    group.add_argument(
        "--db-setup",
        action="store_true",
        help="Create the PostgreSQL RAG database + pgvector extension and print the DSN.",
    )

    args = parser.parse_args()

    # Pre-flight environment check
    if not check_environment():
        print("[Error] Workspace validation failed. Aborting execution.")
        sys.exit(1)

    if args.setup:
        install_dependencies()

    elif args.train:
        execute_submodule(str(Path("training") / "train_multitask.py"))

    elif args.start:
        # Fail fast on a missing/broken DB before the app boots.
        if not check_database():
            print("[Error] Database check failed. Aborting.")
            sys.exit(1)

        # Startup must not silently retrain the model on every launch.
        weights: Path = Path("./multi-task-final/pytorch_model.bin")
        labels: Path = Path("./multi-task-final/labels.json")
        if not weights.exists() or not labels.exists():
            print(
                "[Error] Trained classifier artifacts are missing. "
                "Run 'python run.py --train' before starting the assistant."
            )
            sys.exit(1)
        execute_submodule("assistant.py")

    elif args.benchmark:
        execute_submodule("benchmarks.py", ["--mode", "all"])

    elif args.review:
        execute_submodule("active_learning.py")

    elif args.db_setup:
        execute_submodule("db_setup.py")


if __name__ == "__main__":
    main()
