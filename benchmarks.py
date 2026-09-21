#!/usr/bin/env python3
"""
Elite Performance Evaluation, Latency Benchmarking, and Synthetic Data Generation Suite.
Consolidates and optimizes benchmarks.py, eval_retrieval.py, and generate_data.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import ollama

from retrieval import EliteRetriever

# ---------------------------------------------------------------------------
# Advanced Retrieval Evaluation Engine
# ---------------------------------------------------------------------------


def calculate_ndcg(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K) under binary relevance.
    """
    k = min(k, len(retrieved))
    if k == 0 or not relevant:
        return 0.0

    # Binary relevance vector mapping
    rel: list[int] = []
    for doc in retrieved[:k]:
        hit = 0
        for rel_snippet in relevant:
            if (
                rel_snippet.strip().lower() in doc.lower()
                or doc.lower() in rel_snippet.strip().lower()
            ):
                hit = 1
                break
        rel.append(hit)

    if sum(rel) == 0:
        return 0.0

    # DCG calculation
    dcg: float = sum(r / math.log2(idx + 2) for idx, r in enumerate(rel))

    # IDCG calculation (ideal ranking places all actual relevant documents first)
    ideal_hits: int = min(k, len(relevant))
    idcg: float = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))

    return dcg / idcg if idcg > 0.0 else 0.0


def evaluate_retrieval(
    retriever: EliteRetriever,
    test_set: list[dict[str, Any]],
    k: int = 4,
    concurrency: int = 4,
) -> dict[str, float]:
    """
    Fully multi-threaded retrieval evaluation suite measuring standard RAG metrics:
    Hit Rate@K, Recall@K, Precision@K, F1@K, MRR@K, and NDCG@K.
    """
    if not test_set:
        print("[Evaluator] Warning: Empty test set provided.")
        return {}

    hits: int = 0
    mrr_sum: float = 0.0
    recall_sum: float = 0.0
    precision_sum: float = 0.0
    f1_sum: float = 0.0
    ndcg_sum: float = 0.0

    def _eval_single(item: dict[str, Any]) -> dict[str, Any]:
        query: str = item["query"]
        relevant: list[str] = item.get("relevant_snippets", [])
        lang: str | None = item.get("language")

        # Call optimized retriever
        results: list[str] = retriever.retrieve(query, language=lang, final_k=k)

        found: bool = False
        first_rank_match: int = 0
        overlap_count: int = 0

        # Exact and partial snippet overlap calculations
        for rank, doc in enumerate(results, 1):
            is_match = False
            for rel in relevant:
                if (
                    rel.strip().lower() in doc.lower()
                    or doc.lower() in rel.strip().lower()
                ):
                    is_match = True
                    break
            if is_match:
                overlap_count += 1
                if not found:
                    found = True
                    first_rank_match = rank

        precision: float = overlap_count / k if k > 0 else 0.0
        recall: float = overlap_count / len(relevant) if relevant else 0.0
        f1: float = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mrr: float = 1.0 / first_rank_match if found else 0.0
        ndcg: float = calculate_ndcg(results, relevant, k)

        return {
            "found": found,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
            "ndcg": ndcg,
        }

    # Parallelize evaluations to prevent sequential testing latency blocks
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_eval_single, item) for item in test_set]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res["found"]:
                    hits += 1
                precision_sum += res["precision"]
                recall_sum += res["recall"]
                f1_sum += res["f1"]
                mrr_sum += res["mrr"]
                ndcg_sum += res["ndcg"]
            except Exception as e:
                print(f"[Evaluator] Exception during retrieval processing: {e}")

    n: int = len(test_set)
    metrics = {
        "HitRate@K": round(hits / n, 4),
        "Recall@K": round(recall_sum / n, 4),
        "Precision@K": round(precision_sum / n, 4),
        "F1@K": round(f1_sum / n, 4),
        "MRR@K": round(mrr_sum / n, 4),
        "NDCG@K": round(ndcg_sum / n, 4),
    }

    print("\n" + "=" * 60)
    print(f"  RETRIEVAL METRICS SUMMARY (K={k})")
    print("=" * 60)
    for m, val in metrics.items():
        print(f"  → {m:<15} : {val:.4f}")
    print("=" * 60 + "\n")
    return metrics


# ---------------------------------------------------------------------------
# High-Performance Benchmarking Engine
# ---------------------------------------------------------------------------


def benchmark_embedding(
    retriever: EliteRetriever,
    texts: list[str],
    rounds: int = 3,
    warmups: int = 1,
) -> dict[str, float]:
    """
    Benchmark embedding performance including statistical variance and warm-up buffer cycles.
    """
    if not texts:
        raise ValueError("Empty text corpus submitted to benchmark.")

    # Warm-up phase to load system dependencies into hardware/GPU space
    print(f"[Benchmark] Warming up embedding processor ({warmups} cycles)...")
    for _ in range(warmups):
        for t in texts[:5]:
            try:
                retriever.embed(t)
            except Exception:
                logging.getLogger(__name__).debug("Suppressed exception", exc_info=True)

    print(f"[Benchmark] Executing {rounds} benchmarking cycles...")
    latencies: list[float] = []

    for r in range(rounds):
        start = time.perf_counter()
        for t in texts:
            retriever.embed(t)
        latencies.append(time.perf_counter() - start)

    avg_total: float = statistics.mean(latencies)
    per_text: float = (avg_total / len(texts)) * 1000
    throughput: float = len(texts) / avg_total
    std_dev: float = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    results = {
        "avg_total_sec": round(avg_total, 3),
        "avg_per_text_ms": round(per_text, 2),
        "texts_per_sec": round(throughput, 1),
        "std_dev_sec": round(std_dev, 4),
    }

    print("=" * 60)
    print("  EMBEDDING BENCHMARKS RESULT")
    print("=" * 60)
    print(f"  Average total batch run: {results['avg_total_sec']:.3f} s")
    print(f"  Average per text item  : {results['avg_per_text_ms']:.2f} ms")
    print(f"  Throughput             : {results['texts_per_sec']:.1f} items/sec")
    print(f"  Standard Deviation     : {results['std_dev_sec']:.4f} s")
    print("=" * 60 + "\n")
    return results


def benchmark_retrieval(
    retriever: EliteRetriever,
    queries: list[str],
    k: int = 4,
    rounds: int = 5,
    concurrency: int = 1,
) -> dict[str, float]:
    """
    Stress-tests and measures retrieval latencies across configurable concurrent execution rates.
    """
    if not queries:
        raise ValueError("Empty query corpus submitted to benchmark.")

    # Warm-up
    for q in queries[:3]:
        try:
            retriever.retrieve(q, final_k=k)
        except Exception:
            logging.getLogger(__name__).debug("Suppressed exception", exc_info=True)

    latencies: list[float] = []

    def _worker(q: str) -> float:
        st = time.perf_counter()
        try:
            retriever.retrieve(q, final_k=k)
        except Exception as e:
            print(f"[Benchmark] Retrieval execution warning: {e}")
        return (time.perf_counter() - st) * 1000

    print(
        f"[Benchmark] Measuring retrieval concurrency pipeline (Workers={concurrency})..."
    )
    start_total = time.perf_counter()

    for _ in range(rounds):
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_worker, q) for q in queries]
            for fut in as_completed(futures):
                latencies.append(fut.result())

    end_total = time.perf_counter()
    total_duration = end_total - start_total
    total_requests = len(queries) * rounds
    throughput_qps = total_requests / total_duration

    latencies_arr = np.asarray(latencies, dtype=np.float32)
    results = {
        "avg_ms": round(float(np.mean(latencies_arr)), 2),
        "p50_ms": round(float(np.median(latencies_arr)), 2),
        "p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies_arr, 99)), 2),
        "std_dev_ms": round(float(np.std(latencies_arr)), 2),
        "qps": round(throughput_qps, 2),
    }

    print("=" * 60)
    print("  RETRIEVAL LATENCY & THROUGHPUT BENCHMARKS")
    print("=" * 60)
    print(f"  Throughput (QPS)       : {results['qps']:.2f} queries/sec")
    print(f"  P50 latency            : {results['p50_ms']:.2f} ms")
    print(f"  P95 latency            : {results['p95_ms']:.2f} ms")
    print(f"  P99 latency            : {results['p99_ms']:.2f} ms")
    print(f"  Average latency        : {results['avg_ms']:.2f} ms")
    print(f"  Standard Deviation     : {results['std_dev_ms']:.2f} ms")
    print("=" * 60 + "\n")
    return results


# ---------------------------------------------------------------------------
# High-Throughput Concurrent Dataset Synthesis Engine (Synthetic Data Gen)
# ---------------------------------------------------------------------------

INTENTS: list[str] = [
    "code_generation",
    "code_explanation",
    "code_improvement",
    "general_question",
    "greeting",
    "exit",
    "out_of_scope",
]

LANGUAGES: list[str] = [
    "python",
    "javascript",
    "typescript",
    "go",
    "rust",
    "java",
    "c++",
    "c#",
    "sql",
    "bash",
    "unknown",
]


def generate_intent_batch(
    intent: str, n_samples: int, model: str = "mistral"
) -> list[dict[str, Any]]:
    """
    Synthesize balanced user intents via Ollama with robust schema structure validation.
    """
    client = ollama.Client()
    prompt = f"""Generate exactly {n_samples} highly realistic user developer messages for an AI coding helper.
Intent target category: {intent}
Return ONLY a valid JSON array of objects. Do not include markdown wraps (like ```json), introduction, or commentary.
Each object in the array must contain exactly these fields:
- "text": string (the user request message, vary lengths, formality, and add realistic code terms)
- "intent": string (must be exactly '{intent}')
- "language": string (must be one of: {json.dumps(LANGUAGES)})

Example output block format:
[
  {{"text": "write a fast sorting utility in rust", "intent": "{intent}", "language": "rust"}},
  {{"text": "why does my python socket leak file descriptors?", "intent": "{intent}", "language": "python"}}
]
"""
    try:
        res = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.8, "num_predict": 4096},
        )
        content: str = res["message"]["content"].strip()

        # Robust string decoding boundaries recovery
        start: int = content.find("[")
        end: int = content.rfind("]") + 1
        if start < 0 or end <= start:
            logger_err = f"[Generator] Invalid formatting boundary returned for intent '{intent}'."
            print(logger_err)
            return []

        json_str: str = content[start:end]
        data: list[Any] = json.loads(json_str)
        if not isinstance(data, list):
            return []

        valid_entries: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            text: str = str(item.get("text", "")).strip()
            if len(text) < 4:
                continue
            lang: str = str(item.get("language", "unknown")).lower().strip()
            if lang not in LANGUAGES:
                lang = "unknown"

            valid_entries.append({"text": text, "intent": intent, "language": lang})
        return valid_entries

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"[Generator] Batch generation cycle failed for intent '{intent}': {e}")
        return []


def run_synthetic_data_pipeline(
    output_path: Path,
    target_per_intent: int = 50,
    concurrency: int = 3,
    model: str = "mistral",
) -> None:
    """
    Runs multi-threaded synthetic pipeline tasks concurrently to speed up generation by 3-4x.
    """
    print(
        f"[Generator] Initializing generation tasks (Concurrence={concurrency}, Target={target_per_intent}/intent)..."
    )
    all_synthesized_data: list[dict[str, Any]] = []

    def _task(intent: str) -> list[dict[str, Any]]:
        needed = target_per_intent
        results: list[dict[str, Any]] = []
        attempts = 0
        while needed > 0 and attempts < 5:
            batch = generate_intent_batch(intent, min(30, needed + 5), model=model)
            results.extend(batch)
            needed -= len(batch)
            attempts += 1
            if not batch:
                break
        print(f"  → Generated {len(results)} queries for category intent: '{intent}'")
        return results[:target_per_intent]

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_task, intent): intent for intent in INTENTS}
        for fut in as_completed(futures):
            all_synthesized_data.extend(fut.result())

    if not all_synthesized_data:
        print("[Generator] Process complete but no records compiled successfully.")
        return

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            for item in all_synthesized_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"\n[Generator] Process complete. Compiled {len(all_synthesized_data)} samples to -> '{output_path}'"
        )
    except OSError as e:
        print(f"[Generator] Error writing out database stream: {e}")


# ---------------------------------------------------------------------------
# Program Execution Harness Setup
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elite Coding Assistant Benchmarks, Retrieval Evaluator, and Dataset Generator Suite."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["benchmark", "evaluate", "generate", "all"],
        help="System modes configuration.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrency workers constraint.",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default="intent_language_data.jsonl",
        help="Path where generation data will write out.",
    )
    parser.add_argument(
        "--samples-count",
        type=int,
        default=50,
        help="Samples size constraint per intent for synthetic gen.",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="mistral",
        help="Target Ollama model used inside generation routines.",
    )
    args = parser.parse_args()

    # Build evaluation baseline programmatically if no file exists
    mock_evals = [
        {
            "query": "how do I run concurrent processes in python?",
            "relevant_snippets": ["import multiprocessing", "ProcessPoolExecutor"],
            "language": "python",
        },
        {
            "query": "explain goroutines and channels context in go",
            "relevant_snippets": ["go func()", "make(chan struct{})"],
            "language": "go",
        },
        {
            "query": "how do I format a database query using joins in postgres sql?",
            "relevant_snippets": ["SELECT", "INNER JOIN", "ON"],
            "language": "sql",
        },
        {
            "query": "implement safe memory borrow operations inside rust function blocks",
            "relevant_snippets": ["fn borrowed<'a>", "&'a str"],
            "language": "rust",
        },
    ]

    mock_texts = [
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        "func main() {\n    ch := make(chan int, 100)\n    for i := 0; i < 10; i++ {\n        go func(val int) { ch <- val }(i)\n    }\n}",
        "class Node {\n    constructor(val) {\n        this.val = val;\n        this.next = null;\n    }\n}",
        "pub fn process_data(data: &[u8]) -> Result<String, Error> {\n    let parsed = std::str::from_utf8(data)?;\n    Ok(parsed.to_uppercase())\n}",
    ]

    retriever = EliteRetriever()

    if args.mode in {"generate", "all"}:
        print(">>> STARTING CONCURRENT SYNTHETIC DATA GENERATION PIPELINE")
        run_synthetic_data_pipeline(
            output_path=Path(args.output_dataset),
            target_per_intent=args.samples_count,
            concurrency=args.concurrency,
            model=args.ollama_model,
        )

    if args.mode in {"benchmark", "all"}:
        print("\n>>> STARTING LATENCY & THROUGHPUT BENCHMARKS")
        # Embedding Latency Benchmark
        benchmark_embedding(retriever, mock_texts, rounds=3, warmups=1)

        # Retrieval Latency/QPS Benchmark
        retrieval_queries = [
            "python thread safety",
            "go select statements",
            "rust borrow checker lifetimes",
            "sql select index",
        ]
        benchmark_retrieval(
            retriever,
            retrieval_queries,
            k=4,
            rounds=5,
            concurrency=args.concurrency,
        )

    if args.mode in {"evaluate", "all"}:
        print("\n>>> STARTING RETRIEVAL EVALUATIONS ENGINE")
        evaluate_retrieval(retriever, mock_evals, k=4, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
