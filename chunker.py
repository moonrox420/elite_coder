#!/usr/bin/env python3
"""
Elite structural + semantic code chunker using tree-sitter.
Supports Python, JS/TS, Go, Rust, Java with concurrent semantic merging.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from tree_sitter import Language, Parser

try:
    import tree_sitter_go as tsgo
    import tree_sitter_java as tsjava
    import tree_sitter_javascript as tsjs
    import tree_sitter_python as tspython
    import tree_sitter_rust as tsrust
    import tree_sitter_typescript as tsts
except ImportError as e:
    raise ImportError(
        "tree-sitter language packages missing. "
        "Install: pip install tree-sitter tree-sitter-python tree-sitter-javascript "
        "tree-sitter-typescript tree-sitter-go tree-sitter-rust tree-sitter-java"
    ) from e

logger: logging.Logger = logging.getLogger("chunker")

# Tree-sitter language loading bindings (with type ignores to bypass Pylance C-extension signature stubs)
LANGUAGE_MAP: dict[str, Any] = {
    "python": Language(tspython.language()),  # type: ignore
    "javascript": Language(tsjs.language()),  # type: ignore
    "js": Language(tsjs.language()),  # type: ignore
    "typescript": Language(tsts.language_typescript()),  # type: ignore
    "ts": Language(tsts.language_typescript()),  # type: ignore
    "go": Language(tsgo.language()),  # type: ignore
    "rust": Language(tsrust.language()),  # type: ignore
    "java": Language(tsjava.language()),  # type: ignore
}

PARSER: Parser = Parser()

TARGETS: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "async_function_definition"},
    "javascript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    },
    "js": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    },
    "ts": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
    },
    "java": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
    },
}

# Pre-compiled regex pattern to avoid hot-path recompilation overhead
COMPLEXITY_RE: re.Pattern = re.compile(
    r"\b(if|for|while|match|case|catch|try|return|await|async)\b"
)


def _text(node: Any, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _name(node: Any, src: bytes, lang: str) -> str:
    n = node.child_by_field_name("name")
    if n:
        return _text(n, src)
    for c in node.children:
        if c.type in ("identifier", "property_identifier", "type_identifier"):
            return _text(c, src)
    return "anonymous"


def _complexity(code: str) -> int:
    keywords: list[str] = COMPLEXITY_RE.findall(code)
    return len(keywords) + code.count("&&") + code.count("||")


def _fallback(code: str, size: int = 1600) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    step: int = size - 200
    for i in range(0, len(code), step):
        c: str = code[i : i + size]
        chunks.append(
            {
                "type": "block",
                "name": f"chunk_{i}",
                "code": c,
                "chars": len(c),
                "complexity": _complexity(c),
                "language": "unknown",
                "id": hashlib.sha256(c.encode("utf-8")).hexdigest()[:16],
            }
        )
    return chunks


def _semantic_merge(
    units: list[dict[str, Any]],
    embed_fn: Callable[[str, str | None], list[float]] | Callable[[str], list[float]],
    threshold: float,
    max_chars: int,
) -> list[dict[str, Any]]:
    """
    Groups logically isolated code blocks using highly parallelized semantic embeddings.
    """

    def _fetch_emb(unit: dict[str, Any]) -> np.ndarray | None:
        try:
            # Safely handle both single and dual parameter embedding APIs
            code_slice: str = unit["code"][:6000]
            if embed_fn.__code__.co_argcount >= 2:
                lang: str | None = unit.get("language")
                emb = embed_fn(code_slice, lang)  # type: ignore
            else:
                emb = embed_fn(code_slice)  # type: ignore
            return np.asarray(emb, dtype=np.float32)
        except Exception as e:
            logger.warning("[Chunker] Concurrent embedding retrieval skipped: %s", e)
            return None

    # Resolve thread pooling over external Ollama connection tasks
    with ThreadPoolExecutor(max_workers=min(8, len(units))) as executor:
        embs: list[np.ndarray | None] = list(executor.map(_fetch_emb, units))

    merged: list[dict[str, Any]] = []
    used: set[int] = set()

    for i, u in enumerate(units):
        if i in used:
            continue
        if embs[i] is None:
            merged.append(u)
            used.add(i)
            continue

        cur: dict[str, Any] = dict(u)
        cur_emb: np.ndarray | None = embs[i]
        used.add(i)

        for j in range(i + 1, min(i + 4, len(units))):
            if j in used or embs[j] is None:
                continue
            a: np.ndarray | None = cur_emb
            b: np.ndarray | None = embs[j]
            if a is None or b is None:
                continue

            denom: float = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            sim: float = float(np.dot(a, b) / denom)

            if sim >= threshold and cur["chars"] + units[j]["chars"] <= max_chars:
                cur["code"] += "\n\n" + units[j]["code"]
                cur["name"] += "+" + units[j]["name"]
                cur["chars"] = len(cur["code"])
                cur["type"] = "semantic_merged"
                cur["complexity"] += units[j]["complexity"]
                try:
                    code_slice_merged: str = cur["code"][:6000]
                    if embed_fn.__code__.co_argcount >= 2:
                        lang_cur: str | None = cur.get("language")
                        cur_emb = np.asarray(
                            embed_fn(code_slice_merged, lang_cur), dtype=np.float32
                        )  # type: ignore
                    else:
                        cur_emb = np.asarray(
                            embed_fn(code_slice_merged), dtype=np.float32
                        )  # type: ignore
                except Exception as e:
                    logger.warning(
                        "[Chunker] Merged element re-embedding failed: %s", e
                    )
                used.add(j)

        merged.append(cur)

    return merged


def elite_chunk(
    code: str,
    language: str,
    embed_fn: (
        Callable[[str, str | None], list[float]] | Callable[[str], list[float]] | None
    ) = None,
    max_chars: int = 1800,
    semantic_threshold: float = 0.76,
) -> list[dict[str, Any]]:
    """
    Produce high-quality structural chunks using Tree-Sitter parsing.
    Optionally merges nearby units that are semantically similar.
    """
    lang: str = language.lower().strip()
    if lang not in LANGUAGE_MAP:
        return _fallback(code)

    PARSER.language = LANGUAGE_MAP[lang]
    src: bytes = code.encode("utf-8")

    try:
        tree = PARSER.parse(src)
    except Exception as e:
        logger.error("[Chunker] AST extraction failed. Reverting to fallback: %s", e)
        return _fallback(code)

    root = tree.root_node
    wanted: set[str] = TARGETS.get(lang, set())

    units: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if node.type in wanted:
            c: str = _text(node, src)
            units.append(
                {
                    "type": node.type,
                    "name": _name(node, src, lang),
                    "code": c,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "chars": len(c),
                    "complexity": _complexity(c),
                    "language": lang,
                }
            )
        for child in node.children:
            walk(child)

    walk(root)

    if not units:
        return _fallback(code)

    if embed_fn is not None and len(units) > 1:
        units = _semantic_merge(units, embed_fn, semantic_threshold, max_chars)

    for u in units:
        u["id"] = hashlib.sha256(u["code"].encode("utf-8")).hexdigest()[:16]

    return units
