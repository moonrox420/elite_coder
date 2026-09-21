#!/usr/bin/env python3
"""
Elite hybrid retriever: dense (Ollama embeddings) + BM25 + cross-encoder reranker.
Persistent ChromaDB store with language filtering, memory caching, and GPU reranking.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import numpy as np
import ollama
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pgstore import PgCollection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger: logging.Logger = logging.getLogger("retrieval")

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class EliteRetriever:
    __slots__ = (
        "_bm25_dirty",
        "_corpus_cache",
        "_ids_cache",
        "_tokenized_cache",
        "bm25",
        "chroma",
        "collection",
        "embed_model",
        "ollama",
        "reranker",
    )

    def __init__(
        self,
        embed_model: str = "nomic-embed-text",
        pg_table: str = "code",
        pg_dsn: str | None = None,
        embed_dim: int = 768,
    ) -> None:
        self.embed_model: str = embed_model
        ollama_host = os.environ.get("OLLAMA_HOST", "")
        if "0.0.0.0" in ollama_host:
            self.ollama: ollama.Client = ollama.Client(host="http://127.0.0.1:11434")
        else:
            self.ollama: ollama.Client = ollama.Client()

        # PostgreSQL + pgvector backed collection (replaces local ChromaDB).
        self.collection: PgCollection = PgCollection(
            table=pg_table,
            dsn=pg_dsn,
            embed_dim=embed_dim,
        )
        self.reranker: CrossEncoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
            device=DEVICE,
        )
        self.bm25: BM25Okapi | None = None

        # O(1) in-memory cache structures to eliminate database rebuild overheads
        self._corpus_cache: list[str] = []
        self._tokenized_cache: list[list[str]] = []
        self._ids_cache: list[str] = []
        self._bm25_dirty: bool = True

        self._load_cache()

    def _load_cache(self) -> None:
        """
        One-time initialization lookup that caches existing ChromaDB documents in-memory
        to avoid repetitive database I/O during dynamic document ingestion.
        """
        try:
            data: dict[str, Any] = self.collection.get(include=["documents"])
            docs: list[str] = data.get("documents") or []
            ids: list[str] = data.get("ids") or []

            self._corpus_cache = []
            self._ids_cache = []
            self._tokenized_cache = []

            for i, d in zip(ids, docs):
                if d:
                    tokens: list[str] = d.lower().split()
                    if len(tokens) >= 4:
                        self._corpus_cache.append(d)
                        self._ids_cache.append(i)
                        self._tokenized_cache.append(tokens)

            self._bm25_dirty = True
        except Exception as e:
            logger.error("Failed to load initial search document cache: %s", e)
            self._corpus_cache = []
            self._ids_cache = []
            self._tokenized_cache = []
            self._bm25_dirty = True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.6, min=1, max=5),
        # Pylance fix: ollama.ResponseError fails in strict evaluation; rely on our raised RuntimeError
        retry=retry_if_exception_type(
            (ConnectionError, TimeoutError, OSError, RuntimeError)
        ),
    )
    def embed(self, text: str, language: str | None = None) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Empty text for embedding")
        payload: str = (
            f"language: {language}\n\n{text}"
            if language and language != "unknown"
            else text
        )
        try:
            # Pylance fix: EmbeddingsResponse typing overrides Dict
            res: Any = self.ollama.embeddings(
                model=self.embed_model,
                prompt=payload[:8192],
            )
            emb: list[float] | None = res.get("embedding")
            if not emb or len(emb) < 32:
                raise RuntimeError("Invalid or missing embedding from Ollama")
            return emb
        except Exception as e:
            raise RuntimeError(f"Ollama network embedding error: {e}") from e

    def _rebuild_bm25_if_dirty(self) -> None:
        """
        Lazily compiles the BM25Okapi index structure only upon search execution,
        converting sequential O(N^2) ingestion bottlenecks into linear O(N) operations.
        """
        if self._bm25_dirty:
            if self._tokenized_cache:
                try:
                    self.bm25 = BM25Okapi(self._tokenized_cache)
                except Exception as e:
                    logger.error("BM25 Lazy index compilation failed: %s", e)
                    self.bm25 = None
            else:
                self.bm25 = None
            self._bm25_dirty = False

    def add(self, code: str, metadata: dict[str, Any]) -> bool:
        try:
            if len(code.strip()) < 20:
                return False
            doc_id: str = (
                metadata.get("id")
                or hashlib.sha256(code.encode("utf-8")).hexdigest()[:20]
            )
            emb: list[float] = self.embed(code, metadata.get("language"))
            self.collection.upsert(
                ids=[doc_id],
                embeddings=[emb],
                documents=[code],
                metadatas=[metadata],
            )

            # O(1) in-memory cache append instead of calling full database GET
            tokens: list[str] = code.lower().split()
            if len(tokens) >= 4:
                self._corpus_cache.append(code)
                self._ids_cache.append(doc_id)
                self._tokenized_cache.append(tokens)
                self._bm25_dirty = True
            return True
        except Exception as e:
            logger.error("Add operation failed: %s", e)
            return False

    def add_batch(self, codes: list[str], metadatas: list[dict[str, Any]]) -> int:
        """
        High-throughput bulk ingestion method utilizing vectorized embeddings and single
        database upsert calls to maximize indexing performance.
        """
        if not codes or len(codes) != len(metadatas):
            return 0

        valid_ids: list[str] = []
        valid_codes: list[str] = []
        valid_metas: list[dict[str, Any]] = []
        valid_embs: list[list[float]] = []

        for code, meta in zip(codes, metadatas):
            if len(code.strip()) < 20:
                continue
            doc_id: str = (
                meta.get("id") or hashlib.sha256(code.encode("utf-8")).hexdigest()[:20]
            )
            try:
                emb = self.embed(code, meta.get("language"))
                valid_ids.append(doc_id)
                valid_codes.append(code)
                valid_metas.append(meta)
                valid_embs.append(emb)
            except Exception as e:
                logger.warning("Batch embedding skipped for doc '%s': %s", doc_id, e)

        if not valid_ids:
            return 0

        try:
            self.collection.upsert(
                ids=valid_ids,
                embeddings=valid_embs,
                documents=valid_codes,
                metadatas=valid_metas,
            )
            for code, did in zip(valid_codes, valid_ids):
                tokens = code.lower().split()
                if len(tokens) >= 4:
                    self._corpus_cache.append(code)
                    self._ids_cache.append(did)
                    self._tokenized_cache.append(tokens)
            self._bm25_dirty = True
            return len(valid_ids)
        except Exception as e:
            logger.error("Batch upsert operation failed: %s", e)
            return 0

    def hybrid_search(
        self,
        query: str,
        language: str | None = None,
        k: int = 12,
    ) -> list[tuple[str, dict[str, Any], float]]:
        if not query.strip() or self.collection.count() == 0:
            return []

        self._rebuild_bm25_if_dirty()
        scores: dict[str, float] = {}

        # Dense retrieval phase
        dense: dict[str, Any] | None = None
        try:
            emb: list[float] = self.embed(query, language)
            where: dict[str, Any] | None = (
                {"language": language} if language and language != "unknown" else None
            )
            dense = self.collection.query(
                query_embeddings=[emb],
                n_results=min(k, self.collection.count()),
                where=where,
                include=["documents", "metadatas"],
            )
            for rank, did in enumerate(dense["ids"][0]):
                scores[did] = scores.get(did, 0.0) + 1.0 / (60 + rank)
        except Exception as e:
            logger.error("Dense search phase failed: %s", e)

        # Sparse BM25 retrieval phase
        if self.bm25 and self._corpus_cache:
            try:
                bm25_scores: np.ndarray = self.bm25.get_scores(query.lower().split())
                for rank, idx in enumerate(np.argsort(bm25_scores)[::-1][:k]):
                    if idx < len(self._ids_cache):
                        did: str = self._ids_cache[idx]
                        scores[did] = scores.get(did, 0.0) + 1.0 / (60 + rank)
            except Exception as e:
                logger.warning("Sparse BM25 evaluation failed: %s", e)

        if not scores:
            return []

        ranked: list[tuple[str, float]] = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )[:k]

        # Reuse documents already returned by the dense query and the in-memory
        # BM25 corpus instead of issuing a second ChromaDB fetch per candidate.
        doc_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
        if dense is not None:
            try:
                dense_ids: list[str] = dense["ids"][0]
                dense_docs: list[str] = dense["documents"][0]
                dense_metas: list[dict[str, Any]] = dense["metadatas"][0]
                for did, doc, meta in zip(dense_ids, dense_docs, dense_metas):
                    doc_by_id[str(did)] = (str(doc), dict(meta))
            except (KeyError, IndexError, NameError) as e:
                logger.warning("Dense result maps incomplete: %s", e)

        # BM25-only hits absent from the dense top-k live in the cached corpus.
        corpus_by_id: dict[str, str] = {
            str(did): doc for did, doc in zip(self._ids_cache, self._corpus_cache)
        }

        results: list[tuple[str, dict[str, Any], float]] = []
        for did, score in ranked:
            sid: str = str(did)
            if sid in doc_by_id:
                doc, meta = doc_by_id[sid]
            elif sid in corpus_by_id:
                doc, meta = corpus_by_id[sid], {}
            else:
                continue
            results.append((doc, meta, score))

        return results

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, dict[str, Any], float]],
        top_k: int = 4,
    ) -> list[tuple[str, dict[str, Any], float]]:
        if not candidates:
            return []
        try:
            pairs: list[list[str]] = [[query, doc[:3500]] for doc, _, _ in candidates]
            # Pylance fix: predict returns either ndarray or Tensor depending on backend
            scores: Any = self.reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [(doc, meta, float(s)) for (doc, meta, _), s in ranked[:top_k]]
        except Exception as e:
            logger.error("Cross-encoder rerank phase failed: %s", e)
            return candidates[:top_k]

    def retrieve(
        self,
        query: str,
        language: str | None = None,
        final_k: int = 4,
    ) -> list[str]:
        cands: list[tuple[str, dict[str, Any], float]] = self.hybrid_search(
            query, language, k=max(12, final_k * 3)
        )
        reranked: list[tuple[str, dict[str, Any], float]] = self.rerank(
            query, cands, top_k=final_k
        )
        return [doc for doc, _, _ in reranked]
