#!/usr/bin/env python3
"""
Elite conversation memory with automatic summarization via Ollama.
Optimized with in-memory ring-buffer caching and constant-time context generation.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ollama
from httpx import HTTPError

logger: logging.Logger = logging.getLogger("memory")


class Memory:
    """Memory component."""

    __slots__ = (
        "_summary_cache",
        "client",
        "model",
        "summary_path",
        "turns",
    )

    def __init__(self, model: str = "gemma4:31b-cloud") -> None:
        """Init.

        Args:
            model (str): Description.
        """
        ollama_host = os.environ.get("OLLAMA_HOST", "")
        if "0.0.0.0" in ollama_host:
            self.client: ollama.Client = ollama.Client(
                host="http://127.0.0.1:11434", timeout=30.0
            )
        else:
            self.client: ollama.Client = ollama.Client(timeout=30.0)
        self.model: str = model
        self.turns: list[dict[str, Any]] = []
        self.summary_path: Path = Path("summaries.jsonl")

        # High-performance O(1) in-memory ring buffer for past summaries
        self._summary_cache: deque[str] = deque(maxlen=3)
        self._cold_start_cache()

    def _cold_start_cache(self) -> None:
        """
        Fast reverse-seek binary read to parse only the trailing portion
        of the summaries log file, keeping cold startup to O(1) runtime.
        """
        if not self.summary_path.exists():
            return
        try:
            with self.summary_path.open("rb") as f:
                f.seek(0, 2)  # Seek to end of file
                file_size: int = f.tell()

                # Fetch only the last 8KB of the file to parse trailing lines
                seek_offset: int = min(8192, file_size)
                f.seek(file_size - seek_offset)

                chunk: str = f.read(seek_offset).decode("utf-8", errors="ignore")
                lines: list[str] = chunk.strip().splitlines()[-3:]

            for line in lines:
                if line.strip():
                    try:
                        self._summary_cache.append(json.loads(line)["summary"])
                    except json.JSONDecodeError:
                        continue
        except (OSError, ValueError) as e:
            logger.warning("[Memory] Failed cold-loading summary cache: %s", e)

    def add(self, role: str, content: str) -> None:
        """Add.

        Args:
            role (str): Description.
            content (str): Description.
        """
        self.turns.append(
            {
                "role": role,
                "content": content,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    def should_summarize(self, limit: int = 12) -> bool:
        """Should summarize.

        Args:
            limit (int): Description.

        Returns:
            Description (bool).
        """
        return len(self.turns) >= limit

    def summarize(self) -> str:
        """Summarize.

        Returns:
            Description (str).
        """
        if len(self.turns) < 4:
            return ""

        conv: str = "\n".join(
            f"{t['role'].upper()}: {t['content'][:800]}" for t in self.turns[:-2]
        )
        prompt: str = (
            "Summarize this coding conversation in under 160 words. "
            "Focus on decisions, code written, and current state. "
            "Output only the summary.\n\n"
            f"{conv}"
        )
        try:
            chunks: list[str] = []
            for res in self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                stream=True,
            ):
                message = getattr(res, "message", None)
                content = (
                    getattr(message, "content", None) if message is not None else None
                )
                if content is not None:
                    chunks.append(content)
                elif isinstance(res, dict):
                    message_content = res.get("message", {}).get("content")
                    if isinstance(message_content, str):
                        chunks.append(message_content)
                else:
                    content = getattr(res, "content", "") or ""
                    if isinstance(content, str):
                        chunks.append(content)

            summary: str = "".join(chunks).strip()
            if not summary:
                return ""

            # Persist summary asynchronously/safely to file stream
            with self.summary_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "summary": summary,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # Direct O(1) update to our in-memory sliding window cache
            self._summary_cache.append(summary)
            self.turns = self.turns[-2:]
            return summary
        except (HTTPError, ollama.ResponseError) as e:
            logger.error("[Memory] Summarization API step failed: %s", e)
            return ""

    def context(self) -> str:
        """
        Retrieves context directly from the in-memory deque, transforming
        this from a disk-bound O(S) operation to an instant O(1) operation.
        """
        if not self._summary_cache:
            return ""
        return "Past context:\n" + "\n---\n".join(self._summary_cache)
