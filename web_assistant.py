#!/usr/bin/env python3
"""
Elite Coding Assistant - web preview.

A thin FastAPI wrapper that reuses the existing assistant.py pipeline
(classifier -> hybrid RAG -> memory -> Ollama generation) and serves a small
chat UI. This file is additive: assistant.py itself is unchanged and remains
the REPL entry point.

Runtime model-tag overrides (no source edits):
  - generation/summarization model: gemma4:31b-cloud (installed; code expects
    the gemma4:cloud alias, which is not present on this machine)
  - embeddings: nomic-embed-text (pulled via `ollama pull`, 768-dim, matches
    the pgvector schema created by db_setup.py)

Run:  python web_assistant.py            (serves on 127.0.0.1:3005)
      python web_assistant.py --port N
"""

from __future__ import annotations

import argparse
import logging
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ollama
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Reuse the existing pipeline. Importing assistant.py executes its module-level
# init (labels load, classifier weights, retriever, memory, ollama client) -
# the same objects the REPL uses.
# ---------------------------------------------------------------------------
import assistant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger: logging.Logger = logging.getLogger("web_assistant")

# ---------------------------------------------------------------------------
# Runtime model-tag overrides.
# ---------------------------------------------------------------------------
GENERATION_MODEL = "gemma4:31b-cloud"

assistant.OLLAMA_MODEL = GENERATION_MODEL  # generation used by the web UI

# OLLAMA_HOST on this machine is 0.0.0.0:11434, which Windows TCP clients
# cannot connect to (WinError 10049). Rebind the pipeline's Ollama clients to
# loopback explicitly instead of changing the user-level env config.
OLLAMA_BASE = "http://127.0.0.1:11434"
_shared_client = ollama.Client(host=OLLAMA_BASE)  # generation (no timeout: streaming)
_memory_client = ollama.Client(host=OLLAMA_BASE, timeout=120.0)  # summarization

assistant.client = _shared_client
assistant.retriever.ollama = _shared_client  # embeddings for dense retrieval

# ---------------------------------------------------------------------------
# FastAPI app + in-memory session state (single-user local preview).
# ---------------------------------------------------------------------------
app = FastAPI(title="Elite Coding Assistant - Web Preview")

_sessions: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def _get_session(session_id: str | None) -> tuple[str, dict[str, Any]]:
    """Return (session_id, state), creating the session when needed."""
    with _lock:
        sid = session_id or uuid.uuid4().hex
        if sid not in _sessions:
            memory = assistant.Memory(model=GENERATION_MODEL)
            memory.client = _memory_client  # loopback binding (see OLLAMA_BASE note)
            _sessions[sid] = {
                "memory": memory,
                "turns": [],  # [{role, content}] for the UI transcript
            }
        return sid, _sessions[sid]


def _pipeline_parts(
    user: str, state: dict[str, Any], analysis: dict[str, Any]
) -> list[str]:
    """Build the prompt parts exactly like assistant.main() does."""
    parts: list[str] = []

    mem_ctx: str = state["memory"].context()
    if mem_ctx:
        parts.append(mem_ctx)

    code_ctx: list[str] = []
    if analysis["intent"] in {
        "code_generation",
        "code_improvement",
        "code_explanation",
    }:
        code_ctx = assistant.retriever.retrieve(
            user,
            language=analysis["language"],
            final_k=4,
        )
    if code_ctx:
        parts.append(
            "Relevant code from knowledge base:\n" + "\n\n---\n\n".join(code_ctx)
        )

    parts.append(f"User request:\n{user}")
    if analysis["language"] != "unknown":
        parts.append(f"Target language: {analysis['language']}")
    return parts


def _stream_response(prompt: str, intent: str) -> Iterator[str]:
    """Stream Ollama deltas with the same fallback logic as assistant.generate."""
    system: str = assistant.SYSTEM_PROMPTS.get(
        intent, assistant.SYSTEM_PROMPTS["default"]
    )
    client: ollama.Client = assistant.client
    try:
        for res in client.chat(
            model=assistant.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.15},
            stream=True,
        ):
            delta: str = ""
            if hasattr(res, "message") and hasattr(res.message, "content"):
                delta = res.message.content
            elif isinstance(res, dict):
                delta = res.get("message", {}).get("content", "")
            else:
                delta = getattr(res, "content", "") or ""
            if delta:
                yield delta
    except Exception as e:
        logger.error("generation failed: %s", e)
        yield f"[Generation error] {e}"


@app.get("/")
def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "web_ui.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "HEALTHY",
        "model": assistant.OLLAMA_MODEL,
        "device": assistant.DEVICE,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    """Non-streaming full-pipeline turn: classify -> retrieve -> generate."""
    sid, state = _get_session(req.session_id)

    user: str = req.message.strip()
    if not user:
        return {"error": "empty message", "session_id": sid}

    # 1) Classify (same model the REPL uses)
    analysis: dict[str, Any] = assistant.predict_intent(user)

    # Special intents handled without generation, mirroring the REPL
    if analysis["intent"] == "greeting":
        reply = "Hello. Ready for coding tasks."
    elif analysis["intent"] == "out_of_scope":
        reply = "I specialize in coding tasks."
    elif analysis["intent"] == "exit":
        reply = "Session terminated."
    else:
        state["memory"].add("user", user)
        if state["memory"].should_summarize():
            state["memory"].summarize()
        prompt: str = "\n\n".join(_pipeline_parts(user, state, analysis))
        reply = "".join(_stream_response(prompt, analysis["intent"]))
        state["memory"].add("assistant", reply)
        if (
            analysis["intent"] in {"code_generation", "code_improvement"}
            and "```" in reply
        ):
            assistant.auto_ingest(reply, analysis["language"], user)

    state["turns"].append({"role": "user", "content": user})
    state["turns"].append({"role": "assistant", "content": reply})

    return {
        "session_id": sid,
        "intent": analysis["intent"],
        "intent_confidence": round(analysis["intent_confidence"], 4),
        "language": analysis["language"],
        "language_confidence": round(analysis["language_confidence"], 4),
        "reply": reply,
    }


@app.get("/api/history/{session_id}")
def history(session_id: str) -> dict[str, Any]:
    state = _sessions.get(session_id)
    if not state:
        return {"turns": []}
    return {"turns": state["turns"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elite Coding Assistant web preview")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3005)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
