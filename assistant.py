#!/usr/bin/env python3
"""
Elite Coding Assistant — full production system with strict static typing.
Intent + Language classification • Hybrid RAG • Memory • Active Learning • Auto-ingest
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import ollama
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from active_learning import is_uncertain, log_uncertain
from chunker import elite_chunk
from memory import Memory
from retrieval import EliteRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger: logging.Logger = logging.getLogger("assistant")

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR: Path = Path("./multi-task-final")
CODE_BLOCK_RE: re.Pattern = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Multi-Task Classification Model
# ---------------------------------------------------------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, model_name: str, num_intents: int, num_languages: int) -> None:
        super().__init__()
        self.encoder: Any = AutoModel.from_pretrained(model_name)
        hidden: int = int(self.encoder.config.hidden_size)
        self.intent_head: nn.Linear = nn.Linear(hidden, num_intents)
        self.language_head: nn.Linear = nn.Linear(hidden, num_languages)
        self.dropout: nn.Dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls: torch.Tensor = self.dropout(outputs.last_hidden_state[:, 0])
        return self.intent_head(cls), self.language_head(cls)


# ---------------------------------------------------------------------------
# Classifier & Processing Initialization
# ---------------------------------------------------------------------------
if not (MODEL_DIR / "labels.json").exists():
    print(f"Model labels not found at {MODEL_DIR}. Run train_multitask.py first.")
    sys.exit(1)

try:
    with (MODEL_DIR / "labels.json").open(encoding="utf-8") as f:
        labels: dict[str, Any] = json.load(f)
except (OSError, json.JSONDecodeError) as e:
    print(f"Critical Error: Failed reading labeling files: {e}")
    sys.exit(1)

intent2id: dict[str, int] = labels["intent2id"]
lang2id: dict[str, int] = labels["lang2id"]
id2intent: dict[int, str] = {v: k for k, v in intent2id.items()}
id2lang: dict[int, str] = {v: k for k, v in lang2id.items()}

# Prevent AutoTokenizer typing issues by using PreTrainedTokenizerBase as type annotation
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(str(MODEL_DIR))  # type: ignore
clf_model: MultiTaskModel = MultiTaskModel(
    "distilbert-base-uncased",
    len(intent2id),
    len(lang2id),
)
state_path: Path = MODEL_DIR / "pytorch_model.bin"
if not state_path.exists():
    print(f"Weights not found: {state_path}")
    sys.exit(1)

try:
    clf_model.load_state_dict(
        torch.load(state_path, map_location=DEVICE, weights_only=True)
    )
except Exception as e:
    print(f"Critical Error loading model weight files: {e}")
    sys.exit(1)

clf_model.to(DEVICE)
clf_model.eval()


def resolve_target_language(
    user_text: str, predicted_lang: str, confidence: float
) -> tuple[str, float]:
    """Resolve target language: explicit user prompt mentions override classifier."""
    text_lower = user_text.lower()
    lang_keywords = {
        "python": r"\b(python|py)\b",
        "javascript": r"\b(javascript|js|node|nodejs)\b",
        "typescript": r"\b(typescript|ts)\b",
        "c#": r"\b(c#|csharp|\.net)\b",
        "c++": r"\b(c\+\+|cpp)\b",
        "go": r"\b(go|golang)\b",
        "rust": r"\b(rust)\b",
        "java": r"\b(java)\b",
        "sql": r"\b(sql|postgres|postgresql|sqlite|mysql)\b",
        "bash": r"\b(bash|shell|sh|powershell|zsh)\b",
    }
    for lang, pattern in lang_keywords.items():
        if re.search(pattern, text_lower):
            return lang, 1.0

    if confidence >= 0.60 and predicted_lang != "unknown":
        return predicted_lang, confidence

    return "unknown", confidence


def predict_intent(text: str) -> dict[str, Any]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.inference_mode():  # Completely bypasses graph metadata overhead
        intent_logits, lang_logits = clf_model(**inputs)

    intent_probs: torch.Tensor = torch.softmax(intent_logits, dim=-1)[0]
    lang_probs: torch.Tensor = torch.softmax(lang_logits, dim=-1)[0]

    intent_id: int = int(torch.argmax(intent_probs).item())
    lang_id: int = int(torch.argmax(lang_probs).item())
    raw_lang: str = id2lang[lang_id]
    lang_conf: float = float(lang_probs[lang_id].item())
    resolved_lang, final_conf = resolve_target_language(text, raw_lang, lang_conf)

    return {
        "intent": id2intent[intent_id],
        "intent_confidence": float(intent_probs[intent_id].item()),
        "language": resolved_lang,
        "language_confidence": final_conf,
    }


# ---------------------------------------------------------------------------
# Core Pipeline Execution Logic
# ---------------------------------------------------------------------------
# OLLAMA_HOST on Windows is often set to 0.0.0.0:11434 to bind the daemon,
# but Windows TCP client sockets cannot connect to 0.0.0.0 (WinError 10049).
if os.environ.get("OLLAMA_HOST", "").startswith("0.0.0.0"):
    os.environ["OLLAMA_HOST"] = os.environ["OLLAMA_HOST"].replace(
        "0.0.0.0", "127.0.0.1"
    )

OLLAMA_BASE: str = "http://127.0.0.1:11434"
client: ollama.Client = ollama.Client(host=OLLAMA_BASE)
retriever: EliteRetriever = EliteRetriever()
memory: Memory = Memory(model="gemma4:31b-cloud")
OLLAMA_MODEL: str = "gemma4:31b-cloud"

SYSTEM_PROMPTS: dict[str, str] = {
    "code_generation": (
        "You are an elite software engineer. Write clean, correct, production-ready code. "
        "Use provided context when relevant. After the code, give a short explanation."
    ),
    "code_explanation": (
        "You are an elite software engineer. Explain the code or concept clearly and precisely."
    ),
    "code_improvement": (
        "You are an elite software engineer. Improve the code for readability, performance, "
        "and correctness. Show the improved version and explain key changes."
    ),
    "general_question": (
        "You are a precise technical assistant. Answer clearly and accurately."
    ),
    "default": "You are a helpful coding assistant.",
}


def generate(prompt: str, intent: str) -> str:
    system: str = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["default"])
    try:
        parts: list[str] = []
        for res in client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.15},
            stream=True,
        ):
            delta: str = ""
            if hasattr(res, "message") and hasattr(res.message, "content"):
                delta = res.message.content or ""
            elif isinstance(res, dict):
                delta = res["message"]["content"]
            else:
                delta = getattr(res, "content", "") or ""
            if delta:
                parts.append(delta)
                print(delta, end="", flush=True)
        print()
        return "".join(parts)
    except Exception as e:
        return f"[Generation error] {e}"


def auto_ingest(response: str, language: str, request: str) -> None:
    blocks: list[str] = CODE_BLOCK_RE.findall(response)
    content: str = "\n\n".join(blocks) if blocks else response
    if len(content) < 80:
        return
    units: list[dict[str, Any]] = elite_chunk(
        content, language or "python", embed_fn=retriever.embed
    )
    count: int = 0
    for u in units:
        meta: dict[str, Any] = {
            "id": u["id"],
            "language": u.get("language", language),
            "name": u["name"],
            "type": u["type"],
            "source": "generation",
            "request": request[:160],
        }
        if retriever.add(u["code"], meta):
            count += 1
    if count:
        print(f"  [Ingest] {count} units stored")


def main() -> None:
    print("=" * 60)
    print("  ELITE CODING ASSISTANT — FULL SYSTEM")
    print("  Intent • Language • Hybrid RAG • Memory • Active Learning")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user: str = input("> ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit", "q"}:
                break

            analysis: dict[str, Any] = predict_intent(user)
            print(
                f"  → Intent: {analysis['intent']} "
                f"({analysis['intent_confidence']:.2f})"
            )
            print(
                f"  → Language: {analysis['language']} "
                f"({analysis['language_confidence']:.2f})"
            )

            if is_uncertain(analysis):
                log_uncertain(user, analysis)
                print("  [Active Learning] logged for review")

            if analysis["intent"] == "exit":
                print("Session terminated.")
                break
            if analysis["intent"] == "greeting":
                print("Hello. Ready for coding tasks.")
                continue
            if analysis["intent"] == "out_of_scope":
                print("I specialize in coding tasks.")
                continue

            memory.add("user", user)

            if memory.should_summarize():
                memory.summarize()
                print("  [Memory] summarized")

            code_ctx: list[str] = []
            if analysis["intent"] in {
                "code_generation",
                "code_improvement",
                "code_explanation",
            }:
                code_ctx = retriever.retrieve(
                    user,
                    language=analysis["language"],
                    final_k=4,
                )

            mem_ctx: str = memory.context()

            parts: list[str] = []
            if mem_ctx:
                parts.append(mem_ctx)
            if code_ctx:
                parts.append(
                    "Relevant code from knowledge base:\n"
                    + "\n\n---\n\n".join(code_ctx)
                )
            parts.append(f"User request:\n{user}")
            if analysis["language"] != "unknown":
                parts.append(f"Target language: {analysis['language']}")

            prompt: str = "\n\n".join(parts)
            response: str = generate(prompt, analysis["intent"])
            print()

            memory.add("assistant", response)

            if (
                analysis["intent"] in {"code_generation", "code_improvement"}
                and "```" in response
            ):
                auto_ingest(response, analysis["language"], user)

        except KeyboardInterrupt:
            print("\nTerminated.")
            break
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()
