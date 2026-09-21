#!/usr/bin/env python3
"""
Elite active-learning queue for uncertain intent/language predictions.
Optimized with single-key numeric option mapping and transactional crash recovery.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger("active_learning")

ACTIVE_FILE: Path = Path("active_learning_queue.jsonl")
TRAIN_FILE: Path = Path("intent_language_data.jsonl")
THRESHOLD: float = 0.62

# Structured mappings to prevent typo corruption during user reviews
VALID_INTENTS: list[str] = [
    "code_generation",
    "code_explanation",
    "code_improvement",
    "general_question",
    "greeting",
    "exit",
    "out_of_scope",
]

VALID_LANGUAGES: list[str] = [
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


def is_uncertain(analysis: dict[str, Any]) -> bool:
    if analysis.get("intent_confidence", 1.0) < THRESHOLD:
        return True
    return bool(
        analysis.get("intent")
        in {"code_generation", "code_improvement", "code_explanation"}
        and analysis.get("language_confidence", 1.0) < 0.55
    )


def log_uncertain(text: str, analysis: dict[str, Any]) -> None:
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "predicted_intent": analysis.get("intent"),
        "intent_confidence": analysis.get("intent_confidence"),
        "predicted_language": analysis.get("language"),
        "language_confidence": analysis.get("language_confidence"),
        "labeled": False,
    }
    try:
        with ACTIVE_FILE.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error("[Active Learning] Failed writing logging queue entry: %s", e)


def _prompt_selection(options: list[str], predicted: str, label_type: str) -> str:
    """
    Renders a dynamic, low-friction single-character key numeric menu to
    prevent user typing mistakes and speed up the manual annotation loop.
    """
    print(f"\nSelect correct {label_type} (current predicted: {predicted}):")
    print("  0: [Keep predicted value]")
    for idx, opt in enumerate(options, 1):
        print(f"  {idx}: {opt}")

    while True:
        try:
            user_val: str = input(
                f"Enter choice [0-{len(options)}] or manual input: "
            ).strip()
            if not user_val:
                return predicted
            if user_val.isdigit():
                choice: int = int(user_val)
                if choice == 0:
                    return predicted
                if 1 <= choice <= len(options):
                    return options[choice - 1]

            # Fallback exact-string verification if user typed a string directly
            if user_val in options:
                return user_val

            confirm: str = (
                input(
                    f"'{user_val}' is not in standard system {label_type}s. Proceed anyway? (y/n): "
                )
                .strip()
                .lower()
            )
            if confirm == "y":
                return user_val
        except ValueError:
            print("[Warning] Invalid selection pattern detected. Try again.")


def review_queue() -> None:
    """
    Transactional CLI queue processor with early abort handling and strict
    data-persistence guarantees on program interrupt (KeyboardInterrupt).
    """
    if not ACTIVE_FILE.exists():
        print("No active learning examples.")
        return

    try:
        examples: list[dict[str, Any]] = [
            json.loads(line)
            for line in ACTIVE_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as e:
        print(f"[Error] Failed to read active learning file: {e}")
        return

    unlabeled: list[dict[str, Any]] = [e for e in examples if not e.get("labeled")]
    if not unlabeled:
        print("All records have been successfully labeled in the active queue.")
        return

    print(f"Found {len(unlabeled)} unlabeled entries.")
    print(
        "Type your selection key or press Ctrl+C to save current progress and exit.\n"
    )

    try:
        for idx, ex in enumerate(unlabeled):
            print("\n" + "=" * 60)
            print(f"Record {idx + 1}/{len(unlabeled)}")
            print(f"Text: {ex['text']}")
            print(
                f"Predicted Intent : {ex['predicted_intent']} ({ex.get('intent_confidence', 0):.2f})"
            )
            print(
                f"Predicted Language: {ex['predicted_language']} ({ex.get('language_confidence', 0):.2f})"
            )

            intent: str = _prompt_selection(
                VALID_INTENTS, ex["predicted_intent"], "intent"
            )
            lang: str = _prompt_selection(
                VALID_LANGUAGES, ex["predicted_language"], "language"
            )

            # Append the validated data immediately to training file
            try:
                with TRAIN_FILE.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(
                        json.dumps(
                            {"text": ex["text"], "intent": intent, "language": lang},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except OSError as e:
                print(f"[Error] Block write failed, entry skipped: {e}")
                continue

            ex["labeled"] = True
            ex["final_intent"] = intent
            ex["final_language"] = lang

    except KeyboardInterrupt:
        print(
            "\n\n[Active Learning] Progress interruption caught. Writing current state to disk..."
        )
    finally:
        # Enforce transactional updates to the active file even on early script terminations
        try:
            with ACTIVE_FILE.open("w", encoding="utf-8", newline="\n") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print("[Active Learning] Session updates persisted successfully.")
        except OSError as e:
            print(f"[Error] Failed to update queue records on disk: {e}")

    print("Progress review finalized.")


if __name__ == "__main__":
    review_queue()
