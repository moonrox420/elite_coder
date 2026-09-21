#!/usr/bin/env python3
"""
Elite multi-task training: intent classification + programming language detection.
DistilBERT encoder with dual classification heads, optimized with Automatic Mixed Precision (AMP).
Includes programmatic bootstrapping fallback to initialize training without external manual steps.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger: logging.Logger = logging.getLogger("trainer")

DATA_FILE: Path = Path("intent_language_data.jsonl")
OUTPUT_DIR: Path = Path("./multi-task-final")
MODEL_NAME: str = "distilbert-base-uncased"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Programmatic Dataset Bootstrap (Developer Convenience Fallback)
# ---------------------------------------------------------------------------
def bootstrap_baseline_data() -> None:
    """
    Synthesize high-fidelity intent/language dataset samples if the external dataset
    is not present, enabling instantaneous verification of the full execution pipeline.
    """
    logger.info(
        "Dataset '%s' not found. Bootstrapping high-fidelity baseline...", DATA_FILE
    )

    languages: list[str] = [
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

    samples: list[dict[str, str]] = []

    # 1. Greetings & Exits
    for _ in range(15):
        samples.append(
            {
                "text": "hello there assistant",
                "intent": "greeting",
                "language": "unknown",
            }
        )
        samples.append(
            {"text": "hi, are you online?", "intent": "greeting", "language": "unknown"}
        )
        samples.append(
            {
                "text": "good morning coding helper",
                "intent": "greeting",
                "language": "unknown",
            }
        )
        samples.append({"text": "exit now", "intent": "exit", "language": "unknown"})
        samples.append(
            {"text": "quit session", "intent": "exit", "language": "unknown"}
        )
        samples.append(
            {
                "text": "shutdown assistant please",
                "intent": "exit",
                "language": "unknown",
            }
        )

    # 2. Out of Scope
    for _ in range(15):
        samples.append(
            {
                "text": "what is the capital of France?",
                "intent": "out_of_scope",
                "language": "unknown",
            }
        )
        samples.append(
            {
                "text": "recommend a good movie to watch",
                "intent": "out_of_scope",
                "language": "unknown",
            }
        )
        samples.append(
            {
                "text": "how do I bake chocolate cookies?",
                "intent": "out_of_scope",
                "language": "unknown",
            }
        )

    # 3. General Questions
    general_queries = [
        ("what is a compiler?", "unknown"),
        ("explain the difference between tcp and udp", "unknown"),
        ("how does garbage collection work inside virtual machines?", "unknown"),
        ("what are acid properties in databases?", "sql"),
        ("how do indexes speed up lookups in databases?", "sql"),
    ]
    for q, l in general_queries:
        for _ in range(5):
            samples.append({"text": q, "intent": "general_question", "language": l})

    # 4. Code Generation
    generation_templates = [
        ("write a function to merge two sorted lists in {lang}", "code_generation"),
        ("implement a binary search algorithm using {lang}", "code_generation"),
        ("how do i write a thread pool in {lang}?", "code_generation"),
        (
            "create an api endpoint to parse payload schemas with {lang}",
            "code_generation",
        ),
        (
            "generate boilerplate code for a database adapter in {lang}",
            "code_generation",
        ),
    ]

    # 5. Code Explanation
    explanation_templates = [
        ("explain this recursive function block written in {lang}", "code_explanation"),
        (
            "what does this complex pointer manipulation do in {lang}?",
            "code_explanation",
        ),
        ("how does dynamic dispatch perform lookup inside {lang}?", "code_explanation"),
        ("break down this decorator framework usage in {lang}", "code_explanation"),
        ("explain the memory layout of arrays in {lang}", "code_explanation"),
    ]

    # 6. Code Improvement
    improvement_templates = [
        ("how can I optimize this slow processing loop in {lang}?", "code_improvement"),
        ("refactor this deeply nested conditional block in {lang}", "code_improvement"),
        ("make this asynchronous function memory-safe in {lang}", "code_improvement"),
        (
            "suggest modern cleaner language features for this old {lang} code",
            "code_improvement",
        ),
        (
            "optimize memory allocations inside this hot path written in {lang}",
            "code_improvement",
        ),
    ]

    for lang in languages:
        if lang == "unknown":
            continue
        for temp, intent in generation_templates:
            samples.append(
                {"text": temp.format(lang=lang), "intent": intent, "language": lang}
            )
        for temp, intent in explanation_templates:
            samples.append(
                {"text": temp.format(lang=lang), "intent": intent, "language": lang}
            )
        for temp, intent in improvement_templates:
            samples.append(
                {"text": temp.format(lang=lang), "intent": intent, "language": lang}
            )

    # Shuffle and write back as jsonl
    random.seed(42)
    random.shuffle(samples)

    try:
        with DATA_FILE.open("w", encoding="utf-8", newline="\n") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(
            "Successfully bootstrapped %d baseline samples to '%s'.",
            len(samples),
            DATA_FILE,
        )
    except OSError as e:
        logger.error("Failed writing bootstrapped baseline data: %s", e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Multi-Task Model Definition
# ---------------------------------------------------------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, model_name: str, num_intents: int, num_languages: int) -> None:
        super().__init__()
        self.encoder: nn.Module = AutoModel.from_pretrained(model_name)
        hidden: int = int(getattr(self.encoder.config, "hidden_size", 768))
        self.intent_head: nn.Linear = nn.Linear(hidden, num_intents)
        self.language_head: nn.Linear = nn.Linear(hidden, num_languages)
        self.dropout: nn.Dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        intent_labels: torch.Tensor | None = None,
        language_labels: torch.Tensor | None = None,
    ) -> dict[str, Any] | tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls: torch.Tensor = self.dropout(outputs.last_hidden_state[:, 0])
        intent_logits: torch.Tensor = self.intent_head(cls)
        language_logits: torch.Tensor = self.language_head(cls)

        if intent_labels is not None and language_labels is not None:
            # Multi-task training branch
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(intent_logits, intent_labels) + loss_fct(
                language_logits, language_labels
            )
            return {
                "loss": loss,
                "logits": (intent_logits, language_logits),
            }

        # Inference signature compatibility layer
        return intent_logits, language_logits


# ---------------------------------------------------------------------------
# Multi-Task Trainer Custom Handler
# ---------------------------------------------------------------------------
class MultiTaskTrainer(Trainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            intent_labels=inputs["intent_labels"],
            language_labels=inputs["language_labels"],
        )
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    (intent_logits, language_logits), labels = eval_pred
    if isinstance(labels, dict):
        intent_labels = labels["intent_labels"]
        language_labels = labels["language_labels"]
    else:
        # Transformers passes label ids as a plain tuple ordered by `label_names`,
        # which follow the model's forward() signature: (intent, language).
        intent_labels, language_labels = labels

    intent_preds = np.argmax(intent_logits, axis=-1)
    lang_preds = np.argmax(language_logits, axis=-1)

    return {
        "intent_acc": float(accuracy_score(intent_labels, intent_preds)),
        "intent_f1": float(f1_score(intent_labels, intent_preds, average="weighted")),
        "lang_acc": float(accuracy_score(language_labels, lang_preds)),
    }


# ---------------------------------------------------------------------------
# Main Training Loop Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    if not DATA_FILE.exists():
        bootstrap_baseline_data()

    try:
        data: list[dict[str, Any]] = [
            json.loads(line)
            for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed reading the training dataset file: %s", e)
        sys.exit(1)

    if not data:
        logger.warning("Dataset file is empty. Regenerating baseline samples...")
        bootstrap_baseline_data()
        data = [
            json.loads(line)
            for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    texts: list[str] = [d["text"] for d in data]
    intents: list[str] = [d["intent"] for d in data]
    languages: list[str] = [d.get("language", "unknown") for d in data]

    intent_list: list[str] = sorted(set(intents))
    lang_list: list[str] = sorted(set(languages))
    intent2id: dict[str, int] = {k: i for i, k in enumerate(intent_list)}
    lang2id: dict[str, int] = {k: i for i, k in enumerate(lang_list)}

    # Stratified split to ensure balanced partition distribution across both datasets
    (
        train_texts,
        val_texts,
        train_intents,
        val_intents,
        train_langs,
        val_langs,
    ) = train_test_split(
        texts,
        intents,
        languages,
        test_size=0.15,
        random_state=42,
        stratify=intents,
    )

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch: dict[str, Any]) -> dict[str, list[Any]]:
        encoded = tokenizer(batch["text"], truncation=True, max_length=128)
        return dict(encoded)

    train_ds: Dataset = Dataset.from_dict(
        {
            "text": train_texts,
            "intent_labels": [intent2id[i] for i in train_intents],
            "language_labels": [lang2id[l] for l in train_langs],
        }
    ).map(tokenize, batched=True)

    val_ds: Dataset = Dataset.from_dict(
        {
            "text": val_texts,
            "intent_labels": [intent2id[i] for i in val_intents],
            "language_labels": [lang2id[l] for l in val_langs],
        }
    ).map(tokenize, batched=True)

    model: MultiTaskModel = MultiTaskModel(MODEL_NAME, len(intent_list), len(lang_list))
    model.to(DEVICE)

    # Performance Optimized Training Arguments
    args = TrainingArguments(
        output_dir="./multi-task-model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="intent_f1",
        greater_is_better=True,
        report_to="none",
        logging_steps=20,
        save_total_limit=1,
        # Performance booster: AMP configuration for dynamic precision scaling
        fp16=torch.cuda.is_available(),
        # Performance booster: directly map pointers on host to GPU
        dataloader_pin_memory=True,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting Multi-Task encoder fine-tuning on: %s", DEVICE)
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save explicit state dict targeting exact weight load patterns
    torch.save(model.state_dict(), OUTPUT_DIR / "pytorch_model.bin")

    with (OUTPUT_DIR / "labels.json").open("w", encoding="utf-8") as f:
        json.dump({"intent2id": intent2id, "lang2id": lang2id}, f, indent=2)

    logger.info(
        "Multi-task optimization & training complete. Artifacts stored inside -> %s",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
