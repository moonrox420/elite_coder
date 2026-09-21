"""
Parsing of ComfyUI workflow JSON files.

A workflow file exposes `nodes` (list of dicts) where each node carries a
`type` string. We collect the distinct node types a workflow depends on, and
optionally the image assets associated with it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@dataclass
class WorkflowAnalysis:
    file_path: str
    referenced_types: Counter = field(default_factory=Counter)
    image_paths: list[str] = field(default_factory=list)


def parse_workflow_file(path: Path, include_images: bool) -> WorkflowAnalysis:
    analysis = WorkflowAnalysis(file_path=str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return analysis

    # ComfyUI stores graph nodes under the "nodes" key.
    graph = payload if isinstance(payload, dict) else {}
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []

    counts: Counter = Counter()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type")
        if isinstance(node_type, str) and node_type:
            counts[node_type] += 1

    analysis.referenced_types = counts

    if include_images:
        analysis.image_paths = _collect_image_paths(path.parent)

    return analysis


def _collect_image_paths(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    paths = [
        str(p)
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    return sorted(paths)
