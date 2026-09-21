"""
AST-based analysis of custom-node Python packages.

Detects node definitions the way ComfyUI resolves them at runtime:
primarily via `NODE_CLASS_MAPPINGS`, falling back to classes that define an
`INPUT_TYPES`/`OUTPUT_NODE` node contract when no explicit mapping exists.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NodeDefinition:
    node_type: str  # ComfyUI node type key (as referenced by workflows).
    class_name: str
    display_name: str
    snippet: str | None = None


_COMFY_NODE_ATTRS = frozenset({"INPUT_TYPES", "OUTPUT_NODE", "RETURN_TYPES"})
_MAPPING_NAMES = frozenset({"NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"})


def _iter_own_assignments(tree: ast.Module, source: str):
    """Yield (mod-name, ast.AST or None) top-level assignments of interest."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _MAPPING_NAMES:
                    yield target.id, node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in _MAPPING_NAMES
        ):
            yield node.target.id, node.value
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "NODE_CLASS_MAPPINGS"
        ):
            yield "NODE_CLASS_MAPPINGS", node.value


def _constant_key(key: ast.AST) -> str | None:
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def _value_class_name(value: ast.AST | None) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _collect_classdefs(tree: ast.Module, source: str) -> dict[str, ast.ClassDef]:
    classes: dict[str, ast.ClassDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.setdefault(node.name, node)
    return classes


def analyze_package_directory(directory: Path) -> list[NodeDefinition]:
    """Analyze a package's Python source for node definitions.

    `directory` may point at a directory (recursively scanned) or a single
    `.py` file (used for loose single-file packages).
    """
    definitions: list[NodeDefinition] = []
    class_snippets: dict[str, str] = {}
    mapped_types: dict[str, str] = {}  # node_type -> class_name

    if directory.is_file():
        py_files = [directory]
    else:
        py_files = [p for p in directory.rglob("*.py") if p.is_file()]
    modules = _parse_modules(py_files)
    if not modules:
        return definitions

    # 1) Collect class snippets for every module.
    for _path, tree, source in modules:
        classes = _collect_classdefs(tree, source)
        for name, node in classes.items():
            snippet = ast.get_source_segment(source, node) or source
            class_snippets[name] = class_snippets.get(name) or snippet

    # 2) Resolve authoritative node types via NODE_CLASS_MAPPINGS.
    for _path, tree, source in modules:
        for mapping_name, value in _iter_own_assignments(tree, source):
            if not isinstance(value, ast.Dict):
                continue
            for key_node, value_node in zip(value.keys, value.values):
                if isinstance(value_node, ast.Starred):
                    continue
                node_type = _constant_key(key_node)
                if not node_type:
                    continue
                class_name = _value_class_name(value_node) or node_type
                mapped_types.setdefault(node_type, class_name)

    # 3) Fall back to classes that satisfy the node contract when a package
    #    has no explicit mapping (decorator/auto-register style packages).
    global_mapped_classes = set(mapped_types.values())
    fallback: list[tuple[str, str]] = []
    for _path, tree, source in modules:
        for node in _collect_classdefs(tree, source).values():
            if node.name in global_mapped_classes:
                continue
            if _looks_like_comfy_node(node):
                fallback.append((node.name, node.name))

    for node_type, class_name in mapped_types.items():
        definitions.append(
            NodeDefinition(
                node_type=node_type,
                class_name=class_name,
                display_name=node_type,
                snippet=class_snippets.get(class_name),
            )
        )

    for node_type, class_name in fallback:
        definitions.append(
            NodeDefinition(
                node_type=node_type,
                class_name=class_name,
                display_name=class_name,
                snippet=class_snippets.get(class_name),
            )
        )

    return definitions


def _looks_like_comfy_node(node: ast.ClassDef) -> bool:
    """Heuristic: a class defining INPUT_TYPES/RETURN_TYPES etc. is a node."""
    for child in node.body:
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name) and target.id in _COMFY_NODE_ATTRS:
                    return True
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            child.name and child.name.upper() in _COMFY_NODE_ATTRS
        ):
            return True
    return False


def _parse_modules(paths):
    """Yield (path, ast.Module, source) for each decodable Python file."""
    results = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            results.append((path, ast.parse(source, filename=str(path)), source))
        except OSError:
            continue
    return results
