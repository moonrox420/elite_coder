"""
Discovery of custom-node packages and workflow files.

A "package root" is a directory holding ComfyUI node definitions
(NODE_CLASS_MAPPINGS etc.). When an explicit root is provided it is either
itself a package, or a `custom_nodes`-style container of packages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Markers that indicate a directory is directly a node package (not a container).
_PACKAGE_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
)

# A file that typically lives at a package's top level only when the dir is a package.
_PACKAGE_MODULE = "__init__.py"


@dataclass
class PackageRoot:
    name: str
    directory: Path


def _is_package_dir(directory: Path) -> bool:
    """Heuristic: does this directory directly contain node source code?"""
    if not directory.is_dir():
        return False
    try:
        entries = list(directory.iterdir())
    except OSError:
        return False

    for entry in entries:
        if entry.is_file():
            if entry.suffix == ".py":
                return True
            if entry.name in _PACKAGE_MARKERS:
                return True
    return False


def _split_package_root(candidate: Path) -> list[PackageRoot]:
    """Resolve a candidate path to zero or more PackageRoot objects.

    Priority order:
      1. A directory with `__init__.py` at its root, or with direct `.py`
         files and no package subdirectories, is itself a package.
      2. A directory containing package subdirectories is a container: each
         package subdir becomes a root, plus any loose single-file modules.
    """
    if not candidate.is_dir():
        return []

    children = list(candidate.iterdir())
    subdirs = [c for c in children if c.is_dir()]
    direct_py = [c for c in children if c.is_file() and c.suffix == ".py"]
    has_init = (candidate / _PACKAGE_MODULE).is_file()
    has_package_subdirs = any(
        _is_package_dir(child) or (child / _PACKAGE_MODULE).exists()
        for child in subdirs
    )

    if has_init or (direct_py and not has_package_subdirs):
        return [PackageRoot(name=candidate.name, directory=candidate)]

    if has_package_subdirs:
        roots: list[PackageRoot] = []
        for child in sorted(subdirs):
            if _is_package_dir(child) or (child / _PACKAGE_MODULE).exists():
                roots.append(PackageRoot(name=child.name, directory=child))
        for child in sorted(direct_py):
            # A bare module dropped straight into the container root;
            # point the package at the file itself so only it is analyzed.
            roots.append(PackageRoot(name=child.stem, directory=child))
        return roots

    if direct_py:
        return [PackageRoot(name=candidate.name, directory=candidate)]
    return []


def discover_package_roots(custom_node_paths: list[str] | None) -> list[PackageRoot]:
    """Locate package roots from overrides, else AUTO_DISCOVERY candidates."""
    roots: list[PackageRoot] = []
    seen: set[Path] = set()

    candidates = (
        [Path(p) for p in (custom_node_paths or [])]
        if custom_node_paths
        else _default_discovery_candidates()
    )

    for candidate in candidates:
        resolved = candidate.expanduser()
        if not resolved.exists() or not resolved.is_dir():
            continue
        for root in _split_package_root(resolved):
            key = root.directory.resolve()
            if key not in seen:
                seen.add(key)
                roots.append(root)

    return roots


def _default_discovery_candidates() -> list[Path]:
    """ComfyUI install locations probed when no explicit roots are given."""
    candidates: list[Path] = []
    env = os.environ.get("COMFYUI_ROOT")
    if env:
        candidates.append(Path(env) / "custom_nodes")

    env_nodes = os.environ.get("COMFYUI_CUSTOM_NODES")
    if env_nodes:
        candidates.append(Path(env_nodes))

    home = Path.home()
    relative = [
        "ComfyUI/custom_nodes",
        "comfyui/custom_nodes",
        "custom_nodes",
    ]
    candidates.extend(home / r for r in relative)

    # Comfy Desktop (Windows) installs each runtime under
    # %LOCALAPPDATA%\Comfy-Desktop\ComfyUI-Installs\<user>\ComfyUI\custom_nodes
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.extend(
            Path(local_appdata)
            / "Comfy-Desktop"
            / "ComfyUI-Installs"
            / user
            / "ComfyUI"
            / "custom_nodes"
            for user in (os.environ.get("USERNAME") or "",)
        )

    return candidates


def discover_workflow_files(workflow_paths: list[str] | None) -> list[Path]:
    """Recursively collect workflow JSON files under the given directories."""
    if not workflow_paths:
        return []

    files: list[Path] = []
    for raw in workflow_paths:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        files.extend(p for p in root.rglob("*.json") if p.is_file())

    # Deduplicate deterministically.
    return sorted(set(files), key=lambda p: str(p))
