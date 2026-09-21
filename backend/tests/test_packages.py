"""
ScannerEngine classification + packages endpoint tests.

Builds real fixture files on disk (a custom node package and a workflow
JSON) and asserts the engine persists the expected classifications.
"""

import json
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.models.package import DiscoveredPackage
from app.models.scan import AuditScan
from app.services.scanner_engine import ScannerEngine

USED_PACKAGE_SOURCE = '''
"""Example custom node package."""
from nodes import NODE_CLASS_MAPPINGS  # noqa: F401  (comfy placeholder)


class HelloWorldNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "go"
    CATEGORY = "audit"

    def go(self, text: str):
        return (text,)


NODE_CLASS_MAPPINGS = {
    "HelloWorld": HelloWorldNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HelloWorld": "Hello World Node",
}
'''

UNUSED_PACKAGE_SOURCE = """
class SecretNode:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self):
        return (None,)


NODE_CLASS_MAPPINGS = {
    "SecretRender": SecretNode,
}
"""


def _write_package(container: Path, name: str, source: str) -> Path:
    package_dir = container / name
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(source, encoding="utf-8")
    return package_dir


def _write_workflow(workflows_dir: Path, filename: str, payload: dict) -> Path:
    workflows_dir.mkdir(parents=True, exist_ok=True)
    path = workflows_dir / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
async def audit_fixture(tmp_path: Path) -> dict:
    """A temp ComfyUI-like tree: two node packages + one workflow."""
    container = tmp_path / "custom_nodes"
    container.mkdir()

    _write_package(container, "awesome_nodes", USED_PACKAGE_SOURCE)
    _write_package(container, "orphan_nodes", UNUSED_PACKAGE_SOURCE)

    workflows_dir = tmp_path / "workflows"
    _write_workflow(
        workflows_dir,
        "portrait.json",
        {
            "nodes": [
                {"id": 1, "type": "HelloWorld"},
                {"id": 2, "type": "KSampler"},
            ]
        },
    )

    return {
        "custom_nodes": str(container),
        "workflows": str(workflows_dir),
    }


@pytest.fixture
async def completed_scan(
    audit_fixture: dict,
    session_factory: async_sessionmaker[AsyncSession],
) -> AuditScan:
    async with session_factory() as session:
        scan = AuditScan(
            status="PENDING",
            initiated_by="auditor@enterprise.com",
            scanned_root_path=audit_fixture["custom_nodes"],
        )
        session.add(scan)
        await session.commit()
        await session.refresh(scan)

        await ScannerEngine.execute_audit_scan(
            db=session,
            scan_id=scan.id,
            custom_node_paths=[audit_fixture["custom_nodes"]],
            workflow_paths=[audit_fixture["workflows"]],
            include_images=True,
        )

        # Eager-load everything the tests assert on while the session is open.
        fresh = await session.get(
            AuditScan,
            scan.id,
            options=[
                selectinload(AuditScan.packages).selectinload(DiscoveredPackage.nodes),
                selectinload(AuditScan.workflows),
            ],
        )
        return fresh


@pytest.mark.anyio
async def test_engine_marks_scan_completed(completed_scan: AuditScan):
    assert completed_scan.status == "COMPLETED"
    assert completed_scan.completed_at is not None
    assert completed_scan.error_detail is None


@pytest.mark.anyio
async def test_engine_classifies_used_and_unused_packages(
    completed_scan: AuditScan,
):
    by_name = {pkg.package_name: pkg for pkg in completed_scan.packages}

    awesome = by_name["awesome_nodes"]
    assert awesome.status == "USED"
    assert awesome.detected_nodes_count == 1
    assert awesome.used_nodes_count == 1

    node = awesome.nodes[0]
    assert node.node_type == "HelloWorld"
    assert node.node_class == "HelloWorldNode"
    assert node.referenced_in_workflow is True
    assert node.reference_count == 1
    assert node.source_snippet is not None
    assert "class HelloWorldNode" in node.source_snippet

    orphan = by_name["orphan_nodes"]
    assert orphan.status == "UNUSED"
    assert orphan.detected_nodes_count == 1
    assert orphan.used_nodes_count == 0


@pytest.mark.anyio
async def test_engine_persists_workflow_references(completed_scan: AuditScan):
    assert len(completed_scan.workflows) == 1
    workflow = completed_scan.workflows[0]
    assert workflow.file_path.endswith("portrait.json")

    referenced = {entry["type"] for entry in workflow.referenced_nodes}
    assert referenced == {"HelloWorld", "KSampler"}


@pytest.mark.anyio
async def test_packages_endpoint_returns_classified_packages(
    client: AsyncClient,
    auth_headers: dict,
    completed_scan: AuditScan,
):
    response = await client.get(
        "/api/v1/packages/",
        params={"scan_id": completed_scan.id},
        headers=auth_headers,
    )

    assert response.status_code == 200
    packages = response.json()
    assert len(packages) == 2

    by_name = {pkg["package_name"]: pkg for pkg in packages}
    assert by_name["awesome_nodes"]["status"] == "USED"
    assert by_name["awesome_nodes"]["nodes"][0]["node_type"] == "HelloWorld"


@pytest.mark.anyio
async def test_packages_endpoint_filters_by_status(
    client: AsyncClient,
    auth_headers: dict,
    completed_scan: AuditScan,
):
    response = await client.get(
        "/api/v1/packages/",
        params={
            "scan_id": completed_scan.id,
            "status": "USED",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    packages = response.json()
    assert len(packages) == 1
    assert packages[0]["package_name"] == "awesome_nodes"


@pytest.mark.anyio
async def test_discovery_splits_container_with_loose_module(tmp_path: Path):
    """A container with a package subdir AND a loose .py at the root must
    yield one root per package plus one for the loose module (regression:
    a real ComfyUI install had `websocket_image_save.py` directly in
    custom_nodes, which previously collapsed the whole tree into one package)."""
    container = tmp_path / "custom_nodes"
    container.mkdir()
    _write_package(container, "real_pack", USED_PACKAGE_SOURCE)
    (container / "websocket_image_save.py").write_text(
        """
class ImageSaver:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save"

    def save(self, image):
        return (image,)


NODE_CLASS_MAPPINGS = {
    "WebsocketImageSave": ImageSaver,
}
""",
        encoding="utf-8",
    )

    from app.services.scanner.discovery import discover_package_roots

    roots = discover_package_roots([str(container)])
    names = {root.name for root in roots}

    assert "real_pack" in names
    assert "websocket_image_save" in names
    assert len(roots) == 2

    # The loose module must be analyzed as its own file, not the whole tree.
    loose = next(r for r in roots if r.name == "websocket_image_save")
    assert loose.directory.is_file()
    assert loose.directory.name == "websocket_image_save.py"


@pytest.mark.anyio
async def test_engine_records_failure_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: async_sessionmaker[AsyncSession],
):
    container = tmp_path / "custom_nodes"
    _write_package(container, "boom_nodes", USED_PACKAGE_SOURCE)

    def _explode(*_args, **_kwargs):
        raise RuntimeError("simulated AST analysis failure")

    monkeypatch.setattr(
        "app.services.scanner_engine.analyze_package_directory",
        _explode,
    )

    async with session_factory() as session:
        scan = AuditScan(
            status="PENDING",
            initiated_by="auditor@enterprise.com",
            scanned_root_path="AUTO_DISCOVERY",
        )
        session.add(scan)
        await session.commit()
        await session.refresh(scan)

        with pytest.raises(RuntimeError):
            await ScannerEngine.execute_audit_scan(
                db=session,
                scan_id=scan.id,
                custom_node_paths=[str(container)],
                workflow_paths=[],
            )

        failed = await session.get(AuditScan, scan.id)
        assert failed.status == "FAILED"
        assert failed.error_detail is not None


@pytest.mark.anyio
async def test_workflows_endpoint_lists_items(
    client: AsyncClient,
    auth_headers: dict,
    completed_scan: AuditScan,
):
    response = await client.get(
        "/api/v1/workflows/",
        params={"scan_id": completed_scan.id},
        headers=auth_headers,
    )

    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["referenced_nodes"][0]["type"] == "HelloWorld"
