"""
ScannerEngine — orchestrates a full ComfyUI audit scan.

Responsible for transitioning a scan PENDING -> RUNNING -> COMPLETED and for
persisting package/workflow results. On any error it records the failure
state itself so background-task callers never need to handle rollback.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.package import DiscoveredPackage, PackageNode
from app.models.scan import AuditScan
from app.models.workflow import WorkflowItem
from app.services.scanner.ast_analysis import analyze_package_directory
from app.services.scanner.discovery import (
    discover_package_roots,
    discover_workflow_files,
)
from app.services.scanner.workflow_parser import parse_workflow_file

logger = logging.getLogger("comfyaudit.scanner")


class ScannerEngine:
    @staticmethod
    async def execute_audit_scan(
        db: AsyncSession,
        scan_id: str,
        custom_node_paths: list[str] | None = None,
        workflow_paths: list[str] | None = None,
        include_images: bool = True,
    ) -> None:
        scan = await db.get(AuditScan, scan_id)
        if not scan:
            logger.error("Scan %s not found; aborting audit.", scan_id)
            return

        scan.status = "RUNNING"
        await db.commit()

        try:
            await ScannerEngine._run(
                db,
                scan,
                custom_node_paths or [],
                workflow_paths or [],
                include_images,
            )
            scan.status = "COMPLETED"
            scan.completed_at = datetime.now(timezone.utc)
            scan.error_detail = None
        except Exception as exc:
            logger.exception("Audit scan %s failed", scan_id)
            scan.status = "FAILED"
            scan.completed_at = datetime.now(timezone.utc)
            scan.error_detail = str(exc)
            db.add(scan)
            await db.commit()
            raise

        db.add(scan)
        await db.commit()

    @staticmethod
    async def _run(
        db: AsyncSession,
        scan: AuditScan,
        custom_node_paths: list[str],
        workflow_paths: list[str],
        include_images: bool,
    ) -> None:
        # --- Analyze workflows first so we can classify package usage. ---
        referenced_counts: Counter = Counter()
        workflow_items: list[WorkflowItem] = []
        for wf_file in discover_workflow_files(workflow_paths):
            analysis = parse_workflow_file(wf_file, include_images)
            referenced_counts.update(analysis.referenced_types)
            item = WorkflowItem(
                scan_id=scan.id,
                file_path=analysis.file_path,
                referenced_nodes=[
                    {"type": t, "count": c}
                    for t, c in analysis.referenced_types.items()
                ],
                image_paths=analysis.image_paths,
            )
            db.add(item)
            workflow_items.append(item)

        scan.workflows = workflow_items

        # --- Discover and analyze custom-node packages. ---
        packages: list[DiscoveredPackage] = []
        for package_root in discover_package_roots(custom_node_paths):
            definitions = analyze_package_directory(package_root.directory)
            package = DiscoveredPackage(
                scan_id=scan.id,
                package_name=package_root.name,
                directory_path=str(package_root.directory),
                status="UNKNOWN",
                detected_nodes_count=len(definitions),
            )

            used_count = 0
            nodes: list[PackageNode] = []
            for definition in definitions:
                reference_count = referenced_counts.get(definition.node_type, 0)
                is_used = reference_count > 0
                if is_used:
                    used_count += 1
                nodes.append(
                    PackageNode(
                        node_name=definition.display_name,
                        node_class=definition.class_name,
                        node_type=definition.node_type,
                        referenced_in_workflow=is_used,
                        reference_count=reference_count,
                        source_snippet=definition.snippet,
                    )
                )

            package.used_nodes_count = used_count
            package.status = ScannerEngine._classify(len(definitions), used_count)
            package.nodes = nodes
            packages.append(package)
            db.add(package)

        # Assign to the relationship so consumers see packages immediately even
        # when the parent scan was loaded with eager (selectin) relationships.
        scan.packages = packages

    @staticmethod
    def _classify(detected: int, used: int) -> str:
        if detected > 0 and used > 0:
            return "USED"
        if detected > 0:
            return "UNUSED"
        return "UNKNOWN"
