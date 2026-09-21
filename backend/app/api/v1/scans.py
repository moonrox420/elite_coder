"""
Audit Scans Execution and History endpoints.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user
from app.database import AsyncSessionLocal, get_db
from app.models.package import DiscoveredPackage
from app.models.scan import AuditScan
from app.models.user import User
from app.schemas.scan import (
    AuditScanDetailResponse,
    AuditScanSummaryResponse,
    ScanRequest,
)
from app.services.scanner_engine import ScannerEngine

router = APIRouter(prefix="/scans", tags=["Audit Scans"])
logger = logging.getLogger("comfyaudit.scans")


async def run_background_scan(scan_id: str, scan_in: ScanRequest) -> None:
    async with AsyncSessionLocal() as session:
        try:
            await ScannerEngine.execute_audit_scan(
                db=session,
                scan_id=scan_id,
                custom_node_paths=scan_in.custom_nodes_roots,
                workflow_paths=scan_in.workflows_roots,
                include_images=scan_in.include_image_metadata,
            )
        except Exception:
            # ScannerEngine is responsible for persisting the scan failure state.
            # Log and swallow: never let a background-task exception kill the worker.
            logger.exception("Background audit scan %s failed", scan_id)


@router.post(
    "/",
    response_model=AuditScanSummaryResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def initiate_scan(
    scan_in: ScanRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scanned_path = (
        ",".join(scan_in.custom_nodes_roots)
        if scan_in.custom_nodes_roots
        else "AUTO_DISCOVERY"
    )

    new_scan = AuditScan(
        status="PENDING",
        initiated_by=current_user.email,
        scanned_root_path=scanned_path,
    )

    db.add(new_scan)
    await db.commit()
    await db.refresh(new_scan)

    background_tasks.add_task(
        run_background_scan,
        new_scan.id,
        scan_in,
    )

    return new_scan


@router.get("/", response_model=list[AuditScanSummaryResponse])
async def list_scans(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(AuditScan).order_by(desc(AuditScan.created_at)).offset(skip).limit(limit)
    )

    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{scan_id}", response_model=AuditScanDetailResponse)
async def get_scan_details(
    scan_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(AuditScan)
        .where(AuditScan.id == scan_id)
        .options(
            selectinload(AuditScan.packages).selectinload(DiscoveredPackage.nodes),
            selectinload(AuditScan.workflows),
        )
    )

    scan = (await db.execute(stmt)).scalar_one_or_none()

    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit scan not found",
        )

    return scan
