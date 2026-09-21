"""
Package & Node query endpoints.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user
from app.database import get_db
from app.models.package import DiscoveredPackage
from app.models.user import User
from app.schemas.package import DiscoveredPackageResponse

router = APIRouter(prefix="/packages", tags=["Packages"])


@router.get("/", response_model=list[DiscoveredPackageResponse])
async def query_packages(
    scan_id: str = Query(..., description="Target Scan ID"),
    status: str | None = Query(
        None,
        description="Filter by status (USED, UNUSED, UNKNOWN)",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(DiscoveredPackage)
        .where(DiscoveredPackage.scan_id == scan_id)
        .options(selectinload(DiscoveredPackage.nodes))
    )

    if status:
        stmt = stmt.where(DiscoveredPackage.status == status.upper())

    result = await db.execute(stmt)
    return result.scalars().all()
