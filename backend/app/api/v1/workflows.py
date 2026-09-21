"""
Workflow analysis endpoints.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.user import User
from app.models.workflow import WorkflowItem
from app.schemas.workflow import WorkflowItemResponse

router = APIRouter(prefix="/workflows", tags=["Workflows"])


@router.get("/", response_model=list[WorkflowItemResponse])
async def list_scan_workflows(
    scan_id: str = Query(..., description="Target Scan ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = select(WorkflowItem).where(WorkflowItem.scan_id == scan_id)

    result = await db.execute(stmt)
    return result.scalars().all()
