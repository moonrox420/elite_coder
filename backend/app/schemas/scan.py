"""
Scan execution request/response schemas.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.package import DiscoveredPackageResponse
from app.schemas.workflow import WorkflowItemResponse


class ScanRequest(BaseModel):
    custom_nodes_roots: list[str] = Field(
        default_factory=list,
        description="Overrides for custom node root directories.",
    )
    workflows_roots: list[str] = Field(
        default_factory=list,
        description="Directories searched for workflow JSON files.",
    )
    include_image_metadata: bool = True


class AuditScanSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    initiated_by: str
    scanned_root_path: str
    created_at: datetime
    completed_at: datetime | None = None
    error_detail: str | None = None


class AuditScanDetailResponse(AuditScanSummaryResponse):
    packages: list[DiscoveredPackageResponse] = Field(default_factory=list)
    workflows: list[WorkflowItemResponse] = Field(default_factory=list)
