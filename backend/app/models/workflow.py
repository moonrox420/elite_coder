"""
Workflow file discovered during a scan and its analyzed references.
"""

from sqlalchemy import JSON, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class WorkflowItem(Base):
    __tablename__ = "workflow_items"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    scan_id: Mapped[str] = mapped_column(
        ForeignKey("audit_scans.id", ondelete="CASCADE"), index=True, nullable=False
    )

    file_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Distinct node types referenced by this workflow.
    referenced_nodes: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    # Set when image metadata analysis is enabled (paths of png/json sidecars).
    image_paths: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    scan = relationship("AuditScan", back_populates="workflows")
