"""
Audit scan execution record model.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _generate_scan_id() -> str:
    return str(uuid.uuid4())


class AuditScan(Base):
    __tablename__ = "audit_scans"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_scan_id
    )
    # PENDING -> RUNNING -> COMPLETED | FAILED
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False)

    initiated_by: Mapped[str] = mapped_column(String(255), nullable=False)
    scanned_root_path: Mapped[str] = mapped_column(Text, default="AUTO_DISCOVERY")
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    packages = relationship(
        "DiscoveredPackage",
        back_populates="scan",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    workflows = relationship(
        "WorkflowItem",
        back_populates="scan",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
