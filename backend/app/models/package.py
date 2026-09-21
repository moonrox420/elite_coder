"""
Discovered custom-node package and its detected nodes.
"""

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class DiscoveredPackage(Base):
    __tablename__ = "discovered_packages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    scan_id: Mapped[str] = mapped_column(
        ForeignKey("audit_scans.id", ondelete="CASCADE"), index=True, nullable=False
    )

    package_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Directory on the scanned machine that defines this package.
    directory_path: Mapped[str] = mapped_column(Text, nullable=False)

    # USED | UNUSED | UNKNOWN
    status: Mapped[str] = mapped_column(String(20), default="UNKNOWN", nullable=False)

    detected_nodes_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    used_nodes_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    scan = relationship("AuditScan", back_populates="packages")
    nodes = relationship(
        "PackageNode",
        back_populates="package",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="PackageNode.node_name",
    )


class PackageNode(Base):
    """A single node definition detected within a custom-node package."""

    __tablename__ = "package_nodes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    package_id: Mapped[int] = mapped_column(
        ForeignKey("discovered_packages.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    node_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Class name (e.g. "MyAwesomeNode") when defined as a class.
    node_class: Mapped[str] = mapped_column(String(255), nullable=False)
    # The node type string as ComfyUI workflows / NODE_CLASS_MAPPINGS would see it.
    node_type: Mapped[str] = mapped_column(String(255), nullable=False)

    referenced_in_workflow: Mapped[bool] = mapped_column(default=False, nullable=False)
    reference_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Canonical source AST snippet so a reviewer can inspect definitions.
    source_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)

    package = relationship("DiscoveredPackage", back_populates="nodes")
