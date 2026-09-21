"""
Discovered package & node response schemas.
"""

from pydantic import BaseModel, ConfigDict


class NodeItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    node_name: str
    node_class: str
    node_type: str
    referenced_in_workflow: bool
    reference_count: int
    source_snippet: str | None = None


class DiscoveredPackageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    scan_id: str
    package_name: str
    directory_path: str
    status: str
    detected_nodes_count: int
    used_nodes_count: int
    nodes: list[NodeItemResponse] = []
