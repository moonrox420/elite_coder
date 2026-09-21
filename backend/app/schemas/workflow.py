"""
Workflow item response schemas.
"""

from pydantic import BaseModel, ConfigDict


class WorkflowItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    scan_id: str
    file_path: str
    referenced_nodes: list
    image_paths: list
