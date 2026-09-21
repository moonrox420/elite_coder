"""
Scan initiation & history endpoint tests.
"""

import pytest
from httpx import AsyncClient

SCANS_URL = "/api/v1/scans/"


@pytest.mark.anyio
async def test_initiate_scan_requires_authentication(client: AsyncClient):
    response = await client.post(SCANS_URL, json={})
    assert response.status_code == 401


@pytest.mark.anyio
async def test_initiate_scan_returns_accepted_pending(
    client: AsyncClient,
    auth_headers: dict,
):
    response = await client.post(
        SCANS_URL,
        headers=auth_headers,
        json={},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "PENDING"
    assert payload["scanned_root_path"] == "AUTO_DISCOVERY"
    assert payload["initiated_by"] == "auditor@enterprise.com"


@pytest.mark.anyio
async def test_initiate_scan_records_custom_roots(
    client: AsyncClient,
    auth_headers: dict,
):
    response = await client.post(
        SCANS_URL,
        headers=auth_headers,
        json={
            "custom_nodes_roots": ["/opt/comfy/custom_nodes"],
            "workflows_roots": ["/opt/comfy/workflows"],
            "include_image_metadata": True,
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["scanned_root_path"] == "/opt/comfy/custom_nodes"


@pytest.mark.anyio
async def test_list_scans_returns_created_scan(
    client: AsyncClient,
    auth_headers: dict,
):
    create_response = await client.post(
        SCANS_URL,
        headers=auth_headers,
        json={},
    )
    scan_id = create_response.json()["id"]

    list_response = await client.get(
        SCANS_URL,
        headers=auth_headers,
    )

    assert list_response.status_code == 200
    scans = list_response.json()
    assert any(scan["id"] == scan_id for scan in scans)


@pytest.mark.anyio
async def test_get_scan_details_not_found(
    client: AsyncClient,
    auth_headers: dict,
):
    response = await client.get(
        f"{SCANS_URL}does-not-exist",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.anyio
async def test_get_scan_details_returns_structure(
    client: AsyncClient,
    auth_headers: dict,
):
    create_response = await client.post(
        SCANS_URL,
        headers=auth_headers,
        json={},
    )
    scan_id = create_response.json()["id"]

    detail_response = await client.get(
        f"{SCANS_URL}{scan_id}",
        headers=auth_headers,
    )

    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["id"] == scan_id
    assert "packages" in detail
    assert "workflows" in detail
