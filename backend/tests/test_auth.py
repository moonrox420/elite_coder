"""
Authentication endpoint tests: register, login and /auth/me.
"""

import pytest
from httpx import AsyncClient

REGISTER_URL = "/api/v1/auth/register"
LOGIN_URL = "/api/v1/auth/login"
ME_URL = "/api/v1/auth/me"


@pytest.mark.anyio
async def test_register_creates_user(client: AsyncClient):
    response = await client.post(
        REGISTER_URL,
        json={
            "email": "new.auditor@enterprise.com",
            "password": "SuperSecretPass123!",
            "full_name": "New Auditor",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["email"] == "new.auditor@enterprise.com"
    assert payload["full_name"] == "New Auditor"
    assert payload["is_active"] is True
    assert "hashed_password" not in payload


@pytest.mark.anyio
async def test_register_duplicate_email_rejected(client: AsyncClient):
    user = {
        "email": "dup@enterprise.com",
        "password": "SuperSecretPass123!",
    }
    first = await client.post(REGISTER_URL, json=user)
    assert first.status_code == 201

    second = await client.post(REGISTER_URL, json=user)
    assert second.status_code == 400
    assert "already exists" in second.json()["detail"]


@pytest.mark.anyio
async def test_login_returns_bearer_token(client: AsyncClient):
    await client.post(
        REGISTER_URL,
        json={
            "email": "login@enterprise.com",
            "password": "SuperSecretPass123!",
        },
    )

    response = await client.post(
        LOGIN_URL,
        data={
            "username": "login@enterprise.com",
            "password": "SuperSecretPass123!",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["access_token"]


@pytest.mark.anyio
async def test_login_wrong_password_rejected(client: AsyncClient):
    await client.post(
        REGISTER_URL,
        json={
            "email": "wrongpw@enterprise.com",
            "password": "SuperSecretPass123!",
        },
    )

    response = await client.post(
        LOGIN_URL,
        data={
            "username": "wrongpw@enterprise.com",
            "password": "not-the-right-password",
        },
    )

    assert response.status_code == 401


@pytest.mark.anyio
async def test_me_requires_authentication(client: AsyncClient):
    response = await client.get(ME_URL)
    assert response.status_code == 401


@pytest.mark.anyio
async def test_me_returns_current_user(
    client: AsyncClient,
    auth_headers: dict,
):
    response = await client.get(ME_URL, headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["email"] == "auditor@enterprise.com"
    assert payload["full_name"] == "Lead Security Auditor"
