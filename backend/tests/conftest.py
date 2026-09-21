"""
Pytest Fixtures and asynchronous test client configuration.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models.user import User
from app.services.security_service import SecurityService

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# :memory: SQLite creates a fresh database per connection; StaticPool pins the
# engine to one connection so all sessions share the same in-memory database.
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
async def init_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def session_factory():
    """Expose the test session factory to tests via dependency injection.

    Importing `tests.conftest` directly would create a *second* copy of the
    module (and thus a second engine + empty in-memory database); fixtures are
    the single source of truth.
    """
    return TestingSessionLocal


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict:
    async with TestingSessionLocal() as session:
        test_user = User(
            email="auditor@enterprise.com",
            hashed_password=SecurityService.get_password_hash("SecurePassword123!"),
            full_name="Lead Security Auditor",
            is_active=True,
        )

        session.add(test_user)
        await session.commit()

    token = SecurityService.create_access_token("auditor@enterprise.com")

    return {
        "Authorization": f"Bearer {token}",
    }
