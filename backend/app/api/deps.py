"""
Shared API dependencies — OAuth2 bearer authentication.
"""

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.services.security_service import SecurityService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{'/api/v1/auth/login'}")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = SecurityService.decode_access_token(token)
    except jwt.PyJWTError as exc:
        raise credentials_exc from exc

    email = payload.get("sub")
    if not email:
        raise credentials_exc

    stmt = select(User).where(User.email == email)
    user = (await db.execute(stmt)).scalar_one_or_none()

    if not user or not user.is_active:
        raise credentials_exc

    return user
