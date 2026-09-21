"""
Password hashing and JWT access-token helpers.

Uses bcrypt directly for hashing (passlib is unmaintained and incompatible
with modern bcrypt releases) and PyJWT for signed tokens.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import settings


class SecurityService:
    @staticmethod
    def get_password_hash(password: str) -> str:
        return bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(),
        ).decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        try:
            return bcrypt.checkpw(
                plain_password.encode("utf-8"),
                hashed_password.encode("utf-8"),
            )
        except ValueError:
            return False

    @staticmethod
    def create_access_token(
        subject: str,
        expires_delta: timedelta | None = None,
    ) -> str:
        expire = datetime.now(timezone.utc) + (
            expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode = {"sub": subject, "exp": expire}
        return jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )

    @staticmethod
    def decode_access_token(token: str) -> dict:
        """Decode and validate a JWT, raising jwt.PyJWTError on failure."""
        return jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
