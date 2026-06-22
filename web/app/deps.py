"""Shared FastAPI dependencies (current user, etc.)."""
from __future__ import annotations

import uuid

import jwt
from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .models import User
from .security import decode_session_token

_settings = get_settings()


def get_current_user_optional(
    session: str | None = Cookie(default=None, alias=_settings.session_cookie),
    db: Session = Depends(get_db),
) -> User | None:
    if not session:
        return None
    try:
        payload = decode_session_token(session)
    except jwt.InvalidTokenError:
        return None
    try:
        uid = uuid.UUID(payload["sub"])
    except (KeyError, ValueError):
        return None
    return db.get(User, uid)


def require_user(user: User | None = Depends(get_current_user_optional)) -> User:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="not signed in",
            headers={"Location": "/login"},
        )
    return user
