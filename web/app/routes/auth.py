"""Auth routes: signup, login, logout.

Accepts both JSON and HTML form posts so the templates can submit forms
directly without JS. JSON responses keep the same shape for a future SPA."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..models import User
from ..schemas import SignupIn, UserOut
from ..security import create_session_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])
_settings = get_settings()


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=_settings.session_cookie,
        value=token,
        max_age=_settings.jwt_expire_minutes * 60,
        httponly=True,
        secure=_settings.cookie_secure,
        samesite=_settings.cookie_samesite,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=_settings.session_cookie, path="/")


async def _read_credentials(
    request: Request,
    name: str | None,
    email: str | None,
    password: str | None,
) -> dict:
    """Accept either application/x-www-form-urlencoded or JSON bodies."""
    if email and password:
        return {"name": name or "", "email": email, "password": password}
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        return {
            "name": body.get("name", ""),
            "email": body.get("email", ""),
            "password": body.get("password", ""),
        }
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="email and password required")


@router.post("/signup")
async def signup(
    request: Request,
    name: str | None = Form(default=None),
    email: str | None = Form(default=None),
    password: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    creds = await _read_credentials(request, name, email, password)
    try:
        data = SignupIn(**creds)
    except Exception as exc:
        return _form_or_json(request, error=str(exc), status_code=400, path="/signup")

    user = User(name=data.name.strip(), email=data.email.lower(), password_hash=hash_password(data.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return _form_or_json(request, error="An account with that email already exists.", status_code=409, path="/signup")
    db.refresh(user)
    token = create_session_token(user.id, user.email)
    return _success_response(request, token, user)


@router.post("/login")
async def login(
    request: Request,
    email: str | None = Form(default=None),
    password: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    creds = await _read_credentials(request, None, email, password)
    user = db.query(User).filter(User.email == creds["email"].lower()).first()
    if not user or not verify_password(creds["password"], user.password_hash):
        return _form_or_json(request, error="Invalid email or password.", status_code=401, path="/login")
    token = create_session_token(user.id, user.email)
    return _success_response(request, token, user)


@router.post("/logout")
async def logout(request: Request):
    if request.headers.get("accept", "").startswith("application/json"):
        resp = JSONResponse({"ok": True})
    else:
        resp = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    _clear_session_cookie(resp)
    return resp


def _success_response(request: Request, token: str, user: User):
    if request.headers.get("accept", "").startswith("application/json"):
        resp = JSONResponse({"ok": True, "user": UserOut.model_validate(user).model_dump(mode="json")})
    else:
        resp = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    _set_session_cookie(resp, token)
    return resp


def _form_or_json(request: Request, error: str, status_code: int, path: str):
    if request.headers.get("accept", "").startswith("application/json"):
        return JSONResponse({"ok": False, "error": error}, status_code=status_code)
    # Show the form again with the error message via query string.
    from urllib.parse import quote
    return RedirectResponse(url=f"{path}?error={quote(error)}", status_code=status.HTTP_303_SEE_OTHER)
