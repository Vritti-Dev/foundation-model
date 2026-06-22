"""FastAPI application entrypoint."""
from __future__ import annotations

import os
from urllib.parse import quote

import jwt
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .routes import auth, pages, progress
from .security import decode_session_token

_settings = get_settings()

app = FastAPI(title=_settings.app_name)

app.mount("/static", StaticFiles(directory="web/app/static"), name="static")

# Mount the Jupyter Book at /course/* when it's been built locally. The middleware
# below gates this so only signed-in users can read it.
_BOOK_HTML = os.path.join("book", "_build", "html")
if os.path.isdir(_BOOK_HTML):
    app.mount("/course", StaticFiles(directory=_BOOK_HTML, html=True), name="course")


# Paths whose subtrees are publicly reachable inside /course (none right now).
# JupyterLite's lite/ deployment is part of /course and also gated.
_PUBLIC_COURSE_PREFIXES: tuple[str, ...] = ()


@app.middleware("http")
async def gate_course_content(request: Request, call_next):
    """Require a valid session cookie for any /course/* request.

    Anonymous users get a 303 redirect to /login?next=<original-url>, which
    /auth/login and /auth/signup honour after a successful auth.
    """
    path = request.url.path
    if path.startswith("/course") and not any(path.startswith(p) for p in _PUBLIC_COURSE_PREFIXES):
        token = request.cookies.get(_settings.session_cookie)
        ok = False
        if token:
            try:
                decode_session_token(token)
                ok = True
            except jwt.InvalidTokenError:
                ok = False
        if not ok:
            # Preserve query string so deep links survive the round-trip.
            nxt = path
            if request.url.query:
                nxt = f"{path}?{request.url.query}"
            return RedirectResponse(url=f"/login?next={quote(nxt, safe='')}", status_code=303)
    return await call_next(request)


app.include_router(pages.router)
app.include_router(auth.router)
app.include_router(progress.router)


@app.get("/healthz")
def healthz():
    return {"ok": True, "app": _settings.app_name}
