"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .routes import auth, pages, progress

_settings = get_settings()

app = FastAPI(title=_settings.app_name)

app.mount("/static", StaticFiles(directory="web/app/static"), name="static")

# Optional: serve the Jupyter Book at /lessons /intro.html /lite/... when it has
# been built. In production the static site is typically served by a separate
# host (GitHub Pages, nginx). Locally, mounting it here lets you click straight
# into the course from the dashboard.
import os
_BOOK_HTML = os.path.join("book", "_build", "html")
if os.path.isdir(_BOOK_HTML):
    app.mount("/course", StaticFiles(directory=_BOOK_HTML, html=True), name="course")

app.include_router(pages.router)
app.include_router(auth.router)
app.include_router(progress.router)


@app.get("/healthz")
def healthz():
    return {"ok": True, "app": _settings.app_name}
