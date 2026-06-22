"""Server-rendered pages: landing, login, signup, dashboard."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import get_current_user_optional, require_user
from ..models import Progress, User

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory="web/app/templates")
_settings = get_settings()


# The complete curriculum, in order, for the dashboard progress grid.
LESSONS = [
    ("s1",   "Warm-up: Tokenizer",            "/lessons/s1_tokenizer_warmup.html"),
    ("m00",  "Module 0. See Your Goal",       "/lessons/m00_orientation.html"),
    ("m01",  "Module 1. Just Enough NumPy",   "/lessons/m01_numpy.html"),
    ("m02",  "Module 2. A Neuron",            "/lessons/m02_neuron.html"),
    ("m03",  "Module 3. Backprop From Scratch","/lessons/m03_autograd.html"),
    ("m04",  "Module 4. Bigram",              "/lessons/m04_bigram.html"),
    ("m05",  "Module 5. Neural LM",           "/lessons/m05_mlp_lm.html"),
    ("m06",  "Module 6. Attention",           "/lessons/m06_attention.html"),
    ("m07",  "Module 7. Build the GPT",       "/lessons/m07_gpt.html"),
    ("m08",  "Module 8. Train on Shakespeare","/lessons/m08_train.html"),
    ("m09",  "Module 9. Capstone",            "/lessons/m09_capstone.html"),
    ("m10",  "Module 10. What's Next",        "/lessons/m10_whats_next.html"),
]


@router.get("/", response_class=HTMLResponse)
def landing(request: Request, user: User | None = Depends(get_current_user_optional)):
    return templates.TemplateResponse(
        "landing.html",
        {"request": request, "user": user, "course_url": _settings.course_url},
    )


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str | None = None, user: User | None = Depends(get_current_user_optional)):
    if user is not None:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, error: str | None = None, user: User | None = Depends(get_current_user_optional)):
    if user is not None:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("signup.html", {"request": request, "error": error})


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    rows = db.query(Progress).filter(Progress.user_id == user.id).all()
    completed = {r.lesson_id for r in rows}
    lessons_with_status = [
        {"id": lid, "title": title, "url": url, "done": lid in completed}
        for (lid, title, url) in LESSONS
    ]
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "lessons": lessons_with_status,
            "completed_count": len(completed),
            "total_count": len(LESSONS),
            "course_url": _settings.course_url,
        },
    )
