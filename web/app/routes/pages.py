"""Server-rendered pages: landing, learn (course intro), login, signup, dashboard."""
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


# The complete curriculum grouped into the 4 phases of the course. Each lesson
# has a stable id (for progress), a title, a one-line summary the course intro
# page renders, a runtime tag, and the path to the lesson in the Jupyter Book
# (which is mounted at /course/ in main.py).
PHASES = [
    {
        "key": "warmup",
        "label": "Warm-up",
        "blurb": "One lesson. Just enough to prove the whole thing works in your browser.",
        "lessons": [
            {
                "id": "s1",
                "title": "Warm-up: Build a Tokenizer",
                "summary": "A character-level tokenizer. The numbers-to-text bridge every model needs.",
                "runtime": "browser",
                "url": "/course/lessons/s1_tokenizer_warmup.html",
            },
        ],
    },
    {
        "key": "by_hand",
        "label": "Build it by hand (in your browser)",
        "blurb": "Seven lessons. Pure Python and NumPy. No frameworks. Understand every line.",
        "lessons": [
            {"id": "m00", "title": "Module 0. See Your Goal",        "summary": "A trained tiny GPT writes Shakespeare-style text in your browser. See the destination first.",          "runtime": "browser", "url": "/course/lessons/m00_orientation.html"},
            {"id": "m01", "title": "Module 1. Just Enough NumPy",    "summary": "Arrays, indexing, broadcasting, dot product. The math language of the course.",                       "runtime": "browser", "url": "/course/lessons/m01_numpy.html"},
            {"id": "m02", "title": "Module 2. A Neuron",             "summary": "One neuron, one hand-derived gradient step. Watch loss go down. The seed of every neural network.", "runtime": "browser", "url": "/course/lessons/m02_neuron.html"},
            {"id": "m03", "title": "Module 3. Backprop From Scratch","summary": "Build a tiny autograd engine. Reverse-mode automatic differentiation in 100 lines.",                  "runtime": "browser", "url": "/course/lessons/m03_autograd.html"},
            {"id": "m04", "title": "Module 4. A Bigram That Writes", "summary": "Your first real generator. Pure statistics, no neural net yet. It writes gibberish, beautifully.",   "runtime": "browser", "url": "/course/lessons/m04_bigram.html"},
            {"id": "m05", "title": "Module 5. A Neural Language Model","summary": "Embeddings + MLP + a real training loop. The Bengio 2003 architecture, the ancestor of every LLM.","runtime": "browser", "url": "/course/lessons/m05_mlp_lm.html"},
            {"id": "m06", "title": "Module 6. Attention From Scratch","summary": "Self-attention with a causal mask. The single most important idea in modern deep learning.",         "runtime": "browser", "url": "/course/lessons/m06_attention.html"},
        ],
    },
    {
        "key": "assemble",
        "label": "Assemble and train (free on Colab)",
        "blurb": "Three lessons. PyTorch GPT, real training, your own corpus.",
        "lessons": [
            {"id": "m07", "title": "Module 7. Build the GPT",        "summary": "Assemble the full decoder-only transformer in PyTorch. nanoGPT structure end to end.",                "runtime": "colab",   "url": "/course/lessons/m07_gpt.html"},
            {"id": "m08", "title": "Module 8. Train on Shakespeare", "summary": "Train your GPT on Tiny Shakespeare. Watch the loss fall, watch the samples become English.",        "runtime": "colab",   "url": "/course/lessons/m08_train.html"},
            {"id": "m09", "title": "Module 9. Capstone: Make It Yours","summary": "Train the same GPT on a text corpus YOU choose. Hear your model speak in a voice you picked.",       "runtime": "colab",   "url": "/course/lessons/m09_capstone.html"},
        ],
    },
    {
        "key": "next",
        "label": "Where to go next",
        "blurb": "One reading. How real LLMs differ from what you just built.",
        "lessons": [
            {"id": "m10", "title": "Module 10. What's Next",         "summary": "Scaling laws, RoPE, FlashAttention, RLHF. Every refinement on the spine you just built.",             "runtime": "reading", "url": "/course/lessons/m10_whats_next.html"},
        ],
    },
]


def _flat_lessons() -> list[dict]:
    """Flatten PHASES into a single ordered list. Used by the dashboard."""
    out = []
    for p in PHASES:
        for l in p["lessons"]:
            out.append({**l, "phase": p["label"]})
    return out


# Total lessons for progress display.
TOTAL_LESSONS = sum(len(p["lessons"]) for p in PHASES)


def _gate(target: str, user: User | None) -> str:
    """Return the lesson URL if the user is signed in; otherwise the signup URL
    with ?next= so they bounce back after auth."""
    if user is not None:
        return target
    from urllib.parse import quote
    return f"/signup?next={quote(target, safe='')}"


@router.get("/", response_class=HTMLResponse)
def landing(
    request: Request,
    user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Unified single-page entry: manifesto + curriculum + mission + CTA."""
    completed: set[str] = set()
    if user is not None:
        rows = db.query(Progress).filter(Progress.user_id == user.id).all()
        completed = {r.lesson_id for r in rows}
    phases = []
    for p in PHASES:
        phases.append({
            **p,
            "lessons": [
                {**l, "done": l["id"] in completed, "auth_url": _gate(l["url"], user)}
                for l in p["lessons"]
            ],
        })
    first_lesson_target = PHASES[0]["lessons"][0]["url"]
    first_lesson_url = _gate(first_lesson_target, user)
    return templates.TemplateResponse(
        "landing.html",
        {
            "request": request,
            "user": user,
            "phases": phases,
            "first_lesson_url": first_lesson_url,
            "completed_count": len(completed),
            "total_count": TOTAL_LESSONS,
            "course_url": _settings.course_url,
        },
    )


@router.get("/learn")
def learn_redirect():
    """Old /learn URL: send everyone to the unified landing's curriculum anchor."""
    return RedirectResponse(url="/#curriculum", status_code=301)


def _sanitize_next(value: str | None) -> str | None:
    if not value:
        return None
    if not value.startswith("/") or value.startswith("//"):
        return None
    return value


@router.get("/login", response_class=HTMLResponse)
def login_page(
    request: Request,
    error: str | None = None,
    next: str | None = None,
    user: User | None = Depends(get_current_user_optional),
):
    nxt = _sanitize_next(next)
    if user is not None:
        return RedirectResponse(url=nxt or "/dashboard")
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": error, "next": nxt, "course_url": _settings.course_url},
    )


@router.get("/signup", response_class=HTMLResponse)
def signup_page(
    request: Request,
    error: str | None = None,
    next: str | None = None,
    user: User | None = Depends(get_current_user_optional),
):
    nxt = _sanitize_next(next)
    if user is not None:
        return RedirectResponse(url=nxt or "/dashboard")
    return templates.TemplateResponse(
        "signup.html",
        {"request": request, "error": error, "next": nxt, "course_url": _settings.course_url},
    )


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    rows = db.query(Progress).filter(Progress.user_id == user.id).all()
    completed = {r.lesson_id for r in rows}
    lessons = [{**l, "done": l["id"] in completed} for l in _flat_lessons()]
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "lessons": lessons,
            "completed_count": len(completed),
            "total_count": TOTAL_LESSONS,
            "course_url": _settings.course_url,
        },
    )
