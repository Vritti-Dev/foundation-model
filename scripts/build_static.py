"""Build a self-contained static site for GitHub Pages.

Produces a ``dist/`` directory with:

* ``index.html``       - the manifesto landing, pre-rendered (no user context).
* ``static/``          - landing CSS, JS, demo weights.
* ``course/``          - the Jupyter Book output (intro, lessons, JupyterLite).
* ``signup.html``      - a stub that links to the GitHub repo (no backend here).
* ``login.html``       - same stub.
* ``404.html``         - friendly not-found page.

The login/dashboard/progress flows are stripped for the Pages build. The course
itself stays fully functional because every browser lesson runs entirely in
the visitor's browser via JupyterLite.

Run from the repo root::

    python scripts/build_static.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "web" / "app" / "templates"
WEB_STATIC = ROOT / "web" / "app" / "static"
BOOK_HTML = ROOT / "book" / "_build" / "html"
DIST = ROOT / "dist"


# Mirror of routes/pages.PHASES so the script doesn't import the FastAPI app
# (and pull in its DB/JWT dependencies for a static build).
PHASES = [
    {
        "key": "warmup",
        "label": "Warm-up",
        "blurb": "One lesson. Just enough to prove the whole thing works in your browser.",
        "lessons": [
            {"id": "s1",  "title": "Warm-up: Build a Tokenizer", "summary": "A character-level tokenizer. The numbers-to-text bridge every model needs.", "runtime": "browser", "url": "course/lessons/s1_tokenizer_warmup.html"},
        ],
    },
    {
        "key": "by_hand",
        "label": "Build it by hand (in your browser)",
        "blurb": "Seven lessons. Pure Python and NumPy. No frameworks. Understand every line.",
        "lessons": [
            {"id": "m00", "title": "Module 0. See Your Goal",         "summary": "A trained tiny GPT writes Shakespeare-style text in your browser. See the destination first.",          "runtime": "browser", "url": "course/lessons/m00_orientation.html"},
            {"id": "m01", "title": "Module 1. Just Enough NumPy",     "summary": "Arrays, indexing, broadcasting, dot product. The math language of the course.",                       "runtime": "browser", "url": "course/lessons/m01_numpy.html"},
            {"id": "m02", "title": "Module 2. A Neuron",              "summary": "One neuron, one hand-derived gradient step. Watch loss go down. The seed of every neural network.", "runtime": "browser", "url": "course/lessons/m02_neuron.html"},
            {"id": "m03", "title": "Module 3. Backprop From Scratch", "summary": "Build a tiny autograd engine. Reverse-mode automatic differentiation in 100 lines.",                 "runtime": "browser", "url": "course/lessons/m03_autograd.html"},
            {"id": "m04", "title": "Module 4. A Bigram That Writes",  "summary": "Your first real generator. Pure statistics, no neural net yet. It writes gibberish, beautifully.",   "runtime": "browser", "url": "course/lessons/m04_bigram.html"},
            {"id": "m05", "title": "Module 5. A Neural Language Model","summary":"Embeddings + MLP + a real training loop. The Bengio 2003 architecture, the ancestor of every LLM.",   "runtime": "browser", "url": "course/lessons/m05_mlp_lm.html"},
            {"id": "m06", "title": "Module 6. Attention From Scratch","summary": "Self-attention with a causal mask. The single most important idea in modern deep learning.",          "runtime": "browser", "url": "course/lessons/m06_attention.html"},
        ],
    },
    {
        "key": "assemble",
        "label": "Assemble and train (free on Colab)",
        "blurb": "Three lessons. PyTorch GPT, real training, your own corpus.",
        "lessons": [
            {"id": "m07", "title": "Module 7. Build the GPT",         "summary": "Assemble the full decoder-only transformer in PyTorch. nanoGPT structure end to end.",                "runtime": "colab",   "url": "course/lessons/m07_gpt.html"},
            {"id": "m08", "title": "Module 8. Train on Shakespeare",  "summary": "Train your GPT on Tiny Shakespeare. Watch the loss fall, watch the samples become English.",         "runtime": "colab",   "url": "course/lessons/m08_train.html"},
            {"id": "m09", "title": "Module 9. Capstone: Make It Yours","summary":"Train the same GPT on a text corpus YOU choose. Hear your model speak in a voice you picked.",         "runtime": "colab",   "url": "course/lessons/m09_capstone.html"},
        ],
    },
    {
        "key": "next",
        "label": "Where to go next",
        "blurb": "One reading. How real LLMs differ from what you just built.",
        "lessons": [
            {"id": "m10", "title": "Module 10. What's Next",          "summary": "Scaling laws, RoPE, FlashAttention, RLHF. Every refinement on the spine you just built.",              "runtime": "reading", "url": "course/lessons/m10_whats_next.html"},
        ],
    },
]


# When rendering for GitHub Pages we want relative-from-root paths (no leading
# slash) so the site works regardless of project-page subpath. Lesson URLs
# already use that form above; this is just a sanity-print.

STUB_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coming back soon. Vritti AI Labs</title>
<link rel="stylesheet" href="static/manifesto.css">
</head>
<body class="page-auth">
<header class="nav">
  <a class="brand" href="index.html"><span class="brand-mark">&#9650;</span><span class="brand-name">Vritti AI Labs</span></a>
  <nav class="nav-links">
    <a href="index.html#mission">Mission</a>
    <a href="https://github.com/Vritti-Dev/foundation-model" class="btn-ghost">GitHub</a>
    <a href="course/lessons/s1_tokenizer_warmup.html" class="btn-primary">Start free</a>
  </nav>
</header>
<main class="auth-shell">
  <div class="auth-card">
    <h1 class="auth-title">Coming back soon</h1>
    <p class="auth-sub">
      Sign in and progress sync need a real backend; this preview lives on
      GitHub Pages, which only serves static files. The course itself is
      fully working below.
    </p>
    <a href="course/lessons/s1_tokenizer_warmup.html" class="btn-primary btn-block">Open the warm-up</a>
    <p class="auth-foot"><a href="index.html">&larr; Back to the landing</a></p>
  </div>
</main>
</body>
</html>
"""


def _phases_for_render() -> list[dict]:
    """No-user version of the phases data the landing template expects."""
    out = []
    for p in PHASES:
        out.append({
            **p,
            "lessons": [{**l, "done": False, "auth_url": l["url"]} for l in p["lessons"]],
        })
    return out


def _render_landing() -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))
    template = env.get_template("landing.html")
    phases = _phases_for_render()
    return template.render(
        request=None,
        user=None,
        static_mode=True,
        phases=phases,
        first_lesson_url=phases[0]["lessons"][0]["url"],
        completed_count=0,
        total_count=sum(len(p["lessons"]) for p in PHASES),
        course_url="course/intro.html",
    )


def _rewrite_root_paths(html: str) -> str:
    """The landing template uses absolute paths like ``/static/...`` and
    ``/course/lessons/...`` so the FastAPI app can serve them. For GitHub
    Pages on a project page (``/foundation-model/``) we need them relative."""
    # Order matters: do longer replacements first.
    replacements = [
        ('href="/static/', 'href="static/'),
        ('src="/static/',  'src="static/'),
        ('href="/course/', 'href="course/'),
        ('src="/course/',  'src="course/'),
        ('href="/signup"', 'href="signup.html"'),
        ('href="/login"',  'href="login.html"'),
        ('href="/dashboard"', 'href="index.html"'),
        ('href="/#',       'href="#'),
        ('href="/"',       'href="index.html"'),
    ]
    for a, b in replacements:
        html = html.replace(a, b)
    return html


def _copy_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    if not BOOK_HTML.exists():
        print(f"ERROR: {BOOK_HTML} does not exist. Run `jupyter-book build book/` first.")
        return 1

    print(f"building static site in {DIST}")
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    # Landing.
    html = _rewrite_root_paths(_render_landing())
    (DIST / "index.html").write_text(html, encoding="utf-8")
    print(" - index.html (landing)")

    # Stubs for /signup and /login so accidental links land somewhere friendly.
    (DIST / "signup.html").write_text(STUB_HTML, encoding="utf-8")
    (DIST / "login.html").write_text(STUB_HTML, encoding="utf-8")
    (DIST / "404.html").write_text(STUB_HTML, encoding="utf-8")
    print(" - signup.html, login.html, 404.html (stubs)")

    # Static assets.
    _copy_dir(WEB_STATIC, DIST / "static")
    print(f" - static/ ({sum(1 for _ in (DIST / 'static').rglob('*') if _.is_file())} files)")

    # Course content.
    _copy_dir(BOOK_HTML, DIST / "course")
    print(f" - course/ ({sum(1 for _ in (DIST / 'course').rglob('*') if _.is_file())} files)")

    # GitHub Pages-specific niceties.
    (DIST / ".nojekyll").touch()  # Disable Jekyll so files starting with _ are served.
    print(" - .nojekyll")

    print(f"done. open {DIST}/index.html or push to GitHub Pages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
