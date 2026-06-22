# Building the book

This project uses **Jupyter Book 1.x (the Sphinx-based engine)**. The pinned line that builds on Python 3.14 is:

```bash
pip install "jupyter-book==1.0.4.post1" jupyterlite-sphinx jupyterlite-pyodide-kernel
jupyter-book build book/
python book/fix_lite_paths.py   # Windows only; no-op elsewhere
```

The output lands in `book/_build/html/`. Serve it locally with:

```bash
python -m http.server 8123 --directory book/_build/html
```

## Windows: backslash in launcher URLs

On Windows, `jupyterlite-sphinx` emits the "Launch in your browser" / "Open as a notebook" button with a backslash separator (e.g. `..\lite/lab/...`). Browsers do not normalize backslashes in HTTP URLs, so the button 404s when served over `http://` (it works only on `file://`). The `book/fix_lite_paths.py` script walks the built HTML and rewrites those to forward slashes. Always run it after `jupyter-book build` on Windows. The CI workflow runs it too; on Linux it does nothing.

## Important: do not install `jupyter-book` without a pin

A bare `pip install jupyter-book` now installs **v2.x**, which is a different engine (mystmd / Node.js based) that does **not** consume `_config.yml`, `_toc.yml`, or the `jupyterlite-sphinx` extension. It will **not** produce `book/_build/html/` and will fail silently for this project's schema. You must pin v1 explicitly.

## Python 3.14 notes

The pinned combination above (jupyter-book 1.0.4.post1, sphinx 7.4.7, jupyterlite-sphinx 0.22) builds cleanly on Python 3.14.3. If you hit an install issue on a newer Python, fall back to a Python 3.11 venv:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install "jupyter-book==1.0.4.post1" jupyterlite-sphinx jupyterlite-pyodide-kernel
jupyter-book build book/
```

## What the build does

- Renders the curriculum from `book/intro.md` and `book/lessons/*.md` to static HTML in `book/_build/html/`.
- Copies `book/_static/gating.js` and the demo weights into `_build/html/_static/`.
- Generates the JupyterLite kernel deployment at `_build/html/lite/`, which is what runs the in-browser notebooks for Modules 0 to 6.
