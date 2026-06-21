# Building the course site

This site is a **Jupyter Book 1.x** (Sphinx-based) project. Browser lessons embed
live JupyterLite/Pyodide notebooks via the `jupyterlite-sphinx` extension; the
PyTorch lessons link out to Google Colab.

## Verified build (Python 3.14, this machine)

The site **builds successfully** on the environment used to author it.

```bash
# 1) Install the build toolchain (NOTE the explicit Jupyter Book 1.x pin — see gotcha)
python -m pip install "jupyter-book==1.0.4.post1" jupyterlite-sphinx jupyterlite-pyodide-kernel

# 2) Build
jupyter-book build book/

# 3) Output
#    book/_build/html/index.html        <- site entry point
#    book/_build/html/lessons/*.html    <- 12 lesson pages
#    book/_build/html/lite/             <- the in-browser JupyterLite deployment
```

### Pinned versions that produced a green build

| Package | Version |
|---|---|
| Python | 3.14.3 |
| jupyter-book | 1.0.4.post1 |
| sphinx | 7.4.7 |
| jupyterlite-sphinx | 0.22.1 |
| jupyterlite-pyodide-kernel | 0.7.2 |
| jupyterlite-core | 0.7.6 |
| myst-nb | 1.4.0 |
| myst-parser | 3.0.1 |
| sphinx-book-theme | 1.2.0 |

## Critical gotcha: Jupyter Book 1.x vs 2.x

As of 2026, **`pip install jupyter-book` installs version 2.x by default**, and
**v2 is a completely different, mystmd (Node.js) based engine** — it does NOT use
`_config.yml` + `_toc.yml` + Sphinx + `jupyterlite-sphinx`. Under v2, `jupyter-book
build book/` will NOT consume the config in this directory and will NOT produce
`book/_build/html/`.

This project's config (`_config.yml`, `_toc.yml`, the `{jupyterlite}` directive,
the `sphinx.extra_extensions` / `html_js_files` keys) is the **Jupyter Book 1.x**
schema. You MUST pin v1 explicitly:

```bash
python -m pip install "jupyter-book==1.0.4.post1"
```

If you accidentally installed v2, uninstall it first (`pip uninstall jupyter-book`)
then install the v1 pin above. Verify with `jupyter-book --version` — it should
print `Jupyter Book : 1.0.4.post1`, not `v2.x`.

## Fallback: Python 3.11 virtual environment

Python 3.14 is brand new; if a transitive dependency fails to build a wheel on
3.14 on your platform, use a Python 3.11 venv (the version range the toolchain is
most heavily tested against):

```bash
py -3.11 -m venv .venv-book
. .venv-book/Scripts/activate          # Windows (Git Bash);  .venv-book\Scripts\activate.bat for cmd
#   source .venv-book/bin/activate     # macOS / Linux
python -m pip install --upgrade pip
python -m pip install "jupyter-book==1.0.4.post1" "sphinx<8" jupyterlite-sphinx jupyterlite-pyodide-kernel
jupyter-book build book/
```

## Dependency on the lesson notebooks

The browser lesson pages embed notebooks from `../notebooks/*.ipynb` (referenced
both by the `{jupyterlite}` directives in `book/lessons/*.md` and by
`jupyterlite_contents` in `_config.yml`). The build therefore requires those
notebook files to exist.

The lesson notebooks are authored by the **notebooks track** using the canonical
filenames (`s1_tokenizer_warmup.ipynb`, `m00_orientation.ipynb` … `m10_whats_next.ipynb`).
If you check out this repo before that track has landed real notebooks, the build
will fail with `FileNotFoundError` on the first missing `.ipynb`. To verify the
site pipeline in isolation, minimal placeholder notebooks with those exact
filenames are sufficient; the real lesson content drops in over them unchanged.

## Static assets (`book/_static/`)

- `gating.js` — client-side progress gating (IndexedDB flags, progress-token
  download, Pyodide load-failure fallback, timeout watchdog). Provided by the
  gating track; wired into every page via `html_js_files` in `_config.yml`.
- `demo_weights.npz` / `demo_meta.json` — the fp16 weights + metadata for the
  Module 0 in-browser demo.

## Repo / Colab URLs

The "Open in Colab" buttons on Modules 7–9 and the repository buttons point at
`https://github.com/your-org/foundation-model` on branch `main`. **Update that
placeholder to the real public repo** (in `_config.yml` under `repository.url`
and in the badge URLs in `book/lessons/m07_gpt.md`, `m08_train.md`,
`m09_capstone.md`) before publishing, or the badges will 404.
