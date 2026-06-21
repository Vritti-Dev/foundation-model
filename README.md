# Build an SLM From Scratch. Course Content-Core

A guided, scaffolded course that takes a complete beginner (basic Python, zero ML) all the way to **building and training their own small GPT from scratch**, ending with them sampling text from a model they built and trained themselves.

The pedagogy is *build-by-hand-first*: learners implement the neuron, backpropagation, the tokenizer, and attention in pure Python/NumPy so they understand the machinery, and only adopt PyTorch once they have earned that understanding and need it to assemble and train the full transformer.

This repository is the **content-core**: the curriculum, the reference "answer-key" SLM build, and the auto-grading/checkpoint engine. It is delivered as free static hosting (Jupyter Book + JupyterLite on GitHub Pages) plus Google Colab for the PyTorch work. A custom LMS is a separate, later sub-project. See the deferred items in the spec.

## Architecture

The course straddles a deliberate pedagogical seam that also happens to be a hard runtime boundary: **PyTorch cannot run in the browser**, and that wall lands exactly where learners graduate from hand-built NumPy to PyTorch.

| Modules | Runtime | Mechanism |
|---|---|---|
| **S1 plus M0 to M6** (tokenizer, neuron, backprop, bigram, MLP LM, attention) | **In-browser** | JupyterLite / Pyodide plus NumPy. Pure Python, no torch. Toy data only. $0, offline after first load. |
| **M7 to M9** (build the GPT, train on Tiny Shakespeare, capstone) | **Google Colab (free T4)** | One-click "Open in Colab" from this public repo. Graded by a pasted submission token. |
| **M10** (what's next) | **Static reading** | Plain HTML/markdown plus a "mark complete" cell. |

The reference build is plain Python packages tested with pytest. Grading has two client-side tiers for the browser modules (in-notebook `assert` checks as the guaranteed floor; structured checks on top), and grades Colab work via a compact pasted submission token (loss graded by tolerance band, deterministic samples by hash). Answers are stored as **hashes only**. The client-side test code is always visible, so this is honest self-paced gating, not anti-cheat.

Browser progress is stored per-browser/per-device in IndexedDB with a downloadable JSON "progress token" as the durable backup. It is **not** a record of completion; a future server-side LMS must re-grade and must not trust these flags. See `book/_static/gating.js`.

## Running the tests

The full suite (reference components, grader self-tests, and integration paths) is run with pytest from the repo root:

```bash
python -m pytest          # runs tests/unit, tests/grader_self, tests/integration
```

The notebook headless verifier executes a solved copy of every browser lesson and runs the Colab lessons under a CPU config:

```bash
python notebooks/verify_notebooks.py
```

## Building the site locally

The static site builds with Jupyter Book 1.x (pinned; v2 is a different engine).

```bash
pip install "jupyter-book==1.0.4.post1" jupyterlite-sphinx jupyterlite-pyodide-kernel
jupyter-book build book/
# serve:
python -m http.server 8123 --directory book/_build/html
```

See `book/BUILD.md` for the pinning details and Python 3.14 caveats.

## Repository layout

```
reference/      The answer-key SLM build (autograd, tokenizer, bigram,
                MLP LM, attention, PyTorch GPT, training, NumPy demo forward).
grader/         Two-tier grader, hash-only answer checks, submission token
                parser, safe student-code execution, per-module checkpoints
                plus broken-variant library (the grader judging itself).
tests/          unit, grader_self (golden-passes-broken-fails), integration.
notebooks/      Generated lesson notebooks (S1 plus M0 to M10) and the
                build_notebooks.py generator and verify_notebooks.py.
book/           Jupyter Book sources (intro, per-lesson pages, gating.js).
docs/           docs/superpowers/specs/...content-core-design.md (the spec)
                docs/superpowers/plans/...content-core.md (the plan).
.github/        GitHub Pages deploy workflow.
```

## Status

Built end to end and verified. 65 of 65 pytest pass (reference build, grader self-tests, integration). 12 of 12 lesson notebooks pass headless verification. `jupyter-book build` produces a clean site with embedded in-browser notebooks. The site has not been pushed to a GitHub remote yet; the Colab badges still carry an `OWNER/REPO` placeholder until a remote exists.
