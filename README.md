# Build an SLM From Scratch — Course Content-Core

A guided, scaffolded course that takes a complete beginner (basic Python, zero
ML) all the way to **building and training their own small GPT from scratch** —
ending with them sampling text from a model they built and trained themselves.

The pedagogy is *build-by-hand-first*: learners implement the neuron,
backpropagation, the tokenizer, and attention in pure Python/NumPy so they
understand the machinery, and only adopt PyTorch once they've earned that
understanding and need it to assemble and train the full transformer.

This repository is the **content-core**: the curriculum, the reference
"answer-key" SLM build, and the auto-grading/checkpoint engine. It is delivered
as free static hosting (Jupyter Book + JupyterLite on GitHub Pages) plus Google
Colab for the PyTorch work. A custom LMS is a separate, later sub-project — see
the deferred items in the spec.

## Architecture

The course straddles a deliberate pedagogical seam that also happens to be a
hard runtime boundary: **PyTorch cannot run in the browser**, and that wall
lands exactly where learners graduate from hand-built NumPy to PyTorch.

| Modules | Runtime | Mechanism |
|---|---|---|
| **S1 + M0–M6** (tokenizer → neuron → backprop → bigram → MLP LM → attention) | **In-browser** | JupyterLite / Pyodide + NumPy. Pure Python, no torch. Toy data only. $0, offline after first load. |
| **M7–M9** (build the GPT → train on Tiny Shakespeare → capstone) | **Google Colab (free T4)** | One-click "Open in Colab" from this public repo. Graded by a pasted submission token. |
| **M10** (what's next) | **Static reading** | Plain HTML/markdown + a "mark complete" cell. |

The reference build is plain Python packages tested with pytest. Grading has two
client-side tiers for the browser modules (in-notebook `assert` checks as the
guaranteed floor; structured checks on top) and grades Colab work via a compact
pasted submission token (loss graded by tolerance band, deterministic samples by
hash). Answers are stored as **hashes only** — the client-side test code is
always visible, so this is honest self-paced gating, not anti-cheat.

Browser progress is stored per-browser/per-device in IndexedDB with a
downloadable JSON "progress token" as the durable backup. It is **not** a
record of completion; a future server-side LMS must re-grade and must not trust
these flags. See `book/_static/gating.js`.

## Running the tests

The full suite (reference components, grader self-tests, and integration paths)
is run with pytest from the repo root:

```bash
python -m pytest          # runs tests/unit, tests/grader_self, tests/integration
```

As of this writing the suite is **65 tests, all green**. The "crown jewel" is
`tests/grader_self/`: it asserts the golden answer key passes every checkpoint
**and** that each deliberately-broken variant fails for the *right* reason (the
specific failing check id, not a crash).

Runtime: Python 3.11+ (developed on 3.14), NumPy 2.x, PyTorch 2.x (CPU is
fine), pytest 8.x. See `requirements.txt`.

## Repository layout

```
reference/                 # the "answer key" SLM build (test-first)
  autograd/engine.py        # M3 micrograd-style scalar Value + backward (understanding-only)
  autograd/nn.py            # M3 Neuron / Layer / MLP on Value
  tokenizer/char_tokenizer.py  # M4 / S1 char tokenizer
  bigram/bigram.py          # M4 count-based bigram LM
  mlp_lm/model.py           # M5 embedding + MLP char-LM (NumPy array gradient path, NOT the scalar engine)
  attention/attention.py    # M6 single- + multi-head causal self-attention (NumPy)
  gpt/model.py              # M7 PyTorch decoder-only GPT + cpu/t4 configs
  gpt/train.py              # M8/M9 training loop + submission-token emitter
  demo/numpy_forward.py     # M0 pure-NumPy transformer forward + sampling
  demo/export_weights.py    # trains a small model, exports fp16 .npz for the M0 demo
grader/
  core.py                   # Result / CheckResult / run_checks()
  hash_check.py             # hash-only answer comparison
  submission_token.py       # parse + grade the Colab paste-back token
  robustness.py             # safe execution of student callables (timeout/exception -> clean FAIL)
  checks/                   # per-module checkpoint suites (m3, m4, m5, m6, m7)
  broken_variants/          # mutation library the grader must reject
tests/
  unit/                     # one test module per reference component + grader units
  grader_self/              # golden-passes / broken-variants-fail (the crown jewel)
  integration/              # Colab handback path + Pyodide feasibility gate
notebooks/                  # shipped lesson notebooks (s1, m00–m10)
book/                       # Jupyter Book sources + JupyterLite embedding config
  _static/gating.js         # client-side progress gating (IndexedDB + progress token)
.github/workflows/deploy.yml  # build the book + deploy to GitHub Pages on push to main
```

## Docs

The design spec and the implementation plan are the source of truth for scope
and decisions:

- **Spec:** [`docs/superpowers/specs/2026-06-22-slm-course-content-core-design.md`](docs/superpowers/specs/2026-06-22-slm-course-content-core-design.md)
- **Plan:** [`docs/superpowers/plans/2026-06-22-slm-course-content-core.md`](docs/superpowers/plans/2026-06-22-slm-course-content-core.md)

## Deployment

`.github/workflows/deploy.yml` builds the Jupyter Book and publishes
`book/_build/html` to GitHub Pages on every push to `main`. It does **not** run
until the repo is pushed to GitHub, and Pages must be enabled first
(**Settings → Pages → Source: GitHub Actions**).
