# Build-an-SLM Course (Content-Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the test-first reference SLM ("answer key"), the client-side auto-grader, the lesson notebooks, and the free static hosting for a beginner "build an SLM from scratch" course.

**Architecture:** Modules 0–6 run in-browser (JupyterLite/Pyodide, pure Python + NumPy); modules 7–9 run on free Colab (PyTorch); module 10 is static reading. The reference build is plain Python packages tested with pytest. The grader has two tiers (in-notebook asserts = guaranteed floor; otter-grader = structured) and grades Colab work via a pasted submission token. Spec: `docs/superpowers/specs/2026-06-22-slm-course-content-core-design.md`.

**Tech Stack:** Python 3.14, NumPy 2.4, PyTorch 2.10 (CPU), pytest 8.4; Jupyter Book + jupyterlite-sphinx + otter-grader; GitHub Pages via Actions; vanilla JS for client gating.

---

## File Structure

```
reference/
  __init__.py
  autograd/engine.py     # M3 micrograd-style scalar Value + backward (understanding-only)
  autograd/nn.py         # M3 Neuron/Layer/MLP on Value
  tokenizer/char_tokenizer.py   # M4 / S1 reference solution
  bigram/bigram.py       # M4 count-based bigram LM
  mlp_lm/model.py        # M5 embedding+MLP char-LM, NumPy array gradient path (NOT the scalar engine)
  attention/attention.py # M6 single- + multi-head causal self-attention (NumPy)
  gpt/model.py           # M7 PyTorch decoder-only GPT + two configs
  gpt/train.py           # M8/M9 training loop + submission-token emitter
  demo/numpy_forward.py  # M0 pure-NumPy transformer forward + sampling
  demo/export_weights.py # trains small model, exports fp16 .npz for M0 demo
grader/
  __init__.py
  core.py                # Result, CheckResult, run_checks()
  hash_check.py          # hash-only answer comparison
  submission_token.py    # parse + grade Colab paste-back token
  robustness.py          # safe execution of student callables (timeout/exception → clean FAIL)
  checks/                # per-module check suites (the checkpoints)
  broken_variants/       # deliberately-broken solutions the grader must reject
tests/
  conftest.py            # numerical_grad helper, fixtures
  unit/                  # one test module per reference component + grader unit
  grader_self/           # golden-passes / broken-variants-fail (crown jewel)
  integration/           # in-browser-loop sim (python-level) + Colab handback path
notebooks/               # shipped lesson .ipynb (M0–M10)
book/                    # Jupyter Book sources (_config.yml, _toc.yml, lite config)
.github/workflows/deploy.yml
pyproject.toml           # pytest pythonpath, project metadata
requirements.txt         # reference + grader runtime deps
requirements-lite.txt    # JupyterLite in-browser deps
```

---

## Task 0: Repo scaffolding

**Files:** Create `pyproject.toml`, `requirements.txt`, `reference/__init__.py`, `grader/__init__.py`, `tests/conftest.py`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "slm-course"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Create `requirements.txt`**

```
numpy>=2.0
torch>=2.4
pytest>=8.0
```

- [ ] **Step 3: Create empty `reference/__init__.py` and `grader/__init__.py`** (and `__init__.py` in each subpackage as created).

- [ ] **Step 4: Create `tests/conftest.py` with the shared numerical-gradient helper**

```python
import numpy as np

def numerical_grad(f, x, h=1e-5):
    """Central-difference gradient of scalar f at scalar x."""
    return (f(x + h) - f(x - h)) / (2 * h)

def rel_err(a, b, eps=1e-12):
    return abs(a - b) / max(abs(a), abs(b), eps)
```

- [ ] **Step 5: Commit** — `git add -A && git commit -m "chore: scaffold reference/grader packages + pytest config"`

---

## Task 1: M3 autograd engine (`reference/autograd/engine.py`)

**Files:** Create `reference/autograd/engine.py`, `reference/autograd/__init__.py`; Test `tests/unit/test_engine.py`.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np, random
from tests.conftest import rel_err
from reference.autograd.engine import Value

def _num_grad(f, x, h=1e-6):
    return (f(x+h) - f(x-h)) / (2*h)

def test_add_mul_grad_matches_numerical():
    a = Value(2.0); b = Value(-3.0)
    out = a*b + b
    out.backward()
    # dout/da = b = -3 ; dout/db = a + 1 = 3
    assert rel_err(a.grad, -3.0) < 1e-6
    assert rel_err(b.grad, 3.0) < 1e-6

def test_grad_accumulates_on_reuse():
    a = Value(3.0); b = a*a
    b.backward()
    assert rel_err(a.grad, 6.0) < 1e-6   # d(a^2)/da = 2a

def test_relu_and_random_graph_vs_numerical():
    random.seed(0)
    for _ in range(20):
        xv = random.uniform(-2, 2)
        f = lambda t: (Value(t).relu()*Value(2.0) + Value(1.0)).data
        x = Value(xv); y = (x.relu()*Value(2.0) + Value(1.0)); y.backward()
        assert rel_err(x.grad if xv>0 else x.grad+1e-12, _num_grad(f, xv)+1e-12) < 1e-3 or xv <= 0
```

- [ ] **Step 2: Run, expect FAIL** — `python -m pytest tests/unit/test_engine.py -v` → ImportError/fail.

- [ ] **Step 3: Implement `Value`** (micrograd engine.py shape: `data, grad, _backward, _prev, _op`; ops `__add__,__radd__,__mul__,__rmul__,__pow__,__neg__,__sub__,__rsub__,__truediv__,relu,tanh`; `backward()` builds topo order via DFS, sets `self.grad=1.0`, calls `_backward` in reverse).

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Add topo-order + torch cross-check tests**, implement until green:

```python
import torch
def test_cross_check_torch():
    a = Value(1.5); b = Value(-2.0)
    out = (a*b + a.relu()) ; out.backward()
    ta = torch.tensor(1.5, requires_grad=True); tb = torch.tensor(-2.0, requires_grad=True)
    tout = ta*tb + torch.relu(ta); tout.backward()
    assert rel_err(a.grad, ta.grad.item()) < 1e-6
    assert rel_err(b.grad, tb.grad.item()) < 1e-6
```

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(m3): scalar autograd Value with backward"`

---

## Task 2: M3 nn (`reference/autograd/nn.py`)

**Files:** Create `reference/autograd/nn.py`; Test `tests/unit/test_nn.py`.

- [ ] **Step 1: Failing test** — Neuron/Layer/MLP produce Value outputs; an MLP overfits 4 tiny examples (loss → near 0 in ≤200 steps, lr 0.05).
- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** Neuron (`w,b` Values, `relu`/linear), Layer (list of Neurons), MLP (list of Layers), `parameters()`.
- [ ] **Step 4: Run, PASS** (assert `final_loss < 0.05*initial_loss`).
- [ ] **Step 5: Commit** — `feat(m3): Neuron/Layer/MLP on Value with overfit test`

---

## Task 3: M4 tokenizer (`reference/tokenizer/char_tokenizer.py`) — S1 reference solution

**Files:** Create `reference/tokenizer/char_tokenizer.py`; Test `tests/unit/test_tokenizer.py`.

- [ ] **Step 1: Failing tests**

```python
from reference.tokenizer.char_tokenizer import CharTokenizer
S = "hello world\nthe quick brown fox"

def test_roundtrip_identity():
    tk = CharTokenizer(S)
    assert tk.decode(tk.encode(S)) == S
    assert tk.encode(tk.decode(list(range(tk.vocab_size)))) == list(range(tk.vocab_size))

def test_vocab_integrity():
    tk = CharTokenizer(S)
    assert tk.vocab_size == len(set(S))
    assert all(0 <= i < tk.vocab_size for i in tk.encode(S))
```

- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** sorted unique vocab, `stoi/itos`, `encode/decode`, `vocab_size`.
- [ ] **Step 4: Run, PASS.**
- [ ] **Step 5: Commit** — `feat(m4): char tokenizer with round-trip tests`

---

## Task 4: M4 bigram (`reference/bigram/bigram.py`)

**Files:** Create `reference/bigram/bigram.py`; Test `tests/unit/test_bigram.py`.

- [ ] **Step 1: Failing tests** — `P` rows sum to 1.0 (atol 1e-6); `generate(seed=0)` reproducible across two calls; all generated ids in `[0, vocab_size)`.
- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** count matrix `N` (+1 smoothing) → row-normalized `P`; `generate(n, seed)` using `np.random.default_rng(seed)`; consumes a `CharTokenizer`.
- [ ] **Step 4: Run, PASS.**
- [ ] **Step 5: Commit** — `feat(m4): count-based bigram LM reusing tokenizer`

---

## Task 5: M5 neural char-LM (`reference/mlp_lm/model.py`) — NumPy array gradient path

**Files:** Create `reference/mlp_lm/model.py`; Test `tests/unit/test_mlp_lm.py`.

> Uses **NumPy arrays with manual backward**, NOT the M3 scalar engine (perf). Document this in a module docstring.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from reference.mlp_lm.model import MLPLM, softmax
def test_forward_shape_and_softmax():
    m = MLPLM(vocab=10, n_embd=8, block=3, hidden=16, seed=0)
    X = np.zeros((4,3), dtype=int); logits = m.forward(X)
    assert logits.shape == (4,10)
    p = softmax(logits); assert np.allclose(p.sum(-1), 1.0, atol=1e-6)
def test_softmax_no_nan_large_logits():
    assert np.isfinite(softmax(np.array([[1000.,1001.,999.]]))).all()
def test_overfit_one_batch():
    m = MLPLM(vocab=5, n_embd=8, block=2, hidden=16, seed=0)
    X = np.array([[0,1],[2,3]]); Y = np.array([2,4])
    init = m.loss(X,Y)
    for _ in range(300): m.step(X,Y,lr=0.1)
    assert m.loss(X,Y) < 0.5*init
```

- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** embedding table `C`, flatten context, hidden `tanh`, output linear → logits; `softmax` with max-subtraction; cross-entropy `loss`; manual `backward` over arrays; `step(lr)`.
- [ ] **Step 4: Run, PASS.**
- [ ] **Step 5: Commit** — `feat(m5): numpy embedding+MLP char-LM with training loop`

---

## Task 6: M6 attention (`reference/attention/attention.py`)

**Files:** Create `reference/attention/attention.py`; Test `tests/unit/test_attention.py`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from reference.attention.attention import self_attention, multi_head
def test_shape_and_rowsum():
    np.random.seed(0); X = np.random.randn(6,8)
    out, w = self_attention(X, head_size=8, seed=0, return_weights=True)
    assert out.shape == (6,8); assert np.allclose(w.sum(-1), 1.0, atol=1e-6)
def test_causality():
    np.random.seed(0); X = np.random.randn(6,8)
    o1,_ = self_attention(X, head_size=8, seed=0, return_weights=True)
    X2 = X.copy(); X2[3:] += 5.0                      # perturb future tokens
    o2,_ = self_attention(X2, head_size=8, seed=0, return_weights=True)
    assert np.allclose(o1[:3], o2[:3])                # positions 0..2 unchanged
def test_multihead_dim():
    np.random.seed(0); X = np.random.randn(6,12)
    out = multi_head(X, n_head=3, seed=0); assert out.shape == (6,12)
```

- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** Q/K/V projections (seeded), scaled scores, causal mask (`-inf` upper triangle), softmax, `out=w@V`; `multi_head` = concat heads + output projection (`n_head*head_size==C`).
- [ ] **Step 4: Run, PASS.**
- [ ] **Step 5: Commit** — `feat(m6): numpy causal self-attention + multi-head`

---

## Task 7: M7 PyTorch GPT (`reference/gpt/model.py`)

**Files:** Create `reference/gpt/model.py`; Test `tests/unit/test_gpt.py`.

- [ ] **Step 1: Failing tests**

```python
import torch
from reference.gpt.model import GPT, GPTConfig, CONFIGS
def test_forward_shape():
    cfg = GPTConfig(**{**CONFIGS['cpu'].__dict__, 'vocab_size':65})
    m = GPT(cfg); x = torch.zeros(2, cfg.block_size, dtype=torch.long)
    logits,_ = m(x); assert logits.shape == (2, cfg.block_size, 65)
def test_param_count_near_estimate():
    m = GPT(CONFIGS['cpu']); n = sum(p.numel() for p in m.parameters())
    assert 0.5e6 < n < 1.2e6      # ~0.8M target band
def test_one_step_reduces_loss():
    torch.manual_seed(0); cfg = CONFIGS['cpu']; m = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    _,l0 = m(x,y); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(10):
        opt.zero_grad(); _,l = m(x,y); l.backward(); opt.step()
    _,l1 = m(x,y); assert l1.item() < l0.item()
def test_generate_in_vocab_and_blocksize():
    cfg = CONFIGS['cpu']; m = GPT(cfg)
    out = m.generate(torch.zeros(1,1,dtype=torch.long), max_new_tokens=20)
    assert out.shape[1] == 21 and out.max().item() < cfg.vocab_size
```

- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** nanoGPT-structure GPT (token+pos embeddings, `Block` = LN→MHA→residual + LN→MLP(4×,GELU)→residual, final LN, lm_head); `CONFIGS = {'cpu': 4L/4H/128d/block64/vocab65, 't4': 6L/6H/384d/block256/vocab65}`; `forward(idx, targets=None)` returns `(logits, loss)`; `generate()`.
- [ ] **Step 4: Run, PASS.**
- [ ] **Step 5: Commit** — `feat(m7): pytorch decoder-only GPT with cpu/t4 configs`

---

## Task 8: M8/M9 training + submission token (`reference/gpt/train.py`)

**Files:** Create `reference/gpt/train.py`; Test `tests/unit/test_train.py`.

- [ ] **Step 1: Failing test** — `train(config='cpu', max_iters=20, seed=0, data=tiny_str)` returns a result whose final loss < initial loss; `make_submission_token(result)` returns a string matching `^SLM-\w+ loss=\d+\.\d+ arch=[0-9a-f]+ shash=[0-9a-f]+$`.
- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** training loop (seed-pinned, device-detect with CPU fallback, periodic loss, greedy sample), `arch_signature(model)` = sha256 of sorted `(name, tuple(shape))` + param count, `sample_hash` = sha256 of fixed-prompt argmax sample, `make_submission_token()`.
- [ ] **Step 4: Run, PASS** (use a ~200-char data string + tiny iters so the test is fast).
- [ ] **Step 5: Commit** — `feat(m8): training loop + submission-token emitter`

---

## Task 9: M0 demo (`reference/demo/numpy_forward.py` + `export_weights.py`)

**Files:** Create `reference/demo/numpy_forward.py`, `reference/demo/export_weights.py`; Test `tests/unit/test_demo.py`.

- [ ] **Step 1: Failing test** — a pure-NumPy GPT forward matches the torch GPT forward on identical (small) weights within rtol 1e-3; greedy sampling is deterministic; exported `.npz` loads and round-trips weight shapes.
- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** `numpy_forward(weights, idx)` mirroring `reference/gpt/model.py` math in NumPy; `generate(weights, ...)`; `export_weights.py` trains a *small* model (n_layer=2, n_embd=64) a few hundred iters, saves fp16 `.npz` (< ~2 MB) to `book/_static/demo_weights.npz`.
- [ ] **Step 4: Run, PASS.**
- [ ] **Step 5: Commit** — `feat(m0): pure-numpy transformer forward + fp16 weight export`

---

## Task 10: Grader core + hash check (`grader/core.py`, `grader/hash_check.py`)

**Files:** Create `grader/core.py`, `grader/hash_check.py`; Test `tests/unit/test_grader_core.py`.

- [ ] **Step 1: Failing tests** — `CheckResult(passed, check_id, message)`; `run_checks(list_of_checks)` aggregates to `Result(passed, failed_checks, score)`; `hash_check(value, stored_hash)` True for correct, False for wrong; `make_hash(value)` = sha256 of a canonical repr; a test asserts the stored hash string is NOT the plaintext value.
- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, PASS.**
- [ ] **Step 5: Commit** — `feat(grader): core result types + hash-only answer check`

---

## Task 11: Submission-token grader (`grader/submission_token.py`)

**Files:** Create `grader/submission_token.py`; Test `tests/unit/test_submission_token.py`.

- [ ] **Step 1: Failing tests** — `parse_token(str)` → dict of fields; `grade(token, policy)` passes when `loss < band` and `shash`/`arch` match, fails when loss above band or hash mismatch; malformed/missing-field token → `TokenParseError`, not a crash.
- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** regex parse, tolerance-band loss check, exact hash checks, structured error.
- [ ] **Step 4: Run, PASS.**
- [ ] **Step 5: Commit** — `feat(grader): colab submission-token parser + tolerance-band grading`

---

## Task 12: Robust execution (`grader/robustness.py`)

**Files:** Create `grader/robustness.py`; Test `tests/unit/test_robustness.py`.

- [ ] **Step 1: Failing tests** — `safe_call(fn, *args, timeout_s=2)` returns `CheckResult(passed=False, category='error')` when `fn` raises; `category='shape'` on shape mismatch via a provided validator; `category='timeout'` when `fn` exceeds the timeout (run in a worker thread; on timeout return FAIL — document that a true CPU-bound loop can't be force-killed in-process, mirroring the browser Web-Worker-restart watchdog).
- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** thread-based runner with `join(timeout)`; map exceptions/None/shape to categories.
- [ ] **Step 4: Run, PASS** (timeout test uses `time.sleep`, not a true tight loop).
- [ ] **Step 5: Commit** — `feat(grader): safe student-callable execution with categorized failures`

---

## Task 13: Checkpoints + broken-variant library (the crown jewel)

**Files:** Create `grader/checks/<module>.py` (one check suite per module M2–M8), `grader/broken_variants/<module>_*.py`, `tests/grader_self/test_golden_and_broken.py`.

- [ ] **Step 1: Write the grader-self test FIRST**

```python
import importlib, pytest
from grader.core import run_checks
GOLDEN = {  # module -> (checks_module, golden_solution_module)
  'm3': ('grader.checks.m3', 'reference.autograd.engine'),
  'm4': ('grader.checks.m4', 'reference.tokenizer.char_tokenizer'),
  # ... m5, m6, m7
}
@pytest.mark.parametrize('mod', list(GOLDEN))
def test_golden_passes(mod):
    checks = importlib.import_module(GOLDEN[mod][0]).build_checks()
    assert run_checks(checks).passed

BROKEN = {  # broken_variant_module -> expected failing check id
  'grader.broken_variants.m3_no_grad_accum': 'grad_accumulation',
  'grader.broken_variants.m4_off_by_one_vocab': 'roundtrip',
  'grader.broken_variants.m6_no_causal_mask': 'causal',
  'grader.broken_variants.m5_no_softmax_maxsub': 'finite_loss',
}
@pytest.mark.parametrize('mod,expected', BROKEN.items())
def test_broken_fails_for_right_reason(mod, expected):
    checks = importlib.import_module(mod).build_checks()
    r = run_checks(checks)
    assert not r.passed and expected in r.failed_checks
```

- [ ] **Step 2: Run, FAIL.**
- [ ] **Step 3: Implement** per-module `build_checks()` (each check uses `safe_call` + asserts the §4.2 property, identified by a stable `check_id`); implement each broken variant as a copy of the reference with one defect.
- [ ] **Step 4: Run, PASS** — golden passes all; each broken fails on the expected `check_id`.
- [ ] **Step 5: Commit** — `feat(grader): per-module checkpoints + broken-variant self-tests`

---

## Task 14: Integration tests (Colab handback + Pyodide feasibility)

**Files:** Create `tests/integration/test_handback.py`, `tests/integration/test_pyodide_feasibility.py`.

- [ ] **Step 1: Write tests** — (a) run `train(config='cpu', tiny)` → `make_submission_token` → `submission_token.grade` ACCEPTS; mutate loss above band / swap shash / truncate → each REJECTED for the right reason; checkpoint `state_dict` save/load round-trips identical logits. (b) feasibility: import every `reference/*` module used in M0–M6 and assert none import `torch` (guards the browser boundary); assert M0–M6 reference functions run within a wall-clock budget on toy data.
- [ ] **Step 2: Run, FAIL → implement any gaps → PASS.**
- [ ] **Step 3: Commit** — `test: colab handback + pyodide-feasibility integration`

---

## Task 15: Lesson notebooks (`notebooks/`)

**Files:** Create `notebooks/m00_orientation.ipynb` … `notebooks/m10_whats_next.ipynb` (+ the S1 `notebooks/s1_tokenizer_warmup.ipynb`).

- [ ] **Step 1:** For each browser module (S1, M0–M6): a notebook with (a) markdown teaching prose, (b) a starter code cell with `# YOUR CODE HERE`, (c) a hidden reference cell, (d) Tier-1 `assert` checks importing the module's `build_checks`, (e) an otter `grader.check('qN')` cell. For Colab modules (M7–M9): an Open-in-Colab badge, device-detect cell, the training/build cells, and a final cell printing the submission token. M10: markdown only + "mark complete".
- [ ] **Step 2:** Author programmatically with `nbformat`; execute each browser notebook headless (`jupyter nbconvert --execute` if available, else `python` extraction) to verify the embedded reference code runs and asserts pass.
- [ ] **Step 3: Commit** — `feat(notebooks): lesson notebooks for M0–M10 + S1 skeleton`

---

## Task 16: Jupyter Book + JupyterLite (`book/`)

**Files:** Create `book/_config.yml`, `book/_toc.yml`, `book/intro.md`, `requirements-lite.txt`, and per-lesson markdown pages embedding the notebooks via `jupyterlite-sphinx`.

- [ ] **Step 1:** `pip install jupyter-book jupyterlite-sphinx jupyterlite-pyodide-kernel`. Configure `_config.yml` to enable jupyterlite-sphinx; `_toc.yml` lists modules in order; `requirements-lite.txt` pins numpy + otter-grader for the in-browser kernel.
- [ ] **Step 2:** `jupyter-book build book/` → verify `book/_build/html/` is produced and the `lite/` path embeds. (If install fails on py3.14, document the exact pinned versions and a `pipx`/venv fallback; do not block the Python build.)
- [ ] **Step 3: Commit** — `feat(book): jupyter-book + jupyterlite embedding config`

---

## Task 17: Client gating + GitHub Action

**Files:** Create `book/_static/gating.js` (IndexedDB gate flags, `navigator.storage.persist()`, download progress-token, Pyodide load-failure fallback, Web-Worker timeout watchdog), `.github/workflows/deploy.yml`.

- [ ] **Step 1:** `gating.js` — write/read per-lesson completion flags; export/import a JSON progress token (the durable backup, since IndexedDB is not authoritative); show a plain-language retry on Pyodide load failure; restart the kernel worker on a grading timeout.
- [ ] **Step 2:** `deploy.yml` — on push to `main`: setup Python, `jupyter-book build`, then `actions/deploy-pages`. (Do NOT push/enable Pages without the user; commit the workflow only.)
- [ ] **Step 3: Commit** — `feat(deploy): client gating + github pages workflow`

---

## Self-Review

- **Spec coverage:** §3 architecture → Tasks 7–9,15–17; §3.3 grading → Tasks 10–13; §3.4 persistence → Task 17; §4 reference build → Tasks 1–9; §5 stories → S1=Task 3+13+15, M0–M10 across Tasks 1–17; §6 TDD/tests → every task is test-first + Tasks 13–14; §7 fixes → Task 5 (numpy path), Task 9 (demo model), Task 8 (arch signature), Task 17 (persistence/watchdog), Task 13/12 (robustness). All covered.
- **Placeholder scan:** notebook/book/JS tasks describe concrete file contents and verification commands; no "TBD".
- **Type consistency:** `CheckResult`/`Result`/`run_checks`/`build_checks`/`safe_call`/`CONFIGS`/`GPTConfig`/`make_submission_token`/`parse_token` used consistently across Tasks 8–14.

**Verification floor (what must be green before "done"):** `python -m pytest` passes all of `tests/unit`, `tests/grader_self`, `tests/integration`. The browser/deploy artifacts (Tasks 16–17) are built and locally buildable but their in-browser/Pages execution is verified by the user post-push.
