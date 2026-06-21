# Design Spec — "Build an SLM From Scratch" Course (Content-Core)

- **Date:** 2026-06-22
- **Status:** Approved design, pre-implementation
- **Scope of this spec:** Sub-project 1 of the larger platform — the **content-core**: the curriculum, the reference "answer-key" SLM build, and the auto-grading/checkpoint engine. Hosted for free via Jupyter Book + JupyterLite. A custom LMS is a later, separate sub-project.

---

## 1. Context & goal

Build a guided, scaffolded learning path that takes a **complete beginner (knows basic Python, zero ML)** all the way to **building and training their own small GPT from scratch**, ending with them sampling text from a model they built and trained themselves.

The pedagogy is build-by-hand-first: learners implement the neuron, backpropagation, the tokenizer, and attention **in pure Python/NumPy** so they *understand* the machinery, and only adopt PyTorch once they've earned that understanding and need it to assemble and train the full transformer.

This is delivered eventually as a hosted platform, but it is sequenced **content-core first**: build and validate the curriculum + reference build + grader as a free, static, in-browser experience; build the custom LMS only once the pedagogy is proven.

## 2. The decisions this design rests on

- **Audience:** complete beginners, Python only, no ML.
- **Compute floor:** early lessons must run on a plain laptop CPU (zero cost, offline-capable in-browser); heavier training defaults to Google Colab's free T4 GPU.
- **Build method:** work in **vertical user stories** (each story cuts end-to-end through every part of the app), **not** horizontal layers. **TDD throughout.** Unit + integration tests in place.
- **Delivery vehicle for this sub-project:** free static hosting (Jupyter Book + JupyterLite on GitHub Pages); Colab for PyTorch work. Custom LMS deferred.

## 3. Architecture

### 3.1 Runtime boundary (verified, non-negotiable)

**PyTorch cannot run in the browser** (no Pyodide/WASM wheel; blocked by C/C++ extensions + cffi + no GPU path — confirmed hard wall). This lands precisely on the pure-Python→PyTorch pedagogical seam:

| Modules | Runtime | Mechanism |
|---|---|---|
| 0–6 (neuron → backprop → tokenizer → attention) | **In-browser** | JupyterLite/Pyodide + NumPy. Toy data only. $0, offline after first load. |
| 7–9 (build GPT → train → capstone) | **Free Google Colab T4** | One-click "Open in Colab" badge from a public GitHub repo. |
| 10 (what's next) | **Static reading** | Plain HTML. |

In-browser constraints the design respects: ~2 GB WASM heap ceiling; single Web Worker (no threads/multiprocessing); pure-Python ~5–10× slower than native (so keep loops NumPy-vectorized, small vocab/dims/step-counts → seconds-to-minutes per run); Pyodide+NumPy cold-load is multi-MB/several seconds (paid once as a Module 0 warm-up).

### 3.2 Hosting stack

Jupyter Book (static curriculum site) + `jupyterlite-sphinx` (embeds live JupyterLite/Pyodide notebooks into book pages) → published to **GitHub Pages via a GitHub Action**. Notebooks run entirely in-browser on Pyodide (WASM); **zero backend, $0 hosting.** Colab notebooks live in the **public** repo and are reached via the canonical Open-in-Colab URL: `https://colab.research.google.com/github/<user>/<repo>/blob/<branch>/<notebook>.ipynb`.

### 3.3 Grading

**Browser modules (0–6) — two-tier, fully client-side:**
- Tier 1 (guaranteed floor): plain in-notebook `assert` checks for instant feedback. Zero dependencies — always works under Pyodide.
- Tier 2: `otter-grader` v6.x in JupyterLite mode (`grader.check('qN')` runs in-browser). **Its install under the pinned Pyodide version must be validated** (see S1); if it fails, Tier 1 is the fallback.
- **Answers stored as hashes only.** Client-side test code is always visible to the learner, so raw answers are never shipped. This is honest self-paced gating — **not** anti-cheat.
- `nbgrader` is **dropped** (server-bound: requires instructor-side autograde, a DB, the formgrader server, and an exchange service).

**Colab modules (7–9) — paste-back token (default):**
- The training notebook's final cell prints one compact line, e.g. `SLM-M8 loss=1.6234 arch=ab12cd34 shash=ff90ee01`. The learner pastes it into the off-Colab browser grader.
- No file movement; immune to Colab's ephemeral-filesystem wipe and the >100 MB `files.download()` failure mode.
- **Loss graded by tolerance band, never bit-exact** (GPU/CPU/WASM float nondeterminism). Deterministic argmax/greedy samples graded by hash match. All seeds (torch, numpy, random) pinned in the answer key.
- Checkpoint upload (`files.download()` → upload to grader) is reserved as an **optional fallback at the capstone only** (model kept tiny, <~10 MB, so download stays reliable).

### 3.4 Persistence (MVP decision)

Browser-local IndexedDB (JupyterLite default) is **best-effort scratch state, not a system of record** — it is per-browser/per-device, not synced, and wiped on cache-clear/eviction. For the MVP:
- Gate flags written to IndexedDB; `navigator.storage.persist()` requested to reduce eviction.
- Every lesson offers a **download backup** (completed notebook + a copy-paste/downloadable JSON "progress token").
- **All browser-side gates are honor-system with zero integrity guarantee.** The eventual LMS must **re-grade server-side** and must NOT import browser gate flags as trusted records.
- A durable, cross-device, verified grade-of-record is **explicitly deferred to the custom LMS** (see §8).

## 4. The reference "answer-key" build

### 4.1 Two pinned configs (grounded in nanoGPT)

| Config | Shape | Params | Trains to | Time |
|---|---|---|---|---|
| **Laptop-CPU floor** | 4 layer / 4 head / 128 dim / block 64, batch 12, lr 1e-3 | ~0.8M | val loss ~1.88 (legible babble) | ~3 min Apple-silicon; **up to ~30 min on a free Colab CPU VM or weak laptop** |
| **Colab-T4 target** | 6 layer / 6 head / 384 dim / block 256, batch 64, dropout 0.2, lr 1e-3, 5000 iters | ~10.6M | val loss ~1.47 (clearly legible Shakespeare) | ~10–20 min on a free T4 |

The CPU fallback's `max_iters` is reduced so it finishes in **<10 min even on a slow machine**; it is positioned as a degraded "it still works" path, not the expected experience. Code detects device (`torch.cuda.is_available()`) and falls back to CPU, since a free GPU is not guaranteed.

Corpus: Tiny Shakespeare (~1.1 MB, 65 unique chars) for Modules 8; learner-chosen corpus for the Module 9 capstone.

### 4.2 Hand-built components & their testable properties

| Module | Component | Key testable property |
|---|---|---|
| M2 | Single neuron + one hand-derived gradient step (NumPy) | Forward correct to 1e-6; one step lowers loss; analytic grad matches central-difference to rel-tol 1e-4 |
| M3 | micrograd-style scalar `Value` autograd (`+ * ** neg / relu`, topo-sort `backward`) + Neuron/Layer/MLP | Analytic grads match central-difference (rel-tol 1e-4) on a random graph; grad **accumulates** on reuse; topo order visits each node once, never before children; cross-check vs torch.autograd |
| M4 | Char tokenizer + count-based bigram | `decode(encode(s))==s` and `encode(decode(ids))==ids` over full vocab; `len(vocab)==unique chars`; P rows sum to 1 (1e-6); seeded sampling reproducible & in-vocab |
| M5 | Embedding + MLP next-char LM + **NumPy array-based** training loop | Forward shape `(B,V)`; softmax rows sum to 1 (max-subtraction → no NaN); finite non-negative loss; **overfit-one-batch** → loss→~0; final loss < 0.5×initial on fixed seed |
| M6 | Single- + multi-head causal self-attention (NumPy) | Output shape == input `(T,C)`; attn-weight rows sum to 1; **causal mask**: output at position t unchanged when future tokens altered; multi-head dim == `n_head*head_size == C` |
| M7 | PyTorch decoder-only GPT (nanoGPT structure) | Forward `(B,T)→(B,T,V)`; param count within 5% of analytic estimate; one optimizer step lowers loss on a fixed batch; `generate()` returns in-vocab ids respecting `block_size` |

> **Critical fix (applied):** M5's training loop uses a **separate NumPy array-based gradient path**, NOT the M3 scalar `Value` engine (which would be unusably slow at MLP scale). M3's scalar engine is **understanding-only** — used to train a tiny demo MLP on a handful of examples. "Gradient *concepts*" carry M3→M5; the *implementation* differs, and the spec/regression tests say so explicitly.

### 4.3 Module 0 demo model (fix applied)

Module 0 ("see the final model generate text") ships the **small ~0.8M-param char model** (NOT the 10.6M target), with a dedicated **`reference/demo/numpy_forward.py`** — a pure-NumPy transformer forward + greedy/temperature sampling pass (no torch). Weights stored **fp16** in a small `.npz` (~1–2 MB) to honor the "well under 10 MB / runs in a few seconds in-browser" requirement.

## 5. Vertical-slice user stories

Every story touches all six parts: **content rendering → execution → reference solution → grader/checkpoint → progress persistence → hosting.** **S1 is the walking skeleton** — the thinnest end-to-end slice that forces every part to exist and proves the pipeline before any breadth is added.

| ID | Lesson | Runtime | Depends on |
|---|---|---|---|
| **S1** | **Tokenizer "warm-up" (WALKING SKELETON)** — standalone, decoupled from the M4 gate | browser | — |
| S2 | Module 0 — pre-trained tiny model generates text (Pyodide warm-up) | browser | S1 |
| S3 | Module 1 — just-enough NumPy | browser | S2 |
| S4 | Module 2 — a neuron + one hand-derived gradient step | browser | S3 |
| S5 | Module 3 — autograd engine + MLP | browser | S4 |
| S6 | Module 4 — bigram that generates text (reuses S1 tokenizer) | browser | S1, S5 |
| S7 | Module 5 — neural char-LM + NumPy training loop | browser | S6 |
| S8 | Module 6 — attention from scratch | browser | S7 |
| S9 | Module 7 — build the PyTorch GPT (first Colab handoff; proves paste-back) | colab | S8 |
| S10 | Module 8 — train on Tiny Shakespeare (tolerance-band + hash grading) | colab | S9 |
| S11 | Module 9 — capstone on learner-chosen corpus (+ optional checkpoint upload) | colab | S10 |
| S12 | Module 10 — what's next + course-completion gate | static | S11 |

### 5.1 S1 walking-skeleton acceptance criteria (the de-risking slice)

The skeleton is the Module-4 char tokenizer (crisp deterministic round-trip property, genuine downstream reuse). It must demonstrate, on one lesson:

1. The published GitHub Pages URL loads a lesson page with prose, a runnable cell, and a "Check my work" affordance — **no backend running**.
2. Pyodide cold-loads NumPy; the learner edits, runs, and sees output **entirely in-browser** (verified offline after first load).
3. The grader returns **PASS** for a correct `encode/decode` and **FAIL** for a deliberately broken one, asserting `decode(encode(s))==s` / `encode(decode(ids))==ids` over the full vocab.
4. The grader stores only a **hash** of expected values; inspecting the shipped test file does not reveal the solution.
5. **otter-grader v6.x installs via `piplite` under the pinned Pyodide version and `grader.check` runs in-browser**; if install fails, plain `assert` grading is the working fallback. *(Explicitly validates the single most fragile dependency.)*
6. On PASS, a completion flag writes to IndexedDB; reload shows the lesson complete without re-running.
7. A **"download my work"** affordance produces an `.ipynb` + a JSON progress token (backup independent of IndexedDB).
8. A documented note states client progress is per-browser/per-device and **not authoritative** (server-of-record deferred to LMS).
9. A push to `main` triggers the GitHub Action that rebuilds and redeploys the static site with embedded JupyterLite.

*(Full acceptance criteria for S2–S12 are carried in the user-story backlog and will be expanded per-story in the implementation plan.)*

## 6. TDD strategy & test taxonomy

### 6.1 The per-lesson authoring loop (test first, content last)

1. **RED** — write the checkpoint/grader test *first* (the learning objective encoded as the testable property from §4.2), as both an in-notebook `assert` and an otter `grader.check`. Add matching **broken variant(s)** to the mutation library. Confirm the checkpoint fails (no solution yet) **and** the broken variants fail — proving the test discriminates.
2. **GREEN (solution)** — write the minimal answer-key code that passes; run the grader's self-tests (golden passes, broken variants fail for the right reason).
3. **GREEN (runtime gate)** — run in the *actual* ship runtime: Pyodide for 0–6 (imports pure-Python/NumPy only, finishes in budget, under heap ceiling); Colab **CPU config with pinned seeds** for 7–9 (reproducible in CI without a GPU; token validates off-Colab).
4. **REFACTOR** — clean the solution for beginner readability without breaking checkpoints; tighten tests (pinned seeds, tolerance bands, softmax max-subtraction, robust loss-decrease properties). Expand the broken-variant library if refactoring revealed a bypass.
5. **CONTENT LAST** — author teaching prose/hints around the verified solution. Run the full in-browser lesson loop as definition-of-done.
6. **REGRESSION** — commit checkpoint test + golden solution + broken-variant library as one unit; CI re-runs everything on every change so shared-code edits can't silently regress an earlier gate.

### 6.2 Unit tests

- **Reference components:** the properties in §4.2 (gradient checks, round-trips, shapes, causal-mask invariance, overfit-one-batch, param-count).
- **Grader self-tests (crown jewel):** (a) the golden answer-key **passes** every checkpoint; (b) a curated **broken-variant library** each **fails for the right reason** (the specific failing check id, not a crash) — e.g. `autograd_missing_grad_accumulation`→fails `grad_accumulation`, `attention_no_causal_mask`→fails `causal`, `tokenizer_off_by_one_vocab`→fails `roundtrip`.
- **Grader robustness:** malformed student input (wrong type/shape, NaN/inf, exceptions, infinite loops) yields a clean FAIL with a useful message, never an unhandled crash. *(See §7 timeout mechanism.)*
- **Hash-check:** test file source never contains the plaintext answer; `hash_check(correct)==True`, `hash_check(wrong)==False`.
- **Submission-token parser (M8/M9):** parses the one-line token; loss graded by **band**; sample-hash by match; architecture signature structurally; malformed/tampered tokens → structured parse error, not crash.

### 6.3 Integration tests

- **Full in-browser lesson loop:** boot pinned Pyodide/JupyterLite → inject golden solution → execute → both grading tiers run → all pass → inject broken variant → correct failure → gate written to IndexedDB **and exported** → simulate cache wipe → progress restored from the exported token (not lost). *(See §7 fix — the export is the downloadable progress token, not a server.)*
- **Per-notebook Pyodide feasibility gate:** every Module 0–6 notebook imports only pure-Python/Pyodide-prebuilt wheels (no torch), runs to completion in budget, under the heap ceiling, with no threading/multiprocessing reachable. Any notebook importing torch fails the gate.
- **Colab handback validation:** run the answer-key training notebook under the **CPU config with pinned seeds** (CI, no GPU) → capture token → off-Colab grader accepts (loss in band, sample-hash match, arch signature valid) → mutated tokens each rejected for the right reason → capstone checkpoint round-trips to identical logits; corrupt/oversized uploads fail gracefully.
- **Cross-runtime determinism contract:** hand-built NumPy components (M3–M6) vs their PyTorch counterparts (M7) agree within tolerance band (~1e-4) on fixed inputs — explicitly **not** bit-exact, so it stays green across the JupyterLite→Colab seam.

## 7. Resolved risks (corrections folded in from adversarial review)

1. **M3→M5 autograd:** M5 uses a separate NumPy array-based gradient path; M3 scalar engine is understanding-only. (§4.2)
2. **Module 0 demo model:** defined as the small ~0.8M model + `reference/demo/numpy_forward.py` + fp16 weights. (§4.3)
3. **S9 verification:** architecture signature `(param-name, shape)` + param count + fixed-seed loss-after-one-step (with torch init seed pinned), replacing the meaningless random-weight hash.
4. **Persistence:** MVP = best-effort local + downloadable progress token; integration test restores from that token, not a server; durable grade-of-record deferred to LMS. (§3.4)
5. **In-browser timeout:** student code runs under a **main-thread watchdog that terminates and restarts the Web Worker** on timeout (Pyodide is single-worker — a CPU-bound loop can't be cooperatively interrupted). The robustness test asserts **page recovery + timeout FAIL**, not "kernel stays responsive without restart."
6. **otter+Pyodide install** validated as an S1 acceptance criterion, with plain `assert` grading as the guaranteed fallback floor.

Minor items also addressed: skeleton renumbered as a standalone tokenizer warm-up (decouples the Module-4 split); Module 0 specifies a Pyodide load-failure fallback (plain-language retry + minimum-browser note + static fallback of the demo output) so lesson 0 never dead-ends a beginner.

## 8. Explicitly deferred to the custom LMS (the documented breakpoint)

The static stack **cannot** do four things — these define the migration trigger, not a surprise:
1. Accounts / identity.
2. Cross-device, verified progress (a durable grade-of-record).
3. Deadlines / cohort management.
4. Real anti-cheat (server-side hidden tests; client-side tests are always visible).

The LMS must **re-grade server-side** and must not trust any browser gate flag.

## 9. Proposed repo layout (greenfield — created test-first)

```
D:\FoundationModel\
  reference\
    autograd\engine.py        # M3 scalar Value + MLP (understanding-only)
    tokenizer\char_tokenizer.py
    bigram\bigram.py          # M4
    mlp_lm\model.py           # M5 (NumPy array gradient path)
    attention\attention.py    # M6
    gpt\model.py              # M7 PyTorch (Colab/local only)
    demo\numpy_forward.py     # M0 pure-NumPy forward + sampling
  grader\
    core.py
    hash_check.py
    submission_token.py
    robustness.py
    broken_variants\          # mutation library
  tests\
    unit\  integration\  grader_self\
  notebooks\                  # shipped lesson notebooks
  book\                       # Jupyter Book sources + jupyterlite-sphinx config
  .github\workflows\          # GitHub Pages deploy action
```

## 10. Definition of done (this sub-project)

- All 12 stories shipped: each lesson renders, executes in its correct runtime, grades correctly (golden passes / broken fails), and records progress with a download backup.
- The full test suite (unit + integration + grader-self) is green in CI, including the Pyodide feasibility gate and the Colab CPU-config handback path.
- A complete beginner can go from Module 0 to a self-trained GPT producing legible Shakespeare-style text, entirely on free tooling.
- The site auto-deploys to GitHub Pages on push to `main`.

## 11. Sources (verified during planning)

- Pyodide packages / WASM constraints; `micropip`; JupyterLite storage — pyodide.org, jupyterlite.readthedocs.io
- Colab FAQ (GPU/session/idle/quota); Open-in-Colab URL scheme — research.google.com/colaboratory, github.com/googlecolab/open_in_colab
- otter-grader JupyterLite support (v4→v6) & client-side test visibility — github.com/ucbds-infra/otter-grader; nbgrader server requirements — nbgrader.readthedocs.io
- nanoGPT configs (`train_shakespeare_char`, CPU recipe) & micrograd `engine.py` — github.com/karpathy/nanoGPT, github.com/karpathy/micrograd
