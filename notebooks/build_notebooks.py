"""Data-driven generator for all 12 lesson notebooks.

Run from the repo root:  ``python notebooks/build_notebooks.py``

Design notes
------------
* The HIDDEN reference-solution cell of every browser lesson is built by reading
  the matching ``reference/*.py`` source at generation time, so the reference
  implementation stays the single source of truth (no copy-paste drift).
* Browser lessons (s1, m00-m06) run in-browser: pure Python + NumPy, NO torch
  import at module load. Their check cells use the grader's parameterized
  checks (m3/m4/m5/m6) or self-contained inline asserts (m01/m02/m04, m00).
* Colab lessons (m07-m09) import from ``reference.gpt`` and print a submission
  token; they carry an "Open in Colab" badge with literal OWNER/REPO
  placeholders.
* m10 is static reading + a trivial "mark complete" cell.
"""

from __future__ import annotations

import os
import sys

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(_REPO_ROOT, "notebooks")

# A tiny sys.path bootstrap prepended to browser check / setup cells so that
# `import grader` and `import reference` resolve when the notebook is executed
# with the working directory at the notebook folder (Jupyter's default).
PATH_BOOTSTRAP = (
    "import sys, os\n"
    "# Make the repo root importable so `grader` / `reference` resolve.\n"
    "_root = os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
    "if _root not in sys.path:\n"
    "    sys.path.insert(0, _root)\n"
)

COURSE_GITHUB_OWNER = "Vritti-Dev"
COURSE_GITHUB_REPO = "foundation-model"
COURSE_REPO_URL = f"https://github.com/{COURSE_GITHUB_OWNER}/{COURSE_GITHUB_REPO}.git"
COLAB_URL = (
    f"https://colab.research.google.com/github/{COURSE_GITHUB_OWNER}/"
    f"{COURSE_GITHUB_REPO}/blob/main/notebooks/"
)

# Bootstrap cell prepended to every Colab notebook so `from reference.gpt...`
# resolves. On Colab the cell clones the course repo and cds into it; locally
# (and during the headless verifier) the `import google.colab` fails so this is
# a no-op and the existing cwd-on-sys.path covers imports.
COLAB_BOOTSTRAP = (
    "# Bootstrap: make the course code importable.\n"
    "# On Colab this clones the repo and cds into it; locally it is a no-op.\n"
    "import os, sys\n"
    "try:\n"
    "    import google.colab  # noqa: F401\n"
    f"    if not os.path.exists('{COURSE_GITHUB_REPO}'):\n"
    f"        !git clone -q {COURSE_REPO_URL}\n"
    f"    %cd {COURSE_GITHUB_REPO}\n"
    "    print('repo ready at', os.getcwd())\n"
    "except ImportError:\n"
    "    pass  # running locally; assume cwd is the repo root\n"
)


def read_ref(rel_path):
    """Read a reference source file and return its text (the source of truth)."""
    with open(os.path.join(_REPO_ROOT, rel_path), encoding="utf-8") as f:
        return f.read()


def hidden_solution_cell(source_text, intro_comment):
    """A code cell whose source is hidden by default (jupyter.source_hidden)."""
    cell = new_code_cell(source=intro_comment + source_text)
    cell.metadata["jupyter"] = {"source_hidden": True}
    return cell


def colab_badge(filename):
    return (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({COLAB_URL}{filename})"
    )


# ---------------------------------------------------------------------------
# Browser lessons
# ---------------------------------------------------------------------------


def nb_s1_tokenizer():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Warm-up: Build a Tokenizer (Walking Skeleton)\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "A neural network can't read text. It only does arithmetic on numbers.\n"
        "So the very first thing any language model needs is a way to turn a\n"
        "string like `\"hello\"` into a list of integers, and back again.\n"
        "That translator is called a **tokenizer**.\n"
        "\n"
        "In this warm-up you'll write the simplest possible tokenizer: one that\n"
        "treats every single **character** as one token. The letter `h` becomes\n"
        "some number, `e` becomes another, and so on. No clever splitting into\n"
        "words or sub-words yet -- just characters.\n"
        "\n"
        "Here is the plan, which is exactly what your `CharTokenizer` will do:\n"
        "\n"
        "1. Look at all the text you're given and collect every **unique**\n"
        "   character. Sort them so the order is always the same. That sorted\n"
        "   list is your **vocabulary**.\n"
        "2. Build two lookup tables: `stoi` (\"string to integer\") maps each\n"
        "   character to its position in the vocabulary, and `itos` (\"integer\n"
        "   to string\") maps back.\n"
        "3. `encode` walks through a string and replaces each character with its\n"
        "   id. `decode` does the reverse.\n"
        "\n"
        "The reason we sort the vocabulary is **determinism**: given the same\n"
        "text, you always get the same ids. That matters later when we want a\n"
        "model trained today to behave the same way tomorrow.\n"
    ))
    nb.cells.append(new_code_cell(
        "class CharTokenizer:\n"
        "    \"\"\"Map characters in `text` to integer ids and back.\"\"\"\n"
        "\n"
        "    def __init__(self, text):\n"
        "        # TODO: build self.vocab as the SORTED set of characters in text.\n"
        "        # TODO: build self.stoi (char -> id) and self.itos (id -> char).\n"
        "        # TODO: set self.vocab_size to the number of unique characters.\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def encode(self, s):\n"
        "        # TODO: return a list of integer ids, one per character in s.\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def decode(self, ids):\n"
        "        # TODO: return the string for a list of integer ids.\n"
        "        raise NotImplementedError\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Reveal the reference solution\n"
        "\n"
        "Try it yourself first. When you're ready, run the hidden cell below to\n"
        "load the reference `CharTokenizer` (click to expand it if you want to\n"
        "read the code)."
    ))
    nb.cells.append(hidden_solution_cell(
        read_ref("reference/tokenizer/char_tokenizer.py"),
        "# Reference solution (single source of truth: reference/tokenizer/char_tokenizer.py)\n\n",
    ))
    nb.cells.append(new_markdown_cell(
        "## Check your work\n"
        "\n"
        "The grader builds a tokenizer from a fixed test string and checks that\n"
        "encoding then decoding round-trips perfectly, and that the vocabulary is\n"
        "consistent. A PASS means your tokenizer behaves exactly like the one a\n"
        "real model would rely on."
    ))
    nb.cells.append(new_code_cell(
        PATH_BOOTSTRAP +
        "from grader.checks.m4 import check_tokenizer\n"
        "from grader.core import run_checks\n"
        "\n"
        "r = run_checks(check_tokenizer(CharTokenizer))\n"
        "print('PASS' if r.passed else 'FAIL', f'score={r.score:.2f}')\n"
        "if not r.passed:\n"
        "    print('failed checks:', r.failed_checks)\n"
    ))
    return nb


def nb_m00_orientation():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 0 - See Your Goal: a Tiny Model Writes\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "Before you build anything, let's see where this is all heading.\n"
        "\n"
        "We've already trained a tiny GPT for you on a small slice of\n"
        "Shakespeare and saved its weights to a compact file. In this module you\n"
        "load those weights and watch the model **write text, one character at a\n"
        "time**, right here in your browser -- no GPU, no PyTorch, just NumPy.\n"
        "\n"
        "How is that possible? A trained model is just a big pile of numbers (the\n"
        "weights) plus a fixed recipe for combining them (the forward pass). The\n"
        "recipe is the same arithmetic whether you run it in PyTorch on a GPU or\n"
        "in plain NumPy on a laptop. We saved the weights as a `.npz` file and\n"
        "re-implemented the exact forward pass in NumPy in\n"
        "`reference/demo/numpy_forward.py`. You can read it side by side with the\n"
        "PyTorch model later and see they compute the same thing.\n"
        "\n"
        "**Generation** works like autocomplete: give the model a starting\n"
        "character, it predicts the most likely next character, you append it,\n"
        "and repeat. Because we always take the single most likely character\n"
        "(\"greedy\" decoding), the output is fully deterministic -- the same\n"
        "start always produces the same continuation.\n"
        "\n"
        "Everything in this course builds up to understanding every line of that\n"
        "forward pass. For now, just run it and enjoy the magic."
    ))
    nb.cells.append(new_code_cell(
        "# TODO: load the demo weights + metadata and generate text.\n"
        "# Steps (fill in where marked):\n"
        "#   1. load book/_static/demo_meta.json (gives stoi / itos / config)\n"
        "#   2. load book/_static/demo_weights.npz (the trained fp16 weights)\n"
        "#   3. build a GPTConfig from the saved config dict\n"
        "#   4. call reference.demo.numpy_forward.generate from a start character\n"
        "import json, os\n"
        "import numpy as np\n"
        "\n"
        "STATIC = os.path.join('..', 'book', '_static')\n"
        "\n"
        "# TODO: set `text` to the generated string starting from the letter 'F'.\n"
        "text = None\n"
        "raise NotImplementedError\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Reveal the reference solution\n"
        "\n"
        "Run the hidden cell to load the weights and generate. It defines `text`\n"
        "(the generated string) which the check below inspects."
    ))
    nb.cells.append(hidden_solution_cell(
        "",
        "# Reference solution: load demo weights + meta, run the NumPy forward pass.\n"
        "import json, os\n"
        "import numpy as np\n"
        "from reference.gpt.model import GPTConfig\n"
        "from reference.demo.numpy_forward import generate\n"
        "\n"
        "STATIC = os.path.join('..', 'book', '_static')\n"
        "with open(os.path.join(STATIC, 'demo_meta.json'), encoding='utf-8') as f:\n"
        "    meta = json.load(f)\n"
        "stoi = meta['stoi']\n"
        "itos = {int(k): v for k, v in meta['itos'].items()}\n"
        "c = meta['config']\n"
        "cfg = GPTConfig(n_layer=c['n_layer'], n_head=c['n_head'], n_embd=c['n_embd'],\n"
        "                block_size=c['block_size'], vocab_size=c['vocab_size'],\n"
        "                dropout=c['dropout'])\n"
        "\n"
        "loaded = np.load(os.path.join(STATIC, 'demo_weights.npz'))\n"
        "weights = {k: loaded[k].astype(np.float32) for k in loaded.files}\n"
        "\n"
        "start = np.array([[stoi['F']]])\n"
        "out = generate(weights, start, cfg, max_new_tokens=200)\n"
        "text = ''.join(itos[i] for i in out[0].tolist())\n"
        "print(text)\n",
    ))
    nb.cells.append(new_markdown_cell(
        "## Check your work\n"
        "\n"
        "The check confirms the model actually produced text and that generation\n"
        "is deterministic: running the same greedy generation from the same start\n"
        "character twice gives the exact same string."
    ))
    nb.cells.append(new_code_cell(
        "# Self-contained check: output is non-empty and deterministic.\n"
        "import json, os\n"
        "import numpy as np\n"
        "from reference.gpt.model import GPTConfig\n"
        "from reference.demo.numpy_forward import generate\n"
        "\n"
        "with open(os.path.join(STATIC, 'demo_meta.json'), encoding='utf-8') as f:\n"
        "    _meta = json.load(f)\n"
        "_stoi = _meta['stoi']\n"
        "_itos = {int(k): v for k, v in _meta['itos'].items()}\n"
        "_c = _meta['config']\n"
        "_cfg = GPTConfig(n_layer=_c['n_layer'], n_head=_c['n_head'], n_embd=_c['n_embd'],\n"
        "                 block_size=_c['block_size'], vocab_size=_c['vocab_size'],\n"
        "                 dropout=_c['dropout'])\n"
        "_loaded = np.load(os.path.join(STATIC, 'demo_weights.npz'))\n"
        "_w = {k: _loaded[k].astype(np.float32) for k in _loaded.files}\n"
        "\n"
        "def _gen():\n"
        "    s = np.array([[_stoi['F']]])\n"
        "    o = generate(_w, s, _cfg, max_new_tokens=40)\n"
        "    return ''.join(_itos[i] for i in o[0].tolist())\n"
        "\n"
        "_a, _b = _gen(), _gen()\n"
        "assert text is not None and len(text) > 0, 'no text generated'\n"
        "assert _a == _b, 'generation is not deterministic'\n"
        "assert all(ch in _itos.values() for ch in _a), 'generated chars out of vocab'\n"
        "print('PASS', f'score=1.00  (generated {len(text)} chars, deterministic)')\n"
    ))
    return nb


def nb_m01_numpy():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 1 - Just Enough NumPy\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "Neural networks are, underneath all the jargon, big stacks of array\n"
        "arithmetic. The library everyone uses for that in Python is **NumPy**.\n"
        "You don't need to be a NumPy wizard for this course, but you do need a\n"
        "handful of moves cold. This module drills exactly those.\n"
        "\n"
        "You'll practise four things:\n"
        "\n"
        "1. **Creating arrays** -- turning a plain Python list into an `np.array`,\n"
        "   and knowing its `.shape`.\n"
        "2. **Indexing** -- pulling out a row, a column, or a single element.\n"
        "3. **Broadcasting** -- NumPy's rule for combining arrays of different\n"
        "   shapes (e.g. adding a vector to every row of a matrix) without writing\n"
        "   a loop.\n"
        "4. **The dot product** -- multiplying matrices with `@`, which is the\n"
        "   single most important operation in a neural network.\n"
        "\n"
        "Fill in each TODO so the values match what's described. The asserts in\n"
        "the check cell will tell you immediately if something's off."
    ))
    nb.cells.append(new_code_cell(
        "import numpy as np\n"
        "\n"
        "# 1. Creating arrays.\n"
        "# TODO: make a 2x3 array with rows [1, 2, 3] and [4, 5, 6].\n"
        "A = None\n"
        "\n"
        "# 2. Indexing.\n"
        "# TODO: pull out the second row of A (the one starting with 4).\n"
        "row1 = None\n"
        "# TODO: pull out the element in row 0, column 2 (the value 3).\n"
        "elem = None\n"
        "\n"
        "# 3. Broadcasting.\n"
        "# TODO: add the vector [10, 20, 30] to EVERY row of A (no loop).\n"
        "bcast = None\n"
        "\n"
        "# 4. Dot product.\n"
        "# TODO: matrix-multiply A (2x3) by B (3x2) using @.\n"
        "B = np.array([[1, 0], [0, 1], [1, 1]])\n"
        "C = None\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Reveal the reference solution\n"
        "\n"
        "Run the hidden cell for one correct way to fill in each blank."
    ))
    nb.cells.append(hidden_solution_cell(
        "",
        "# Reference solution.\n"
        "import numpy as np\n"
        "A = np.array([[1, 2, 3], [4, 5, 6]])\n"
        "row1 = A[1]\n"
        "elem = A[0, 2]\n"
        "bcast = A + np.array([10, 20, 30])\n"
        "B = np.array([[1, 0], [0, 1], [1, 1]])\n"
        "C = A @ B\n",
    ))
    nb.cells.append(new_markdown_cell(
        "## Check your work\n"
        "\n"
        "These asserts confirm each array has the right shape and values."
    ))
    nb.cells.append(new_code_cell(
        "import numpy as np\n"
        "assert A.shape == (2, 3), 'A should be 2x3'\n"
        "assert np.array_equal(A, [[1, 2, 3], [4, 5, 6]]), 'A values wrong'\n"
        "assert np.array_equal(row1, [4, 5, 6]), 'row1 wrong'\n"
        "assert int(elem) == 3, 'elem wrong'\n"
        "assert np.array_equal(bcast, [[11, 22, 33], [14, 25, 36]]), 'broadcasting wrong'\n"
        "assert C.shape == (2, 2), 'C should be 2x2'\n"
        "assert np.array_equal(C, [[4, 5], [10, 11]]), 'dot product wrong'\n"
        "print('PASS', 'score=1.00')\n"
    ))
    return nb


def nb_m02_neuron():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 2 - A Neuron and a Gradient Step\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "A neural network is built out of one tiny repeated unit: the **neuron**.\n"
        "A neuron takes some inputs, multiplies each by a **weight**, adds them\n"
        "up, adds a **bias**, and squashes the result through a nonlinearity.\n"
        "That's it. Stack thousands of these and you get a language model.\n"
        "\n"
        "In this module you'll do two things by hand:\n"
        "\n"
        "1. **Forward pass.** Compute a single neuron's output:\n"
        "   `y = tanh(w . x + b)`. The `tanh` keeps the output between -1 and 1.\n"
        "2. **One gradient step.** We'll define a simple loss -- how far the\n"
        "   neuron's output is from a target -- and nudge the weights in the\n"
        "   direction that *reduces* that loss. This is the heartbeat of all\n"
        "   training: measure error, compute the slope (gradient) of the error\n"
        "   with respect to each weight, take a small step downhill.\n"
        "\n"
        "We'll derive the gradient by hand here so there's no magic. The key\n"
        "fact: for loss `L = (y - target)^2` with `y = tanh(z)` and `z = w.x + b`,\n"
        "the chain rule gives `dL/dw = 2*(y - target) * (1 - y^2) * x`. After one\n"
        "step of `w <- w - lr * dL/dw`, the loss should go **down**."
    ))
    nb.cells.append(new_code_cell(
        "import numpy as np\n"
        "\n"
        "x = np.array([0.5, -1.0, 2.0])     # inputs\n"
        "w = np.array([0.1, 0.2, -0.1])     # weights\n"
        "b = 0.0                            # bias\n"
        "target = 1.0\n"
        "lr = 0.1\n"
        "\n"
        "# 1. Forward pass.\n"
        "# TODO: compute z = w . x + b, then y = tanh(z).\n"
        "z = None\n"
        "y = None\n"
        "\n"
        "# 2. Loss (squared error) BEFORE the step.\n"
        "# TODO: loss_before = (y - target) ** 2\n"
        "loss_before = None\n"
        "\n"
        "# 3. One gradient step.\n"
        "# dL/dy = 2*(y - target);  dy/dz = 1 - y**2;  dz/dw = x\n"
        "# TODO: compute grad_w (a length-3 array) via the chain rule, then\n"
        "#       update: w_new = w - lr * grad_w\n"
        "grad_w = None\n"
        "w_new = None\n"
        "\n"
        "# 4. Loss AFTER the step (recompute y with w_new).\n"
        "# TODO: loss_after using w_new\n"
        "loss_after = None\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Reveal the reference solution\n"
        "\n"
        "Run the hidden cell to see the forward pass and the hand-derived\n"
        "gradient step."
    ))
    nb.cells.append(hidden_solution_cell(
        "",
        "# Reference solution (self-contained: re-declares the inputs too).\n"
        "import numpy as np\n"
        "x = np.array([0.5, -1.0, 2.0])\n"
        "w = np.array([0.1, 0.2, -0.1])\n"
        "b = 0.0\n"
        "target = 1.0\n"
        "lr = 0.1\n"
        "z = w @ x + b\n"
        "y = np.tanh(z)\n"
        "loss_before = (y - target) ** 2\n"
        "grad_w = 2 * (y - target) * (1 - y ** 2) * x\n"
        "w_new = w - lr * grad_w\n"
        "y_after = np.tanh(w_new @ x + b)\n"
        "loss_after = (y_after - target) ** 2\n",
    ))
    nb.cells.append(new_markdown_cell(
        "## Check your work\n"
        "\n"
        "The check confirms the forward pass is finite and that one gradient step\n"
        "actually lowered the loss."
    ))
    nb.cells.append(new_code_cell(
        "import numpy as np\n"
        "assert np.isfinite(y), 'y is not finite'\n"
        "assert -1.0 <= float(y) <= 1.0, 'tanh output must be in [-1, 1]'\n"
        "assert loss_before is not None and loss_after is not None, 'compute the losses'\n"
        "assert float(loss_after) < float(loss_before), (\n"
        "    f'gradient step did not lower loss: {loss_before} -> {loss_after}')\n"
        "print('PASS', f'score=1.00  (loss {float(loss_before):.4f} -> {float(loss_after):.4f})')\n"
    ))
    return nb


def nb_m03_autograd():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 3 - Backprop From Scratch\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "In Module 2 you derived a gradient by hand. That's fine for one neuron,\n"
        "but real models have millions of parameters -- nobody derives those\n"
        "gradients by hand. Instead we build an **autograd engine**: a small\n"
        "system that records every arithmetic operation as it happens, then\n"
        "replays them backwards to compute all the gradients automatically.\n"
        "This is exactly what PyTorch does under the hood, just bigger.\n"
        "\n"
        "You'll write a `Value` class. Each `Value` wraps a single number and\n"
        "remembers:\n"
        "\n"
        "* its `data` (the forward number),\n"
        "* its `grad` (how much the final output changes if you nudge this\n"
        "  number -- starts at 0),\n"
        "* which other `Value`s produced it, and a local `_backward` function\n"
        "  that knows how to push gradient to those producers.\n"
        "\n"
        "When you write `c = a * b`, the engine creates `c` and stashes a closure\n"
        "that, when called, adds `b.data * c.grad` to `a.grad` and `a.data *\n"
        "c.grad` to `b.grad` -- the product rule. Calling `.backward()` on the\n"
        "final node sorts the whole graph topologically and runs every local\n"
        "`_backward` in reverse. That's backpropagation.\n"
        "\n"
        "Implement `+`, `*`, `**`, the helpers built on them (`-`, `/`), the\n"
        "`relu`/`tanh` nonlinearities, and `backward()`."
    ))
    nb.cells.append(new_code_cell(
        "import math\n"
        "\n"
        "class Value:\n"
        "    \"\"\"A single scalar with a gradient and a recorded backward function.\"\"\"\n"
        "\n"
        "    def __init__(self, data, _children=(), _op=''):\n"
        "        self.data = float(data)\n"
        "        self.grad = 0.0\n"
        "        self._backward = lambda: None\n"
        "        self._prev = set(_children)\n"
        "        self._op = _op\n"
        "\n"
        "    def __add__(self, other):\n"
        "        # TODO: return a Value for self + other, and set its _backward so\n"
        "        #       that d(out)/d(self) = 1 and d(out)/d(other) = 1.\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def __mul__(self, other):\n"
        "        # TODO: product rule. d(a*b)/da = b, d(a*b)/db = a.\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def __pow__(self, other):\n"
        "        # TODO: only int/float exponents. d(a**n)/da = n * a**(n-1).\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def relu(self):\n"
        "        # TODO: max(0, x); gradient flows only where the input was positive.\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def tanh(self):\n"
        "        # TODO: tanh(x); d(tanh)/dx = 1 - tanh(x)**2.\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def backward(self):\n"
        "        # TODO: topologically sort the graph, seed self.grad = 1.0, then\n"
        "        #       run each node's _backward in reverse order.\n"
        "        raise NotImplementedError\n"
        "\n"
        "    # These are provided for you, built on the ops above.\n"
        "    def __radd__(self, other): return self + other\n"
        "    def __rmul__(self, other): return self * other\n"
        "    def __neg__(self): return self * -1\n"
        "    def __sub__(self, other): return self + (-other)\n"
        "    def __rsub__(self, other): return other + (-self)\n"
        "    def __truediv__(self, other): return self * other ** -1\n"
        "    def __rtruediv__(self, other): return other * self ** -1\n"
        "    def __repr__(self): return f'Value(data={self.data}, grad={self.grad})'\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Reveal the reference solution\n"
        "\n"
        "Run the hidden cell to load the reference `Value` engine. This\n"
        "**redefines** `Value` in the notebook with the complete implementation,\n"
        "so the check below grades a working engine."
    ))
    nb.cells.append(hidden_solution_cell(
        read_ref("reference/autograd/engine.py"),
        "# Reference solution (single source of truth: reference/autograd/engine.py)\n\n",
    ))
    nb.cells.append(new_markdown_cell(
        "## Check your work\n"
        "\n"
        "The grader runs two properties: a hand-checkable gradient is correct,\n"
        "and gradients accumulate properly when a value is used more than once."
    ))
    nb.cells.append(new_code_cell(
        PATH_BOOTSTRAP +
        "from grader.checks.m3 import check_autograd\n"
        "from grader.core import run_checks\n"
        "\n"
        "r = run_checks(check_autograd(Value))\n"
        "print('PASS' if r.passed else 'FAIL', f'score={r.score:.2f}')\n"
        "if not r.passed:\n"
        "    print('failed checks:', r.failed_checks)\n"
    ))
    return nb


def nb_m04_bigram():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 4 - A Bigram Model That Writes\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "Time to build your first model that actually generates text. The\n"
        "**bigram model** is the simplest \"real\" language model: it assumes the\n"
        "next character depends only on the *single* character right before it.\n"
        "\n"
        "The recipe is pure counting:\n"
        "\n"
        "1. Reuse the `CharTokenizer` from the warm-up to turn text into ids.\n"
        "2. Build a `vocab x vocab` table `N` where `N[i, j]` counts how often\n"
        "   character `j` followed character `i`. Start every count at 1\n"
        "   (\"add-one smoothing\") so nothing is ever impossible.\n"
        "3. Normalise each row so it sums to 1 -- now `P[i]` is a probability\n"
        "   distribution over \"what comes after character i\".\n"
        "4. To **generate**, start at some character, sample the next one from\n"
        "   its row of `P`, move there, and repeat. A seeded random generator\n"
        "   makes the output reproducible.\n"
        "\n"
        "It won't write Shakespeare -- with only one character of memory it\n"
        "produces babble -- but it's a genuine probabilistic model, and it sets\n"
        "up everything that follows."
    ))
    nb.cells.append(new_code_cell(
        "import numpy as np\n"
        "from reference.tokenizer.char_tokenizer import CharTokenizer\n"
        "\n"
        "class BigramLM:\n"
        "    def __init__(self, text):\n"
        "        self.tokenizer = CharTokenizer(text)\n"
        "        V = self.tokenizer.vocab_size\n"
        "        # TODO: build N (V x V) starting at 1 (add-one smoothing),\n"
        "        #       count each consecutive (current, next) pair in the text,\n"
        "        #       then row-normalise into self.P (each row sums to 1).\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def generate(self, n, seed):\n"
        "        # TODO: with np.random.default_rng(seed), start at id 0 and draw\n"
        "        #       n characters from the current row of self.P, returning the\n"
        "        #       decoded string.\n"
        "        raise NotImplementedError\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Reveal the reference solution\n"
        "\n"
        "Run the hidden cell to load the reference `BigramLM` (it redefines the\n"
        "class with the complete implementation)."
    ))
    nb.cells.append(hidden_solution_cell(
        read_ref("reference/bigram/bigram.py"),
        "# Reference solution (single source of truth: reference/bigram/bigram.py)\n\n",
    ))
    nb.cells.append(new_markdown_cell(
        "## Check your work\n"
        "\n"
        "These asserts confirm each row of `P` is a valid probability\n"
        "distribution (sums to 1), that seeded generation is reproducible, and\n"
        "that every generated character is in the vocabulary."
    ))
    nb.cells.append(new_code_cell(
        "import numpy as np\n"
        "_text = 'hello world\\nthe quick brown fox jumps over the lazy dog'\n"
        "_m = BigramLM(_text)\n"
        "\n"
        "assert _m.P.shape == (_m.tokenizer.vocab_size, _m.tokenizer.vocab_size), 'P wrong shape'\n"
        "assert np.allclose(_m.P.sum(axis=1), 1.0), 'rows of P must sum to 1'\n"
        "\n"
        "_a = _m.generate(20, seed=42)\n"
        "_b = _m.generate(20, seed=42)\n"
        "assert _a == _b, 'seeded generation should be reproducible'\n"
        "assert len(_a) == 20, 'should generate 20 characters'\n"
        "assert all(ch in _m.tokenizer.vocab for ch in _a), 'generated chars out of vocab'\n"
        "print('PASS', f'score=1.00  (sample: {_a!r})')\n"
    ))
    return nb


def nb_m05_mlp_lm():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 5 - A Neural Language Model\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "The bigram model only remembers one character. Now we give the model a\n"
        "wider **context window** and let it *learn* (not just count) how to\n"
        "predict the next character. This is the classic Bengio-style MLP\n"
        "language model, written in plain NumPy with backprop derived by hand.\n"
        "\n"
        "Two pieces to implement:\n"
        "\n"
        "1. **`softmax(logits)`** -- turns raw scores into probabilities. The\n"
        "   trick that matters: subtract the per-row max before exponentiating,\n"
        "   so large logits don't overflow to infinity. Mathematically identical,\n"
        "   numerically safe.\n"
        "2. **`MLPLM`** -- the model itself:\n"
        "   * an **embedding** table `C` that maps each token id to a small dense\n"
        "     vector,\n"
        "   * a hidden layer `W1, b1` with a `tanh` nonlinearity,\n"
        "   * an output layer `W2, b2` producing one logit per vocabulary token.\n"
        "   `forward` flattens the embedded context and runs it through the two\n"
        "   layers. `loss` is mean cross-entropy. `step` does one SGD update with\n"
        "   **hand-derived gradients** for every parameter -- read the comments\n"
        "   in the reference to see each gradient spelled out.\n"
        "\n"
        "Train it on a tiny batch and the loss should drop sharply: the model is\n"
        "memorising the mapping, which proves the gradients are correct."
    ))
    nb.cells.append(new_code_cell(
        "import numpy as np\n"
        "\n"
        "def softmax(logits):\n"
        "    # TODO: numerically stable row-wise softmax. Subtract the per-row max,\n"
        "    #       exponentiate, divide by the row sum.\n"
        "    raise NotImplementedError\n"
        "\n"
        "class MLPLM:\n"
        "    def __init__(self, vocab, n_embd, block, hidden, seed):\n"
        "        # TODO: store dims; with np.random.default_rng(seed) init\n"
        "        #       C (vocab x n_embd), W1 (block*n_embd x hidden), b1,\n"
        "        #       W2 (hidden x vocab), b2. See the reference for the scales.\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def forward(self, X):\n"
        "        # TODO: embed X (B, block) -> (B, block, n_embd), flatten,\n"
        "        #       tanh hidden layer, then linear to (B, vocab) logits.\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def loss(self, X, Y):\n"
        "        # TODO: mean cross-entropy of the next-token predictions.\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def step(self, X, Y, lr):\n"
        "        # TODO: one SGD update with hand-derived gradients for C, W1, b1,\n"
        "        #       W2, b2.\n"
        "        raise NotImplementedError\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Reveal the reference solution\n"
        "\n"
        "Run the hidden cell to load the reference `softmax` and `MLPLM` (it\n"
        "redefines both with the complete implementation)."
    ))
    nb.cells.append(hidden_solution_cell(
        read_ref("reference/mlp_lm/model.py"),
        "# Reference solution (single source of truth: reference/mlp_lm/model.py)\n\n",
    ))
    nb.cells.append(new_markdown_cell(
        "## Check your work\n"
        "\n"
        "The grader checks the forward shape, that softmax stays finite on huge\n"
        "logits, that the loss is finite and non-negative, and -- the real test\n"
        "-- that training on a tiny batch overfits it (loss drops below half)."
    ))
    nb.cells.append(new_code_cell(
        PATH_BOOTSTRAP +
        "from grader.checks.m5 import check_mlp_lm\n"
        "from grader.core import run_checks\n"
        "\n"
        "r = run_checks(check_mlp_lm(softmax, MLPLM))\n"
        "print('PASS' if r.passed else 'FAIL', f'score={r.score:.2f}')\n"
        "if not r.passed:\n"
        "    print('failed checks:', r.failed_checks)\n"
    ))
    return nb


def nb_m06_attention():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 6 - Attention From Scratch\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "This is the idea that makes transformers work: **self-attention**. The\n"
        "MLP in Module 5 mashed the whole context into one flat vector. Attention\n"
        "is smarter -- it lets every position look back at the earlier positions\n"
        "and decide, on the fly, which ones are worth paying attention to.\n"
        "\n"
        "You'll implement a single causal attention head in NumPy:\n"
        "\n"
        "1. Project the input `X` (shape `T x C`) into three things via seeded\n"
        "   weight matrices: **queries** `Q`, **keys** `K`, and **values** `V`.\n"
        "2. Score every pair of positions: `scores = Q @ K.T / sqrt(head_size)`.\n"
        "   The scaling keeps the numbers in a sane range.\n"
        "3. Apply a **causal mask**: set the strict upper triangle to `-inf` so a\n"
        "   position can only attend to itself and earlier positions (a model\n"
        "   predicting the next token must not peek at the future).\n"
        "4. `softmax` each row into attention weights, then mix the values:\n"
        "   `out = weights @ V`.\n"
        "\n"
        "Then wrap several heads into **`multi_head`**: run `n_head` independent\n"
        "heads (each with its own seed), concatenate their outputs, and apply a\n"
        "final seeded output projection. That's the whole attention mechanism --\n"
        "the rest of a GPT is embeddings, MLPs, and LayerNorm around this core."
    ))
    nb.cells.append(new_code_cell(
        "import numpy as np\n"
        "\n"
        "def self_attention(X, head_size, seed, return_weights=False):\n"
        "    # TODO: T, C = X.shape; rng = np.random.default_rng(seed).\n"
        "    #   - build Wq, Wk, Wv each (C x head_size) from the SAME rng\n"
        "    #   - Q, K, V = X @ Wq, X @ Wk, X @ Wv\n"
        "    #   - scores = Q @ K.T / sqrt(head_size)\n"
        "    #   - causal mask: strict upper triangle -> -inf\n"
        "    #   - weights = softmax(scores) row-wise; out = weights @ V\n"
        "    #   - return (out, weights) if return_weights else out\n"
        "    raise NotImplementedError\n"
        "\n"
        "def multi_head(X, n_head, seed):\n"
        "    # TODO: head_size = C // n_head; run n_head heads with seeds seed+i,\n"
        "    #       concatenate to (T, C), then apply a seeded output proj Wo (C x C).\n"
        "    raise NotImplementedError\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Reveal the reference solution\n"
        "\n"
        "Run the hidden cell to load the reference `self_attention` and\n"
        "`multi_head` (it redefines both with the complete implementation,\n"
        "including the small `_softmax` helper they use)."
    ))
    nb.cells.append(hidden_solution_cell(
        read_ref("reference/attention/attention.py"),
        "# Reference solution (single source of truth: reference/attention/attention.py)\n\n",
    ))
    nb.cells.append(new_markdown_cell(
        "## Check your work\n"
        "\n"
        "The grader checks output shape, that attention weights sum to 1 per row,\n"
        "**causality** (changing future inputs must not change earlier outputs),\n"
        "and that `multi_head` preserves the channel dimension."
    ))
    nb.cells.append(new_code_cell(
        PATH_BOOTSTRAP +
        "from grader.checks.m6 import check_attention\n"
        "from grader.core import run_checks\n"
        "\n"
        "r = run_checks(check_attention(self_attention, multi_head))\n"
        "print('PASS' if r.passed else 'FAIL', f'score={r.score:.2f}')\n"
        "if not r.passed:\n"
        "    print('failed checks:', r.failed_checks)\n"
    ))
    return nb


# ---------------------------------------------------------------------------
# Colab lessons
# ---------------------------------------------------------------------------


def nb_m07_gpt():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 7 - Build the GPT (PyTorch, on Colab)\n"
        "\n"
        f"{colab_badge('m07_gpt.ipynb')}\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "You've built every piece in NumPy. Now we assemble the real thing in\n"
        "**PyTorch**: a small decoder-only GPT, nanoGPT-style. PyTorch gives us\n"
        "autograd (the Module 3 idea, industrial strength), GPU support, and\n"
        "battle-tested layers, so we can stack the components into a working\n"
        "transformer and train it.\n"
        "\n"
        "> **Run this on Colab.** It uses PyTorch and benefits from a GPU. Click\n"
        "> the *Open in Colab* badge above. A free **Google account** is\n"
        "> required. Grading is by **pasting the printed token** into the browser\n"
        "> grader -- the notebook never uploads your weights.\n"
        "\n"
        "The reference `GPT` (in `reference/gpt/model.py`) is a stack of pre-LayerNorm\n"
        "blocks: token + positional embeddings, then `n_layer` blocks of causal\n"
        "self-attention + MLP with residual connections, a final LayerNorm, and a\n"
        "linear language-model head. The `CONFIGS['cpu']` preset is a ~0.8M\n"
        "parameter model that runs fine even without a GPU."
    ))
    nb.cells.append(new_code_cell(
        COLAB_BOOTSTRAP +
        "\n"
        "# Device detection: use a GPU if Colab gave you one, else fall back to CPU.\n"
        "import torch\n"
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "print('device:', device)\n"
    ))
    nb.cells.append(new_code_cell(
        "# Build the reference GPT from the CPU-sized config and sanity-check it.\n"
        "import torch\n"
        "from reference.gpt.model import GPT, CONFIGS\n"
        "\n"
        "cfg = CONFIGS['cpu']\n"
        "model = GPT(cfg).to(device)\n"
        "n_params = sum(p.numel() for p in model.parameters())\n"
        "print(f'parameters: {n_params:,}')\n"
        "\n"
        "# Forward a batch of zeros to confirm the output shape (B, T, vocab).\n"
        "x = torch.zeros(2, cfg.block_size, dtype=torch.long, device=device)\n"
        "logits, loss = model(x)\n"
        "print('logits shape:', tuple(logits.shape), '| loss (no targets):', loss)\n"
    ))
    nb.cells.append(new_code_cell(
        "# A quick training sanity check: a few AdamW steps on a random batch\n"
        "# should drive the loss down, confirming autograd + the model wiring work.\n"
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), device=device)\n"
        "y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), device=device)\n"
        "_, l0 = model(x, y)\n"
        "opt = torch.optim.AdamW(model.parameters(), lr=1e-3)\n"
        "for _ in range(20):\n"
        "    opt.zero_grad(); _, l = model(x, y); l.backward(); opt.step()\n"
        "_, l1 = model(x, y)\n"
        "print(f'loss {l0.item():.4f} -> {l1.item():.4f}')\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Print your submission token\n"
        "\n"
        "The final cell trains the model briefly and prints a one-line\n"
        "**submission token**. Copy that line and paste it into the browser\n"
        "grader for Module 7. The token encodes your final loss (graded by a\n"
        "tolerance band), an architecture fingerprint, and a deterministic\n"
        "sample hash -- never your raw weights."
    ))
    nb.cells.append(new_code_cell(
        "# Train briefly and print the submission token for Module 7.\n"
        "from reference.gpt.train import train, make_submission_token\n"
        "\n"
        "# A small built-in corpus so the notebook is self-contained.\n"
        "data = (\n"
        "    'First Citizen:\\nBefore we proceed any further, hear me speak.\\n'\n"
        "    'All:\\nSpeak, speak.\\nFirst Citizen:\\nYou are all resolved rather to '\n"
        "    'die than to famish?\\nAll:\\nResolved. resolved.\\n'\n"
        ") * 8\n"
        "\n"
        "res = train(config='cpu', max_iters=50, seed=1337, data=data)\n"
        "print(make_submission_token(res, 'M7'))\n"
    ))
    return nb


def nb_m08_train():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 8 - Train on Tiny Shakespeare\n"
        "\n"
        f"{colab_badge('m08_train.ipynb')}\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "Now you train the GPT for real, on the **Tiny Shakespeare** corpus, and\n"
        "watch it go from random noise to recognisable (if nonsensical) English.\n"
        "This is the same training loop used everywhere, just scaled down: sample\n"
        "random windows of text, predict the next character at every position,\n"
        "backprop the cross-entropy loss, and step AdamW.\n"
        "\n"
        "> **Run this on Colab** with a GPU for the best experience (it works on\n"
        "> CPU too, just slower). Click the *Open in Colab* badge. A free\n"
        "> **Google account** is required. Grading is by **pasting the printed\n"
        "> token** into the browser grader.\n"
        "\n"
        "The `train` helper in `reference/gpt/train.py` handles everything:\n"
        "building the tokenizer, resizing the config's vocab, the AdamW loop, and\n"
        "reporting initial vs final loss. You pick the corpus and the number of\n"
        "iterations; more iterations means lower loss (down to a point)."
    ))
    nb.cells.append(new_code_cell(
        COLAB_BOOTSTRAP +
        "\n"
        "import torch\n"
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "print('device:', device, '| GPU:', torch.cuda.is_available())\n"
    ))
    nb.cells.append(new_code_cell(
        "# Get the Tiny Shakespeare text. On Colab you can download the full file;\n"
        "# here we fall back to a small embedded sample so the notebook always runs.\n"
        "SAMPLE = (\n"
        "    'First Citizen:\\nBefore we proceed any further, hear me speak.\\n\\n'\n"
        "    'All:\\nSpeak, speak.\\n\\nFirst Citizen:\\nYou are all resolved rather '\n"
        "    'to die than to famish?\\n\\nAll:\\nResolved. resolved.\\n\\n'\n"
        "    'First Citizen:\\nFirst, you know Caius Marcius is chief enemy to the '\n"
        "    'people.\\n\\nAll:\\nWe know\\'t, we know\\'t.\\n'\n"
        ")\n"
        "\n"
        "try:\n"
        "    import urllib.request\n"
        "    url = ('https://raw.githubusercontent.com/karpathy/char-rnn/master/'\n"
        "           'data/tinyshakespeare/input.txt')\n"
        "    data = urllib.request.urlopen(url, timeout=10).read().decode('utf-8')\n"
        "    print('downloaded tiny shakespeare:', len(data), 'chars')\n"
        "except Exception as e:\n"
        "    data = SAMPLE * 20\n"
        "    print('download failed (', e, '); using embedded sample:', len(data), 'chars')\n"
    ))
    nb.cells.append(new_code_cell(
        "# Train. On a GPU bump max_iters up (e.g. 2000+); on CPU keep it small.\n"
        "from reference.gpt.train import train, make_submission_token\n"
        "import torch\n"
        "\n"
        "max_iters = 1000 if torch.cuda.is_available() else 100\n"
        "config = 't4' if torch.cuda.is_available() else 'cpu'\n"
        "res = train(config=config, max_iters=max_iters, seed=1337, data=data)\n"
        "print(f\"initial loss {res['initial_loss']:.4f} -> final loss {res['final_loss']:.4f}\")\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Print your submission token\n"
        "\n"
        "Copy the printed token line and paste it into the browser grader for\n"
        "Module 8."
    ))
    nb.cells.append(new_code_cell(
        "print(make_submission_token(res, 'M8'))\n"
    ))
    return nb


def nb_m09_capstone():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 9 - Capstone: Make It Yours\n"
        "\n"
        f"{colab_badge('m09_capstone.ipynb')}\n"
        "\n"
        "## What you'll build\n"
        "\n"
        "This is your victory lap. Train the GPT on a corpus **you** choose --\n"
        "your favourite book, song lyrics, code, chat logs, anything that's a\n"
        "plain-text file -- and watch it learn that style. The pipeline is\n"
        "identical to Module 8; only the data changes.\n"
        "\n"
        "> **Run this on Colab** (GPU recommended). Click the *Open in Colab*\n"
        "> badge. A free **Google account** is required. Grading is by **pasting\n"
        "> the printed token** into the browser grader.\n"
        "\n"
        "### Picking a corpus\n"
        "\n"
        "* Aim for at least a few hundred KB of text -- more is better.\n"
        "* It must be a **character-level** friendly plain-text file (`.txt`).\n"
        "* Keep it legal and appropriate: use public-domain text (Project\n"
        "  Gutenberg is a goldmine) or text you own.\n"
        "\n"
        "On Colab, upload your file with the snippet below, or paste text\n"
        "directly into the `learner_corpus` string."
    ))
    nb.cells.append(new_code_cell(
        COLAB_BOOTSTRAP +
        "\n"
        "import torch\n"
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "print('device:', device)\n"
    ))
    nb.cells.append(new_code_cell(
        "# --- Provide YOUR corpus here ---\n"
        "# Option A: paste/keep text directly in this string.\n"
        "learner_corpus = (\n"
        "    'To be, or not to be, that is the question:\\n'\n"
        "    'Whether tis nobler in the mind to suffer\\n'\n"
        "    'The slings and arrows of outrageous fortune,\\n'\n"
        ") * 40\n"
        "\n"
        "# Option B (Colab): upload a .txt file and load it instead.\n"
        "#   from google.colab import files\n"
        "#   up = files.upload()\n"
        "#   learner_corpus = open(next(iter(up)), encoding='utf-8').read()\n"
        "\n"
        "assert len(learner_corpus) > 200, 'corpus is very small; add more text'\n"
        "print('corpus chars:', len(learner_corpus))\n"
    ))
    nb.cells.append(new_code_cell(
        "# Train on your corpus.\n"
        "from reference.gpt.train import train, make_submission_token\n"
        "import torch\n"
        "\n"
        "max_iters = 1000 if torch.cuda.is_available() else 100\n"
        "config = 't4' if torch.cuda.is_available() else 'cpu'\n"
        "res = train(config=config, max_iters=max_iters, seed=1337, data=learner_corpus)\n"
        "print(f\"initial loss {res['initial_loss']:.4f} -> final loss {res['final_loss']:.4f}\")\n"
    ))
    nb.cells.append(new_code_cell(
        "# (Optional) Save a checkpoint you can download from Colab.\n"
        "import torch\n"
        "torch.save(res['model'].state_dict(), 'capstone_model.pt')\n"
        "print('saved capstone_model.pt')\n"
        "# On Colab, download it with:\n"
        "#   from google.colab import files; files.download('capstone_model.pt')\n"
    ))
    nb.cells.append(new_markdown_cell(
        "## Print your submission token\n"
        "\n"
        "Copy the printed token line and paste it into the browser grader for\n"
        "Module 9. Congratulations -- you built a language model from scratch and\n"
        "made it your own."
    ))
    nb.cells.append(new_code_cell(
        "print(make_submission_token(res, 'M9'))\n"
    ))
    return nb


# ---------------------------------------------------------------------------
# Static lesson
# ---------------------------------------------------------------------------


def nb_m10_whats_next():
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(
        "# Module 10 - What's Next: How Real LLMs Differ\n"
        "\n"
        "You've built a working GPT from first principles. The frontier models you\n"
        "hear about -- the ones that write essays and code -- are the *same idea*\n"
        "you just implemented, scaled and refined along a few key axes. Here's\n"
        "what changes between your tiny Shakespeare model and a production LLM.\n"
        "\n"
        "## 1. Scale\n"
        "\n"
        "Your model has under a million parameters. Frontier models have hundreds\n"
        "of billions to trillions. The architecture is nearly identical -- more\n"
        "layers, wider embeddings, more heads, a longer context window. Scaling\n"
        "laws showed that loss falls predictably as you grow model size, data,\n"
        "and compute together, which is why the industry kept making models\n"
        "bigger. Training one run can cost millions of dollars and weeks on\n"
        "thousands of GPUs.\n"
        "\n"
        "## 2. Data\n"
        "\n"
        "You trained on a few KB of Shakespeare. Real models train on trillions\n"
        "of tokens scraped from the web, books, and code, then heavily filtered\n"
        "and deduplicated. They also use **subword tokenizers** (BPE) instead of\n"
        "your character-level one, so each token carries more meaning and\n"
        "sequences are shorter. Data quality and mixture matter as much as\n"
        "quantity.\n"
        "\n"
        "## 3. Fine-tuning\n"
        "\n"
        "A raw pretrained model just predicts the next token -- it doesn't\n"
        "naturally follow instructions. **Supervised fine-tuning (SFT)** continues\n"
        "training on curated (instruction, response) pairs so the model learns to\n"
        "be a helpful assistant rather than an autocomplete engine.\n"
        "\n"
        "## 4. RLHF and alignment\n"
        "\n"
        "**Reinforcement Learning from Human Feedback** goes further: humans rank\n"
        "model responses, a reward model learns those preferences, and the LLM is\n"
        "optimised to produce answers people prefer -- more helpful, honest, and\n"
        "harmless. Newer variants (DPO, RLAIF, constitutional methods) chase the\n"
        "same goal of aligning model behaviour with human intent.\n"
        "\n"
        "## 5. Inference tricks\n"
        "\n"
        "Production systems add KV caching for speed, quantization to shrink\n"
        "models, better sampling (temperature, top-p) instead of greedy decoding,\n"
        "and long-context techniques. The core forward pass, though, is the one\n"
        "you wrote.\n"
        "\n"
        "## Where to go from here\n"
        "\n"
        "* Read Karpathy's *nanoGPT* and *makemore* -- the spiritual parents of\n"
        "  this course.\n"
        "* Try a BPE tokenizer and compare loss per character.\n"
        "* Scale your capstone model up on a bigger corpus and longer context.\n"
        "* Explore SFT: fine-tune on instruction data and see the behaviour shift.\n"
        "\n"
        "You now understand, end to end, what an LLM actually is. That intuition\n"
        "doesn't go out of date."
    ))
    nb.cells.append(new_code_cell(
        "# Mark this module complete.\n"
        "complete = True\n"
        "print('Module 10 complete:', complete)\n"
        "print('You finished the course. Go build something.')\n"
    ))
    return nb


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

NOTEBOOKS = {
    "s1_tokenizer_warmup.ipynb": nb_s1_tokenizer,
    "m00_orientation.ipynb": nb_m00_orientation,
    "m01_numpy.ipynb": nb_m01_numpy,
    "m02_neuron.ipynb": nb_m02_neuron,
    "m03_autograd.ipynb": nb_m03_autograd,
    "m04_bigram.ipynb": nb_m04_bigram,
    "m05_mlp_lm.ipynb": nb_m05_mlp_lm,
    "m06_attention.ipynb": nb_m06_attention,
    "m07_gpt.ipynb": nb_m07_gpt,
    "m08_train.ipynb": nb_m08_train,
    "m09_capstone.ipynb": nb_m09_capstone,
    "m10_whats_next.ipynb": nb_m10_whats_next,
}


def main():
    os.makedirs(NB_DIR, exist_ok=True)
    for filename, builder in NOTEBOOKS.items():
        nb = builder()
        path = os.path.join(NB_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"wrote {path}  ({len(nb.cells)} cells)")
    print(f"\ngenerated {len(NOTEBOOKS)} notebooks into {NB_DIR}")


if __name__ == "__main__":
    main()
