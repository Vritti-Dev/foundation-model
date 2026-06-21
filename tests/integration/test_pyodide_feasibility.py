"""Integration: the in-browser runtime boundary (spec 3.1).

Modules 0-6 must run in JupyterLite/Pyodide, which CANNOT load PyTorch. So the
reference modules those lessons import must not import torch at module load
time. We enforce this statically with an AST scan (lazy imports inside a
function — e.g. demo.weights_from_torch — are allowed, since the browser never
calls them). Modules 7-9 (Colab) are expected to import torch.
"""

import ast
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]

# M0-M6 reference modules: these load in the browser, so NO module-level torch.
BROWSER_MODULES = [
    "reference/autograd/engine.py",
    "reference/autograd/nn.py",
    "reference/tokenizer/char_tokenizer.py",
    "reference/bigram/bigram.py",
    "reference/mlp_lm/model.py",
    "reference/attention/attention.py",
    "reference/demo/numpy_forward.py",
]

# M7/M8 reference modules: Colab-only, torch at module level is expected.
COLAB_MODULES = ["reference/gpt/model.py", "reference/gpt/train.py"]


def _top_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:  # only module top level; not inside functions
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_browser_modules_do_not_import_torch_at_top_level():
    for rel in BROWSER_MODULES:
        imports = _top_level_imports(ROOT / rel)
        assert "torch" not in imports, f"{rel} imports torch at module level (breaks Pyodide)"


def test_colab_modules_do_import_torch():
    # Sanity: confirm the boundary is real, not that torch merely isn't used anywhere.
    for rel in COLAB_MODULES:
        assert "torch" in _top_level_imports(ROOT / rel), f"{rel} expected to use torch"


def test_browser_reference_runs_on_toy_data_within_budget():
    """Each browser-module path runs to completion on toy data quickly."""
    from reference.autograd.engine import Value
    from reference.tokenizer.char_tokenizer import CharTokenizer
    from reference.bigram.bigram import BigramLM
    from reference.mlp_lm.model import MLPLM
    from reference.attention.attention import self_attention
    from reference.demo.numpy_forward import numpy_forward

    start = time.time()
    a = Value(2.0); (a * a + a).backward()
    tk = CharTokenizer("hello world")
    assert tk.decode(tk.encode("hello")) == "hello"
    BigramLM("hello world the lazy dog").generate(20, seed=0)
    m = MLPLM(vocab=5, n_embd=8, block=2, hidden=16, seed=0)
    m.step(np.array([[0, 1]]), np.array([2]), lr=0.1)
    self_attention(np.random.default_rng(0).standard_normal((4, 8)), head_size=8, seed=0)
    assert time.time() - start < 10.0, "toy browser path too slow for in-browser use"
