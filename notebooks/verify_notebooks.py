"""Headless verification of every generated lesson notebook.

Strategy
--------
* BROWSER notebooks (s1, m00-m06): build a SOLVED copy by replacing the
  starter code cell's source with the hidden reference-solution cell's source
  (the cell carrying ``metadata.jupyter.source_hidden == True``). Then drop the
  hidden solution cell so the symbol isn't defined twice. Execute headless with
  nbclient and assert (a) no cell raised, and (b) the check cell's stdout
  contains a ``PASS`` marker.
* COLAB notebooks (m07, m08): execute as-is headless (torch is installed
  locally). Iters are already small for CPU. Assert no cell raised and that a
  submission-token line (``SLM-...``) was printed.
* m09 / m10: smoke-execute (m09 is a Colab notebook with a CPU fallback path
  identical to m08; m10 is static). Treated as informational, not gating.

All notebooks execute with the working directory set to ``notebooks/`` -- the
same place a browser/Jupyter session opens them -- and a path bootstrap is
prepended so ``import grader`` / ``import reference`` resolve from the repo root.

Run from the repo root:  ``python notebooks/verify_notebooks.py``
Exit code 0 == every gated notebook passed.
"""

from __future__ import annotations

import copy
import os
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(_REPO_ROOT, "notebooks")

BROWSER = [
    "s1_tokenizer_warmup.ipynb",
    "m00_orientation.ipynb",
    "m01_numpy.ipynb",
    "m02_neuron.ipynb",
    "m03_autograd.ipynb",
    "m04_bigram.ipynb",
    "m05_mlp_lm.ipynb",
    "m06_attention.ipynb",
]
COLAB_GATED = ["m07_gpt.ipynb", "m08_train.ipynb"]
SMOKE = ["m09_capstone.ipynb", "m10_whats_next.ipynb"]

# Prepended to the first cell of every solved/executed notebook so that
# `import grader` and `import reference` resolve when cwd is notebooks/.
BOOTSTRAP = (
    "import sys, os\n"
    f"_repo = {_REPO_ROOT!r}\n"
    "if _repo not in sys.path:\n"
    "    sys.path.insert(0, _repo)\n"
)


def code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def find_hidden_solution(nb):
    """Return the source of the hidden (source_hidden) solution code cell."""
    for c in nb.cells:
        if c.cell_type == "code" and c.get("metadata", {}).get("jupyter", {}).get("source_hidden"):
            return c.source
    return None


def build_solved(nb):
    """Return a copy where the starter cell is replaced by the solution.

    The starter is the FIRST code cell; the hidden solution is the
    source_hidden code cell. We overwrite the starter with the solution source
    and remove the hidden cell so the symbol is defined exactly once.
    """
    nb = copy.deepcopy(nb)
    solution_src = find_hidden_solution(nb)

    code = code_cells(nb)
    starter = code[0]
    if solution_src is not None:
        starter.source = solution_src

    # Drop the hidden solution cell(s) from the notebook entirely.
    nb.cells = [
        c for c in nb.cells
        if not (c.cell_type == "code"
                and c.get("metadata", {}).get("jupyter", {}).get("source_hidden"))
    ]

    # Prepend the path bootstrap to the (new) first code cell.
    first_code = code_cells(nb)[0]
    first_code.source = BOOTSTRAP + "\n" + first_code.source
    return nb


def prepend_bootstrap(nb):
    nb = copy.deepcopy(nb)
    first_code = code_cells(nb)[0]
    first_code.source = BOOTSTRAP + "\n" + first_code.source
    return nb


def execute(nb):
    """Execute a notebook in-memory with cwd = notebooks/. Returns the nb."""
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": NB_DIR}},
    )
    client.execute()
    return nb


def cell_text_outputs(cell):
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            out.append(o.get("text", ""))
        elif o.get("output_type") in ("execute_result", "display_data"):
            out.append(o.get("data", {}).get("text/plain", ""))
        elif o.get("output_type") == "error":
            out.append("\n".join(o.get("traceback", [])))
    return "\n".join(out)


def all_outputs_text(nb):
    return "\n".join(cell_text_outputs(c) for c in nb.cells if c.cell_type == "code")


def verify_browser(filename):
    path = os.path.join(NB_DIR, filename)
    nb = nbformat.read(path, as_version=4)
    solved = build_solved(nb)
    try:
        execute(solved)
    except CellExecutionError as e:
        # Surface the actual exception line (last line of the traceback).
        last = [ln for ln in str(e).splitlines() if ln.strip()]
        return False, "cell error: " + (last[-1] if last else "unknown")
    text = all_outputs_text(solved)
    if "PASS" not in text:
        return False, "no PASS marker in outputs"
    if "FAIL" in text:
        return False, "a check printed FAIL"
    return True, "PASS marker found"


def verify_colab(filename):
    path = os.path.join(NB_DIR, filename)
    nb = nbformat.read(path, as_version=4)
    nb = prepend_bootstrap(nb)
    try:
        execute(nb)
    except CellExecutionError as e:
        # Surface the actual exception line (last line of the traceback).
        last = [ln for ln in str(e).splitlines() if ln.strip()]
        return False, "cell error: " + (last[-1] if last else "unknown")
    text = all_outputs_text(nb)
    if "SLM-" not in text:
        return False, "no submission token (SLM-...) printed"
    # Pull the token line for the summary.
    token = next((ln for ln in text.splitlines() if ln.strip().startswith("SLM-")), "")
    return True, f"token: {token.strip()}"


def verify_smoke(filename):
    path = os.path.join(NB_DIR, filename)
    nb = nbformat.read(path, as_version=4)
    nb = prepend_bootstrap(nb)
    try:
        execute(nb)
    except CellExecutionError as e:
        # Surface the actual exception line (last line of the traceback).
        last = [ln for ln in str(e).splitlines() if ln.strip()]
        return False, "cell error: " + (last[-1] if last else "unknown")
    return True, "executed clean"


def main():
    results = []  # (name, gated, ok, detail)

    print("=== BROWSER notebooks (solved copy) ===")
    for fn in BROWSER:
        ok, detail = verify_browser(fn)
        results.append((fn, True, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {fn}: {detail}")

    print("=== COLAB notebooks (executed locally) ===")
    for fn in COLAB_GATED:
        ok, detail = verify_colab(fn)
        results.append((fn, True, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {fn}: {detail}")

    print("=== SMOKE notebooks ===")
    for fn in SMOKE:
        ok, detail = verify_smoke(fn)
        results.append((fn, False, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {fn}: {detail}")

    gated = [r for r in results if r[1]]
    gated_ok = all(r[2] for r in gated)
    smoke_ok = all(r[2] for r in results if not r[1])

    print("\n=== SUMMARY ===")
    print(f"  gated notebooks: {sum(1 for r in gated if r[2])}/{len(gated)} passed")
    print(f"  smoke notebooks: {sum(1 for r in results if not r[1] and r[2])}/"
          f"{sum(1 for r in results if not r[1])} passed")
    all_ok = gated_ok and smoke_ok
    print(f"  OVERALL: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
