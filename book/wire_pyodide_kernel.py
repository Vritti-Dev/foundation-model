"""Wire the JupyterLite Pyodide kernel labextension into the built lite app.

`jupyter-lite build` (via jupyterlite-sphinx) does not discover the
`@jupyterlite/pyodide-kernel-extension` labextension on this Windows setup,
even when JUPYTER_PATH points at the right share/jupyter directory. The build
ships pyodide.js and the kernel config but leaves `federated_extensions: []`,
so the live app starts with "No Kernel".

This script does what the build should have:
1. Copies the labextension files into book/_build/html/lite/extensions/.
2. Reads its package.json to get the remoteEntry filename.
3. Patches book/_build/html/lite/jupyter-lite.json to register it in
   `federated_extensions`.

Idempotent. Safe to run multiple times. Run after `jupyter-book build book/`.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def _find_labextension() -> Path:
    """Locate the pyodide-kernel-extension labextension shipped with the
    installed jupyterlite-pyodide-kernel package."""
    import jupyterlite_pyodide_kernel  # noqa: F401

    candidates = [
        Path(os.environ.get("APPDATA", "")) / "Python/share/jupyter/labextensions/@jupyterlite/pyodide-kernel-extension",
        Path.home() / "AppData/Roaming/Python/share/jupyter/labextensions/@jupyterlite/pyodide-kernel-extension",
        Path(sys.prefix) / "share/jupyter/labextensions/@jupyterlite/pyodide-kernel-extension",
    ]
    for c in candidates:
        if c.exists() and (c / "package.json").exists():
            return c
    raise SystemExit(f"could not find pyodide-kernel-extension labextension; tried: {candidates}")


def main() -> int:
    lite_root = Path(__file__).resolve().parent / "_build" / "html" / "lite"
    if not lite_root.exists():
        print(f"no lite build at {lite_root}; run jupyter-book build first")
        return 1

    src = _find_labextension()
    print(f"labextension source: {src}")

    dst = lite_root / "extensions" / "@jupyterlite" / "pyodide-kernel-extension"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"copied to: {dst}")

    pkg = json.loads((src / "package.json").read_text(encoding="utf-8"))
    static = src / "static"
    remote_entries = sorted(static.glob("remoteEntry.*.js"))
    if not remote_entries:
        raise SystemExit("no remoteEntry.*.js in labextension static; cannot register")
    remote = remote_entries[0].name

    je = pkg.get("jupyterlab", {})
    entry = {
        "name": pkg["name"],
        "load": f"static/{remote}",
    }
    if je.get("extension"):
        entry["extension"] = "./extension"
    if je.get("mimeExtension"):
        entry["mimeExtension"] = "./mimeExtension"

    lite_json = lite_root / "jupyter-lite.json"
    cfg = json.loads(lite_json.read_text(encoding="utf-8"))
    data = cfg.setdefault("jupyter-config-data", {})
    fed = [e for e in data.get("federated_extensions", []) if e.get("name") != pkg["name"]]
    fed.append(entry)
    data["federated_extensions"] = fed
    lite_json.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"registered federated extension {pkg['name']} (load={entry['load']})")
    print("federated_extensions now:", [e["name"] for e in fed])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
