"""Post-build fixup for the Windows-built site.

`jupyterlite-sphinx` on Windows emits the "Open as a notebook" launcher with a
backslash in the URL (e.g. `..\\lite/lab/index.html`). Browsers do not normalize
backslashes in HTTP URLs, so the button 404s when served over HTTP (works only
on file://). This script walks the built HTML and normalizes those separators.

Run after `jupyter-book build book/`. No-op on Linux/Mac builds.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "_build" / "html"


def main() -> int:
    if not ROOT.exists():
        print(f"no build at {ROOT}; nothing to fix")
        return 0
    needle = "..\\lite/"
    fixed_files = 0
    fixed_total = 0
    for p in ROOT.rglob("*.html"):
        s = p.read_text(encoding="utf-8")
        if needle not in s:
            continue
        new = s.replace(needle, "../lite/")
        if new != s:
            p.write_text(new, encoding="utf-8")
            fixed_files += 1
            fixed_total += s.count(needle)
    print(f"fixed {fixed_total} backslash-prefixed lite/ URLs across {fixed_files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
