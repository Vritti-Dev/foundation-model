"""Hash-only answer checking.

Client-side test code is always visible to the learner (it runs in their
browser), so we never ship the plaintext expected answer. Instead we ship the
SHA-256 of a canonical representation and compare hashes. This is honest
self-paced gating, NOT anti-cheat (a determined learner can read the hashed
value but not invert it cheaply; the eventual LMS must re-grade server-side).
"""

from __future__ import annotations

import hashlib
import json


def _canonical(value) -> str:
    """Deterministic string form of a value, stable across runs/processes."""
    if isinstance(value, str):
        return "s:" + value
    if isinstance(value, bool):
        return "b:" + str(value)
    if isinstance(value, int):
        return "i:" + str(value)
    if isinstance(value, float):
        # Round to avoid float-repr drift; hash-checks are for discrete answers.
        return "f:" + format(round(value, 6), ".6f")
    if isinstance(value, (list, tuple)):
        return "l:" + json.dumps([_canonical(v) for v in value], separators=(",", ":"))
    if isinstance(value, dict):
        items = sorted((str(k), _canonical(v)) for k, v in value.items())
        return "d:" + json.dumps(items, separators=(",", ":"))
    return "r:" + repr(value)


def make_hash(value) -> str:
    """SHA-256 hex digest of the canonical form of ``value``."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def hash_check(value, stored_hash: str) -> bool:
    """True iff ``value`` hashes to ``stored_hash``."""
    return make_hash(value) == stored_hash
