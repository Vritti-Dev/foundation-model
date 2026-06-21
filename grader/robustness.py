"""Safe execution of student-supplied callables.

A student's function may raise, return the wrong shape/type, return ``None``,
or loop forever. The grader must turn each of those into a clean, categorized
failure rather than crashing the grading pipeline.

Timeout note: Python (like Pyodide's single Web Worker in the browser) cannot
forcibly kill a CPU-bound thread. We run the callable in a daemon thread and
``join`` with a timeout; on timeout we report a ``timeout`` failure and let the
orphan thread die with the process. In the browser the equivalent recovery is
terminating and restarting the Web Worker (see book/_static/gating.js).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class SafeResult:
    ok: bool
    value: object = None
    category: str = "ok"  # ok|error|shape|timeout|value
    message: str = ""


def safe_call(fn, args=(), kwargs=None, timeout_s: float = 2.0, validator=None) -> SafeResult:
    """Run ``fn(*args, **kwargs)`` defensively.

    - exception        -> SafeResult(ok=False, category='error')
    - exceeds timeout   -> SafeResult(ok=False, category='timeout')
    - ``validator`` (called on the return value) returns False
                        -> SafeResult(ok=False, category='shape')
    - otherwise         -> SafeResult(ok=True, value=<return>)
    """
    kwargs = kwargs or {}
    box: dict = {}

    def _run():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - student code may raise anything
            box["error"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return SafeResult(ok=False, category="timeout", message=f"exceeded {timeout_s}s")
    if "error" in box:
        exc = box["error"]
        return SafeResult(ok=False, category="error", message=f"{type(exc).__name__}: {exc}")
    value = box.get("value")
    if validator is not None:
        try:
            valid = bool(validator(value))
        except BaseException as exc:  # noqa: BLE001
            return SafeResult(ok=False, category="shape", message=f"validator raised: {exc}")
        if not valid:
            return SafeResult(ok=False, category="shape", message="validator rejected return value")
    return SafeResult(ok=True, value=value)
