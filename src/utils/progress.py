"""Optional progress callback for live API traces.

The HTTP layer binds a callback on the worker thread; retrieve / agent / agentic
emit events. Missing callback is a no-op so CLI and benchmarks are unchanged.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional

_tls = threading.local()


def emit(stage: str, title: str, preview: str = "", **extra: Any) -> None:
    cb = getattr(_tls, "cb", None)
    if not callable(cb):
        return
    text = str(preview or "")
    if len(text) > 480:
        text = text[:480] + "…"
    ev = {
        "type": "step",
        "stage": str(stage or ""),
        "title": str(title or ""),
        "preview": text,
    }
    ev.update(extra)
    try:
        cb(ev)
    except Exception:
        pass


@contextmanager
def bind(cb: Optional[Callable[[dict], None]]):
    prev = getattr(_tls, "cb", None)
    _tls.cb = cb
    try:
        yield
    finally:
        _tls.cb = prev
