from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

_langfuse: Any | None = None
_initialized: bool = False


def _get_client() -> Any | None:
    global _langfuse, _initialized
    if _initialized:
        return _langfuse
    _initialized = True
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse import get_client

        _langfuse = get_client()
    except Exception:
        logger.warning("Langfuse init failed — tracing disabled", exc_info=True)
    return _langfuse


class _NoopObs:
    def update(self, **_: Any) -> None:
        pass


@contextmanager
def observe_span(*, name: str) -> Generator[Any, None, None]:
    lf = _get_client()
    if lf is None:
        yield _NoopObs()
        return
    try:
        cm = lf.start_as_current_observation(name=name, as_type="span")
    except Exception:
        logger.warning("Langfuse trace failed for '%s'", name, exc_info=True)
        yield _NoopObs()
        return
    with cm as span:
        yield span
    try:
        lf.flush()
    except Exception:
        logger.warning("Langfuse flush failed for '%s'", name, exc_info=True)


@contextmanager
def observe_generation(
    *,
    name: str,
    model: str,
    input: Any,
    metadata: dict | None = None,
) -> Generator[Any, None, None]:
    lf = _get_client()
    if lf is None:
        yield _NoopObs()
        return
    try:
        cm = lf.start_as_current_observation(
            name=name,
            as_type="generation",
            model=model,
            input=input,
            metadata=metadata or {},
        )
    except Exception:
        logger.warning("Langfuse trace failed for '%s'", name, exc_info=True)
        yield _NoopObs()
        return
    with cm as gen:
        yield gen
    try:
        lf.flush()
    except Exception:
        logger.warning("Langfuse flush failed for '%s'", name, exc_info=True)
