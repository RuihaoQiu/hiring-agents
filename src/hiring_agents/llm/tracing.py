from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
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


def create_trace_id() -> str | None:
    lf = _get_client()
    if lf is None:
        return None
    return lf.create_trace_id()


def get_langchain_handler(*, trace_id: str | None = None) -> Any | None:
    lf = _get_client()
    if lf is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        ctx = {"trace_id": trace_id} if trace_id else None
        return CallbackHandler(trace_context=ctx)
    except Exception:
        logger.warning("Langfuse LangChain handler init failed", exc_info=True)
        return None


class _NoopObs:
    def update(self, **_: Any) -> None:
        pass


@contextmanager
def observe_trace(
    *,
    name: str,
    session_id: str | None = None,
    trace_id: str | None = None,
    metadata: dict | None = None,
) -> Generator[Any, None, None]:
    lf = _get_client()
    if lf is None:
        yield _NoopObs()
        return
    try:
        cm = lf.start_as_current_observation(
            name=name,
            as_type="span",
            trace_context={"trace_id": trace_id} if trace_id else None,
            metadata=metadata or {},
        )
    except Exception:
        logger.warning("Langfuse trace failed for '%s'", name, exc_info=True)
        yield _NoopObs()
        return
    prop = _prop_ctx(session_id)
    with prop:
        with cm as span:
            yield span
    try:
        lf.flush()
    except Exception:
        logger.warning("Langfuse flush failed for '%s'", name, exc_info=True)


def update_current_span(**kwargs: Any) -> None:
    lf = _get_client()
    if lf is None:
        return
    try:
        from opentelemetry import trace as otel_trace_api

        if otel_trace_api.get_current_span() is otel_trace_api.INVALID_SPAN:
            return
        lf.update_current_span(**kwargs)
    except Exception:
        pass


def _prop_ctx(session_id: str | None) -> Any:
    if not session_id:
        return nullcontext()
    try:
        from langfuse import propagate_attributes

        return propagate_attributes(session_id=session_id)
    except Exception:
        return nullcontext()


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
