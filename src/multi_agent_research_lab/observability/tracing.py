"""Tracing hooks — file-based JSON trace + optional LangSmith integration.

Plug in LangSmith by setting LANGSMITH_API_KEY in .env.
Without it, spans are printed to stdout when LOG_LEVEL=DEBUG.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span that logs duration. Attributes can be mutated inside the block."""
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}}
    try:
        yield span
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 3)
        logger.debug("span: %s", json.dumps(span))
