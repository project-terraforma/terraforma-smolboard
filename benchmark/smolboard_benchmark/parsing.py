"""Centralized, deliberately strict response parsing."""

from __future__ import annotations

import re
from typing import Any

from .types import Prediction


_LABEL_PATTERN = re.compile(
    r"^(?:ANSWER\s*:\s*)?(MATCH|NO[\s_-]?MATCH)\s*[.!]?\s*$", re.IGNORECASE
)


def parse_prediction(raw_output: Any) -> Prediction:
    """Accept only a single standalone label and benign presentation formatting."""
    if not isinstance(raw_output, str):
        return Prediction.INVALID
    cleaned = raw_output.strip().strip('`"\'').strip()
    match = _LABEL_PATTERN.fullmatch(cleaned)
    if not match:
        return Prediction.INVALID
    normalized = re.sub(r"[\s-]", "_", match.group(1).upper())
    return Prediction.NO_MATCH if normalized == "NO_MATCH" else Prediction.MATCH
