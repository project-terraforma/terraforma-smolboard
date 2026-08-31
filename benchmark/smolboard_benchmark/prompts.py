"""Prompt selection and safe record rendering."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from .types import ConflationExample, PromptSpec


def load_prompt(name: str, config: Mapping[str, Any]) -> PromptSpec:
    prompt = config["prompts"][name]
    template = str(prompt["template"])
    fields = tuple(str(field) for field in prompt["exposed_fields"])
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return PromptSpec(
        name=name,
        version=str(prompt["version"]),
        template=template,
        exposed_fields=fields,
        sha256=digest,
    )


def build_prompt(example: ConflationExample, prompt: PromptSpec) -> str:
    return prompt.template.format(
        record_a=serialize_record(example.record_a.fields, prompt.exposed_fields),
        record_b=serialize_record(example.record_b.fields, prompt.exposed_fields),
    )


def serialize_record(fields: Mapping[str, Any], exposed_fields: tuple[str, ...]) -> str:
    visible = {field: normalize_value(fields.get(field)) for field in exposed_fields}
    return json.dumps(visible, ensure_ascii=False, separators=(", ", ": "))


def normalize_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return value
