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
    record_format = str(prompt.get("record_format", "json"))
    if record_format not in {"json", "text"}:
        raise ValueError(f"Unsupported prompt record_format: {record_format!r}")
    # Include rendering choices so a checkpoint cannot mix different model inputs.
    contract = json.dumps({
        "template": template, "exposed_fields": fields,
        "record_format": record_format, "renderer_version": "1",
    }, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(contract.encode("utf-8")).hexdigest()
    return PromptSpec(
        name=name,
        version=str(prompt["version"]),
        template=template,
        exposed_fields=fields,
        sha256=digest,
        record_format=record_format,
    )


def build_prompt(example: ConflationExample, prompt: PromptSpec) -> str:
    return prompt.template.format(
        record_a=serialize_record(example.record_a.fields, prompt.exposed_fields, prompt.record_format),
        record_b=serialize_record(example.record_b.fields, prompt.exposed_fields, prompt.record_format),
    )


def serialize_record(
    fields: Mapping[str, Any], exposed_fields: tuple[str, ...], record_format: str = "json"
) -> str:
    visible = {field: normalize_value(fields.get(field)) for field in exposed_fields}
    if record_format == "json":
        return json.dumps(visible, ensure_ascii=False, separators=(", ", ": "))
    if record_format == "text":
        return "\n".join(_text_lines(visible))
    raise ValueError(f"Unsupported prompt record_format: {record_format!r}")


def _text_lines(value: Any, indent: str = "") -> list[str]:
    value = normalize_value(value)
    if value is None or value == {} or value == []:
        return [f"{indent}Not provided"]
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            label = str(key).replace("_", " ").capitalize()
            item = normalize_value(item)
            if isinstance(item, (Mapping, list, tuple)) and item:
                lines.append(f"{indent}{label}:")
                lines.extend(_text_lines(item, indent + "  "))
            else:
                lines.append(f"{indent}{label}: {_text_lines(item)[0]}")
        return lines
    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            rendered = _text_lines(item, indent + "  ")
            lines.append(f"{indent}- {rendered[0][len(indent) + 2:]}")
            lines.extend(rendered[1:])
        return lines
    # Keep a field value on its own line, including source strings with newlines.
    return [f"{indent}{' '.join(str(value).split())}"]


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
