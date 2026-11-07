import pytest

from smolboard_benchmark.prompts import build_prompt, load_prompt, serialize_record
from smolboard_benchmark.types import ConflationExample, Label, PlaceRecord


def prompt_config(**overrides):
    return {"prompts": {"test": {
        "version": "1", "template": "Place A:\n{record_a}\nPlace B:\n{record_b}",
        "exposed_fields": ["names", "addresses", "phones"], **overrides,
    }}}


def test_text_renders_nested_and_json_encoded_fields_without_metadata():
    example = ConflationExample(
        example_id="secret_pair_id", label=Label.MATCH,
        record_a=PlaceRecord({
            "names": '{"primary":"Café A","common":{"fr":"Café français"}}',
            "addresses": [
                {"freeform": "123 Main St", "locality": "Seattle"},
                {"freeform": "Suite 2", "country": "US"},
            ],
            "phones": '["+1 555 0100", "+1 555 0101"]',
            "id": "secret_record_id", "geometry": "secret_coordinates",
            "llm_pair_type": "secret_pair_type", "label": "secret_label",
        }),
        record_b=PlaceRecord({"names": {"primary": "Cafe A"}}),
        metadata={"llm_pair_type": "secret_metadata"},
    )
    prompt = load_prompt("test", prompt_config(record_format="text"))
    rendered = build_prompt(example, prompt)
    assert "Names:\n  Primary: Café A\n  Common:\n    Fr: Café français" in rendered
    assert "Addresses:\n  - Freeform: 123 Main St\n    Locality: Seattle" in rendered
    assert "  - Freeform: Suite 2\n    Country: US" in rendered
    assert "Phones:\n  - +1 555 0100\n  - +1 555 0101" in rendered
    assert "Place B:\nNames:\n  Primary: Cafe A\nAddresses: Not provided\nPhones: Not provided" in rendered
    assert "secret_" not in rendered
    assert "{" not in rendered and "[" not in rendered


def test_text_marks_empty_fields_and_keeps_source_newlines_within_value():
    rendered = serialize_record({
        "names": None, "websites": [], "addresses": {},
        "phones": float("nan"), "emails": "  ", "socials": "first\nsecond",
    }, ("names", "websites", "addresses", "phones", "emails", "socials"), "text")
    assert rendered.count("Not provided") == 5
    assert "Socials: first second" in rendered


def test_unspecified_format_preserves_json_rendering():
    prompt = load_prompt("test", prompt_config())
    assert prompt.record_format == "json"
    assert serialize_record({"names": '{"primary":"A"}'}, ("names",)) == '{"names": {"primary": "A"}}'


def test_fingerprint_covers_format_fields_and_template():
    base = load_prompt("test", prompt_config())
    variants = [
        {"record_format": "text"},
        {"exposed_fields": ["names"]},
        {"exposed_fields": ["phones", "addresses", "names"]},
        {"template": "A={record_a}; B={record_b}"},
    ]
    for variant in variants:
        assert load_prompt("test", prompt_config(**variant)).sha256 != base.sha256


def test_unknown_format_fails_before_inference():
    with pytest.raises(ValueError, match="record_format"):
        load_prompt("test", prompt_config(record_format="xml"))
