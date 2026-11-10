"""Configuration for multi-model classification."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets" / "geo"
RESULTS_DIR = DATASETS_DIR / "multi_model_results"

LABELS = {
    "parent_child": "one record is a parent/child of the other (e.g. a store inside a mall/airport/campus, a department of an institution, a building and its tenant)",
    "same_name_different_address": "the names are the same or nearly the same but the addresses clearly differ (e.g. chain branches)",
    "typo_differences": "the places look the same but differ by typos, spelling, abbreviations, casing, punctuation or formatting",
    "incomplete_data": "one or both records are missing key fields (name, address, etc.) so the pair cannot be judged well",
    "unsolvable_conflicting": "the evidence conflicts or is ambiguous so even a human could not decide reliably",
    "languages": "the records differ mainly because of language or script (translation/transliteration)",
}
FALLBACK = "other"

SHOWN_FIELDS = ["names", "addresses", "websites", "phones", "emails", "socials", "brand", "categories"]

PROMPT_TEMPLATE = """You are analysing pairs of place records used for entity resolution.
Pick the single best label describing the relationship/difficulty of this pair.

Labels:
{label_desc}
- {fallback}: none of the above fit

Place A:
{a}

Place B:
{b}

Reply with only the label name."""

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

@dataclass
class ClassificationConfig:
    """Configuration for a classification run."""
    model_names: list[str]
    batch_size: int = 32
    max_workers: int = 1
    temperature: float = 0.0
    max_tokens: int = 16
    device: str = "cuda"
    use_int8: bool = True
    limit: int | None = None
    resume: bool = True
