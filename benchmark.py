import hashlib
import json
import os
import random
import re
import struct
from datetime import datetime
from typing import Literal

import pyarrow.parquet as pq
import torch
import dspy
from transformers import LogitsProcessor, LogitsProcessorList

import runtime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


MODELS = {
    "qwen3-4b-instruct":        "Qwen/Qwen3-4B-Instruct-2507",
    "phi-3-mini-4k-instruct":   "microsoft/Phi-3-mini-4k-instruct",
    "gemma-3-4b-it":            "google/gemma-3-4b-it",
    "llama-3.2-3b-instruct":    "meta-llama/Llama-3.2-3B-Instruct",
    "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "smollm2-1.7b-instruct":    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "granite-3.1-2b-instruct":  "ibm-granite/granite-3.1-2b-instruct",
    "yi-1.5-6b-chat":           "01-ai/Yi-1.5-6B-Chat",
    "falcon3-7b-instruct":      "tiiuae/Falcon3-7B-Instruct",
    "olmo-2-7b-instruct":       "allenai/OLMo-2-1124-7B-Instruct",
}

MODEL_REVISIONS = {
    "qwen3-4b-instruct":        "cdbee75f17c01a7cc42f958dc650907174af0554",
    "phi-3-mini-4k-instruct":   "f39ac1d28e925b323eae81227eaba4464caced4e",
    "gemma-3-4b-it":            "093f9f388b31de276ce2de164bdc2081324b9767",
    "llama-3.2-3b-instruct":    "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "mistral-7b-instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    "smollm2-1.7b-instruct":    "31b70e2e869a7173562077fd711b654946d38674",
    "granite-3.1-2b-instruct":  "bbc2aed595bd38bd770263dc3ab831db9794441d",
    "yi-1.5-6b-chat":           "771924d1c83d67527d665913415d7086f11ea9c0",
    "falcon3-7b-instruct":      "1e57a0ecd176c7c139f289c60a74e57f887c3dfb",
    "olmo-2-7b-instruct":       "470b1fba1ae01581f270116362ee4aa1b97f4c84",
}

CONDITION_NAMES = ["json", "text", "dspy"]


TERRAFORMA_REPO = "https://github.com/project-terraforma/terraforma-smolboard.git"
TERRAFORMA_BRANCH = "add-location-aware-golden-dataset"
TERRAFORMA_COMMIT = "c0e52a70a8e09f82b7643275130c1bc84366a24c"
GOLDEN_DATASET_SHA256 = "7e41de31b1e3ae0c98bcec89f74d4576a8a7ea9e77651da127a16d6966248235"
TERRAFORMA_DIR = os.path.join(PROJECT_DIR, "data", "terraforma-smolboard")
GOLDEN_DATASET_FILE = os.path.join(TERRAFORMA_DIR, "datasets", "geo", "micro_pairs.parquet")
GOLDEN_DATASET_URL = ("https://raw.githubusercontent.com/project-terraforma/terraforma-smolboard/"
                      f"{TERRAFORMA_COMMIT}/datasets/geo/micro_pairs.parquet")

FULL_RUN = True
N_PER_PAIR_TYPE = 15
SAMPLE_SEED = 42
SHOW_COORDINATES = False


MAX_NEW_TOKENS = 20
DO_SAMPLE = False

CHAT_TEMPLATE_DATE = datetime(2024, 7, 26)


DRY_RUN_N_EXAMPLES = 3


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_dataset():
    import httpx
    os.makedirs(os.path.dirname(GOLDEN_DATASET_FILE), exist_ok=True)
    part = GOLDEN_DATASET_FILE + ".part"
    try:
        with httpx.stream("GET", GOLDEN_DATASET_URL, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"could not download the golden dataset ({type(e).__name__}: {e}). Copy data/ over from a "
            f"machine that has it, or save {GOLDEN_DATASET_URL} as {GOLDEN_DATASET_FILE}."
        ) from e
    os.replace(part, GOLDEN_DATASET_FILE)


def ensure_dataset():
    if not os.path.exists(GOLDEN_DATASET_FILE):
        _download_dataset()
    digest = _sha256(GOLDEN_DATASET_FILE)
    if digest != GOLDEN_DATASET_SHA256:
        raise RuntimeError(
            f"{GOLDEN_DATASET_FILE} has sha256 {digest}, expected {GOLDEN_DATASET_SHA256} "
            f"(terraforma-smolboard @ {TERRAFORMA_COMMIT}). Results keyed by example_id would "
            f"no longer line up with the existing checkpoint. Delete that file to re-fetch it."
        )
    return GOLDEN_DATASET_FILE


def load_rows():
    return pq.read_table(ensure_dataset()).to_pylist()


def decode_lonlat(geom_bytes):
    _, _, lon, lat = struct.unpack("<BIdd", geom_bytes)
    return lon, lat


def flat_record(row, side):
    names = row.get(f"{side}_names") or {}
    addrs = row.get(f"{side}_addresses") or []
    addr = addrs[0] if addrs else {}
    brand = row.get(f"{side}_brand") or {}
    brand_name = (brand.get("names") or {}).get("primary") if brand else None
    phones = row.get(f"{side}_phones") or []
    websites = row.get(f"{side}_websites") or []
    d = {
        "name": names.get("primary"),
        "address": addr.get("freeform"),
        "city": addr.get("locality"),
        "state": addr.get("region"),
        "postal": addr.get("postcode"),
        "country": addr.get("country"),
        "phone": phones[0] if phones else None,
        "website": websites[0] if websites else None,
        "brand": brand_name,
        "category": row.get(f"{side}_basic_category"),
    }
    if SHOW_COORDINATES and row.get(f"{side}_geometry"):
        lon, lat = decode_lonlat(row[f"{side}_geometry"])
        d["lat"], d["lon"] = lat, lon
    return d


PDF_JSON_FIELDS = [
    "id", "geometry", "categories", "confidence", "websites", "emails", "socials", "phones",
    "brand", "addresses", "basic_category", "taxonomy", "operating_status", "version", "names",
    "sources", "provider", "corpus_state",
]


def raw_record(row, side):
    d = {}
    for field in PDF_JSON_FIELDS:
        if field == "id":
            continue
        if field == "geometry":
            if SHOW_COORDINATES and row.get(f"{side}_geometry"):
                lon, lat = decode_lonlat(row[f"{side}_geometry"])
                d["geometry"] = {"type": "Point", "coordinates": [lon, lat]}
            continue
        d[field] = row.get(f"{side}_{field}")
    return d


def build_dataset(all_rows, full_run, n_per_pair_type, seed):
    if full_run:
        chosen_rows = all_rows
    else:
        by_bucket = {"match": [], "hard_negative_geo": [], "hard_negative_brand": [], "easy_negative": []}
        for r in all_rows:
            key = f"hard_negative_{r['hard_neg_source']}" if r["pair_type"] == "hard_negative" else r["pair_type"]
            by_bucket.setdefault(key, []).append(r)
        rng = random.Random(seed)
        n_half = (n_per_pair_type + 1) // 2
        chosen_rows = (
            rng.sample(by_bucket["match"], min(n_per_pair_type, len(by_bucket["match"])))
            + rng.sample(by_bucket["hard_negative_geo"], min(n_half, len(by_bucket["hard_negative_geo"])))
            + rng.sample(by_bucket["hard_negative_brand"], min(n_per_pair_type - n_half, len(by_bucket["hard_negative_brand"])))
            + rng.sample(by_bucket["easy_negative"], min(n_per_pair_type, len(by_bucket["easy_negative"])))
        )
        rng.shuffle(chosen_rows)
    return [
        {
            "id": i,
            "label": "MATCH" if row["label"] == 1 else "NOT_MATCH",
            "pair_type": row["pair_type"],
            "hard_neg_source": row.get("hard_neg_source") or None,
            "_row": row,
        }
        for i, row in enumerate(chosen_rows)
    ]


def dry_run_slice(dataset, n_examples):
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(dataset, min(n_examples, len(dataset)))


PROMPT_VERSION = "v2"


def build_json_prompt(loc_a: dict, loc_b: dict) -> list:
    payload = {"location_a": loc_a, "location_b": loc_b}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    text += "\n\nAnswer with exactly: MATCH or NOT_MATCH"
    return [{"role": "user", "content": text}]


def build_text_prompt(loc_a: dict, loc_b: dict) -> list:
    def g(d, *keys):
        for k in keys:
            v = d.get(k)
            if v not in (None, ""):
                return v
        return ""

    def coord_line(d):
        if "lat" not in d and "lon" not in d:
            return ""
        return f'Coordinates: {g(d, "lat", "latitude")}, {g(d, "lon", "longitude")}\n'

    text = f"""Determine whether these two records describe the SAME physical location.

Record A:
Name: {g(loc_a, "name")}
Address: {g(loc_a, "address")}
City: {g(loc_a, "city")}
State/Region: {g(loc_a, "state", "region")}
Postal Code: {g(loc_a, "postal", "postal_code")}
Country: {g(loc_a, "country")}
{coord_line(loc_a)}
Record B:
Name: {g(loc_b, "name")}
Address: {g(loc_b, "address")}
City: {g(loc_b, "city")}
State/Region: {g(loc_b, "state", "region")}
Postal Code: {g(loc_b, "postal", "postal_code")}
Country: {g(loc_b, "country")}
{coord_line(loc_b)}
Rules:

● Formatting differences and abbreviations do not necessarily mean the records are different.
● Small spelling differences do not necessarily mean the records are different.
● Locations with similar names are NOT a match if they represent different branches or physical places.
● Conflicting cities, addresses, building numbers, or coordinates are strong evidence of NOT MATCH.
● Missing information alone is not evidence of NOT MATCH.

Answer with exactly: MATCH or NOT_MATCH"""
    return [{"role": "user", "content": text}]


class LocationEntityResolution(dspy.Signature):

    location_a: str = dspy.InputField(
        desc="JSON record for the first location: name, address, city/region, "
             "postal code, country, and if present phone, website, brand, category."
    )
    location_b: str = dspy.InputField(
        desc="JSON record for the second location, same field set as location_a."
    )
    match: Literal["MATCH", "NOT_MATCH"] = dspy.OutputField(
        desc="MATCH if location_a and location_b are the same physical location, else NOT_MATCH."
    )


dspy_predict = dspy.Predict(LocationEntityResolution)

DSPY_ADAPTER = dspy.ChatAdapter(use_json_adapter_fallback=False)


class LocalHFEngine(dspy.BaseLM):

    forward_contract = "typed_lm"

    def __init__(self, hf_model, hf_tokenizer, label):
        super().__init__(model=label, cache=False, num_retries=0)
        self._hf_model = hf_model
        self._hf_tokenizer = hf_tokenizer
        self.last_raw = None

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        messages = [{"role": m.role, "content": m.text or ""} for m in request.messages]
        self.last_raw = generate(self._hf_model, self._hf_tokenizer, messages)
        return dspy.LMResponse.from_text(self.last_raw, model=request.model)


def _predict(lm, loc_a: dict, loc_b: dict):
    with dspy.context(lm=lm, adapter=DSPY_ADAPTER, disable_history=True):
        dspy_predict(
            location_a=json.dumps(loc_a, ensure_ascii=False),
            location_b=json.dumps(loc_b, ensure_ascii=False),
        )


def dspy_condition_raw(hf_model, hf_tokenizer, short_name, loc_a: dict, loc_b: dict) -> str:
    engine = LocalHFEngine(hf_model, hf_tokenizer, f"local/{short_name}")
    try:
        _predict(engine, loc_a, loc_b)
    except Exception:
        if engine.last_raw is None:
            raise
    return engine.last_raw or ""


PROMPT_SHA256 = {
    "json": "2b7413a66ee782bd1fef0cf9129011962480efb18652a2ea97ccdf9d0604b94f",
    "text": "30119a6679ae01eadb84c26c4f09ef685efc9302e3330e45da19aa327a3e3918",
    "dspy": "68001c63bf2d01a077592dae28bd0cdfffcb4293eb2d51fa485061b0dc34a488",
}


class _CaptureLM(dspy.BaseLM):
    """Records the messages DSPy sends, without a model."""

    forward_contract = "typed_lm"

    def __init__(self):
        super().__init__(model="capture", cache=False, num_retries=0)
        self.messages = []

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.messages.append([{"role": m.role, "content": m.text or ""} for m in request.messages])
        return dspy.LMResponse.from_text("[[ ## match ## ]]\nMATCH\n\n[[ ## completed ## ]]", model=request.model)


def prompt_fingerprints(dataset) -> dict:
    digests = {name: hashlib.sha256() for name in CONDITION_NAMES}
    for example in dataset:
        row = example["_row"]
        lm = _CaptureLM()
        _predict(lm, flat_record(row, "left"), flat_record(row, "right"))
        if len(lm.messages) != 1:
            raise RuntimeError(f"DSPy issued {len(lm.messages)} LM calls for one example; expected exactly 1")
        messages = {
            "json": build_json_prompt(raw_record(row, "left"), raw_record(row, "right")),
            "text": build_text_prompt(flat_record(row, "left"), flat_record(row, "right")),
            "dspy": lm.messages[0],
        }
        for name, msgs in messages.items():
            digests[name].update(json.dumps(msgs, ensure_ascii=False).encode("utf-8"))
    return {name: d.hexdigest() for name, d in digests.items()}


_NEGATIVE = re.compile(r"\b(?:NOT|NO|NON)[\s_\-]*(?:A\s+)?MATCH\b", re.IGNORECASE)
_POSITIVE = re.compile(r"\bMATCH\b", re.IGNORECASE)


def parse_match(raw_text: str) -> str:
    """Extract MATCH / NOT_MATCH from raw model output. 'UNPARSEABLE' if neither is found.
    A negative form anywhere in the reply wins over a bare MATCH, as before."""
    if not raw_text:
        return "UNPARSEABLE"
    cleaned = raw_text.strip()
    if _NEGATIVE.search(cleaned):
        return "NOT_MATCH"
    if _POSITIVE.search(cleaned):
        return "MATCH"
    return "UNPARSEABLE"


def _pinned_strftime(fmt):
    return CHAT_TEMPLATE_DATE.strftime(fmt)


def chat_template_kwargs():
    return {} if CHAT_TEMPLATE_DATE is None else {"strftime_now": _pinned_strftime}


class NonFiniteLogitsError(RuntimeError):
    """The forward pass produced NaN/inf logits: a numerical failure, not a model answer."""


class _NonFiniteLogitsGuard(LogitsProcessor):
    """Read-only: records whether any step's logits were NaN or +inf and returns them unchanged,
    so the generated tokens are exactly what they would be without it. The flag stays on the
    device and is read once after generation -- a per-step read would force a host sync every
    token, which transformers deliberately avoids on MPS."""

    def __init__(self):
        self.flag = None

    def __call__(self, input_ids, scores):
        bad = torch.isnan(scores).any() | torch.isposinf(scores).any()
        self.flag = bad if self.flag is None else self.flag | bad
        return scores

    def tripped(self):
        return self.flag is not None and bool(self.flag)


_last_hit_token_cap = False


def last_generation_hit_token_cap() -> bool:
    return _last_hit_token_cap


def generate(model, tokenizer, messages) -> str:
    global _last_hit_token_cap
    _last_hit_token_cap = False
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           **chat_template_kwargs())
    bos = tokenizer.bos_token
    add_special_tokens = not (bos and prompt.startswith(bos))
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(model.device)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [] if eos_ids is None else ([eos_ids] if isinstance(eos_ids, int) else list(eos_ids))
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)
    guard = _NonFiniteLogitsGuard()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids or None,
                logits_processor=LogitsProcessorList([guard]),
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            runtime.empty_cache()
        raise
    if guard.tripped():
        raise NonFiniteLogitsError(
            f"NaN/inf logits with {str(model.dtype).replace('torch.', '')} compute (numerical "
            f"overflow, not a model answer). Run this model with --dtype bfloat16, or --dtype "
            f"float32 on GPUs without bf16 support (e.g. T4, V100)."
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    ids = new_tokens.tolist()
    _last_hit_token_cap = len(ids) >= MAX_NEW_TOKENS and not any(t in eos_ids for t in ids)
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_json(model, tokenizer, short_name, example):
    loc_a = raw_record(example["_row"], "left")
    loc_b = raw_record(example["_row"], "right")
    return generate(model, tokenizer, build_json_prompt(loc_a, loc_b))


def run_text(model, tokenizer, short_name, example):
    loc_a = flat_record(example["_row"], "left")
    loc_b = flat_record(example["_row"], "right")
    return generate(model, tokenizer, build_text_prompt(loc_a, loc_b))


def run_dspy(model, tokenizer, short_name, example):
    loc_a = flat_record(example["_row"], "left")
    loc_b = flat_record(example["_row"], "right")
    return dspy_condition_raw(model, tokenizer, short_name, loc_a, loc_b)


CONDITIONS = {"json": run_json, "text": run_text, "dspy": run_dspy}
