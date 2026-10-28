"""Label each pair in a copy of medium_pairs.parquet with an LLM-assigned case type.

Talks to any OpenAI-compatible chat endpoint (e.g. llama-server from the benchmark,
default http://127.0.0.1:8192). Results are checkpointed to a JSONL file so the run
can be resumed. Writes a parquet copy with a new `llm_pair_type` column and a
pairtype_stats.json summary. The source parquet is never modified.

Example:
    .venv\\Scripts\\python golden_dataset_formation\\classify_pair_types.py --limit 200
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from collections import Counter
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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

PROMPT = """You are analysing pairs of place records used for entity resolution.
Pick the single best label describing the relationship/difficulty of this pair.

Labels:
{label_desc}
- {fallback}: none of the above fit

Place A:
{a}

Place B:
{b}

Reply with only the label name."""


def render(row: pd.Series, side: str) -> str:
    out = {}
    for f in SHOWN_FIELDS:
        v = row.get(f"{side}_{f}")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if hasattr(v, "tolist"):
            v = v.tolist()
        out[f] = v
    return json.dumps(out, ensure_ascii=False, default=str)[:1500]


def parse_label(text: str | None) -> str:
    if not text:
        return FALLBACK
    t = text.strip().lower()
    # prefer an exact/first-occurring label name in the reply
    hits = [(m.start(), k) for k in LABELS for m in [re.search(re.escape(k), t)] if m]
    return min(hits)[1] if hits else FALLBACK


def classify(client: httpx.Client, model: str, prompt: str, retries: int = 3) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
    }
    for attempt in range(retries):
        try:
            r = client.post("/v1/chat/completions", json=payload)
            r.raise_for_status()
            return parse_label(r.json()["choices"][0]["message"].get("content"))
        except (httpx.HTTPError, KeyError, ValueError):
            time.sleep(attempt + 1)
    return "error"


class HFClassifier:
    """Runs a Hugging Face causal LM in-process (no server)."""

    def __init__(self, model_id: str, batch_size: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.batch_size = torch, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        ).eval()

    def classify_batch(self, prompts: list[str]) -> list[str]:
        try:
            return self._classify(prompts)
        except (self.torch.cuda.OutOfMemoryError, self.torch.AcceleratorError) as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, self.torch.cuda.OutOfMemoryError):
                raise
            self.torch.cuda.empty_cache()
            if len(prompts) == 1:
                return ["error"]
            mid = len(prompts) // 2
            return self.classify_batch(prompts[:mid]) + self.classify_batch(prompts[mid:])

    def _classify(self, prompts: list[str]) -> list[str]:
        texts = [
            self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=self.tok.pad_token_id)
        replies = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [parse_label(r) for r in replies]


def build_stats(df: pd.DataFrame, col: str) -> dict:
    total = len(df)
    counts = Counter(df[col])
    stats = {
        "total_pairs": total,
        "counts": {k: counts.get(k, 0) for k in [*LABELS, FALLBACK, "error"]},
        "percentages": {k: round(100 * counts.get(k, 0) / total, 2) if total else 0.0 for k in [*LABELS, FALLBACK, "error"]},
    }
    for src in ("label", "pair_type"):
        if src in df.columns:
            stats[f"by_{src}"] = {
                str(k): {str(t): int(n) for t, n in g[col].value_counts().items()}
                for k, g in df.groupby(src)
            }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=ROOT / "datasets/geo/medium_pairs.parquet")
    ap.add_argument("--output", type=Path, default=ROOT / "datasets/geo/medium_pairs_typed.parquet")
    ap.add_argument("--stats", type=Path, default=ROOT / "datasets/geo/pairtype_stats.json")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "datasets/geo/pairtype_checkpoint.jsonl")
    ap.add_argument("--hf-model", default=None, help="Hugging Face model id to run in-process (e.g. Qwen/Qwen3-4B-Instruct-2507); omit to use a server")
    ap.add_argument("--batch-size", type=int, default=8, help="batch size for --hf-model")
    ap.add_argument("--base-url", default="http://127.0.0.1:8192")
    ap.add_argument("--model", default="local")
    ap.add_argument("--workers", type=int, default=4, help="match the server's --parallel slots")
    ap.add_argument("--limit", type=int, default=None, help="only classify the first N pairs")
    ap.add_argument("--column", default="llm_pair_type", help="new column (pair_type already exists in the source)")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    done: dict[str, str] = {}
    if args.checkpoint.exists():
        for line in args.checkpoint.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["pair_id"]] = rec["type"]
    todo = [i for i in df.index if str(df.at[i, "pair_id"]) not in done]
    print(f"{len(df)} pairs, {len(done)} already done, {len(todo)} to classify")

    label_desc = "\n".join(f"- {k}: {v}" for k, v in LABELS.items())
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def make_prompt(i: int) -> str:
        row = df.loc[i]
        return PROMPT.format(label_desc=label_desc, fallback=FALLBACK, a=render(row, "left"), b=render(row, "right"))

    def results():
        if args.hf_model:
            hf = HFClassifier(args.hf_model, args.batch_size)
            for s in range(0, len(todo), args.batch_size):
                chunk = todo[s : s + args.batch_size]
                yield from zip((str(df.at[i, "pair_id"]) for i in chunk), hf.classify_batch([make_prompt(i) for i in chunk]))
        else:
            with httpx.Client(base_url=args.base_url, timeout=180) as client, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                yield from ex.map(lambda i: (str(df.at[i, "pair_id"]), classify(client, args.model, make_prompt(i))), todo)

    with args.checkpoint.open("a", encoding="utf-8") as ckpt:
        for n, (pid, label) in enumerate(results(), 1):
            done[pid] = label
            ckpt.write(json.dumps({"pair_id": pid, "type": label}) + "\n")
            ckpt.flush()
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}")

    df[args.column] = df["pair_id"].astype(str).map(done)
    df.to_parquet(args.output, index=False)
    stats = build_stats(df, args.column)
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats["counts"], indent=2))
    print(f"wrote {args.output} and {args.stats}")


if __name__ == "__main__":
    main()
