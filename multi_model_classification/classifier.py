"""Base classifier and HuggingFace implementation."""

from __future__ import annotations

import json
import importlib.util
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import FALLBACK, ClassificationConfig, PROMPT_TEMPLATE
from .utils import (
    build_label_descriptions,
    compute_statistics,
    get_checkpoint_path,
    get_output_parquet_path,
    get_stats_path,
    load_checkpoint,
    parse_label,
    prompt_hash,
    render_record,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


class LLMClassifier(ABC):
    """Abstract base class for LLM classifiers."""

    def __init__(self, model_name: str, config: ClassificationConfig):
        self.model_name = model_name
        self.config = config
        if not config.resume:
            checkpoint_path = get_checkpoint_path(model_name)
            if checkpoint_path.exists():
                checkpoint_path.write_text("", encoding="utf-8")
        self.checkpoint = load_checkpoint(model_name) if config.resume else {}

    @abstractmethod
    def classify_single(self, prompt: str) -> str:
        """Classify a single pair. Returns the label string."""
        pass

    def classify_pair(self, pair_id: str, row: pd.Series) -> str:
        """Classify a pair and save checkpoint."""
        a_text = render_record(row, "left")
        b_text = render_record(row, "right")

        label_desc = build_label_descriptions()
        prompt = PROMPT_TEMPLATE.format(
            label_desc=label_desc,
            fallback=FALLBACK,
            a=a_text,
            b=b_text,
        )
        digest = prompt_hash(json.dumps({
            "prompt": prompt,
            "model": self.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "device": self.config.device,
            "use_int8": self.config.use_int8,
        }, sort_keys=True))
        if pair_id in self.checkpoint:
            record = self.checkpoint[pair_id]
            if record["prompt_hash"] != digest:
                raise ValueError(f"Checkpoint prompt mismatch for {self.model_name}, pair {pair_id}; use --no-resume to restart this model.")
            return record["llm_pair_type"]

        raw_output = self.classify_single(prompt)
        classification = parse_label(raw_output)

        save_checkpoint(self.model_name, pair_id, classification, digest, raw_output)
        self.checkpoint[pair_id] = {"llm_pair_type": classification, "prompt_hash": digest}

        return classification

    def run(self, input_parquet: Path) -> Path:
        """Run classification on all pairs and save results."""
        logger.info(f"Starting classification with {self.model_name}")

        df = pd.read_parquet(input_parquet)
        if "pair_id" not in df or df["pair_id"].isna().any() or df["pair_id"].astype(str).duplicated().any():
            raise ValueError("Input must have unique, non-null pair_id values")
        if self.config.limit:
            df = df.head(self.config.limit)
            logger.info(f"Limited to {self.config.limit} pairs")

        results = []
        for idx, row in df.iterrows():
            pair_id = str(row.get("pair_id", idx))
            classification = self.classify_pair(pair_id, row)
            results.append({**dict(row), "llm_pair_type": classification})

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} pairs")

        result_df = pd.DataFrame(results)
        output_path = get_output_parquet_path(self.model_name)
        result_df.to_parquet(output_path)
        logger.info(f"Saved results to {output_path}")

        stats = compute_statistics(output_path)
        stats_path = get_stats_path(self.model_name)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")

        return output_path


class HuggingFaceClassifier(LLMClassifier):
    """Classifier using HuggingFace transformers."""

    def __init__(self, model_name: str, config: ClassificationConfig):
        super().__init__(model_name, config)
        logger.info(f"Loading model {model_name}")

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this PyTorch installation; use --device cpu or install a CUDA-enabled PyTorch build.")
        if config.device == "cuda" and config.use_int8 and importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("CUDA int8 inference requires bitsandbytes. Install it or use --no-int8 (requires more GPU memory).")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        if self.tokenizer.chat_template is None:
            raise ValueError(f"Model {model_name} does not provide a chat template")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {
            "device_map": config.device,
            "dtype": torch.float16 if config.device == "cuda" else torch.float32,
        }
        if config.use_int8 and config.device == "cuda":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )
        self.model.eval()

    def classify_single(self, prompt: str) -> str:
        """Classify using the loaded model."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response
