import json
import re
import time
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== CONFIGURATION ==========
DATA_PATH = "data/samples_3k_project_c_updated.parquet"
MAX_NEW_TOKENS = 50          # Increased to allow longer answers
MAX_SAMPLES = 500            # Adjust if needed
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(42)

# 10 small models (Gemma removed; added others)
MODEL_IDS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "stabilityai/stablelm-2-zephyr-1_6b",
    "microsoft/phi-1_5",
]

# ========== HELPER FUNCTIONS ==========
def clean_field(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)

def build_prompt(row, prompt_type):
    base = {
        "names": clean_field(row.get("base_names")),
        "categories": clean_field(row.get("base_categories")),
        "addresses": clean_field(row.get("base_addresses")),
        "phones": clean_field(row.get("base_phones")),
        "websites": clean_field(row.get("base_websites")),
        "brand": clean_field(row.get("base_brand")),
    }
    candidate = {
        "names": clean_field(row.get("names")),
        "categories": clean_field(row.get("categories")),
        "addresses": clean_field(row.get("addresses")),
        "phones": clean_field(row.get("phones")),
        "websites": clean_field(row.get("websites")),
        "brand": clean_field(row.get("brand")),
    }

    if prompt_type == "plain_text":
        return (
            "You are a location-matching assistant. "
            "Decide whether the two records describe the same real-world place.\n\n"
            f"Base record:\n{json.dumps(base, ensure_ascii=False, indent=2)}\n\n"
            f"Candidate record:\n{json.dumps(candidate, ensure_ascii=False, indent=2)}\n\n"
            "Answer only with 1 if they are the same place, or 0 if they are not."
        )
    elif prompt_type == "structured_instructions":
        return (
            "Determine whether these two location records refer to the same real-world place.\n\n"
            "Consider the following fields for both records:\n"
            "- Names (exact or similar)\n"
            "- Categories (e.g., restaurant, cafe)\n"
            "- Addresses (street, city, postcode)\n"
            "- Phones\n"
            "- Websites\n"
            "- Brand\n\n"
            "Do not require every field to match. Use your best judgment.\n\n"
            f"Base record:\n{json.dumps(base, ensure_ascii=False, indent=2)}\n\n"
            f"Candidate record:\n{json.dumps(candidate, ensure_ascii=False, indent=2)}\n\n"
            "Return 1 if they most likely refer to the same place, otherwise return 0.\n"
            "Answer with only the number 1 or 0."
        )
    elif prompt_type == "json_output":
        return (
            "Determine whether these two records refer to the same real-world place.\n"
            "Return JSON only, in this format:\n"
            '{"match": true/false, "confidence": 0.0 to 1.0}\n\n'
            f"Base record: {json.dumps(base, ensure_ascii=False)}\n"
            f"Candidate record: {json.dumps(candidate, ensure_ascii=False)}"
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

def parse_output(text, prompt_type):
    text_clean = text.strip().lower()
    if prompt_type == "json_output":
        # Try to parse JSON object
        try:
            start = text_clean.find('{')
            end = text_clean.rfind('}') + 1
            json_str = text_clean[start:end]
            data = json.loads(json_str)
            return 1 if data.get("match", False) else 0
        except:
            pass  # fall through to other checks
    # Check for standalone 0 or 1 (word boundary)
    match = re.search(r"\b[01]\b", text_clean)
    if match:
        return int(match.group(0))
    # Check for any digit 0 or 1 anywhere (e.g., "answer: 1")
    match = re.search(r"[01]", text_clean)
    if match:
        return int(match.group(0))
    # Check for yes/no
    if "yes" in text_clean or ("match" in text_clean and "no match" not in text_clean):
        return 1
    if "no" in text_clean:
        return 0
    return -1

# ========== EVALUATION FUNCTION ==========
def evaluate_model(model_id, prompt_type, df, max_samples=MAX_SAMPLES):
    print(f"\n{'='*60}\nEvaluating: {model_id} | Prompt: {prompt_type}\n{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.to(DEVICE)
    model.eval()

    df_eval = df.head(max_samples).copy()
    predictions = []
    latencies = []

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc=f"{model_id} / {prompt_type}"):
        prompt = build_prompt(row, prompt_type)
        messages = [{"role": "user", "content": prompt}]
        if tokenizer.chat_template is not None:
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            chat_text = prompt

        enc = tokenizer(chat_text, return_tensors="pt", truncation=True)
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - start_time
        new_tokens = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred = parse_output(generated_text, prompt_type)

        predictions.append(pred)
        latencies.append(latency)

    labels = df_eval["label"].astype(int).to_numpy()
    pred_arr = np.array(predictions)
    valid = pred_arr != -1
    valid_count = valid.sum()
    total = len(pred_arr)

    if valid_count > 0:
        tp = ((pred_arr == 1) & (labels == 1)).sum()
        fp = ((pred_arr == 1) & (labels == 0)).sum()
        fn = ((pred_arr == 0) & (labels == 1)).sum()
        tn = ((pred_arr == 0) & (labels == 0)).sum()
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    else:
        accuracy = precision = recall = f1 = 0.0

    failure_rate = 1.0 - (valid_count / total)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_id,
        "prompt_type": prompt_type,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parsing_failure_rate": failure_rate,
        "avg_latency_sec": avg_latency,
        "samples_evaluated": total,
        "valid_predictions": valid_count,
    }

# ========== MAIN ==========
if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    print(f"Dataset shape: {df.shape}")
    print(df["label"].value_counts())

    prompt_types = ["plain_text", "structured_instructions", "json_output"]
    results = []

    for model_id in MODEL_IDS:
        for prompt_type in prompt_types:
            try:
                result = evaluate_model(model_id, prompt_type, df)
                results.append(result)
                print(result)
            except Exception as e:
                print(f"ERROR {model_id}/{prompt_type}: {e}")
                results.append({
                    "model": model_id,
                    "prompt_type": prompt_type,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "parsing_failure_rate": None,
                    "avg_latency_sec": None,
                    "samples_evaluated": 0,
                    "valid_predictions": 0,
                    "error": str(e),
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv("prompt_comparison_results.csv", index=False)
    print("\nSaved to prompt_comparison_results.csv")
    print(results_df)
