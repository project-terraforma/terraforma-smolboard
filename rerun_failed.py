import json
import re
import time
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_PATH = "data/samples_3k_project_c_updated.parquet"
MAX_NEW_TOKENS = 50
MAX_SAMPLES = 500
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(42)

JOBS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "plain_text"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "structured_instructions"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "json_output"),
    ("microsoft/phi-1_5", "plain_text"),
]

def clean_field(v):
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)

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
            "You are a location-matching assistant. Decide whether the two records describe the same real-world place.\n\n"
            f"Base record:\n{json.dumps(base, ensure_ascii=False, indent=2)}\n\n"
            f"Candidate record:\n{json.dumps(candidate, ensure_ascii=False, indent=2)}\n\n"
            "Answer only with 1 if they are the same place, or 0 if they are not."
        )
    elif prompt_type == "structured_instructions":
        return (
            "Determine whether these two location records refer to the same real-world place.\n\n"
            "Consider the following fields for both records:\n"
            "- Names (exact or similar)\n- Categories (e.g., restaurant, cafe)\n- Addresses (street, city, postcode)\n"
            "- Phones\n- Websites\n- Brand\n\n"
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
        raise ValueError("bad prompt type")

def parse_output(text, prompt_type):
    t = text.strip().lower()
    if "true" in t:
        return 1
    if "false" in t:
        return 0
    if prompt_type == "json_output":
        try:
            start = t.find('{')
            end = t.rfind('}') + 1
            data = json.loads(t[start:end])
            if "match" in data:
                return 1 if data["match"] else 0
        except:
            pass
    m = re.search(r"\b[01]\b", t)
    if m:
        return int(m.group(0))
    m = re.search(r"[01]", t)
    if m:
        return int(m.group(0))
    if "yes" in t or "match" in t:
        return 1
    if "no" in t or "not match" in t or "not_match" in t:
        return 0
    return -1

def evaluate(model_id, prompt_type, df):
    print(f"Running {model_id} / {prompt_type}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager"
    )
    model.to(DEVICE)
    model.eval()
    sub = df.head(MAX_SAMPLES).copy()
    preds = []
    lats = []
    for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"{model_id}/{prompt_type}"):
        prompt = build_prompt(row, prompt_type)
        messages = [{"role": "user", "content": prompt}]
        if tokenizer.chat_template is not None:
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            chat = prompt
        enc = tokenizer(chat, return_tensors="pt", truncation=True)
        input_ids = enc["input_ids"].to(DEVICE)
        att = enc["attention_mask"].to(DEVICE)
        st = time.time()
        with torch.no_grad():
            out = model.generate(input_ids, attention_mask=att, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        lat = time.time() - st
        new = out[0][input_ids.shape[1]:]
        txt = tokenizer.decode(new, skip_special_tokens=True).strip()
        pred = parse_output(txt, prompt_type)
        preds.append(pred)
        lats.append(lat)
    labels = sub["label"].astype(int).to_numpy()
    p = np.array(preds)
    valid = p != -1
    vc = valid.sum()
    total = len(p)
    if vc > 0:
        tp = ((p == 1) & (labels == 1)).sum()
        fp = ((p == 1) & (labels == 0)).sum()
        fn = ((p == 0) & (labels == 1)).sum()
        tn = ((p == 0) & (labels == 0)).sum()
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    else:
        acc = prec = rec = f1 = 0.0
    fail = 1 - vc / total
    avg_lat = sum(lats) / len(lats) if lats else 0.0
    return {
        "model": model_id,
        "prompt_type": prompt_type,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "parsing_failure_rate": fail,
        "avg_latency_sec": avg_lat,
        "samples_evaluated": total,
        "valid_predictions": vc,
    }

if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH).dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    results = []
    for mid, pt in JOBS:
        try:
            r = evaluate(mid, pt, df)
            results.append(r)
            print(r)
        except Exception as e:
            print(f"ERROR {mid}/{pt}: {e}")
            results.append({"model": mid, "prompt_type": pt, "error": str(e)})
    res_df = pd.DataFrame(results)
    res_df.to_csv("rerun_results.csv", index=False)
    print("\nSaved rerun_results.csv")
    print(res_df)
