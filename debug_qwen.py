import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_PATH = "data/samples_3k_project_c_updated.parquet"
MODEL_IDS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def clean_field(v):
    if v is None: return ""
    if isinstance(v, float) and pd.isna(v): return ""
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

df = pd.read_parquet(DATA_PATH).dropna(subset=["label"]).head(3)

for model_id in MODEL_IDS:
    print(f"\n===== {model_id} =====")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager")
    model.to(DEVICE); model.eval()
    for i, row in df.iterrows():
        for pt in ["plain_text", "structured_instructions", "json_output"]:
            prompt = build_prompt(row, pt)
            messages = [{"role":"user","content":prompt}]
            if tokenizer.chat_template is not None:
                chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                chat = prompt
            inputs = tokenizer(chat, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"Row {i}, Prompt {pt}: {repr(text)}")
    del model, tokenizer
