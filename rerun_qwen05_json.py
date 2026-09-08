import json, re, time
import pandas as pd, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DATA_PATH = "data/samples_3k_project_c_updated.parquet"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_TYPE = "json_output"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_SAMPLES = 500
MAX_NEW_TOKENS = 80

def clean_field(v):
    if v is None: return ""
    if isinstance(v, float) and pd.isna(v): return ""
    return str(v)

def build_prompt(row):
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
    return (
        "Determine whether these two records refer to the same real-world place.\n"
        "Return JSON only, in this format:\n"
        '{"match": true/false, "confidence": 0.0 to 1.0}\n\n'
        f"Base record: {json.dumps(base, ensure_ascii=False)}\n"
        f"Candidate record: {json.dumps(candidate, ensure_ascii=False)}"
    )

def parse_output(text):
    text_clean = text.strip()
    # Remove markdown code fences
    text_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_clean, flags=re.IGNORECASE)
    # Try to extract JSON object
    try:
        start = text_clean.find("{")
        end = text_clean.rfind("}") + 1
        json_str = text_clean[start:end]
        data = json.loads(json_str)
        if "match" in data:
            return 1 if data["match"] else 0
    except:
        pass
    # Fallback to true/false or 0/1
    t = text_clean.lower()
    if "true" in t: return 1
    if "false" in t: return 0
    m = re.search(r"\b[01]\b", t)
    if m: return int(m.group(0))
    return -1

def evaluate():
    df = pd.read_parquet(DATA_PATH).dropna(subset=["label"]).head(MAX_SAMPLES).copy()
    df["label"] = df["label"].astype(int)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager"
    )
    model.to(DEVICE); model.eval()

    preds = []; lats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{MODEL_ID} / {PROMPT_TYPE}"):
        prompt = build_prompt(row)
        messages = [{"role":"user","content":prompt}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(chat, return_tensors="pt", truncation=True)
        input_ids = enc["input_ids"].to(DEVICE)
        att = enc["attention_mask"].to(DEVICE)
        st = time.time()
        with torch.no_grad():
            out = model.generate(input_ids, attention_mask=att, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        lat = time.time()-st
        new = out[0][input_ids.shape[1]:]
        txt = tokenizer.decode(new, skip_special_tokens=True).strip()
        pred = parse_output(txt)
        preds.append(pred); lats.append(lat)

    labels = df["label"].to_numpy()
    p = np.array(preds)
    valid = p != -1
    vc = valid.sum(); total = len(p)
    if vc>0:
        tp = ((p==1)&(labels==1)).sum(); fp = ((p==1)&(labels==0)).sum()
        fn = ((p==0)&(labels==1)).sum(); tn = ((p==0)&(labels==0)).sum()
        acc = (tp+tn)/total
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec = tp/(tp+fn) if (tp+fn) else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    else:
        acc=prec=rec=f1=0.0
    fail = 1 - vc/total
    avg_lat = sum(lats)/len(lats) if lats else 0.0
    result = {
        "model": MODEL_ID, "prompt_type": PROMPT_TYPE, "accuracy": acc,
        "precision": prec, "recall": rec, "f1": f1, "parsing_failure_rate": fail,
        "avg_latency_sec": avg_lat, "samples_evaluated": total, "valid_predictions": vc
    }
    print(result)
    pd.DataFrame([result]).to_csv("qwen05_json_fixed.csv", index=False)

if __name__ == "__main__":
    evaluate()
