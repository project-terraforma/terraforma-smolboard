"""Tag SMOLBoard pairs by case. Reads {size}_pairs.parquet, writes a sidecar tags file.

Usage: python tag_cases.py <pairs.parquet> <out_dir> [--review 30]

Tags are multi-label and rule-based (no LLM). Thresholds are first guesses and
need hand-checking against the review sample before anyone trusts them.
"""
import argparse, csv, difflib, json, random, re, sys, unicodedata
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

TAGS = [
    "same_name_diff_address",
    "incomplete_data",
    "name_variant",
    "non_latin_script",
    "colocated_same_address",   # proxy for parent-child
    "conflicting_evidence",     # proxy for unsolvable/conflicting
]
VISIBLE = ["websites", "socials", "emails", "phones", "addresses"]
INCOMPLETE_MAX_FIELDS = 3      # of the 6 visible fields, either side
NAME_VARIANT_MIN, NAME_VARIANT_MAX = 0.5, 1.0
CONFLICT_NAME_SIM = 0.5


def primary(names):
    return (names or {}).get("primary") or ""


def norm(s):
    # NFKC turns full-width text like "１２−１" into "12-1". \w keeps letters from
    # every script, so Japanese or Arabic names don't all collapse to "".
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]|_", "", s)).strip()


def sim(a, b):
    na, nb = norm(a), norm(b)
    if na and nb and na != nb and (na in nb or nb in na):
        # One name inside the other ("Blue Beacon" vs "Blue Beacon Truck Wash")
        # is a name variant, not conflicting evidence.
        return 0.99
    return difflib.SequenceMatcher(None, na, nb).ratio()


def filled(row, side):
    n = 1 if primary(row[f"{side}_names"]) else 0
    n += sum(1 for k in VISIBLE if row[f"{side}_{k}"])
    return n


def street(row, side):
    addrs = row[f"{side}_addresses"] or []
    return norm((addrs[0] or {}).get("freeform") or "") if addrs else ""


def digits(p):
    d = re.sub(r"\D", "", p or "")
    return d[-8:]


def url_key(u):
    u = re.sub(r"^https?://(www\.)?", "", (u or "").lower()).split("?")[0].rstrip("/")
    return u


def shared_contact(row):
    def keys(side):
        s = set()
        s |= {url_key(u) for u in row[f"{side}_websites"] or [] if u}
        s |= {url_key(u) for u in row[f"{side}_socials"] or [] if u}
        s |= {(e or "").lower() for e in row[f"{side}_emails"] or [] if e}
        s |= {digits(p) for p in row[f"{side}_phones"] or [] if digits(p)}
        return s
    if keys("left") & keys("right"):
        return True
    la, ra = street(row, "left"), street(row, "right")
    return bool(la and la == ra)


def non_latin(s):
    return any(c.isalpha() and ord(c) > 0x024F for c in s)


def tag_row(r):
    L, R = primary(r["left_names"]), primary(r["right_names"])
    nsim = sim(L, R) if L and R else 0.0
    la, ra = street(r, "left"), street(r, "right")
    t = dict.fromkeys(TAGS, False)
    if r["label"] == 0:
        same_name = bool(L and R and norm(L) == norm(R))
        if (same_name or r["hard_neg_source"] == "brand") and la != ra:
            t["same_name_diff_address"] = True
        if r["hard_neg_source"] == "geo" and la and la == ra and norm(L) != norm(R):
            t["colocated_same_address"] = True
    else:
        if L and R and NAME_VARIANT_MIN <= nsim < NAME_VARIANT_MAX:
            t["name_variant"] = True
        if nsim < CONFLICT_NAME_SIM and not shared_contact(r):
            t["conflicting_evidence"] = True
    if min(filled(r, "left"), filled(r, "right")) <= INCOMPLETE_MAX_FIELDS:
        t["incomplete_data"] = True
    if non_latin(L) or non_latin(R):
        t["non_latin_script"] = True
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("out_dir")
    ap.add_argument("--review", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.pairs).name.replace("_pairs.parquet", "")
    cols = ["pair_id", "label", "hard_neg_source"] + [
        f"{s}_{k}" for s in ("left", "right")
        for k in ["names"] + VISIBLE]
    rows = pq.read_table(a.pairs, columns=cols).to_pylist()
    tags = [tag_row(r) for r in rows]

    table = pa.table({"pair_id": [r["pair_id"] for r in rows],
                      **{k: [t[k] for t in tags] for k in TAGS}})
    pq.write_table(table, out / f"{stem}_case_tags.parquet")

    counts = {k: sum(t[k] for t in tags) for k in TAGS}
    stats = {"pairs": len(rows), "counts": counts,
             "thresholds": {"incomplete_max_fields": INCOMPLETE_MAX_FIELDS,
                            "name_variant_range": [NAME_VARIANT_MIN, NAME_VARIANT_MAX],
                            "conflict_name_sim": CONFLICT_NAME_SIM}}
    (out / f"{stem}_case_stats.json").write_text(json.dumps(stats, indent=2))

    random.seed(42)
    with open(out / f"{stem}_review_sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "pair_id", "label", "left_name", "right_name",
                    "left_address", "right_address", "left_phones", "right_phones",
                    "left_websites", "right_websites", "my_verdict", "notes"])
        for k in TAGS:
            idx = [i for i, t in enumerate(tags) if t[k]]
            for i in random.sample(idx, min(a.review, len(idx))):
                r = rows[i]
                addr = lambda s: (r[f"{s}_addresses"] or [{}])[0].get("freeform", "")
                w.writerow([k, r["pair_id"], r["label"], primary(r["left_names"]),
                            primary(r["right_names"]), addr("left"), addr("right"),
                            r["left_phones"], r["right_phones"],
                            r["left_websites"], r["right_websites"], "", ""])
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
