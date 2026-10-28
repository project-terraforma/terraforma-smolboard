"""Local review UI for LLM pair-type labels.

    .venv\\Scripts\\python golden_dataset_formation\\review_pairs.py datasets\\geo\\medium_pairs_typed_part1.parquet

Opens http://127.0.0.1:8765. Each pair shows the left/right record side by side with the
current classification on top. "Correct" marks it reviewed; "Wrong" reveals a dropdown of
labels and writes the chosen label to `llm_pair_type` (original kept in
`original_llm_pair_type`, progress in `review_status`). The parquet is saved on every click.
"""

import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [
    "parent_child",
    "same_name_different_address",
    "typo_differences",
    "incomplete_data",
    "unsolvable_conflicting",
    "languages",
    "other",
]
FIELDS = ["names", "addresses", "categories", "brand", "websites", "phones", "emails", "socials",
          "confidence", "operating_status", "sources", "geometry"]
COL = "llm_pair_type"

lock = threading.Lock()
df: pd.DataFrame
path: Path


def plain(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, bytes):
        return f"<{len(v)} bytes>"
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def pair_payload(i: int) -> dict:
    row = df.iloc[i]
    return {
        "index": i,
        "total": len(df),
        "reviewed": int((df["review_status"] != "unreviewed").sum()),
        "pair_id": str(row["pair_id"]),
        "label": plain(row.get("label")),
        "source_pair_type": plain(row.get("pair_type")),
        "distance_m": plain(row.get("distance_m")),
        "classification": row[COL],
        "status": row["review_status"],
        "fields": [
            {"name": f, "left": plain(row.get(f"left_{f}")), "right": plain(row.get(f"right_{f}"))}
            for f in FIELDS
            if f"left_{f}" in df.columns
        ],
    }


def save() -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


PAGE = """<!doctype html><meta charset=utf-8><title>Pair review</title>
<style>
body{font:14px system-ui;margin:0;background:#f4f5f7;color:#222}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:12px 20px;text-align:center}
#cls{font-size:24px;font-weight:700;color:#0b5cad}
#meta{color:#666;margin-top:4px}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;background:#ddd;margin-left:6px;font-size:12px}
main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px 20px 110px}
.card{background:#fff;border:1px solid #ccc;border-radius:8px;padding:12px}
.card h3{margin:0 0 8px}
.f{margin-bottom:10px}.f b{display:block;font-size:11px;text-transform:uppercase;color:#777}
.f pre{margin:2px 0;white-space:pre-wrap;word-break:break-word;font:13px ui-monospace,monospace}
.diff pre{background:#fff3c4}
footer{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #ccc;padding:14px;text-align:center}
button,select{font-size:16px;padding:10px 28px;border-radius:6px;border:1px solid #888;cursor:pointer;margin:0 8px}
#ok{background:#2e9e4f;color:#fff;border:0}#bad{background:#d23b3b;color:#fff;border:0}
#pick{display:none}
</style>
<header><div id=cls></div><div id=meta></div></header>
<main><div class=card id=L></div><div class=card id=R></div></main>
<footer>
<button id=ok>Correct</button><button id=bad>Wrong</button>
<span id=pick><select id=sel></select><button id=apply>Apply</button></span>
<div style="margin-top:8px"><button onclick="go(-1)">&larr; Prev</button><button onclick="go(1)">Next &rarr;</button>
<button onclick="next()">Next unreviewed</button></div>
</footer>
<script>
const LABELS=__LABELS__;let cur=0;
const $=id=>document.getElementById(id);
$('sel').innerHTML=LABELS.map(l=>`<option>${l}</option>`).join('');
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const show=v=>v===null||v===undefined?'-':(typeof v==='object'?JSON.stringify(v,null,1):String(v));
async function load(i){
  const r=await fetch('/api/pair?i='+i);const d=await r.json();cur=d.index;
  $('cls').innerHTML=esc(d.classification)+(d.status!=='unreviewed'?`<span class=badge>${d.status}</span>`:'');
  $('meta').textContent=`Pair ${d.index+1}/${d.total} · reviewed ${d.reviewed} · ground-truth label ${d.label} · source type ${d.source_pair_type} · ${d.distance_m==null?'':Math.round(d.distance_m)+' m apart'}`;
  for(const [side,id,t] of [['left','L','Location A'],['right','R','Location B']])
    $(id).innerHTML=`<h3>${t}</h3>`+d.fields.map(f=>{
      const same=show(f.left)===show(f.right);
      return `<div class="f ${same?'':'diff'}"><b>${f.name}</b><pre>${esc(show(f[side]))}</pre></div>`}).join('');
  $('pick').style.display='none';$('sel').value=d.classification;
  window.scrollTo(0,0);
}
async function post(body){
  await fetch('/api/review',{method:'POST',body:JSON.stringify(body)});
  const d=await (await fetch('/api/next?i='+cur)).json();load(d.index);
}
function go(n){load(Math.min(Math.max(cur+n,0),1e9))}
async function next(){const d=await (await fetch('/api/next?i='+cur)).json();load(d.index)}
$('ok').onclick=()=>post({index:cur,action:'correct'});
$('bad').onclick=()=>{$('pick').style.display='inline'};
$('apply').onclick=()=>post({index:cur,action:'wrong',label:$('sel').value});
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;
  if(e.key==='ArrowRight')go(1);if(e.key==='ArrowLeft')go(-1)});
load(0);
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json", code)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse

        u = urlparse(self.path)
        q = parse_qs(u.query)
        i = int(q.get("i", ["0"])[0])
        with lock:
            if u.path == "/":
                self._send(PAGE.replace("__LABELS__", json.dumps(LABELS)).encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/pair":
                self._json(pair_payload(min(max(i, 0), len(df) - 1)))
            elif u.path == "/api/next":
                todo = np.flatnonzero((df["review_status"] == "unreviewed").to_numpy())
                after = todo[todo > i]
                nxt = int(after[0]) if len(after) else (int(todo[0]) if len(todo) else i)
                self._json({"index": nxt})
            else:
                self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/api/review":
            return self._send(b"not found", "text/plain", 404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        i = int(body["index"])
        with lock:
            if body["action"] == "correct":
                df.iat[i, df.columns.get_loc("review_status")] = "correct"
            elif body["action"] == "wrong" and body.get("label") in LABELS:
                df.iat[i, df.columns.get_loc(COL)] = body["label"]
                df.iat[i, df.columns.get_loc("review_status")] = "corrected"
            else:
                return self._json({"error": "bad request"}, 400)
            save()
        self._json({"ok": True})


def main() -> None:
    global df, path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    path = args.parquet
    df = pd.read_parquet(path)
    if "original_llm_pair_type" not in df.columns:
        df["original_llm_pair_type"] = df[COL]
    if "review_status" not in df.columns:
        df["review_status"] = "unreviewed"
    url = f"http://127.0.0.1:{args.port}"
    print(f"Reviewing {path} ({len(df)} pairs) at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
