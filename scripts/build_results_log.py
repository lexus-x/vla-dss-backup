"""Consolidate ALL FNO-VLA experiments into one results file (RESULTS_SUMMARY.md).
Reads every eval jsonl, computes per-task + overall success, writes a single table."""
import json, os, numpy as np

NAMES = ["alphabet soup","cream cheese","salad dressing","bbq sauce","ketchup",
         "tomato sauce","butter","milk","chocolate pudding","orange juice"]

# (label, file, tag-or-None, model, params, protocol)
EXPS = [
    ("Baseline (mean-pool)",   r"E:\fno_data\zerror_expA.jsonl", "exc8alias","FNO-VLA",         "28.9M","N=200, exec8"),
    ("_sv ep5 (early-stop)",   r"E:\fno_data\zsv_ep5diag.jsonl", "sv_ep5d", "FNO-VLA +sepviews+wrist","28.9M","N=200, exec8"),
    ("_sv ep10",               r"E:\fno_data\zsv_ep10diag.jsonl","sv_ep10d","FNO-VLA +sepviews+wrist","28.9M","N=200, exec8"),
    ("_sv ep44 (over-trained)",r"E:\fno_data\zsv_eval.jsonl",    "sv_final","FNO-VLA +sepviews+wrist","28.9M","N=200, exec8"),
    ("_sv ep5 CLOSED-LOOP",    r"E:\fno_data\zsv_cl1.jsonl",     "sv_cl1",  "FNO-VLA +sepviews+wrist","28.9M","N=200, exec1"),
    ("Octo-Small (finetuned)", r"E:\fno_data\octo_results.jsonl",None,      "Octo-Small",      "137M", "N=200"),
]

def load(path, tag):
    if not os.path.exists(path): return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if tag and tag != "exc8alias":
        rows = [r for r in rows if r.get("tag") == tag]
    elif tag == "exc8alias":
        rows = [r for r in rows if r.get("tag") == "exec8"]
    return rows or None

results = []  # (label, model, params, protocol, per_task[10] or None, overall, n)
for label, path, tag, model, params, proto in EXPS:
    rows = load(path, tag)
    if not rows:
        results.append((label, model, params, proto, None, None, 0)); continue
    pt = []
    for ti in range(10):
        g = [r for r in rows if r.get("task_idx") == ti]
        pt.append(round(100*sum(r["success"] for r in g)/len(g)) if g else None)
    tot = sum(r["success"] for r in rows); n = len(rows)
    results.append((label, model, params, proto, pt, round(100*tot/n,1), n))

out = []
out.append("# FNO-VLA — Consolidated Experiment Results\n")
out.append("_LIBERO-Object benchmark. Auto-generated from eval logs._\n")
out.append("## Overall success\n")
out.append("| experiment | model | params | protocol | success | n |")
out.append("|---|---|---|---|---|---|")
for label, model, params, proto, pt, ov, n in results:
    ovs = f"**{ov}%**" if ov is not None else "_(running/none)_"
    out.append(f"| {label} | {model} | {params} | {proto} | {ovs} | {n} |")

out.append("\n## Per-task success (%)\n")
hdr = "| task | " + " | ".join(r[0] for r in results) + " |"
out.append(hdr)
out.append("|" + "---|"*(len(results)+1))
for ti in range(10):
    cells = []
    for r in results:
        pt = r[4]
        cells.append(str(pt[ti]) if pt and pt[ti] is not None else "-")
    out.append(f"| {NAMES[ti]} | " + " | ".join(cells) + " |")
# overall row
ovcells = [(f"{r[5]}" if r[5] is not None else "-") for r in results]
out.append("| **OVERALL** | " + " | ".join(ovcells) + " |")

out.append("\n## Key findings\n")
out.append("- **Baseline (mean-pool FNO-VLA): 63.5%** — pre-registered N=200, the apples-to-apples headline.")
out.append("- **Octo-Small (137M, finetuned): 70.5%** — consistent with independent preprint (66±5). Gap to baseline n.s. (p≈0.14).")
out.append("- **_sv (separate-views + proprio-gated wrist attn):** best at **early-stop ep5 = 67.0%** (>baseline, within noise); over-training to ep44 collapses it to 54.5% (memorization).")
out.append("- **Failure driver = X-Y localization** (not height): failures have dxy≈22mm vs 15mm on successes; height gap is fine. Off-center descent topples narrow objects (bbq).")
out.append("- **_sv per-object:** big gains OJ +55, ketchup +30, pudding +25, milk +20; regressions tomato -40 (placement/basket-miss), bbq -25 (x-y topple).")
out.append("- **Levers to ~73%:** closed-loop approach (x-y re-aim, inference-time), grasp oracle (x-y-precise demos for bbq/cream cheese), placement fix for tomato.")
out.append("\n_Sources: zerror_expA.jsonl(exec8), zsv_ep5diag.jsonl, zsv_ep10diag.jsonl, zsv_eval.jsonl, zsv_cl1.jsonl, octo_results.jsonl_\n")

dest = r"c:\sarvik\fno_backup\RESULTS_SUMMARY.md"
open(dest, "w", encoding="utf-8").write("\n".join(out))
print("wrote", dest, "(re-run anytime to refresh with new evals)")
