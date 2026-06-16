"""Convert an eval jsonl into a clean per-test summary JSON (success+CI, jerk, per-task).
Usage: python summarize_eval.py <suite> <jsonl_path> [<jsonl_path> ...]
Writes <results_json>/<tag>.json for each.
"""
import os, sys, json, math

OUT = r"c:\sarvik\fno_backup\results_json"; os.makedirs(OUT, exist_ok=True)

def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 0.0]
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [round(100*(c-h), 1), round(100*(c+h), 1)]

def summarize(suite, path):
    if not os.path.exists(path):
        print("MISSING", path); return
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        print("EMPTY", path); return
    tag = rows[0].get("tag", os.path.basename(path))
    n = len(rows); k = sum(int(r.get("success", 0)) for r in rows)
    jl = [r["jerk"] for r in rows if r.get("jerk") is not None and r["jerk"] == r["jerk"]]
    per = {}
    for ti in sorted(set(r.get("task_idx") for r in rows)):
        g = [r for r in rows if r.get("task_idx") == ti]
        gk = sum(int(r.get("success", 0)) for r in g)
        per[str(ti)] = {"task": g[0].get("task", ""), "successes": gk, "n": len(g),
                        "pct": round(100*gk/len(g), 1)}
    summary = {
        "suite": suite, "tag": tag, "checkpoint": rows[0].get("checkpoint", ""),
        "execute": rows[0].get("execute"), "N": n, "successes": k,
        "success_pct": round(100*k/n, 1), "wilson_ci_95": wilson(k, n),
        "mean_jerk": round(sum(jl)/len(jl), 4) if jl else None,
        "per_task": per,
    }
    outp = os.path.join(OUT, f"{tag}.json")
    with open(outp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{tag}: {summary['success_pct']}% (N={n}) CI{summary['wilson_ci_95']} jerk={summary['mean_jerk']} -> {outp}")

if __name__ == "__main__":
    suite = sys.argv[1]
    for p in sys.argv[2:]:
        summarize(suite, p)
