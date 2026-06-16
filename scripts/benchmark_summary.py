"""
Reportable benchmark summary: (success% + Wilson 95% CI, mean jerk, N) per model,
plus a two-proportion z-test vs the aux-71 reference -> decide DAgger vs aux.
Reads the eval jsonls; writes benchmark_summary.csv. CPU-only, safe anytime.
"""
import os, json, math, csv

MODELS = [
    ("aux-71 (ep5)",      "E:/fno_data/zaux_5.jsonl",          "aux5"),
    ("dagger ep5",        "E:/fno_data/zdag_dag5n20.jsonl",    "dag5n20"),
    ("dagger best(ep8)",  "E:/fno_data/zdag_dagbestn20.jsonl", "dagbestn20"),
    ("aux ep5 (matchedN)","E:/fno_data/zaux_5_n20.jsonl",      "aux5n20"),
    ("Octo (ref)",        "E:/fno_data/octo_results.jsonl",    None),
]
JERK_REF = {"Octo (ref)": "E:/fno_data/octo_jerk.jsonl"}


def load(path, tag):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if tag is not None and d.get("tag") != tag:
                continue
            rows.append(d)
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (100*(c-half), 100*(c+half))


def ztest(k1, n1, k2, n2):
    """two-proportion z, returns (z, approx p two-sided)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1/n1, k2/n2
    p = (k1+k2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1+1/n2))
    if se == 0:
        return (float("nan"), float("nan"))
    z = (p1-p2)/se
    p_two = math.erfc(abs(z)/math.sqrt(2))
    return (z, p_two)


def mean_jerk(rows, jpath=None, tag=None):
    js = [r["jerk"] for r in rows if r.get("jerk") is not None and r["jerk"] == r["jerk"]] if rows else []
    if not js and jpath:
        jr = load(jpath, tag) or []
        js = [r["jerk"] for r in jr if r.get("jerk") is not None and r["jerk"] == r["jerk"]]
    return sum(js)/len(js) if js else float("nan")


ref = None
table = []
for label, path, tag in MODELS:
    rows = load(path, tag)
    if not rows:
        table.append((label, 0, 0, None, None, None, None)); continue
    n = len(rows); k = sum(int(r.get("success", 0)) for r in rows)
    lo, hi = wilson(k, n)
    mj = mean_jerk(rows, JERK_REF.get(label), tag)
    table.append((label, n, k, 100*k/n, lo, hi, mj))
    if label.startswith("aux-71"):
        ref = (k, n)

print(f"{'model':<20} {'N':>4} {'succ%':>7} {'95% CI':>16} {'jerk':>8} {'vs aux-71':>22}")
print("-"*82)
for label, n, k, rate, lo, hi, mj in table:
    if n == 0:
        print(f"{label:<20} {'(pending)':>4}"); continue
    ci = f"[{lo:.0f},{hi:.0f}]"
    js = f"{mj:.4f}" if mj == mj else "  -"
    vs = ""
    if ref and not label.startswith("aux-71"):
        z, p = ztest(k, n, ref[0], ref[1])
        if z == z:
            sig = "SIG" if p < 0.05 else "ns"
            vs = f"d{rate-100*ref[0]/ref[1]:+.1f} z={z:+.2f} p={p:.2f} {sig}"
    print(f"{label:<20} {n:>4} {rate:>6.1f}% {ci:>16} {js:>8} {vs:>22}")

with open("c:/sarvik/fno_backup/benchmark_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "N", "successes", "success_pct", "wilson_lo", "wilson_hi", "mean_jerk"])
    for label, n, k, rate, lo, hi, mj in table:
        w.writerow([label, n, k, "" if n == 0 else round(rate, 1),
                    "" if n == 0 else round(lo, 1), "" if n == 0 else round(hi, 1),
                    "" if (mj is None or mj != mj) else round(mj, 4)])
print("\n-> wrote benchmark_summary.csv")
