#!/usr/bin/env python3
"""python analyze_big.py out/<tag> -> analysis.json, chart.png, fix_chart.png in that folder."""
import json, sys, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = sys.argv[1]; r = np.random.default_rng(0); B = 10000
it = json.load(open(f"{D}/items.json")); per = json.load(open(f"{D}/perlayer_items.json"))
var = json.load(open(f"{D}/variants.json")); ho = json.load(open(f"{D}/heldout.json")); SK = list(per)
def boot(x): i = r.integers(0, len(x), (B, len(x))); return [float(x.mean()), *map(float, np.percentile(x[i].mean(1), [2.5, 97.5]))]
out = {"n": {s: len(it["items"][s]) for s in SK}, "bits": {k: v["bits"] for k, v in var.items()}}
out["drop_flat"] = {s: boot(1 - np.array(var["flat"]["kept"][s])) for s in SK}
pw = {}
for i, x in enumerate(SK):
    for y in SK[i+1:]:
        a, b = 1 - np.array(var["flat"]["kept"][x]), 1 - np.array(var["flat"]["kept"][y])
        d = a[r.integers(0, len(a), (B, len(a)))].mean(1) - b[r.integers(0, len(b), (B, len(b)))].mean(1)
        pw[f"{x}-{y}"] = [float(a.mean() - b.mean()), *map(float, np.percentile(d, [2.5, 97.5]))]
out["drop_diff"] = pw; out["kill_fired"] = all(lo <= 0 <= hi for _, lo, hi in pw.values())
Ls = sorted(int(l) for l in per[SK[0]]); out["perlayer_drop"] = {s: [1 - float(np.mean(per[s][str(l)])) for l in Ls] for s in SK}
from scipy.stats import spearmanr
out["perlayer_spearman"] = {f"{x}-{y}": float(spearmanr(out["perlayer_drop"][x], out["perlayer_drop"][y])[0]) for i, x in enumerate(SK) for y in SK[i+1:]} if len(Ls) > 2 else {}
out["variants_retention"] = {k: {s: float(np.mean(v["kept"][s])) for s in SK} for k, v in var.items()}
best = ho["best_variant"]; out["heldout"] = {"best_variant": best, "bits": {k: ho[k]["bits"] for k in ["flat", best]}}
for s in SK:
    a, b = np.array(ho["flat"]["kept"][s]), np.array(ho[best]["kept"][s]); n = len(a); i = r.integers(0, n, (B, n))
    out["heldout"][s] = {"n": n, "flat": boot(a), "fix": boot(b), "gain": [float((b-a).mean()), *map(float, np.percentile((b[i]-a[i]).mean(1), [2.5, 97.5]))]}
json.dump(out, open(f"{D}/analysis.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 2.6]})
for k, s in enumerate(SK):
    m, lo, hi = [100*v for v in out["drop_flat"][s]]; ax[0].bar(k, m, yerr=[[m-lo], [hi-m]], capsize=4)
ax[0].set_xticks(range(len(SK)), [f"{s}\nn={out['n'][s]}" for s in SK], fontsize=8); ax[0].set_ylabel("% originally-correct lost"); ax[0].set_title("Flat recipe, 95% CI")
w = 0.8 / len(SK)
for k, s in enumerate(SK): ax[1].bar(np.array(Ls) + (k - len(SK)/2) * w, np.array(out["perlayer_drop"][s]) * 100, w, label=s)
ax[1].set_xlabel("layer quantized alone"); ax[1].set_ylabel("% lost"); ax[1].legend(fontsize=7); ax[1].set_title(os.path.basename(D.rstrip("/")))
plt.tight_layout(); plt.savefig(f"{D}/chart.png", dpi=130); plt.close()
x = np.arange(len(SK))
for k, key in enumerate(["flat", "fix"]):
    v = np.array([out["heldout"][s][key] for s in SK]) * 100
    plt.bar(x + (k - .5) * .38, v[:, 0], .38, yerr=[v[:, 0]-v[:, 1], v[:, 2]-v[:, 0]], capsize=3,
            label=f"{'flat' if key=='flat' else best} ({out['heldout']['bits']['flat' if key=='flat' else best]:.3f} b)")
plt.xticks(x, [f"{s}\nn={out['heldout'][s]['n']}" for s in SK], fontsize=8); plt.ylabel("% retained (held-out)"); plt.ylim(0, 100); plt.legend(); plt.tight_layout(); plt.savefig(f"{D}/fix_chart.png", dpi=130)
print(json.dumps({k: out[k] for k in ["n", "drop_flat", "kill_fired", "heldout"]}, indent=1))
