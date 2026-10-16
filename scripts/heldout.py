# Held-out validation (fresh items: seed 1, CounterFact slice 700:2500) of patched flat_spec(protect_layers=(0,)).
import os
import json, sys, torch, numpy as np
SEED = 1; FA_SLICE = (700, 2500)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")).read().split('if __name__ == "__main__":')[0])
full = evalset({s: SK[s][0] for s in SK}); items = {s: [it for it, ok in zip(SK[s][0], full[s]) if ok] for s in SK}
print("heldout n", {s: len(v) for s, v in items.items()}, flush=True)
orig = {n: p.data.clone() for n, p in model.named_parameters()}; res = {"n": {s: len(v) for s, v in items.items()}, "items": items}
for k, prot in [("flat", ()), ("protect_L0", (0,))]:
    with torch.no_grad():
        for n, p in model.named_parameters(): p.data.copy_(orig[n])
    spec = ad.flat_spec(model, 3.0, 4.0, 128, protect_layers=prot); b = MQ.effective_bits(model, spec)["avg_bits_all"]
    MQ.apply_ptq(model, spec, verbose=False); r = evalset(items)
    res[k] = {"bits": b, "kept": {s: v.tolist() for s, v in r.items()}}
    print(k, f"{b:.3f}", {s: round(float(v.mean()), 3) for s, v in r.items()}, flush=True)
json.dump(res, open(OUT+"heldout.json", "w"))
