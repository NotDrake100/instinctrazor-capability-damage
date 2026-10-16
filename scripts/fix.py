# Protection variants of the flat recipe on the discovery items.
import os
import json, sys, torch, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")).read().split('if __name__ == "__main__":')[0])
items = {s: [tuple(x) for x in v] for s, v in json.load(open(OUT+"items.json"))["items"].items()}
orig = {n: p.data.clone() for n, p in model.named_parameters()}
def variant(emb16=False, exp4=(), attn16=()):
    with torch.no_grad():
        for n, p in model.named_parameters(): p.data.copy_(orig[n])
    lb = {}
    for n, m in MQ.iter_quant_linears(model):
        if emb16 and (n.endswith("embed_tokens") or n.endswith("lm_head")): lb[n] = 16
        elif MQ.layer_idx_of(n) in attn16: lb[n] = 16
    eb = {n: torch.full((p.shape[0],), 4.0 if MQ.layer_idx_of(n) in exp4 else 3.0) for n, p in MQ.iter_expert_tensors(model)}
    spec = MQ.AllocSpec(expert_bits=eb, linear_bits=lb, default_linear_bits=4, group=128)
    b = MQ.effective_bits(model, spec)["avg_bits_all"]; MQ.apply_ptq(model, spec, verbose=False)
    return b, {s: v.tolist() for s, v in evalset(items).items()}
V = {"baseline_flat": {}, "embed_bf16": {"emb16": True}, "layer0_protected": {"exp4": (0,), "attn16": (0,)},
     "embed_bf16+layer0": {"emb16": True, "exp4": (0,), "attn16": (0,)},
     "embed_bf16+layers0-2_int4exp": {"emb16": True, "exp4": (0, 1, 2)}, "all_experts_int4": {"exp4": tuple(range(L))}}
res = {}
for k, kw in V.items():
    b, r = variant(**kw); res[k] = {"bits": b, "kept": r}
    print(k, f"bits={b:.3f}", {s: round(float(np.mean(v)), 3) for s, v in r.items()}, flush=True)
    json.dump(res, open(OUT+"fix.json", "w"))
