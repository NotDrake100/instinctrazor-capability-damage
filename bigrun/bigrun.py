#!/usr/bin/env python3
"""Capability-specific damage of the InstinctRazor flat recipe + --protect-layers fix, for any MoE InstinctRazor supports.
Usage: python bigrun.py --model Qwen/Qwen3-30B-A3B-Base --tag qwen3_30b [--n 400] [--device cuda]
Env: IR_SRC (InstinctRazor/src/quant, patched branch), COUNTERFACT (counterfact.json). Outputs -> out/<tag>/."""
import argparse, json, os, random, sys, time, torch, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=400, help="max kept items per skill (discovery); held-out uses up to 2x")
ap.add_argument("--pool", type=int, default=600); ap.add_argument("--fact-pool", type=int, default=3000)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--device", default="cuda")
ap.add_argument("--clip-steps", type=int, default=16); ap.add_argument("--layers", default="", help="subset for smoke, e.g. 0,1")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("IR_SRC", os.path.join(HERE, "InstinctRazor", "src", "quant")))
import moe_quant as MQ, model_adapters as MA
import transformers; from transformers import AutoTokenizer, AutoModelForCausalLM
OUT = os.path.join(HERE, "out", a.tag); os.makedirs(OUT, exist_ok=True)
CF = os.environ.get("COUNTERFACT", os.path.join(HERE, "counterfact.json"))
torch.manual_seed(0); T0 = time.time()
def log(*x):
    s = " ".join(str(y) for y in x); print(s, flush=True); open(os.path.join(OUT, "log.txt"), "a").write(s + "\n")
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
dkw = {"dtype": torch.bfloat16} if int(transformers.__version__.split(".")[0]) >= 5 else {"torch_dtype": torch.bfloat16}
def load():
    m = AutoModelForCausalLM.from_pretrained(a.model, **dkw).to(a.device).eval()
    ad = MA.get_adapter(m.config); MQ.set_active_adapter(ad); MQ.set_clip_search(a.clip_steps); return m, ad
model, ad = load(); L = model.config.num_hidden_layers
nE = sum(1 for _ in MQ.iter_expert_tensors(model)); nLin = sum(1 for _ in MQ.iter_quant_linears(model))
log(f"model={a.model} adapter={type(ad).__name__} layers={L} expert_tensors={nE} linears={nLin} transformers={transformers.__version__}")
assert nE > 0 or ad.include_expert_linears, "no expert tensors found: wrong transformers version (need 5.x fused experts)"

def gen(prompts, k):
    outs = []
    for s in range(0, len(prompts), a.bs):
        enc = tok(prompts[s:s+a.bs], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad(): o = model.generate(**enc, max_new_tokens=k, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs
def g_num(o, t): o = o.strip(); return o.startswith(t) and not o[len(t):len(t)+1].isdigit()
def g_pre(o, t): return o.strip().startswith(t)
cf = json.load(open(CF)); random.Random(0).shuffle(cf)
def suites(seed, fslice):
    rng = random.Random(seed)
    # arithmetic: 3-digit addition (2-digit saturates big models) and 2-digit multiplication
    add = [("123+456=579\n305+217=522\n640+199=839\n" + f"{x}+{y}=", str(x+y)) for x, y in [(rng.randint(100,999), rng.randint(100,999)) for _ in range(a.pool)]]
    mul = [("12*11=132\n23*14=322\n31*25=775\n" + f"{x}*{y}=", str(x*y)) for x, y in [(rng.randint(10,99), rng.randint(10,99)) for _ in range(a.pool)]]
    fa = [(c["requested_rewrite"]["prompt"].format(c["requested_rewrite"]["subject"]), c["requested_rewrite"]["target_true"]["str"]) for c in cf[fslice[0]:fslice[1]]]
    words = "apple river stone cloud tiger lamp chair forest piano rocket garden mirror candle violin bridge desert".split()
    def kvp():
        ks = rng.sample(words, 8); vs = [str(rng.randint(100, 999)) for _ in ks]; q = rng.randrange(8)
        return " ".join(f"{k}={v}" for k, v in zip(ks, vs)) + f"\nWhat is {ks[q]}? {ks[q]}=", vs[q]
    ex = kvp(); kv = [(ex[0] + ex[1] + "\n\n" + p, t) for p, t in (kvp() for _ in range(a.pool))]
    # code: predict the output of a tiny Python snippet (exact match)
    def code():
        x, y, z = rng.randint(2, 9), rng.randint(2, 9), rng.randint(1, 20); f = rng.randrange(3)
        if f == 0: s, v = f"x = {x}\nfor i in range({y}):\n    x += {z}\nprint(x)", x + y*z
        elif f == 1: s, v = f"a = [{x}, {y}, {z}]\nprint(sum(a) * {x})", (x+y+z)*x
        else: s, v = f"s = 'ab' * {x}\nprint(len(s) + {z})", 2*x + z
        return s, str(v)
    cshot = "".join(f"```python\n{s}\n```\nOutput: {v}\n\n" for s, v in (code() for _ in range(3)))
    co = [(cshot + f"```python\n{s}\n```\nOutput: ", v) for s, v in (code() for _ in range(a.pool))]
    # multi-step reasoning: two-step word problems
    names = ["Sam", "Ana", "Lee", "Mia", "Raj", "Zoe"]
    def wp2():
        nm = rng.choice(names); n0, b, g, k = rng.randint(10, 50), rng.randint(2, 15), rng.randint(1, 9), rng.randint(2, 4)
        return f"{nm} has {n0} coins, finds {b} more, spends {g}, then the total is multiplied by {k}. How many coins? Answer: ", str((n0+b-g)*k)
    wshot = "".join(p + t + "\n" for p, t in (wp2() for _ in range(3)))
    mr = [(wshot + p, t) for p, t in (wp2() for _ in range(a.pool))]
    return {"add3": (add, g_num, 5), "mul2": (mul, g_num, 6), "fact": (fa, g_pre, 6), "icl_kv": (kv, g_num, 4),
            "code_out": (co, g_num, 5), "multistep": (mr, g_num, 6)}
def evalset(SK, sets):
    return {s: np.array([SK[s][1](o, t) for o, (_, t) in zip(gen([p for p, _ in sets[s]], SK[s][2]), sets[s])], dtype=int) for s in sets}
def spec_for(prot):
    return ad.flat_spec(model, 3.0, 4.0, 128, protect_layers=prot) if prot else ad.flat_spec(model, 3.0, 4.0, 128)
def save(name, obj): json.dump(obj, open(os.path.join(OUT, name), "w"))

# ---------- discovery ----------
SK = suites(0, (0, a.fact_pool))
full = evalset(SK, {s: SK[s][0] for s in SK})
items = {s: [it for it, ok in zip(SK[s][0], full[s]) if ok][:a.n] for s in SK}
log("orig pool acc", {s: round(float(v.mean()), 3) for s, v in full.items()}, "kept", {s: len(v) for s, v in items.items()}, f"{time.time()-T0:.0f}s")
SK = {s: v for s, v in SK.items() if len(items[s]) >= 30}  # drop suites the base model can't do
save("items.json", {"items": {s: items[s] for s in SK}, "orig_pool": {s: full[s].tolist() for s in SK}})
# ---------- per-layer scan (quantize only layer l) ----------
layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(L))
per = {s: {} for s in SK}
for l in layers:
    t1 = time.time(); pre = f"layers.{l}."
    ps = [p for n, p in MQ.iter_expert_tensors(model) if pre in n]; ws = [m.weight for n, m in MQ.iter_quant_linears(model) if pre in n]
    bak = [(p, p.data.clone()) for p in ps + ws]
    with torch.no_grad():
        for p in ps:
            for e0 in range(0, p.shape[0], 8): p.data[e0:e0+8] = MQ.fakequant(p.data[e0:e0+8], 3.0, 128)
        for w in ws: w.data.copy_(MQ.fakequant(w.data, 4.0, 128))
    r = evalset(SK, {s: items[s] for s in SK})
    for s in SK: per[s][l] = r[s].tolist()
    for p, b in bak: p.data.copy_(b)
    del bak; log(f"layer {l}: " + " ".join(f"{s}={r[s].mean():.3f}" for s in SK), f"{time.time()-t1:.0f}s")
    save("perlayer_items.json", per)
# ---------- full recipe + protection variants ----------
md = {l: np.mean([1 - np.mean(per[s][l]) for s in SK]) for l in layers}
worst = sorted(md, key=md.get, reverse=True)
variants = {"flat": (), "protect_L0": (0,), f"protect_worst1_L{worst[0]}": (worst[0],), "protect_worst2": tuple(sorted(worst[:2]))}
res = {}
seen = set()
for k, prot in variants.items():
    if prot in seen: continue
    seen.add(prot)
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model, ad = load(); sp = spec_for(prot); b = MQ.effective_bits(model, sp)["avg_bits_all"]; MQ.apply_ptq(model, sp, verbose=False)
    r = evalset(SK, {s: items[s] for s in SK}); res[k] = {"protect": list(prot), "bits": b, "kept": {s: v.tolist() for s, v in r.items()}}
    log(k, f"bits={b:.3f}", {s: round(float(v.mean()), 3) for s, v in r.items()}); save("variants.json", res)
# ---------- held-out (fresh items) for flat vs best protection ----------
best = max([k for k in res if k != "flat"], key=lambda k: np.mean([np.mean(res[k]["kept"][s]) for s in SK]) - 0.0 * res[k]["bits"])
del model; model, ad = load()
H = suites(1, (a.fact_pool, 2 * a.fact_pool)); H = {s: H[s] for s in SK}
hf = evalset(H, {s: H[s][0] for s in H}); hitems = {s: [it for it, ok in zip(H[s][0], hf[s]) if ok][:2*a.n] for s in H}
ho = {"best_variant": best, "n": {s: len(v) for s, v in hitems.items()}, "items": hitems}
for k in ["flat", best]:
    del model; model, ad = load(); sp = spec_for(tuple(res[k]["protect"])); b = MQ.effective_bits(model, sp)["avg_bits_all"]
    MQ.apply_ptq(model, sp, verbose=False); r = evalset(H, hitems)
    ho[k] = {"bits": b, "kept": {s: v.tolist() for s, v in r.items()}}; log("heldout", k, f"bits={b:.3f}", {s: round(float(v.mean()), 3) for s, v in r.items()})
save("heldout.json", ho); log("DONE", f"{time.time()-T0:.0f}s")
