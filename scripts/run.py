# Original vs InstinctRazor recipe (INT3 experts / INT4 backbone+embed / BF16 router+norms, g128, 16-step clip search)
# on granite-3.0-1b-a400m-base, using InstinctRazor's own quant code (InstinctRazor/src/quant).
import json, os, random, sys, time, torch, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, os.environ.get("IR_SRC", os.path.join(HERE, "..", "InstinctRazor", "src", "quant")))
import moe_quant as MQ, model_adapters as MA
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(8); torch.manual_seed(0)
MODEL = "ibm-granite/granite-3.0-1b-a400m-base"; OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "..", "results")) + "/"
SEED = int(globals().get("SEED", 0)); FA_SLICE = globals().get("FA_SLICE", (0, 700))
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval()
ad = MA.get_adapter(model.config); MQ.set_active_adapter(ad); MQ.set_clip_search(16)
print("adapter", type(ad).__name__, flush=True)
L = model.config.num_hidden_layers
def gen(prompts, k, bs=32):
    outs = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s+bs], return_tensors="pt", padding=True)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=k, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs
rng = random.Random(SEED)
shots = "12+7=19\n33+45=78\n58+26=84\n"
ar = [(shots + f"{a}+{b}=", str(a+b)) for a, b in [(rng.randint(10,99), rng.randint(10,99)) for _ in range(300)]]
cf = json.load(open(os.environ.get("COUNTERFACT", os.path.join(HERE, "..", "data", "counterfact.json")))); random.Random(0).shuffle(cf)
fa = [(c["requested_rewrite"]["prompt"].format(c["requested_rewrite"]["subject"]), c["requested_rewrite"]["target_true"]["str"]) for c in cf[FA_SLICE[0]:FA_SLICE[1]]]
words = "apple river stone cloud tiger lamp chair forest piano rocket garden mirror candle violin bridge desert".split()
def kvp():
    ks = rng.sample(words, 6); vs = [str(rng.randint(100, 999)) for _ in ks]; q = rng.randrange(6)
    return " ".join(f"{k}={v}" for k, v in zip(ks, vs)) + f"\nWhat is {ks[q]}? {ks[q]}=", vs[q]
ex = kvp(); kv = [(ex[0] + ex[1] + "\n\n" + p, t) for p, t in (kvp() for _ in range(300))]
def g_num(o, t): o = o.strip(); return o.startswith(t) and not o[len(t):len(t)+1].isdigit()
def g_fa(o, t): return o.strip().startswith(t)
SK = {"arith": (ar, g_num, 4), "fact": (fa, g_fa, 5), "icl_kv": (kv, g_num, 4)}
def evalset(sets):
    return {s: np.array([SK[s][1](o, t) for o, (_, t) in zip(gen([p for p, _ in sets[s]], SK[s][2]), sets[s])], dtype=int) for s in sets}
if __name__ == "__main__":
    t0 = time.time()
    full = evalset({s: SK[s][0] for s in SK})
    print("orig pool acc", {s: float(v.mean()) for s, v in full.items()}, flush=True)
    items = {s: [it for it, ok in zip(SK[s][0], full[s]) if ok][:200] for s in SK}
    json.dump({"items": items, "orig_pool": {s: v.tolist() for s, v in full.items()}}, open(OUT + "items.json", "w"))
    per = {s: np.zeros((L, len(items[s])), dtype=int) for s in SK}
    for l in range(L):
        t1 = time.time(); pre = f"model.layers.{l}."
        ex_ = [p for n, p in MQ.iter_expert_tensors(model) if n.startswith(pre)]
        li = [m.weight for n, m in MQ.iter_quant_linears(model) if n.startswith(pre)]
        bak = [(p, p.data.clone()) for p in ex_ + li]
        with torch.no_grad():
            for p in ex_: p.data.copy_(MQ.fakequant(p.data, 3.0, 128))
            for w in li: w.data.copy_(MQ.fakequant(w.data, 4.0, 128))
        r = evalset(items)
        for s in SK: per[s][l] = r[s]
        for p, b in bak: p.data.copy_(b)
        print(f"layer {l}: " + " ".join(f"{s}={r[s].mean():.3f}" for s in SK), f"{time.time()-t1:.0f}s", flush=True)
    np.savez(OUT + "perlayer.npz", **per)
    json.dump({s: per[s].tolist() for s in SK}, open(OUT + "perlayer_items.json", "w"))
    spec = ad.flat_spec(model, 3.0, 4.0, 128); eb = MQ.effective_bits(model, spec); st = MQ.apply_ptq(model, spec)
    ck = evalset(items); cp = evalset({s: SK[s][0] for s in SK})
    print("compressed retention", {s: float(v.mean()) for s, v in ck.items()}, flush=True)
    json.dump({"effective_bits": eb, "ptq": st, "comp_kept": {s: v.tolist() for s, v in ck.items()},
               "comp_pool": {s: v.tolist() for s, v in cp.items()}}, open(OUT + "compressed.json", "w"))
    print("done", time.time()-t0, flush=True)
