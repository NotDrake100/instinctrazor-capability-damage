"""mathfix PHASE 1 -- diagnosis of the GSM8K collapse in arm b of sleep_shield.py.

Arm b only (LoRA rank 16 on all layers / all 7 projection types, lr 2e-4, 120 steps/night, batch 16, fp16 autocast),
seeds 0,1,2, 7 nights x 100 fictional facts (accumulating). Model, probes, fact stream, parsers, chat format and
training loop are verbatim from ../sleep_shield.py (hashes asserted).

  (a) raw GSM8K generations (base + every night, all base-correct items) -> gens/*.jsonl
  (b) failure categories (transparent rules in common.categorize), strict vs lenient score, 50 random failures per
      night for nights 1/4/7 -> failures_review.md
  (c) per-layer drift vs base on the GSM8K prompts: cosine distance of the residual stream (all prompt tokens and the
      last prompt token), shift coherence (mean pairwise cosine of the per-prompt displacement), logit-lens KL
  (d) logit lens at the first answer position: per layer P(reasoning-start tokens) vs P(short-answer-start tokens)
  (e) layer swap on night-7 adapters: LoRA off on one layer / one block of 3 -> GSM8K strict on a fixed <=100-item
      subset + fact-recall cost
  (f) shared-direction test (Palit et al., arXiv 2610.08718): remove the top-1 direction of the mean LoRA update
      (mean over seeds x nights) from each matrix; plus own-top-1 and random rank-1 controls -> GSM8K + recall
  (g) extra, cheap: prefill test at base and night 7 -- assistant turn pre-filled with "Let's solve this step by step.\n"
Outputs: results.json (after every night / every analysis unit), adapters/s{seed}/night{n}/, log.txt,
phase1_summary.json (+ automatic hypothesis classification per predictions.md), phase1_report.md.
Resumable: re-run the same command. CPU smoke test: MATHFIX_SMOKE=1 python phase1.py --out /tmp/p1smoke
"""
import os, sys, json, re, math, time, random, gc, argparse
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import DEV, AMP

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1_out"))
ap.add_argument("--swap_gsm_max_new", type=int, default=None, help="cap GSM max_new_tokens in the layer-swap / shared-direction "
                "re-scoring only (default: same as training eval, 512). Use 384 only if the runtime estimate is too long.")
args = ap.parse_args()
SMOKE = os.environ.get("MATHFIX_SMOKE", "0") == "1"
CFG = dict(C.BASE_CFG); CFG.update(swap_subset=100, prefill="Let's solve this step by step.\n", block=3)
if SMOKE: CFG.update(C.SMOKE_OVERRIDES); CFG.update(swap_subset=4)
CFG["swap_gsm_max_new"] = args.swap_gsm_max_new or CFG["gsm_max_new"]
OUT = args.out; os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(OUT, "gens"), exist_ok=True)
log = C.Logger(OUT); RES = os.path.join(OUT, "results.json")
R = json.load(open(RES)) if os.path.exists(RES) else {"cfg": CFG, "nights": {}, "timing": {}}
for k in ["model", "n_probe", "nights", "facts_per_night", "lr", "steps_per_night", "lora_r", "lora_alpha", "batch_size", "gsm_max_new",
          "gsm_parser", "seeds", "swap_subset", "prefill", "swap_gsm_max_new"]:
    assert R["cfg"].get(k) == CFG[k], f"config changed vs saved results ({k}: {R['cfg'].get(k)} -> {CFG[k]}); use a new --out"
def save(): C.save_json(RES, R)
T0 = time.time()
log("=" * 30, "mathfix phase 1 start | device", DEV, "| smoke", SMOKE, "| out", OUT)
if DEV == "cuda": log("GPU:", torch.cuda.get_device_name(0))

PROBES, PH = C.load_probes(CFG, SMOKE); FACTS, FH, FACT_VOCAB = C.load_facts(CFG, SMOKE)
R["probe_hash"], R["fact_hash"] = PH, FH; save()
log("probe hash", PH, "| fact hash", FH, "(match sleep_shield.py)" if not SMOKE else "(smoke)")
tok, base = C.load_model(CFG["model"]); gen = C.Gen(tok, base); TR = C.Trainer(CFG, tok, gen.chat)
N_LAYERS = len(base.model.layers); FP0 = C.fingerprint(base); log("layers", N_LAYERS); R["n_layers"] = N_LAYERS
GSM = PROBES["gsm8k"]; MAXG = CFG["gsm_max_new"]

def write_jsonl(path, recs, extra):
    with open(path, "w") as f:
        for r in recs: f.write(json.dumps({**extra, **r}) + "\n")
def read_jsonl(path): return [json.loads(l) for l in open(path)] if os.path.exists(path) else []

# ------------------------------------------------------------------ base eval (all scored items)
if "base" not in R:
    t = time.time()
    outs, nt, tr = gen(base, [x["q"] for x in GSM], MAXG, CFG["gen_bs"]); first = gen.first
    recs = C.score_gsm(GSM, list(range(len(GSM))), outs, nt, tr, FACT_VOCAB, first)
    t_gsm = time.time() - t
    write_jsonl(os.path.join(OUT, "gens", "base.jsonl"), recs, {"seed": None, "night": 0, "cond": "base"})
    m_acc, m_c = C.eval_simple(gen, base, PROBES, "mult", list(range(len(PROBES["mult"]))), CFG)
    a_acc, a_c = C.eval_simple(gen, base, PROBES, "arc", list(range(len(PROBES["arc"]))), CFG)
    t = time.time(); brec = C.eval_recall(gen, base, FACTS, CFG); t_rec = time.time() - t
    R["base"] = {"gsm_items": [r["strict_ok"] for r in recs], "mult_items": m_c, "arc_items": a_c,
                 "acc": {"gsm8k_strict": 100 * np.mean([r["strict_ok"] for r in recs]), "gsm8k_lenient": 100 * np.mean([r["lenient_ok"] for r in recs]),
                         "mult": m_acc, "arc": a_acc}, "recall_all_facts": brec}
    R["timing"].update(base_gsm_200=t_gsm, recall_all=t_rec); save()
    log(f"base eval: {R['base']['acc']} | base recall on all facts {brec:.1f} | gsm 200 items {t_gsm:.0f}s, recall {len(FACTS)*2} qs {t_rec:.0f}s")
BASE_OK = {"gsm8k": [i for i, c in enumerate(R["base"]["gsm_items"]) if c], "mult": [i for i, c in enumerate(R["base"]["mult_items"]) if c],
           "arc": [i for i, c in enumerate(R["base"]["arc_items"]) if c]}
if SMOKE:
    for k in BASE_OK:
        if not BASE_OK[k]: BASE_OK[k] = list(range(len(PROBES[k]))); log(f"SMOKE: tiny model has 0 base-correct {k} items -> using all items to exercise code")
log("base-correct counts:", {k: len(v) for k, v in BASE_OK.items()})
_orig = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vast_run", "results.json")
if os.path.exists(_orig) and not SMOKE and "orig_overlap" not in R:
    o = json.load(open(_orig))["base_items"]
    R["orig_overlap"] = {k: {"orig": int(sum(o[k])), "now": int(sum(R["base"][f"{k}_items" if k != "gsm8k" else "gsm_items"])),
                             "both": int(sum(a and b for a, b in zip(o[k], R["base"][f"{k}_items" if k != "gsm8k" else "gsm_items"])))} for k in ["mult", "gsm8k", "arc"]}
    save(); log("base-correct overlap with the Vast sleep_shield run:", R["orig_overlap"])
SUB = sorted(random.Random(0).sample(BASE_OK["gsm8k"], min(CFG["swap_subset"], len(BASE_OK["gsm8k"]))))
R["swap_subset_items"] = SUB; save()

if "base_prefill" not in R:
    outs, nt, tr = gen(base, [GSM[i]["q"] for i in BASE_OK["gsm8k"]], MAXG, CFG["gen_bs"], prefill=CFG["prefill"])
    recs = C.score_gsm(GSM, BASE_OK["gsm8k"], [CFG["prefill"] + o for o in outs], nt, tr, FACT_VOCAB, gen.first)
    write_jsonl(os.path.join(OUT, "gens", "base_prefill.jsonl"), recs, {"seed": None, "night": 0, "cond": "base_prefill"})
    R["base_prefill"] = C.agg_gsm(recs); save(); log("base prefill on base-correct:", {k: R["base_prefill"][k] for k in ["strict", "lenient", "mean_tokens", "pct_steps"]})
base_recs = [r for r in read_jsonl(os.path.join(OUT, "gens", "base.jsonl")) if r["item"] in set(BASE_OK["gsm8k"])]
R["base_gsm_on_base_correct"] = C.agg_gsm(base_recs); save()

# ------------------------------------------------------------------ runtime estimate
def estimate(t_night_train=None, t_night_eval=None):
    tm = R["timing"]; tg = tm["base_gsm_200"] / 2.0          # ~100 long GSM generations
    tr_all = tm["recall_all"]
    tt = t_night_train if t_night_train is not None else 25.0 * (1 if DEV == "cuda" else 20)
    te = t_night_eval if t_night_eval is not None else (tg * 0.5 + tr_all * 0.6 + 10)
    nn = len(CFG["seeds"]) * CFG["nights"]; nsw = N_LAYERS + math.ceil(N_LAYERS / CFG["block"]) + 1
    parts = {"training+nightly eval": nn * (tt + te), "prefill": (len(CFG["seeds"]) + 1) * tg,
             "drift+lens": (nn + 1) * 6, "layer swap": len(CFG["seeds"]) * nsw * (tg * CFG["swap_gsm_max_new"] / MAXG + tr_all),
             "shared direction": len(CFG["seeds"]) * 4 * (tg + tr_all + 5) + 60}
    return parts, sum(parts.values())
p, tot = estimate(); log(f"RUNTIME ESTIMATE (before training, rough): {tot/3600:.2f} h total | " + ", ".join(f"{k} {v/60:.0f} min" for k, v in p.items()))

# ------------------------------------------------------------------ training (arm b)
from peft import get_peft_model, PeftModel
ALL = list(range(N_LAYERS))
def adir(seed, night): return os.path.join(OUT, "adapters", f"s{seed}", f"night{night}")
def train_one(seed, night):
    global base
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = get_peft_model(base, C.lora_cfg(CFG, ALL)) if night == 1 else PeftModel.from_pretrained(base, adir(seed, night - 1), is_trainable=True)
    got = sorted({int(re.search(r"layers\.(\d+)\.", n).group(1)) for n, p in model.named_parameters() if p.requires_grad})
    assert got == ALL, f"LoRA layer filter wrong: {got}"
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    seen = [f for f in FACTS if f["night"] <= night]
    t = time.time(); loss = TR.train_night(model, seen, seed, night); tt = time.time() - t
    model.save_pretrained(adir(seed, night)); base = model.unload(); del model; gc.collect()
    assert abs(C.fingerprint(base) - FP0) < 1e-6, "base modified"
    return loss, ntrain, tt
def ensure_adapter(seed, night):
    """rebuild a missing adapter by replaying the seeded training (identical data order) for nights 1..night"""
    if os.path.isdir(adir(seed, night)): return
    for n in range(1, night + 1):
        if not os.path.isdir(adir(seed, n)):
            loss, _, _ = train_one(seed, n); log(f"[s{seed}] replayed missing adapter night {n} (train loss {loss:.3f})")
def load_adapter(seed, night):
    ensure_adapter(seed, night); m = PeftModel.from_pretrained(base, adir(seed, night)); m.eval(); return m
def unload(m):
    global base
    base = m.unload(); del m; gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()

def eval_gsm(model, idx, tag_extra, path, prefill="", max_new=None):
    outs, nt, tr = gen(model, [GSM[i]["q"] for i in idx], max_new or MAXG, CFG["gen_bs"], prefill=prefill)
    recs = C.score_gsm(GSM, idx, [prefill + o for o in outs] if prefill else outs, nt, tr, FACT_VOCAB, gen.first)
    if path: write_jsonl(path, recs, tag_extra)
    return recs

t_train_start = time.time(); n_done_now = 0
total_nights = len(CFG["seeds"]) * CFG["nights"]
for seed in CFG["seeds"]:
    key = f"s{seed}"; run = R["nights"].setdefault(key, {})
    for night in range(1, CFG["nights"] + 1):
        if str(night) in run: continue
        if night > 1: ensure_adapter(seed, night - 1)
        t0 = time.time(); loss, ntrain, tt = train_one(seed, night)
        model = load_adapter(seed, night)                       # reload from disk, like sleep_shield.py
        te0 = time.time(); seen = [f for f in FACTS if f["night"] <= night]
        rec = C.eval_recall(gen, model, seen, CFG); rec_new = C.eval_recall(gen, model, [f for f in seen if f["night"] == night], CFG)
        m_ret, _ = C.eval_simple(gen, model, PROBES, "mult", BASE_OK["mult"], CFG)
        a_ret, _ = C.eval_simple(gen, model, PROBES, "arc", BASE_OK["arc"], CFG)
        recs = eval_gsm(model, BASE_OK["gsm8k"], {"seed": seed, "night": night, "cond": "trained"}, os.path.join(OUT, "gens", f"s{seed}_n{night}.jsonl"))
        g = C.agg_gsm(recs); ent = {"recall": rec, "recall_tonight": rec_new, "mult": m_ret, "arc": a_ret, "gsm": g, "train_loss": loss,
                                    "n_trainable": ntrain, "sec_train": tt}
        if night == CFG["nights"]:
            pr = eval_gsm(model, BASE_OK["gsm8k"], {"seed": seed, "night": night, "cond": "prefill"}, os.path.join(OUT, "gens", f"s{seed}_n{night}_prefill.jsonl"), prefill=CFG["prefill"])
            ent["gsm_prefill"] = C.agg_gsm(pr)
        if night == 1 or night == CFG["nights"]:
            with model.disable_adapter(): ent["recall_adapter_off"] = C.eval_recall(gen, model, seen, CFG)
        unload(model); ent["sec_eval"] = time.time() - te0; ent["sec"] = time.time() - t0
        run[str(night)] = ent; save(); n_done_now += 1
        left = total_nights - sum(len(v) for v in R["nights"].values())
        log(f"[b {key} n{night}] recall {rec:.1f} (tonight {rec_new:.1f}) | ret mult {m_ret:.1f} arc {a_ret:.1f} | GSM strict {g['strict']:.1f} "
            f"lenient {g['lenient']:.1f} tok {g['mean_tokens']:.0f} steps {g['pct_steps']:.0f}% trunc {g['pct_truncated']:.0f}% fails {g['fail_categories']}"
            + (f" | PREFILL strict {ent['gsm_prefill']['strict']:.1f}" if "gsm_prefill" in ent else "")
            + f" | loss {loss:.3f} train {tt:.0f}s eval {ent['sec_eval']:.0f}s | training ETA {(time.time()-t_train_start)/n_done_now*left/60:.0f} min")
        if n_done_now == 1:
            p, tot = estimate(tt, ent["sec_eval"]); R["timing"]["estimate_after_night1_h"] = tot / 3600; save()
            log(f"RUNTIME ESTIMATE (measured night 1): {tot/3600:.2f} h total | " + ", ".join(f"{k} {v/60:.0f} min" for k, v in p.items())
                + ("  !!! > 2 h target: consider --swap_gsm_max_new 384 (only affects (e)/(f) re-scoring) on a fresh --out" if tot > 7200 and DEV == "cuda" else ""))
log(f"training done ({(time.time()-t_train_start)/60:.1f} min this session)")
LAST = CFG["nights"]

# ------------------------------------------------------------------ (c)+(d) drift and logit lens
from collections import Counter
def first_token_sets():
    br = [r for r in base_recs if r["has_steps"] and r["first_token"] is not None]
    noteR = ""
    if not br: br = [r for r in base_recs if r["first_token"] is not None]; noteR = "no base output with reasoning steps; R from all base outputs. "
    cR = Counter(r["first_token"] for r in br); nR = max(len(br), 1)
    Rset = sorted(t for t, c in cR.items() if c / nR >= 0.05) or [t for t, _ in cR.most_common(1)]
    n7 = [r for s in CFG["seeds"] for r in read_jsonl(os.path.join(OUT, "gens", f"s{s}_n{LAST}.jsonl"))
          if not r["strict_ok"] and r["category"] in ("no_steps", "fact_style") and r["first_token"] is not None]
    cS = Counter(r["first_token"] for r in n7); nS = max(len(n7), 1)
    Sset = sorted(t for t, c in cS.items() if c / nS >= 0.05 and t not in Rset)
    note = noteR
    if not Sset:
        Sset = [t for t, _ in cS.most_common(3) if t not in Rset]; note += "no short-answer start token reached 5% outside R; used top-3 non-R"
    digits = sorted({tok.convert_tokens_to_ids(str(d)) for d in range(10)} - {None, tok.unk_token_id})
    return {"R": Rset, "S": Sset, "digits": digits, "R_str": [tok.decode([t]) for t in Rset], "S_str": [tok.decode([t]) for t in Sset],
            "R_counts": {tok.decode([t]): cR[t] for t in Rset}, "S_counts": {tok.decode([t]): cS[t] for t in Sset}, "n_base": nR, "n_short_fail": nS, "note": note}
if "token_sets" not in R:
    R["token_sets"] = first_token_sets(); save()
TS = R["token_sets"]; log("logit-lens token sets: reasoning-start", TS["R_str"], "| short-answer-start", TS["S_str"], TS["note"])

@torch.no_grad()
def layer_states_batches(model, idx, bs=20):
    """yields per batch: (list over layers of residual-stream outputs [n, T, d] fp32 on CPU, attention mask)"""
    layers = base.model.layers; caps = {}
    def mk(i):
        def h(mod, inp, out): caps[i] = (out[0] if isinstance(out, (tuple, list)) else out)
        return h
    hs = [layers[i].register_forward_hook(mk(i)) for i in range(N_LAYERS)]
    tok.padding_side = "left"
    try:
        for s in range(0, len(idx), bs):
            enc = tok([gen.chat(GSM[i]["q"]) for i in idx[s:s + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            caps.clear()
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP): model(**enc)
            yield [caps[i].float().cpu() for i in range(N_LAYERS)], enc["attention_mask"].cpu()
    finally:
        for h in hs: h.remove()
@torch.no_grad()
def lens_logp(h_last):  # residual at last prompt position [n, d] -> log-probs of the logit lens (final norm + lm_head)
    x = base.model.norm(h_last.to(DEV)); return F.log_softmax(base.lm_head(x).float(), -1)
@torch.no_grad()
def drift_lens(st_base, model):
    """st_base: list of per-batch (layers fp16, mask) for the base model; model: adapted model, processed batch by batch"""
    keys = ["cos_all", "cos_last", "kl_last", "pR", "pS", "pD", "pR_base", "pS_base", "pD_base"]
    acc = {k: [[] for _ in range(N_LAYERS)] for k in keys}; disp = [[] for _ in range(N_LAYERS)]
    for (hb_all, m), (hm_all, _) in zip(st_base, layer_states_batches(model, BASE_OK["gsm8k"])):
        mask = m.bool()
        for L in range(N_LAYERS):
            b, a = hb_all[L].float(), hm_all[L]
            cos = F.cosine_similarity(b, a, dim=-1)
            acc["cos_all"][L].append((1 - cos).masked_fill(~mask, 0.0).sum(1) / mask.sum(1)); acc["cos_last"][L].append(1 - cos[:, -1])
            disp[L].append(a[:, -1] - b[:, -1])
            lb, la = lens_logp(b[:, -1]), lens_logp(a[:, -1])
            acc["kl_last"][L].append((lb.exp() * (lb - la)).sum(-1).cpu())
            for k, lp, S in (("pR", la, TS["R"]), ("pS", la, TS["S"]), ("pD", la, TS["digits"]), ("pR_base", lb, TS["R"]), ("pS_base", lb, TS["S"]), ("pD_base", lb, TS["digits"])):
                acc[k][L].append(lp[:, S].exp().sum(-1).cpu() if S else torch.zeros(lp.shape[0]))
    out = {k: [float(torch.cat(acc[k][L]).mean()) for L in range(N_LAYERS)] for k in keys}
    out["coherence_last"] = []
    for L in range(N_LAYERS):   # mean pairwise cosine of per-prompt displacement at the last prompt token (1 = one shared shift)
        dn = F.normalize(torch.cat(disp[L]), dim=-1); cm = dn @ dn.T; n = cm.shape[0]
        out["coherence_last"].append(float((cm.sum() - n) / max(n * (n - 1), 1)))
    return out
R.setdefault("drift_lens", {})
need = [(s, n) for s in CFG["seeds"] for n in range(1, LAST + 1) if f"s{s}_n{n}" not in R["drift_lens"]]
if need:
    t = time.time(); ST_BASE = [([h.half() for h in hl], m) for hl, m in layer_states_batches(base, BASE_OK["gsm8k"])]
    for s, n in need:
        m = load_adapter(s, n); d = drift_lens(ST_BASE, m); unload(m)
        R["drift_lens"][f"s{s}_n{n}"] = d; save()
        log(f"[drift/lens s{s} n{n}] cos-dist(last tok) by layer {[round(x,3) for x in d['cos_last']]} | final-layer KL {d['kl_last'][-1]:.3f} "
            f"| final P(R) {d['pR'][-1]:.3f} (base {d['pR_base'][-1]:.3f}) P(S) {d['pS'][-1]:.3f} (base {d['pS_base'][-1]:.3f})")
    del ST_BASE; gc.collect(); R["timing"]["drift_lens"] = time.time() - t; save()

# ------------------------------------------------------------------ (e) layer swap on night-7 adapters
def set_lora_scale(model, layers_off, orig=None):
    """multiply LoRA scaling by 0 on the given layers (restorable). Returns the original scalings."""
    from peft.tuners.lora import LoraLayer
    if orig is None:
        orig = {}
        for name, mod in model.named_modules():
            if isinstance(mod, LoraLayer): orig[name] = dict(mod.scaling)
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLayer):
            li = int(re.search(r"layers\.(\d+)\.", name).group(1))
            for a in mod.scaling: mod.scaling[a] = 0.0 if li in layers_off else orig[name][a]
    return orig
BLOCKS = [list(range(i, min(i + CFG["block"], N_LAYERS))) for i in range(0, N_LAYERS, CFG["block"])]
SWAPS = {"none": []} | {f"L{i}": [i] for i in range(N_LAYERS)} | {f"B{b[0]}-{b[-1]}": b for b in BLOCKS} | {"all": ALL}
R.setdefault("swap", {})
SEEN_ALL = [f for f in FACTS if f["night"] <= LAST]
for s in CFG["seeds"]:
    done = R["swap"].setdefault(f"s{s}", {})
    todo = [k for k in SWAPS if k not in done]
    if not todo: continue
    m = load_adapter(s, LAST); orig = set_lora_scale(m, [])
    for k in todo:
        t = time.time(); set_lora_scale(m, SWAPS[k], orig)
        recs = eval_gsm(m, SUB, {"seed": s, "night": LAST, "cond": f"swap_{k}"}, None, max_new=CFG["swap_gsm_max_new"])
        g = C.agg_gsm(recs); rc = C.eval_recall(gen, m, SEEN_ALL, CFG)
        done[k] = {"layers_off": SWAPS[k], "gsm_strict": g["strict"], "gsm_lenient": g["lenient"], "mean_tokens": g["mean_tokens"],
                   "pct_steps": g["pct_steps"], "recall": rc, "sec": time.time() - t}
        save(); log(f"[swap s{s}] {k:7s} off -> GSM strict {g['strict']:.1f} (subset n={len(SUB)}) steps {g['pct_steps']:.0f}% tok {g['mean_tokens']:.0f} | recall {rc:.1f} | {time.time()-t:.0f}s")
    set_lora_scale(m, [], orig); unload(m)

# ------------------------------------------------------------------ (f) shared-direction test
from safetensors.torch import load_file
SCALE = CFG["lora_alpha"] / CFG["lora_r"]
def adapter_factors(seed, night):
    ensure_adapter(seed, night)
    sd = load_file(os.path.join(adir(seed, night), "adapter_model.safetensors"))
    fac = {}
    for k, v in sd.items():
        m = re.search(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])", k)
        if m: fac.setdefault((int(m.group(1)), m.group(2), m.group(3)), {})[m.group(4)] = v.float()
    return fac
def top1_of_product(B, A, scale):
    """exact top singular triplet of scale * B @ A (B: out x k, A: k x in) via QR"""
    Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
    U, S, Vh = torch.linalg.svd(scale * Rb @ Ra.T, full_matrices=False)
    return Qb @ U[:, 0], S[0], Qa @ Vh[0]
def target_weight(key):
    li, grp, name = key; return getattr(getattr(base.model.layers[li], grp), name).weight
R.setdefault("shared", {})
if any(f"s{s}" not in R["shared"] or len(R["shared"][f"s{s}"]) < 5 for s in CFG["seeds"]):
    t = time.time()
    allfac = {(s, n): adapter_factors(s, n) for s in CFG["seeds"] for n in range(1, LAST + 1)}
    keys = sorted(allfac[(CFG["seeds"][0], LAST)].keys())
    shared_dir, stats = {}, {"cos_seed_pairs": [], "cos_own_vs_shared": [], "energy_shared": [], "energy_own_top1": []}
    for key in keys:
        Bc = torch.cat([allfac[sn][key]["B"] for sn in allfac], 1).to(DEV); Ac = torch.cat([allfac[sn][key]["A"] for sn in allfac], 0).to(DEV)
        u, sv, v = top1_of_product(Bc, Ac, SCALE / len(allfac)); shared_dir[key] = (u, v)
        owns = []
        for s in CFG["seeds"]:
            B, A = allfac[(s, LAST)][key]["B"].to(DEV), allfac[(s, LAST)][key]["A"].to(DEV)
            uo, so, vo = top1_of_product(B, A, SCALE); owns.append(uo)
            dW = SCALE * B @ A; fro2 = float((dW ** 2).sum()) + 1e-20
            stats["energy_own_top1"].append(float(so ** 2) / fro2)
            stats["energy_shared"].append(float((u @ dW @ v) ** 2) / fro2)
            stats["cos_own_vs_shared"].append(abs(float(uo @ u)))
        for i in range(len(owns)):
            for j in range(i + 1, len(owns)): stats["cos_seed_pairs"].append(abs(float(owns[i] @ owns[j])))
    R["shared_stats"] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in stats.items()}
    R["shared_stats"]["note"] = "means over all 168 matrices; cos = |cosine| of top-1 left singular vectors; energy = fraction of ||dW||_F^2 at night 7"
    save(); log("shared-direction stats:", R["shared_stats"])
    W0 = {key: target_weight(key).detach().clone().cpu() for key in keys}
    for s in CFG["seeds"]:
        done = R["shared"].setdefault(f"s{s}", {})
        for variant in ["merged_unedited", "minus_shared_top1", "minus_own_top1", "minus_random_rank1", "base_no_adapter"]:
            if variant in done: continue
            t1 = time.time(); g_ = torch.Generator(device="cpu").manual_seed(4242 + s)
            with torch.no_grad():
                for ki, key in enumerate(keys):
                    W = target_weight(key); W.copy_(W0[key].to(DEV))
                    if variant == "base_no_adapter": continue
                    B, A = allfac[(s, LAST)][key]["B"].to(DEV), allfac[(s, LAST)][key]["A"].to(DEV); dW = SCALE * B @ A
                    if variant == "minus_shared_top1":
                        u, v = shared_dir[key]; dW = dW - (u @ dW @ v) * torch.outer(u, v)
                    elif variant == "minus_own_top1":
                        uo, so, vo = top1_of_product(B, A, SCALE); dW = dW - so * torch.outer(uo, vo)
                    elif variant == "minus_random_rank1":
                        uo, so, vo = top1_of_product(B, A, SCALE)
                        ru = F.normalize(torch.randn(dW.shape[0], generator=g_), dim=0).to(DEV); rv = F.normalize(torch.randn(dW.shape[1], generator=g_), dim=0).to(DEV)
                        dW = dW - so * torch.outer(ru, rv)
                    W.add_(dW.to(W.dtype))
            recs = eval_gsm(base, SUB, {"seed": s, "night": LAST, "cond": f"shared_{variant}"},
                            os.path.join(OUT, "gens", f"s{s}_n{LAST}_{variant}.jsonl"), max_new=CFG["swap_gsm_max_new"])
            g = C.agg_gsm(recs); rc = C.eval_recall(gen, base, SEEN_ALL, CFG)
            mr, _ = C.eval_simple(gen, base, PROBES, "mult", BASE_OK["mult"], CFG); ar, _ = C.eval_simple(gen, base, PROBES, "arc", BASE_OK["arc"], CFG)
            done[variant] = {"gsm_strict": g["strict"], "gsm_lenient": g["lenient"], "mean_tokens": g["mean_tokens"], "pct_steps": g["pct_steps"],
                             "recall": rc, "mult": mr, "arc": ar, "sec": time.time() - t1}
            save(); log(f"[shared s{s}] {variant:20s} GSM strict {g['strict']:.1f} steps {g['pct_steps']:.0f}% tok {g['mean_tokens']:.0f} | recall {rc:.1f} | mult {mr:.1f} arc {ar:.1f}")
    with torch.no_grad():
        for key in keys: target_weight(key).copy_(W0[key].to(DEV))
    assert abs(C.fingerprint(base) - FP0) < 1e-3, "base not restored after shared-direction edits"
    del allfac, W0; gc.collect(); R["timing"]["shared"] = time.time() - t; save()

# ------------------------------------------------------------------ summary, classification, failure dump, report
import phase1_summarize
phase1_summarize.main(OUT)
log(f"PHASE 1 DONE in {(time.time()-T0)/60:.1f} min (this session). Summary: {os.path.join(OUT, 'phase1_summary.json')}")
