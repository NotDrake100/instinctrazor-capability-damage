"""mathfix PHASE 2 -- the fix that targets the mechanism found in phase 1 (3 seeds, 7 nights, pre-registered criteria in
PREREG_phase2.md, written before any phase-2 run).

Reads <phase1_out>/phase1_summary.json -> phase2_choice.mechanism (override with --mechanism H1|H2|H3):
  H1 (format overwrite)  : arms b (facts only), r (facts + GSM8K-train step-by-step replay, 20% of training examples),
                           c (facts + the same number of supervised tokens of non-math Dolly-15k text)
  H2 (localized layers)  : arms b, p (no LoRA on the phase-1 layers whose swap restored math), rnd (no LoRA on the same
                           number of random other layers), bot (no LoRA on the bottom k layers)
  H3 (shared shift)      : arms b, x (after each night remove the top-1 singular component of every LoRA update),
                           rr (after each night remove a random rank-1 component of the same norm inside the update's span)
Configs: --config phase2_0p5b (Qwen2.5-0.5B-Instruct, default) or phase2_1p5b (Qwen2.5-1.5B-Instruct confirmation).
Everything else (facts, probes, parsers, LoRA r16, lr 2e-4, 120 steps, batch 16, fp16 autocast) as sleep_shield.py.
Resumable (results.json after every night, adapters per night). CPU smoke: MATHFIX_SMOKE=1 python phase2.py --phase1 <dir> --out <dir>
"""
import os, sys, json, re, math, time, random, gc, argparse
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import DEV, AMP

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="phase2_0p5b", choices=["phase2_0p5b", "phase2_1p5b"])
ap.add_argument("--phase1", default=os.path.join(HERE, "phase1_out"), help="phase-1 output dir (reads phase1_summary.json)")
ap.add_argument("--mechanism", default=None, choices=[None, "H1", "H2", "H3"])
ap.add_argument("--out", default=None)
args = ap.parse_args()
SMOKE = os.environ.get("MATHFIX_SMOKE", "0") == "1"
CFG = dict(C.BASE_CFG)
if args.config == "phase2_1p5b": CFG["model"] = "Qwen/Qwen2.5-1.5B-Instruct"
CFG.update(replay_frac=0.2, replay_slots=[3, 3, 3, 3, 4], replay_seed=2024, filler_seed=2025, filler_source="databricks/databricks-dolly-15k")
if SMOKE: CFG.update(C.SMOKE_OVERRIDES); CFG.update(replay_slots=[1])
CFG["replay_K"] = CFG["steps_per_night"] * sum(CFG["replay_slots"]) // len(CFG["replay_slots"])   # 120*3.2 = 384 = 20% of 1920 examples
S1 = json.load(open(os.path.join(args.phase1, "phase1_summary.json")))
choice = dict(S1["phase2_choice"]); MECH = args.mechanism or choice["mechanism"]
CFG["mechanism"] = MECH
OUT = args.out or os.path.join(HERE, f"{args.config}_{MECH}_out"); os.makedirs(os.path.join(OUT, "gens"), exist_ok=True)
log = C.Logger(OUT); RES = os.path.join(OUT, "results.json")
T0 = time.time(); log("=" * 30, "mathfix phase 2 start |", args.config, "| mechanism", MECH, "(override)" if args.mechanism else f"(phase-1 choice: {choice['reason']})", "| device", DEV)

PROBES, PH = C.load_probes(CFG, SMOKE); FACTS, FH, FACT_VOCAB = C.load_facts(CFG, SMOKE)
tok, base = C.load_model(CFG["model"]); gen = C.Gen(tok, base); TR = C.Trainer(CFG, tok, gen.chat)
N_LAYERS = len(base.model.layers); FP0 = C.fingerprint(base); ALL = list(range(N_LAYERS)); GSM = PROBES["gsm8k"]; MAXG = CFG["gsm_max_new"]
if DEV == "cuda": log("GPU:", torch.cuda.get_device_name(0))

# ------------------------------------------------------------------ arms
ARMS = {"b": {"layers": ALL, "extra": None, "edit": None}}
if MECH == "H1":
    ARMS["r"] = {"layers": ALL, "extra": "replay", "edit": None}; ARMS["c"] = {"layers": ALL, "extra": "filler", "edit": None}
    FIX, CONTROLS = "r", ["c"]
elif MECH == "H2":
    p1_layers = choice.get("protect_layers") or S1["evidence"]["H2"]["best_block_layers"]
    n1 = S1.get("n_layers") or (2 if SMOKE else 24)
    prot = sorted({min(N_LAYERS - 1, round(l * (N_LAYERS - 1) / max(n1 - 1, 1))) for l in p1_layers}) if N_LAYERS != n1 else sorted(p1_layers)
    if len(prot) >= N_LAYERS:
        assert SMOKE, f"protect layers {prot} cover every layer"; prot = [0]
    k = len(prot); CFG["protect_layers"] = prot; CFG["protect_layers_phase1"] = p1_layers
    bottom = list(range(k)) if set(range(k)) != set(prot) else list(range(N_LAYERS - k, N_LAYERS))
    ARMS["p"] = {"layers": [l for l in ALL if l not in prot], "extra": None, "edit": None}
    ARMS["bot"] = {"layers": [l for l in ALL if l not in bottom], "extra": None, "edit": None}
    ARMS["rnd"] = {"layers": None, "extra": None, "edit": None}          # per seed, below
    FIX, CONTROLS = "p", ["rnd", "bot"]
    log("H2 protect layers", prot, "(phase-1 layers", p1_layers, ") | bottom control", bottom)
elif MECH == "H3":
    ARMS["x"] = {"layers": ALL, "extra": None, "edit": "top1"}; ARMS["rr"] = {"layers": ALL, "extra": None, "edit": "rand"}
    FIX, CONTROLS = "x", ["rr"]
CFG["arms"] = list(ARMS); CFG["fix_arm"] = FIX; CFG["control_arms"] = CONTROLS
def arm_layers(arm, seed):
    if arm == "rnd":
        pool = [l for l in ALL if l not in CFG["protect_layers"]]
        ex = sorted(random.Random(777 + seed).sample(pool, min(len(CFG["protect_layers"]), len(pool))))
        return [l for l in ALL if l not in ex]
    return ARMS[arm]["layers"]

R = json.load(open(RES)) if os.path.exists(RES) else {"cfg": CFG, "runs": {}}
for k in ["model", "nights", "facts_per_night", "lr", "steps_per_night", "lora_r", "batch_size", "gsm_max_new", "arms", "mechanism", "replay_K", "replay_slots"]:
    assert R["cfg"].get(k) == CFG[k], f"config changed vs saved results ({k}); use a new --out"
R["probe_hash"], R["fact_hash"] = PH, FH
def save(): C.save_json(RES, R)
save()

# ------------------------------------------------------------------ replay + filler data (H1)
EXTRA = {}
if MECH == "H1":
    from datasets import load_dataset
    gtr = load_dataset("openai/gsm8k", "main", split="train")
    kl_idx = set(random.Random(99).sample(range(len(gtr)), 32))          # sleep_shield's KL texts: keep them unused
    probe_qs = {x["q"].split("\nSolve step by step.")[0] for x in GSM}
    pool = [i for i in range(len(gtr)) if i not in kl_idx and gtr[i]["question"] not in probe_qs]
    ridx = random.Random(CFG["replay_seed"]).sample(pool, CFG["replay_K"])
    SUFFIX = "\nSolve step by step. End with a final line exactly of the form 'The answer is N.' where N is a number."
    def solution(a):
        body, ans = a.split("####"); body = re.sub(r"<<[^>]*>>", "", body).strip()
        return body + "\nThe answer is " + ans.strip().replace(",", "") + "."
    replay = [TR.encode(gtr[i]["question"] + SUFFIX, solution(gtr[i]["answer"])) for i in ridx]
    sup = lambda x: sum(1 for t in x[1] if t != -100)
    # filler: Dolly-15k, no digits anywhere (non-math), response at least as long as the paired replay answer, truncated to
    # exactly the same number of supervised tokens (+ end-of-turn token) as its paired replay item.
    dol = load_dataset(CFG["filler_source"], split="train")
    cand = [i for i in range(len(dol)) if not re.search(r"\d", dol[i]["instruction"] + dol[i]["context"] + dol[i]["response"])]
    random.Random(CFG["filler_seed"]).shuffle(cand)
    rlen = {i: len(tok(dol[i]["response"], add_special_tokens=False)["input_ids"]) for i in cand}
    used, filler = set(), []
    END_ID = tok(TR.END, add_special_tokens=False)["input_ids"]
    for x in replay:
        need = sup(x) - len(END_ID)                                   # answer tokens excluding the end-of-turn token(s)
        j = next((i for i in cand if i not in used and need <= rlen[i] <= 3 * need), None)
        if j is None: j = next(i for i in cand if i not in used and rlen[i] >= need)
        used.add(j); d = dol[j]
        q = d["instruction"] + (("\n\n" + d["context"]) if d["context"].strip() else "")
        p = gen.chat(q); pi = tok(p, add_special_tokens=False)["input_ids"]
        ai = tok(d["response"], add_special_tokens=False)["input_ids"][:need] + END_ID
        filler.append((pi + ai, [-100] * len(pi) + ai))
    assert all(sup(a) == sup(b) for a, b in zip(replay, filler)), "filler/replay supervised-token mismatch"
    EXTRA = {"replay": replay, "filler": filler}
    R["extra_stats"] = {"K": len(replay), "replay_gsm8k_train_idx_first10": ridx[:10],
                        "replay_supervised_tokens": sum(map(sup, replay)), "filler_supervised_tokens": sum(map(sup, filler)),
                        "replay_total_tokens": sum(len(x[0]) for x in replay), "filler_total_tokens": sum(len(x[0]) for x in filler),
                        "dolly_candidates_no_digits": len(cand), "replay_example": solution(gtr[ridx[0]]["answer"]),
                        "filler_example": tok.decode(filler[0][0])[-400:]}
    save(); log("H1 data:", {k: v for k, v in R["extra_stats"].items() if not k.endswith("example")})

# ------------------------------------------------------------------ base eval
if "base" not in R:
    outs, nt, tr = gen(base, [x["q"] for x in GSM], MAXG, CFG["gen_bs"])
    recs = C.score_gsm(GSM, list(range(len(GSM))), outs, nt, tr, FACT_VOCAB, gen.first)
    with open(os.path.join(OUT, "gens", "base.jsonl"), "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    m_acc, m_c = C.eval_simple(gen, base, PROBES, "mult", list(range(len(PROBES["mult"]))), CFG)
    a_acc, a_c = C.eval_simple(gen, base, PROBES, "arc", list(range(len(PROBES["arc"]))), CFG)
    R["base"] = {"gsm_items": [r["strict_ok"] for r in recs], "mult_items": m_c, "arc_items": a_c,
                 "acc": {"gsm8k_strict": 100 * float(np.mean([r["strict_ok"] for r in recs])), "mult": m_acc, "arc": a_acc}}
    R["base_recall"] = C.eval_recall(gen, base, FACTS, CFG); save()
BASE_OK = {"gsm8k": [i for i, c in enumerate(R["base"]["gsm_items"]) if c], "mult": [i for i, c in enumerate(R["base"]["mult_items"]) if c],
           "arc": [i for i, c in enumerate(R["base"]["arc_items"]) if c]}
if SMOKE:
    for k in BASE_OK:
        if not BASE_OK[k]: BASE_OK[k] = list(range(len(PROBES[k])))
log("base acc", R["base"]["acc"], "| base-correct", {k: len(v) for k, v in BASE_OK.items()}, "| base recall (all facts)", R["base_recall"])

# ------------------------------------------------------------------ per-night edits (H3)
@torch.no_grad()
def edit_lora(model, kind, seed, night):
    """kind 'top1': remove the top singular component of every LoRA update dW = s*B@A (exact, stays rank<=r).
    kind 'rand': remove sigma_1 * a b^T with a, b random unit vectors inside span(U), span(V) of dW (same norm as top1,
    random direction, stays representable at rank r). Re-factor: B = U sqrt(S/s), A = sqrt(S/s) V^T."""
    from peft.tuners.lora import LoraLayer
    g = torch.Generator(device="cpu").manual_seed(seed * 1000 + night); n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LoraLayer): continue
        for a in mod.lora_A:
            A, B = mod.lora_A[a].weight, mod.lora_B[a].weight; s = mod.scaling[a]
            Qb, Rb = torch.linalg.qr(B.float()); Qa, Ra = torch.linalg.qr(A.float().T)
            U, Sv, Vh = torch.linalg.svd(s * Rb @ Ra.T, full_matrices=False); U, Vh = Qb @ U, Vh @ Qa.T          # dW = U diag(Sv) Vh
            if kind == "top1":
                Sv = Sv.clone(); Sv[0] = 0.0
            else:
                ca = F.normalize(torch.randn(U.shape[1], generator=g), dim=0).to(U.device); cb = F.normalize(torch.randn(Vh.shape[0], generator=g), dim=0).to(U.device)
                core = torch.diag(Sv) - Sv[0] * torch.outer(ca, cb)                          # in the (U, V) basis
                u2, s2, v2 = torch.linalg.svd(core, full_matrices=False); U, Sv, Vh = U @ u2, s2, v2 @ Vh
            r = A.shape[0]; m = min(r, Sv.shape[0]); sq = torch.sqrt(Sv[:m].clamp_min(0) / s)
            Bn = torch.zeros_like(B, dtype=torch.float32); An = torch.zeros_like(A, dtype=torch.float32)   # rank-deficient dims stay 0
            Bn[:, :m] = U[:, :m] * sq; An[:m] = sq[:, None] * Vh[:m]
            B.copy_(Bn.to(B.dtype)); A.copy_(An.to(A.dtype)); n += 1
    return n

# ------------------------------------------------------------------ training + eval loop
from peft import get_peft_model, PeftModel
def adir(key, night): return os.path.join(OUT, "adapters", key, f"night{night}")
def train_one(arm, seed, night):
    global base
    key = f"{arm}_s{seed}"; layers = arm_layers(arm, seed)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = get_peft_model(base, C.lora_cfg(CFG, layers)) if night == 1 else PeftModel.from_pretrained(base, adir(key, night - 1), is_trainable=True)
    got = sorted({int(re.search(r"layers\.(\d+)\.", n).group(1)) for n, p in model.named_parameters() if p.requires_grad})
    assert got == layers, f"LoRA layer filter wrong: {got} vs {layers}"
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    seen = [f for f in FACTS if f["night"] <= night]; ex = ARMS[arm]["extra"]
    t = time.time(); loss = TR.train_night(model, seen, seed, night, extra=EXTRA.get(ex), extra_slots=CFG["replay_slots"] if ex else None); tt = time.time() - t
    nedit = edit_lora(model, ARMS[arm]["edit"], seed, night) if ARMS[arm]["edit"] else 0
    model.save_pretrained(adir(key, night)); base = model.unload(); del model; gc.collect()
    assert abs(C.fingerprint(base) - FP0) < 1e-6, "base modified"
    return loss, ntrain, tt, layers, nedit
def ensure_adapter(arm, seed, night):
    for n in range(1, night + 1):
        if not os.path.isdir(adir(f"{arm}_s{seed}", n)):
            loss, *_ = train_one(arm, seed, n); log(f"[{arm}_s{seed}] replayed missing adapter night {n} (loss {loss:.3f})")

total = len(ARMS) * len(CFG["seeds"]) * CFG["nights"]; t_start = time.time(); done_now = 0
for seed in CFG["seeds"]:
    for arm in ARMS:
        key = f"{arm}_s{seed}"; run = R["runs"].setdefault(key, {"nights": {}})
        for night in range(1, CFG["nights"] + 1):
            if str(night) in run["nights"]: continue
            if night > 1: ensure_adapter(arm, seed, night - 1)
            t0 = time.time(); loss, ntrain, tt, layers, nedit = train_one(arm, seed, night)
            run["excluded_layers"] = [l for l in ALL if l not in layers]; run["n_trainable"] = ntrain
            model = PeftModel.from_pretrained(base, adir(key, night)); model.eval()
            seen = [f for f in FACTS if f["night"] <= night]
            rec = C.eval_recall(gen, model, seen, CFG); rec_new = C.eval_recall(gen, model, [f for f in seen if f["night"] == night], CFG)
            m_ret, _ = C.eval_simple(gen, model, PROBES, "mult", BASE_OK["mult"], CFG); a_ret, _ = C.eval_simple(gen, model, PROBES, "arc", BASE_OK["arc"], CFG)
            outs, nt, tr = gen(model, [GSM[i]["q"] for i in BASE_OK["gsm8k"]], MAXG, CFG["gen_bs"])
            recs = C.score_gsm(GSM, BASE_OK["gsm8k"], outs, nt, tr, FACT_VOCAB, gen.first)
            with open(os.path.join(OUT, "gens", f"{key}_n{night}.jsonl"), "w") as f:
                for r in recs: f.write(json.dumps(r) + "\n")
            g = C.agg_gsm(recs)
            off = None
            if night in (1, CFG["nights"]):
                with model.disable_adapter(): off = C.eval_recall(gen, model, seen, CFG)
            base = model.unload(); del model; gc.collect()
            if DEV == "cuda": torch.cuda.empty_cache()
            run["nights"][str(night)] = {"recall": rec, "recall_tonight": rec_new, "mult": m_ret, "arc": a_ret, "gsm": g, "recall_adapter_off": off,
                                         "train_loss": loss, "sec_train": tt, "sec": time.time() - t0, "n_edited_modules": nedit}
            save(); done_now += 1; left = total - sum(len(v["nights"]) for v in R["runs"].values())
            log(f"[{key} n{night}] recall {rec:.1f} (tonight {rec_new:.1f}) | GSM strict {g['strict']:.1f} lenient {g['lenient']:.1f} tok {g['mean_tokens']:.0f} "
                f"steps {g['pct_steps']:.0f}% | mult {m_ret:.1f} arc {a_ret:.1f} | loss {loss:.3f} | train {tt:.0f}s total {time.time()-t0:.0f}s | "
                f"ETA {(time.time()-t_start)/done_now*left/60:.0f} min")
            if done_now == 1:
                est = left * (time.time() - t0) * (1.6 if MECH == "H1" else 1.1) + (time.time() - T0)
                log(f"RUNTIME ESTIMATE (after first night): ~{est/3600:.2f} h total (crude: first night is arm b; replay/filler arms cost more per step)")

# ------------------------------------------------------------------ verdict (pre-registered, PREREG_phase2.md)
N = CFG["nights"]; S = CFG["seeds"]
def fin(arm, f):
    vals = []
    for s in S:
        e = R["runs"][f"{arm}_s{s}"]["nights"][str(N)]
        vals.append(e["gsm"][f] if f in e["gsm"] else e[f])
    return vals
summ = {a: {f: C.mean_std(fin(a, f)) for f in ["strict", "lenient", "mean_tokens", "pct_steps", "recall", "mult", "arc"]} for a in ARMS}
F_ = summ[FIX]
best_c = max(CONTROLS, key=lambda c: summ[c]["strict"][0])
comb = math.sqrt(F_["strict"][1] ** 2 + summ[best_c]["strict"][1] ** 2)
fails = []
if not F_["strict"][0] >= 50.0: fails.append(f"P1: {FIX} GSM8K strict retention {F_['strict'][0]:.1f} < 50")
if not F_["strict"][0] - summ[best_c]["strict"][0] > comb: fails.append(f"P2: {FIX} - {best_c} = {F_['strict'][0]-summ[best_c]['strict'][0]:.1f} <= combined std {comb:.1f}")
if not abs(F_["recall"][0] - summ["b"]["recall"][0]) <= 5.0: fails.append(f"P3: |recall {FIX} - b| = {abs(F_['recall'][0]-summ['b']['recall'][0]):.1f} > 5")
offs = [R["runs"][f"{a}_s{s}"]["nights"][str(n)]["recall_adapter_off"] for a in ARMS for s in S for n in sorted({1, N})]
valid = all(o is None or o <= R["base_recall"] + 5 for o in offs)
if not valid: fails.append("validity: adapter-off recall not near base")
b_gap = summ["b"]["lenient"][0] - summ["b"]["strict"][0]
V = {"verdict": "PASS" if not fails else "FAIL", "fails": fails, "mechanism": MECH, "fix": FIX, "controls": CONTROLS, "best_control": best_c,
     "combined_std": comb, "night_final": summ, "b_lenient_minus_strict": b_gap, "b_format_loss_parser_level": bool(b_gap >= 20.0), "config": args.config}
R["verdict"] = V; save()
print(json.dumps(V, indent=1)); log(f"VERDICT ({args.config}, {MECH}): {V['verdict']}" + ("" if not fails else " | " + " ; ".join(fails)))
log(f"PHASE 2 DONE in {(time.time()-T0)/60:.1f} min (this session)")
