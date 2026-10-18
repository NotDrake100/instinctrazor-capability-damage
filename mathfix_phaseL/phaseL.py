"""Phase L driver: is arm-b format damage localized in a few LoRA layers?

Pre-registered in PREREG_phaseL.md. Do not change k, N-after-pilot, calibration
metric, sets, or seeds after any test-set eval.

Modes:
  python phaseL.py --pilot     # timing + N choice; no test-set eval
  python phaseL.py --smoke     # tiny-model path (meaningless numbers)
  python phaseL.py             # full experiment (reads frozen N)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "mathfix"))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from common import AMP, DEV  # noqa: E402
from zeroing import restore_lora_weights, snapshot_lora_weights, zero_lora_delta_layers  # noqa: E402

SUFFIX = "\nSolve step by step. End with a final line exactly of the form 'The answer is N.' where N is a number."
CALIB_SEED = 20261018
RANDOM_SUBSET_SEED = 1234
K_VERDICT = 4
K_POSTHOC = (2, 8)
N_CALIB = 32
N_RANDOM = 8
PREREG_PATH = os.path.join(HERE, "PREREG_phaseL.md")
REPO = os.path.abspath(os.path.join(HERE, ".."))


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_time() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "log", "-1", "--format=%cI"], text=True).strip()
    except Exception:
        return ""


ap = argparse.ArgumentParser()
ap.add_argument("--pilot", action="store_true")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--nights", type=int, default=None, help="only for --smoke or after N is frozen in results")
args = ap.parse_args()
SMOKE = args.smoke or os.environ.get("MATHFIX_SMOKE", "0") == "1"

OUT = args.out or (
    os.path.join(HERE, "work", "smoke") if SMOKE else os.path.join(REPO, "results", "phaseL")
)
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "gens"), exist_ok=True)
os.makedirs(os.path.join(OUT, "adapters"), exist_ok=True)
log = C.Logger(OUT)
RES = os.path.join(OUT, "results.json")


def save(obj):
    C.save_json(RES, obj)


def load() -> dict:
    return json.load(open(RES)) if os.path.exists(RES) else {}


CFG = dict(C.BASE_CFG)
if SMOKE:
    CFG.update(C.SMOKE_OVERRIDES)
    CFG["nights"] = args.nights or 1
    CFG["n_calib_phaseL"] = min(N_CALIB, CFG["n_probe"])
else:
    CFG["n_calib_phaseL"] = N_CALIB

# CPU: smaller generate batches. Greedy + left pad as in common.Gen; not a design change.
if DEV == "cpu":
    CFG["gen_bs"] = 4 if not SMOKE else min(CFG.get("gen_bs", 8), 4)
    CFG["recall_bs"] = 16 if not SMOKE else min(CFG.get("recall_bs", 8), 8)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

CFG["k_verdict"] = K_VERDICT
CFG["calib_seed"] = CALIB_SEED
CFG["random_subset_seed"] = RANDOM_SUBSET_SEED
CFG["gsm_max_new"] = 16 if SMOKE else C.BASE_CFG["gsm_max_new"]  # 512 on the real run


def random_subsets(n_layers: int, k: int, n: int, seed: int):
    k = min(k, n_layers)
    rng = random.Random(seed)
    out, guard = [], 0
    while len(out) < n and guard < 10000:
        s = tuple(sorted(rng.sample(range(n_layers), k)))
        if s not in out:
            out.append(list(s))
        guard += 1
    return out


def build_calib(n, probe_gsm, smoke):
    if smoke:
        items = []
        for i, x in enumerate(probe_gsm[:n]):
            items.append({"q": x["q"], "a": x["a"], "train_idx": i})
        return list(range(len(items))), items
    from datasets import load_dataset
    gtr = load_dataset("openai/gsm8k", "main", split="train")
    probe_qs = {x["q"].split("\nSolve step by step.")[0] for x in probe_gsm}
    eligible = [i for i in range(len(gtr)) if gtr[i]["question"] not in probe_qs]
    idx = sorted(random.Random(CALIB_SEED).sample(eligible, n))
    items = [{
        "q": gtr[i]["question"] + SUFFIX,
        "a": gtr[i]["answer"].split("####")[-1].strip().replace(",", ""),
        "train_idx": i,
    } for i in idx]
    return idx, items


def adir(seed, night):
    return os.path.join(OUT, "adapters", f"s{seed}", f"night{night}")


def train_one(base, tok, gen, TR, CFG, FACTS, ALL, seed, night, FP0):
    from peft import PeftModel, get_peft_model
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if night == 1:
        model = get_peft_model(base, C.lora_cfg(CFG, ALL))
    else:
        model = PeftModel.from_pretrained(base, adir(seed, night - 1), is_trainable=True)
    seen = [f for f in FACTS if f["night"] <= night]
    t = time.time()
    loss = TR.train_night(model, seen, seed, night)
    tt = time.time() - t
    model.save_pretrained(adir(seed, night))
    base = model.unload()
    del model
    gc.collect()
    assert abs(C.fingerprint(base) - FP0) < 1e-3, "base weights changed"
    return base, float(loss), tt


def load_adapter(base, seed, night):
    from peft import PeftModel
    m = PeftModel.from_pretrained(base, adir(seed, night))
    m.eval()
    return m


def unload(model):
    base = model.unload()
    del model
    gc.collect()
    return base


def eval_gsm_items(gen, model, items, idx, CFG, fact_vocab, path=None, extra=None):
    qs = [items[i]["q"] for i in idx]
    outs, nt, tr = gen(model, qs, CFG["gsm_max_new"], CFG["gen_bs"])
    recs = C.score_gsm(items, idx, outs, nt, tr, fact_vocab, gen.first)
    if path:
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps({**(extra or {}), **r}) + "\n")
    return recs


def retention_and_agg(recs):
    g = C.agg_gsm(recs)
    return float(g["strict"]), g


def rank_layers(mean_tokens):
    # descending mean_tokens; ties: lower index first
    return sorted(range(len(mean_tokens)), key=lambda i: (-mean_tokens[i], i))


def predict_hours(pilot, N, n_layers, n_base_correct_guess=99):
    """Wall-time prediction from measured step/token rates. For N-choice only."""
    step_s = pilot["sec_per_train_step"]
    tok_s = max(pilot["gen_tokens_per_sec"], 1e-6)
    # Training: 3 seeds × N × 120 steps
    train = 3 * N * CFG["steps_per_night"] * step_s
    # Base GSM: 200 items × assumed 218 tokens (prior mathfix mean; timing only)
    base = 200 * 218 / tok_s
    # Calib: 3 seeds × n_layers × 32 items. Most layers stay short (~15 tok);
    # assume 4 layers restore ~180 tok (conservative mix).
    short_tok, long_tok = 15.0, 180.0
    n_long = min(4, n_layers)
    calib_tok = 3 * ((n_layers - n_long) * N_CALIB * short_tok + n_long * N_CALIB * long_tok)
    calib = calib_tok / tok_s
    # Test GSM: full_b short; selected maybe long; 8 random mixed
    gsm_test = (
        3 * n_base_correct_guess * 15
        + 3 * n_base_correct_guess * 180
        + 24 * n_base_correct_guess * 40
    ) / tok_s
    # Recall: ~ N*100*2 questions × ~8 tokens, for full_b + selected + 8 random + k2/k8
    n_rec_cond = 1 + 1 + 8 + 2  # full_b, selected_k4, 8 random, k2, k8
    recall = 3 * n_rec_cond * (N * 100 * 2) * 8 / tok_s
    overhead = 600  # load/save/gc
    parts = {
        "train": train,
        "base_gsm": base,
        "calib_scan": calib,
        "test_gsm": gsm_test,
        "recall": recall,
        "overhead": overhead,
    }
    return parts, sum(parts.values()) / 3600.0


def run_pilot():
    """A few train steps + a few generations. No test-set scoring."""
    log("=" * 30, "Phase L timing pilot | device", DEV)
    t_all = time.time()
    CFG_P = dict(C.BASE_CFG)
    CFG_P["nights"] = 2  # fact prefix only; we take night-1 facts
    FACTS, _, _ = C.load_facts({**CFG_P, "nights": 2}, smoke=True)  # skip 7-night hash
    # Use real night-1 facts so step time matches arm b
    facts = [f for f in FACTS if f["night"] == 1][: CFG_P["facts_per_night"]]
    tok, base = C.load_model(CFG_P["model"])
    n_layers = len(base.model.layers)
    gen = C.Gen(tok, base)
    TR = C.Trainer(CFG_P, tok, gen.chat)
    from peft import get_peft_model
    model = get_peft_model(base, C.lora_cfg(CFG_P, list(range(n_layers))))
    # time 4 train steps (drop the first as warmup)
    CFG_P["steps_per_night"] = 4
    t0 = time.time()
    loss = TR.train_night(model, facts, seed=0, night=1)
    train_sec = time.time() - t0
    sec_per_step = train_sec / 4.0
    # generations: 4 short GSM-like prompts, cap 64 new tokens
    qs = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" + SUFFIX,
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?" + SUFFIX,
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. How much more does she need?" + SUFFIX,
        "Julie is reading a 120-page book. She has already read 1/3 of it. How many pages does she have left?" + SUFFIX,
    ]
    t1 = time.time()
    outs, nt, tr = gen(model, qs, 64, 2)
    gen_sec = time.time() - t1
    ntok = int(sum(nt))
    tok_s = ntok / max(gen_sec, 1e-6)
    # one longer cap to see whether generate overhead dominates
    t2 = time.time()
    _, nt2, _ = gen(model, qs[:2], 128, 1)
    gen2_sec = time.time() - t2
    ntok2 = int(sum(nt2))
    base = model.unload()
    del model
    gc.collect()
    pilot = {
        "device": DEV,
        "model": CFG_P["model"],
        "n_layers": n_layers,
        "n_cpu": os.cpu_count(),
        "train_steps_timed": 4,
        "train_loss_last10mean": float(loss),
        "sec_train_4steps": train_sec,
        "sec_per_train_step": sec_per_step,
        "gen_prompts": 4,
        "gen_max_new": 64,
        "gen_n_tokens": ntok,
        "gen_sec": gen_sec,
        "gen_tokens_per_sec": tok_s,
        "gen2_max_new": 128,
        "gen2_n_tokens": ntok2,
        "gen2_sec": gen2_sec,
        "sample_out0_head": (outs[0] or "")[:200],
        "sample_ntok": nt,
        "wall_sec": time.time() - t_all,
    }
    pred = {}
    for N in (1, 2):
        parts, hours = predict_hours(pilot, N, n_layers)
        pred[str(N)] = {"hours": hours, "parts_sec": parts}
    # Decision rule from PREREG: if N=2 predicted > ~5 h, choose N=1
    chosen = 1 if pred["2"]["hours"] > 5.0 else 2
    pilot["predicted_hours"] = pred
    pilot["chosen_N"] = chosen
    pilot["decision_rule"] = "N=2 unless predicted wall for the full plan exceeds ~5 h, then N=1"
    C.save_json(os.path.join(OUT, "timing_pilot.json"), pilot)
    log("PILOT", json.dumps({k: pilot[k] for k in [
        "sec_per_train_step", "gen_tokens_per_sec", "gen_n_tokens", "gen_sec",
        "chosen_N",
    ]}, indent=1))
    log(f"predicted N=2 {pred['2']['hours']:.2f} h | N=1 {pred['1']['hours']:.2f} h | CHOSEN N={chosen}")
    return pilot


def append_prereg_n(pilot):
    """Write Appendix A. Called only from --pilot, before any test eval."""
    text = open(PREREG_PATH).read()
    marker = "## Appendix A — timing pilot and N choice"
    block = f"""## Appendix A — timing pilot and N choice

Written {time.strftime("%Y-%m-%d %H:%M:%S %z")} after the timing pilot and **before any test-set eval**.
Prereg commit at first lock: see git history of this file before this appendix.

### Measured (real Qwen2.5-0.5B-Instruct, CPU, not test-set)

| quantity | value |
|---|---|
| device | {pilot["device"]} |
| n_layers | {pilot["n_layers"]} |
| n_cpu | {pilot["n_cpu"]} |
| sec / train step (4 steps incl. first) | {pilot["sec_per_train_step"]:.4f} |
| gen tokens / sec (4 prompts, max_new=64) | {pilot["gen_tokens_per_sec"]:.4f} |
| tokens produced in that gen | {pilot["gen_n_tokens"]} |
| gen wall sec | {pilot["gen_sec"]:.2f} |
| second gen (2 prompts, max_new=128) tokens / sec | {pilot["gen2_n_tokens"] / max(pilot["gen2_sec"], 1e-6):.4f} |
| pilot wall sec | {pilot["wall_sec"]:.1f} |

### Predicted full-plan wall (from those rates; not measured)

| N | predicted hours | train h | base GSM h | calib h | test GSM h | recall h |
|---|---|---|---|---|---|---|
| 1 | {pilot["predicted_hours"]["1"]["hours"]:.2f} | {pilot["predicted_hours"]["1"]["parts_sec"]["train"]/3600:.2f} | {pilot["predicted_hours"]["1"]["parts_sec"]["base_gsm"]/3600:.2f} | {pilot["predicted_hours"]["1"]["parts_sec"]["calib_scan"]/3600:.2f} | {pilot["predicted_hours"]["1"]["parts_sec"]["test_gsm"]/3600:.2f} | {pilot["predicted_hours"]["1"]["parts_sec"]["recall"]/3600:.2f} |
| 2 | {pilot["predicted_hours"]["2"]["hours"]:.2f} | {pilot["predicted_hours"]["2"]["parts_sec"]["train"]/3600:.2f} | {pilot["predicted_hours"]["2"]["parts_sec"]["base_gsm"]/3600:.2f} | {pilot["predicted_hours"]["2"]["parts_sec"]["calib_scan"]/3600:.2f} | {pilot["predicted_hours"]["2"]["parts_sec"]["test_gsm"]/3600:.2f} | {pilot["predicted_hours"]["2"]["parts_sec"]["recall"]/3600:.2f} |

Prediction assumptions (timing only, never used as experimental numbers): base mean 218 tokens;
calib 20 layers × 15 tok + 4 layers × 180 tok; test mix short/long as in `predict_hours`.

### N choice

**N = {pilot["chosen_N"]}**

Rule applied: default N=2; if predicted wall for the full plan exceeds ~5 h, set N=1.
N=2 predicted {pilot["predicted_hours"]["2"]["hours"]:.2f} h, which is
{"above" if pilot["predicted_hours"]["2"]["hours"] > 5 else "at or below"} the 5 h cutoff.
"""
    if marker in text:
        pre = text.split(marker)[0].rstrip() + "\n\n"
        open(PREREG_PATH, "w").write(pre + block)
    else:
        open(PREREG_PATH, "a").write("\n\n" + block)
    return pilot["chosen_N"]


def summarize_verdict(R):
    seeds = R["cfg"]["seeds"]
    ret_sel, rec_sel, ret_b, rec_b = [], [], [], []
    random_rets = []
    for s in seeds:
        run = R["runs"][f"s{s}"]
        ret_sel.append(run["selected_k4"]["retention"])
        rec_sel.append(run["selected_k4"]["recall"])
        ret_b.append(run["full_b"]["retention"])
        rec_b.append(run["full_b"]["recall"])
        for rr in run["random"]:
            random_rets.append(rr["retention"])
    mean_sel = float(np.mean(ret_sel))
    mean_sel_rec = float(np.mean(rec_sel))
    mean_b = float(np.mean(ret_b))
    mean_b_rec = float(np.mean(rec_b))
    mean_rand = float(np.mean(random_rets))
    std_rand = float(np.std(random_rets, ddof=1)) if len(random_rets) > 1 else 0.0
    c1 = mean_sel >= 50.0
    c2 = mean_sel_rec >= mean_b_rec - 5.0
    c3 = (mean_sel - mean_rand) > (2.0 * std_rand)
    valid = mean_b < 50.0
    if not valid:
        label = "INVALID"
    elif c1 and c2 and c3:
        label = "PASS"
    else:
        label = "FAIL"
    return {
        "mean_selected_retention": mean_sel,
        "mean_selected_recall": mean_sel_rec,
        "mean_full_b_retention": mean_b,
        "mean_full_b_recall": mean_b_rec,
        "mean_random_retention": mean_rand,
        "std_random_retention_ddof1": std_rand,
        "selected_minus_random": mean_sel - mean_rand,
        "threshold_2std": 2.0 * std_rand,
        "C1_retention_ge_50": c1,
        "C2_recall_ge_fullb_minus_5": c2,
        "C3_beats_random": c3,
        "validity_full_b_lt_50": valid,
        "label": label,
        "per_seed_selected_retention": ret_sel,
        "per_seed_selected_recall": rec_sel,
        "per_seed_full_b_retention": ret_b,
        "per_seed_full_b_recall": rec_b,
        "n_random_runs": len(random_rets),
    }


def write_results_md(R, path):
    v = R["verdict"]
    lines = [
        "# Phase L results",
        "",
        f"**Verdict: {v['label']}**",
        "",
        "Claim (verbatim):",
        "",
        "> On Qwen2.5-0.5B-Instruct after N nights of arm-`b` terse fact training (exactly the mathfix `b` recipe and fact/probe sets, 3 seeds 0/1/2), zeroing the LoRA delta in the k=4 layers whose single-layer zeroing most increases mean GSM8K output length on a 32-item calibration set (GSM8K train split, disjoint from the test probes, chosen per seed from calibration only) keeps GSM8K strict retention >= 50% of base-correct test items (mean over seeds) AND keeps recall >= full-b recall - 5 pts (mean over seeds) AND beats random 4-layer zeroing: mean selected retention - mean of the 24 random runs (8 seeded random 4-layer subsets x 3 seeds, seed 1234) > 2 x sample std of those 24 random runs.",
        "",
        f"N = {R['cfg']['nights']} (frozen after timing pilot). k = 4. max_new_tokens = {R['cfg']['gsm_max_new']}. greedy.",
        "",
        "## Conditions vs thresholds",
        "",
        "| condition | number | threshold | hold |",
        "|---|---|---|---|",
        f"| C1 selected retention (mean seeds) | {v['mean_selected_retention']:.4f} | >= 50 | {v['C1_retention_ge_50']} |",
        f"| C2 selected recall (mean seeds) | {v['mean_selected_recall']:.4f} | >= full-b recall - 5 = {v['mean_full_b_recall'] - 5:.4f} | {v['C2_recall_ge_fullb_minus_5']} |",
        f"| C3 selected - random mean | {v['selected_minus_random']:.4f} | > 2 * sd_random = {v['threshold_2std']:.4f} | {v['C3_beats_random']} |",
        f"| Validity full-b retention (mean seeds) | {v['mean_full_b_retention']:.4f} | < 50 | {v['validity_full_b_lt_50']} |",
        "",
        f"Random: mean {v['mean_random_retention']:.4f}, sample std (ddof=1) {v['std_random_retention_ddof1']:.4f}, n={v['n_random_runs']}.",
        "",
        "## Per seed",
        "",
        "| seed | full-b ret | full-b rec | selected layers | selected ret | selected rec | selected tok | selected %steps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in R["cfg"]["seeds"]:
        run = R["runs"][f"s{s}"]
        sel = run["selected_k4"]
        fb = run["full_b"]
        lines.append(
            f"| {s} | {fb['retention']:.4f} | {fb['recall']:.4f} | {sel['layers']} | "
            f"{sel['retention']:.4f} | {sel['recall']:.4f} | {sel['agg']['mean_tokens']:.2f} | {sel['agg']['pct_steps']:.2f} |"
        )
    lines += [
        "",
        f"Mean full-b retention {v['mean_full_b_retention']:.4f}, recall {v['mean_full_b_recall']:.4f}.",
        f"Mean selected retention {v['mean_selected_retention']:.4f}, recall {v['mean_selected_recall']:.4f}.",
        "",
        "## Base",
        "",
        f"n_probe = {R['base']['n_probe']}, n_base_correct = {R['base']['n_base_correct']}, "
        f"strict on all probes = {R['base']['acc_strict_all']:.4f}, "
        f"mean tokens on base-correct = {R['base']['agg_on_base_correct']['mean_tokens']:.2f}.",
        f"probe_hash = `{R.get('probe_hash')}`, fact_hash = `{R.get('fact_hash')}`.",
        "",
        "## Random 4-layer subsets (control, seed 1234)",
        "",
        f"subsets = {R['random_subsets']}",
        "",
        "| seed | subset | layers | retention | recall | mean_tokens | pct_steps |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in R["cfg"]["seeds"]:
        for i, rr in enumerate(R["runs"][f"s{s}"]["random"]):
            lines.append(
                f"| {s} | {i} | {rr['layers']} | {rr['retention']:.4f} | {rr['recall']:.4f} | "
                f"{rr['agg']['mean_tokens']:.2f} | {rr['agg']['pct_steps']:.2f} |"
            )
    lines += [
        "",
        "## Post-hoc (not the verdict): selected k=2 and k=8",
        "",
        "| seed | k | layers | retention | recall | mean_tokens | pct_steps |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in R["cfg"]["seeds"]:
        run = R["runs"][f"s{s}"]
        for kk in ("selected_k2", "selected_k8"):
            if kk not in run:
                continue
            x = run[kk]
            lines.append(
                f"| {s} | {kk} | {x['layers']} | {x['retention']:.4f} | {x['recall']:.4f} | "
                f"{x['agg']['mean_tokens']:.2f} | {x['agg']['pct_steps']:.2f} |"
            )
    lines += [
        "",
        "## Calibration ranking (mean tokens after single-layer zeroing)",
        "",
    ]
    for s in R["cfg"]["seeds"]:
        cal = R["runs"][f"s{s}"]["calib"]
        lines.append(f"### seed {s}")
        lines.append("")
        lines.append(f"ranking (desc tokens, tie low index): {cal['ranking']}")
        lines.append("")
        lines.append("| layer | mean_tokens |")
        lines.append("|---|---|")
        for i, mt in enumerate(cal["per_layer_mean_tokens"]):
            lines.append(f"| {i} | {mt:.4f} |")
        lines.append("")
    lines += [
        "## Commits and wall time",
        "",
        f"- prereg commit: `{R.get('prereg_commit')}` at {R.get('prereg_commit_time')}",
        f"- results commit (filled after this file is committed): `{R.get('results_commit', 'TBD')}`",
        f"- wall seconds (this run object): {R.get('wall_sec')}",
        f"- timing pilot: {R.get('timing_pilot', {}).get('chosen_N')}",
        "",
        "Raw JSON: `results/phaseL/results.json` and per-condition jsonl under `results/phaseL/gens/`.",
        "",
    ]
    open(path, "w").write("\n".join(lines) + "\n")


def run_experiment():
    R = load()
    if not R:
        R = {"cfg": CFG, "runs": {}, "timing": {}}
    # Freeze nights: smoke uses 1; real run must already have chosen_N from pilot file
    if not SMOKE:
        if args.nights is not None:
            raise SystemExit("refusing --nights on the real run; N is taken from the committed timing pilot")
        tpath = os.path.join(os.path.join(REPO, "results", "phaseL"), "timing_pilot.json")
        if not os.path.exists(tpath):
            raise SystemExit("timing_pilot.json missing; run --pilot first and commit N")
        pilot = json.load(open(tpath))
        CFG["nights"] = int(pilot["chosen_N"])
        R["timing_pilot"] = pilot
        R["prereg_commit"] = R.get("prereg_commit") or "9b5a2d6c0d9489a90d070b7c2975923fa02b43ce"
        R["prereg_commit_time"] = R.get("prereg_commit_time") or "2026-10-18T08:10:34+00:00"
    R["cfg"] = {**CFG, **{"nights": CFG["nights"]}}
    log("=" * 30, "Phase L start | smoke", SMOKE, "| N", CFG["nights"], "| device", DEV, "| out", OUT)

    t_all = time.time()
    PROBES, PH = C.load_probes(CFG, SMOKE)
    FACTS, FH, FACT_VOCAB = C.load_facts(CFG, SMOKE)
    R["probe_hash"], R["fact_hash"] = PH, FH
    save(R)
    log("probe hash", PH, "| fact hash", FH)

    tok, base = C.load_model(CFG["model"])
    n_layers = len(base.model.layers)
    ALL = list(range(n_layers))
    FP0 = C.fingerprint(base)
    gen = C.Gen(tok, base)
    TR = C.Trainer(CFG, tok, gen.chat)
    GSM = PROBES["gsm8k"]
    R["n_layers"] = n_layers
    k_use = min(K_VERDICT, n_layers)
    n_rand = N_RANDOM if not SMOKE else min(N_RANDOM, 2)
    R["random_subsets"] = random_subsets(n_layers, k_use, n_rand, RANDOM_SUBSET_SEED)
    calib_idx, CALIB = build_calib(CFG["n_calib_phaseL"], GSM, SMOKE)
    R["calib"] = {"train_indices": calib_idx, "n": len(CALIB), "seed": CALIB_SEED}
    save(R)
    log("layers", n_layers, "| random subsets", R["random_subsets"], "| calib train idx", calib_idx)

    # ---- base eval (defines base-correct test items)
    if "base" not in R:
        log("base GSM on all", len(GSM), "test probes")
        recs = eval_gsm_items(gen, base, GSM, list(range(len(GSM))), CFG, FACT_VOCAB,
                              os.path.join(OUT, "gens", "base.jsonl"), {"cond": "base"})
        ok = [r["item"] for r in recs if r["strict_ok"]]
        if SMOKE and not ok:
            ok = list(range(len(GSM)))
            log("SMOKE: 0 base-correct GSM -> using all items to exercise code")
        base_recs = [r for r in recs if r["item"] in set(ok)]
        seen = [f for f in FACTS if f["night"] <= CFG["nights"]]
        brec = C.eval_recall(gen, base, seen, CFG)
        R["base"] = {
            "n_probe": len(GSM),
            "gsm_items": [bool(r["strict_ok"]) for r in recs],
            "base_correct": ok,
            "n_base_correct": len(ok),
            "acc_strict_all": 100.0 * float(np.mean([r["strict_ok"] for r in recs])),
            "agg_on_base_correct": C.agg_gsm(base_recs),
            "recall_seen": brec,
        }
        save(R)
        log("base-correct", len(ok), "strict_all", R["base"]["acc_strict_all"], "recall", brec)
    BASE_OK = R["base"]["base_correct"]

    # ---- train 3 seeds
    for seed in CFG["seeds"]:
        for night in range(1, CFG["nights"] + 1):
            if os.path.isdir(os.path.join(adir(seed, night), "adapter_config.json")) or os.path.exists(
                os.path.join(adir(seed, night), "adapter_model.safetensors")
            ):
                log(f"[s{seed} n{night}] adapter exists, skip train")
                continue
            if night > 1 and not os.path.isdir(adir(seed, night - 1)):
                raise SystemExit(f"missing previous adapter s{seed} n{night-1}")
            log(f"[s{seed} n{night}] training arm b ...")
            base, loss, tt = train_one(base, tok, gen, TR, CFG, FACTS, ALL, seed, night, FP0)
            R.setdefault("train", {}).setdefault(f"s{seed}", {})[str(night)] = {"loss": loss, "sec": tt}
            save(R)
            log(f"[s{seed} n{night}] loss {loss:.4f} train {tt:.1f}s")

    LAST = CFG["nights"]
    seen = [f for f in FACTS if f["night"] <= LAST]

    def eval_condition(model, tag, layers_zero, do_recall=True):
        snap = snapshot_lora_weights(model)
        nmod = zero_lora_delta_layers(model, layers_zero)
        recs = eval_gsm_items(
            gen, model, GSM, BASE_OK, CFG, FACT_VOCAB,
            os.path.join(OUT, "gens", f"{tag}.jsonl"),
            {"cond": tag, "layers_zero": list(layers_zero)},
        )
        ret, agg = retention_and_agg(recs)
        rec = C.eval_recall(gen, model, seen, CFG) if do_recall else float("nan")
        restore_lora_weights(model, snap)
        return {
            "layers": list(layers_zero),
            "n_modules_zeroed": nmod,
            "retention": ret,
            "recall": rec,
            "agg": {k: agg[k] for k in [
                "strict", "lenient", "mean_tokens", "median_tokens", "pct_steps",
                "pct_truncated", "pct_exact_marker", "n", "n_fail", "fail_categories",
            ]},
        }

    # ---- per seed: full_b, calib, selected, random
    for seed in CFG["seeds"]:
        key = f"s{seed}"
        run = R["runs"].setdefault(key, {})
        log(f"==== seed {seed} load night-{LAST} adapter ====")
        model = load_adapter(base, seed, LAST)

        if "full_b" not in run:
            log(f"[s{seed}] eval full_b")
            run["full_b"] = eval_condition(model, f"s{seed}_full_b", [], do_recall=True)
            # sanity: zeroing none should not have touched modules
            save(R)
            log(f"[s{seed}] full_b ret {run['full_b']['retention']:.2f} rec {run['full_b']['recall']:.2f} "
                f"tok {run['full_b']['agg']['mean_tokens']:.1f}")

        if "calib" not in run:
            log(f"[s{seed}] calibration scan ({n_layers} layers × {len(CALIB)} items)")
            snap = snapshot_lora_weights(model)
            means = []
            for L in ALL:
                zero_lora_delta_layers(model, [L])
                recs = eval_gsm_items(
                    gen, model, CALIB, list(range(len(CALIB))), CFG, FACT_VOCAB,
                    os.path.join(OUT, "gens", f"s{seed}_calib_L{L}.jsonl"),
                    {"cond": "calib", "layer": L, "seed": seed},
                )
                mt = float(np.mean([r["n_tokens"] for r in recs])) if recs else float("nan")
                means.append(mt)
                restore_lora_weights(model, snap)
                log(f"[s{seed} calib] L{L} mean_tokens {mt:.2f}")
                run["calib"] = {"per_layer_mean_tokens": means, "ranking": rank_layers(means)}
                save(R)  # incremental
            ranking = rank_layers(means)
            run["calib"] = {"per_layer_mean_tokens": means, "ranking": ranking}
            save(R)
            log(f"[s{seed}] ranking {ranking[:8]}...")

        ranking = run["calib"]["ranking"]
        sel4 = ranking[:k_use]
        if "selected_k4" not in run:
            log(f"[s{seed}] eval selected_k4 {sel4}")
            run["selected_k4"] = eval_condition(model, f"s{seed}_selected_k4", sel4)
            save(R)
            log(f"[s{seed}] selected_k4 ret {run['selected_k4']['retention']:.2f} rec {run['selected_k4']['recall']:.2f}")

        for kpost in K_POSTHOC:
            kk = f"selected_k{kpost}"
            layers = ranking[: min(kpost, n_layers)]
            if kk not in run:
                log(f"[s{seed}] eval POST-HOC {kk} {layers}")
                run[kk] = eval_condition(model, f"s{seed}_{kk}", layers)
                save(R)

        run.setdefault("random", [])
        done_l = {tuple(x["layers"]) for x in run["random"]}
        for i, subset in enumerate(R["random_subsets"]):
            if tuple(subset) in done_l:
                continue
            log(f"[s{seed}] eval random {i} {subset}")
            ent = eval_condition(model, f"s{seed}_rand{i}", subset)
            run["random"].append(ent)
            save(R)
            log(f"[s{seed}] random {i} ret {ent['retention']:.2f} rec {ent['recall']:.2f}")

        base = unload(model)
        assert abs(C.fingerprint(base) - FP0) < 1e-3

    R["wall_sec"] = time.time() - t_all
    R["verdict"] = summarize_verdict(R)
    R["head_at_results"] = git_hash()
    save(R)
    md = os.path.join(REPO, "RESULTS_phaseL.md") if not SMOKE else os.path.join(OUT, "RESULTS_phaseL.md")
    write_results_md(R, md)
    log("VERDICT", R["verdict"]["label"], R["verdict"])
    log("wrote", md)
    return R


if __name__ == "__main__":
    if args.pilot:
        if SMOKE:
            raise SystemExit("pilot is the real 0.5B timing run; do not combine with --smoke")
        OUT = os.path.join(REPO, "results", "phaseL")
        os.makedirs(OUT, exist_ok=True)
        log = C.Logger(OUT)
        pilot = run_pilot()
        N = append_prereg_n(pilot)
        log("N frozen to", N, "in PREREG appendix A")
    else:
        run_experiment()
