#!/usr/bin/env python3
"""Pre-registered Qwen2.5-0.5B-Instruct critical-layer replication.

See PREREG_qwen05b_critical_layer.md. Do not change B / L* / seeds / scoring
after any test-set number has been written.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from cpu_fakequant import (  # noqa: E402
    GROUP_SIZE,
    DiskWeightBank,
    fakequant_asymmetric,
    iter_block_linears,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SHOTS = "12*11=132\n23*14=322\n31*25=775\n"
SHOT_PAIRS = {(12, 11), (23, 14), (31, 25)}
PILOT_SEED, CAL_SEED, TEST_SEED, RANDOM_SEED = 11, 22, 33, 44
PILOT_N, CAL_N, TEST_N = 50, 64, 300
BIT_CANDIDATES = [4, 3, 2]
DROP_THRESHOLD = 0.15
MAX_NEW_TOKENS = 6
GEN_BATCH = 4
N_RANDOM = 8
N_BOOT = 10_000
BOOT_SEED = 0
PREREG = "PREREG_qwen05b_critical_layer.md"
OUT_DEFAULT = os.path.join(ROOT, "results_qwen05b")


def log(msg: str, log_path: str | None = None) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if log_path:
        with open(log_path, "a") as f:
            f.write(line)


def dump_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def prereg_is_committed(repo: str = ROOT) -> bool:
    r = subprocess.run(
        ["git", "log", "--pretty=format:%H", "--", PREREG],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(r.stdout.strip())


def g_num(output: str, target: str) -> bool:
    """Repo scoring: exact integer prefix, next char must not be a digit."""
    o = output.strip()
    return o.startswith(target) and not o[len(target) : len(target) + 1].isdigit()


def make_items(n: int, seed: int, exclude: set[tuple[int, int]]) -> list[dict]:
    rng = random.Random(seed)
    seen = set(exclude) | set(SHOT_PAIRS)
    items = []
    while len(items) < n:
        x, y = rng.randint(10, 99), rng.randint(10, 99)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        items.append({"x": x, "y": y, "prompt": SHOTS + f"{x}*{y}=", "target": str(x * y)})
    return items


def generate_splits() -> dict:
    pilot = make_items(PILOT_N, PILOT_SEED, set())
    used = {(it["x"], it["y"]) for it in pilot}
    cal = make_items(CAL_N, CAL_SEED, used)
    used |= {(it["x"], it["y"]) for it in cal}
    test = make_items(TEST_N, TEST_SEED, used)
    pairs = [(it["x"], it["y"]) for it in pilot + cal + test]
    assert len(pairs) == len(set(pairs)) == PILOT_N + CAL_N + TEST_N
    return {
        "model": MODEL_ID,
        "shots": SHOTS,
        "shot_pairs": [list(p) for p in sorted(SHOT_PAIRS)],
        "seeds": {"pilot": PILOT_SEED, "calibration": CAL_SEED, "test": TEST_SEED, "random_layers": RANDOM_SEED},
        "pilot": pilot,
        "calibration": cal,
        "test": test,
    }


def bootstrap_ci(correct, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    x = np.asarray(correct, dtype=float)
    n = int(x.size)
    if n == 0:
        return {"acc": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n": 0}
    acc = float(x.mean())
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n_boot, n), replace=True).mean(axis=1)
    return {
        "acc": acc,
        "ci95_lo": float(np.percentile(bs, 2.5)),
        "ci95_hi": float(np.percentile(bs, 97.5)),
        "n": n,
    }


def gold_token_ids(tok, prompt: str, target: str) -> tuple[torch.Tensor, int, str]:
    p = tok(prompt, add_special_tokens=True, return_tensors="pt")
    f = tok(prompt + target, add_special_tokens=True, return_tensors="pt")
    p_ids = p.input_ids[0]
    f_ids = f.input_ids[0]
    method = "prefix"
    if f_ids[: p_ids.numel()].tolist() != p_ids.tolist():
        t_ids = tok(target, add_special_tokens=False, return_tensors="pt").input_ids[0]
        f_ids = torch.cat([p_ids, t_ids], dim=0)
        method = "fallback_concat"
    return f_ids, int(p_ids.numel()), method


@torch.no_grad()
def mean_gold_nll(model, tok, items: list[dict]) -> tuple[float, int, list[str]]:
    """Mean of per-item mean NLLs of gold answer tokens. Returns (nll, n_used, notes)."""
    notes = []
    per_item = []
    device = next(model.parameters()).device
    for k, it in enumerate(items):
        f_ids, prompt_len, method = gold_token_ids(tok, it["prompt"], it["target"])
        gold_len = f_ids.numel() - prompt_len
        if gold_len <= 0:
            notes.append(f"empty_gold_suffix item={k} method={method}")
            continue
        if method != "prefix":
            notes.append(f"prefix_unstable item={k}")
        inp = f_ids.unsqueeze(0).to(device)
        logits = model(input_ids=inp).logits[0].float()
        pred = logits[prompt_len - 1 : prompt_len - 1 + gold_len]
        gold = f_ids[prompt_len:].to(device)
        logp = torch.log_softmax(pred, dim=-1)
        nll = -logp.gather(1, gold.unsqueeze(1)).squeeze(1).mean()
        per_item.append(float(nll))
    if not per_item:
        return float("nan"), 0, notes + ["no_items_with_gold_tokens"]
    return float(np.mean(per_item)), len(per_item), notes


@torch.no_grad()
def generate_texts(model, tok, prompts: list[str], max_new: int = MAX_NEW_TOKENS, bs: int = GEN_BATCH) -> list[str]:
    outs = []
    device = next(model.parameters()).device
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s : s + bs], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        o = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        plen = enc["input_ids"].shape[1]
        outs.extend(tok.batch_decode(o[:, plen:], skip_special_tokens=True))
    return outs


def score_items(model, tok, items: list[dict]) -> dict:
    texts = generate_texts(model, tok, [it["prompt"] for it in items])
    correct = [int(g_num(o, it["target"])) for o, it in zip(texts, items)]
    ci = bootstrap_ci(correct)
    return {
        **ci,
        "correct": correct,
        "outputs": texts,
    }


def load_real_model(model_id: str = MODEL_ID):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"low_cpu_mem_usage": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, **kw)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        try:
            model.generation_config.temperature = None
        except Exception:
            pass
        model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


class TinyBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.o_proj(torch.relu(self.q_proj(x)))


class TinyInner(nn.Module):
    def __init__(self, n_layers: int, d: int, vocab: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TinyBlock(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)


class TinyLM(nn.Module):
    """Randomly initialized stand-in with the Qwen block-linear layout."""

    def __init__(self, n_layers: int = 3, d: int = 64, vocab: int = 32):
        super().__init__()
        self.model = TinyInner(n_layers, d, vocab)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.generation_config = type("GC", (), {"do_sample": False, "temperature": None, "pad_token_id": 0})()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        for layer in self.model.layers:
            x = x + layer(x)
        logits = self.lm_head(self.model.norm(x))
        return type("Out", (), {"logits": logits})()

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=6, do_sample=False, pad_token_id=0, **kw):
        ids = input_ids
        mask = attention_mask if attention_mask is not None else torch.ones_like(ids)
        for _ in range(max_new_tokens):
            logits = self.forward(ids, attention_mask=mask).logits[:, -1]
            nxt = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
        return ids


class TinyTok:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def __call__(self, text, add_special_tokens=True, return_tensors=None, padding=False):
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = list(text)
            single = False
        ids = []
        for t in texts:
            seq = [min(31, 2 + (ord(ch) % 30)) for ch in t] or [2]
            if add_special_tokens:
                seq = [1] + seq
            ids.append(seq)
        if padding and len(ids) > 1:
            m = max(len(s) for s in ids)
            if self.padding_side == "left":
                ids = [[0] * (m - len(s)) + s for s in ids]
            else:
                ids = [s + [0] * (m - len(s)) for s in ids]
        if return_tensors == "pt":
            if single and not padding:
                input_ids = torch.tensor(ids, dtype=torch.long)
            else:
                m = max(len(s) for s in ids)
                if self.padding_side == "left":
                    ids = [[0] * (m - len(s)) + s for s in ids]
                else:
                    ids = [s + [0] * (m - len(s)) for s in ids]
                input_ids = torch.tensor(ids, dtype=torch.long)
            mask = (input_ids != 0).long()

            class Enc(dict):
                def __getattr__(self, key):
                    return self[key]

            return Enc(input_ids=input_ids, attention_mask=mask)
        return ids[0] if single else ids

    def batch_decode(self, ids, skip_special_tokens=True):
        out = []
        for row in ids:
            chars = []
            for t in row.tolist():
                if skip_special_tokens and t <= 1:
                    continue
                chars.append(chr(97 + (int(t) % 26)))
            out.append("".join(chars))
        return out


@contextmanager
def timed(label: str, notes: list, lp: str | None):
    t0 = time.time()
    log(f"START {label}", lp)
    try:
        yield
    finally:
        dt = time.time() - t0
        notes.append(f"{label}_sec={dt:.1f}")
        log(f"END {label} {dt:.1f}s", lp)


def choose_B(acc_fp32: float, acc_by_bits: dict[int, float]) -> int | None:
    for b in BIT_CANDIDATES:
        if acc_fp32 - acc_by_bits[b] >= DROP_THRESHOLD:
            return b
    return None


def recovery_and_verdict(acc_fp32: float, acc_q: float, acc_l: float, random_accs: list[float]) -> dict:
    denom = acc_fp32 - acc_q
    if denom == 0:
        R = None
        cond1 = False
        cond1_note = "denominator_zero"
    else:
        R = (acc_l - acc_q) / denom
        cond1 = R >= 0.5
        cond1_note = "ok"
    mu = float(np.mean(random_accs)) if random_accs else float("nan")
    if len(random_accs) < 2:
        std = float("nan")
        cond2 = False
        cond2_note = "need_8_random"
    else:
        std = float(np.std(random_accs, ddof=1))
        thresh = mu + 2 * std
        cond2 = acc_l > thresh if std > 0 or True else acc_l > mu
        # std==0: pre-reg says strict greater than mean, which is the same as > mean+0
        cond2_note = "ok"
    passed = bool(cond1 and cond2)
    return {
        "recovery_fraction": R,
        "random_mean": mu,
        "random_std_ddof1": std,
        "pass_recovery": cond1,
        "pass_vs_random": cond2,
        "verdict": "PASS" if passed else "FAIL",
        "notes": {"recovery": cond1_note, "random": cond2_note},
    }


def write_results_md(path: str, payload: dict) -> None:
    stop = payload.get("stopped_before_test")
    v = payload.get("verdict") or {}
    lines = [
        "# Qwen2.5-0.5B-Instruct critical-layer replication",
        "",
        "Pre-registration: `PREREG_qwen05b_critical_layer.md` (committed before test-set evaluation).",
        "Quantizer: CPU asymmetric RTN, group 128, dequantized to fp32. InstinctRazor's MoE/symmetric",
        "fake-quant was **not** used (dense model, no GPU, pre-reg specified this grid).",
        "",
        f"**Verdict: {v.get('verdict', 'N/A — stopped before test') }**",
        "",
        "## Locked choices",
        "",
        f"- Model: `{payload.get('model')}` on CPU fp32",
        f"- B: {payload.get('B')}",
        f"- L*: {payload.get('L_star')}",
        f"- Random blocks (seed 44, excluding L*): {payload.get('random_blocks')}",
        f"- Wall time (s): {payload.get('wall_time_sec')}",
        "",
        "## Pilot (n=50, choose B)",
        "",
    ]
    pilot = payload.get("pilot") or {}
    if pilot:
        lines.append(f"- fp32 accuracy: {pilot.get('fp32', {}).get('acc')}")
        for b in BIT_CANDIDATES:
            ab = (pilot.get("bits") or {}).get(str(b)) or (pilot.get("bits") or {}).get(b)
            if ab:
                lines.append(f"- {b}-bit full-quant accuracy: {ab.get('acc')}")
        lines.append(f"- B chosen: {pilot.get('B')}")
        if pilot.get("stop_reason"):
            lines.append(f"- Stopped: {pilot.get('stop_reason')}")
    else:
        lines.append("- Pilot did not run.")
    lines += ["", "## Calibration NLL (n=64, choose L*)", ""]
    cal = payload.get("calibration") or {}
    if cal:
        lines.append(f"- Full-quant NLL: {cal.get('nll_quant')}")
        lines.append(f"- L* = {cal.get('L_star')}  (NLL {cal.get('nll_L_star')}, drop vs quant {cal.get('drop_L_star')})")
        lines.append(f"- Runner-up = {cal.get('runner_up')}  (NLL {cal.get('nll_runner_up')}, drop {cal.get('drop_runner_up')})")
        lines.append(f"- L* is block 0: {cal.get('L_star') == 0}")
        lines.append(f"- L* is last block: {cal.get('L_star_is_last')}")
        lines.append("")
        lines.append("| block | NLL | drop vs full-quant |")
        lines.append("|---|---|---|")
        for row in cal.get("table") or []:
            lines.append(f"| {row['block']} | {row['nll']} | {row['drop']} |")
    else:
        lines.append("- Calibration did not run.")
    lines += ["", "## Test (n=300, once)", ""]
    test = payload.get("test") or {}
    if stop:
        lines.append(f"Test set was **not** scored. Reason: {stop}")
    elif test:
        def _row(name, d):
            if not d:
                return f"- {name}: missing"
            return (
                f"- {name}: {d.get('acc'):.4f}  "
                f"95% CI [{d.get('ci95_lo'):.4f}, {d.get('ci95_hi'):.4f}]  n={d.get('n')}"
            )
        lines.append(_row("fp32", test.get("fp32")))
        lines.append(_row("full-quant", test.get("quant")))
        lines.append(_row("protect L*", test.get("protect_L_star")))
        for rb in test.get("random") or []:
            lines.append(_row(f"protect random block {rb.get('block')}", rb))
        lines += [
            "",
            f"- Recovery fraction R = {v.get('recovery_fraction')}",
            f"- mean(random) = {v.get('random_mean')}, std(random, ddof=1) = {v.get('random_std_ddof1')}",
            f"- PASS recovery (R >= 0.5): {v.get('pass_recovery')}",
            f"- PASS vs random (L* > mean + 2 std): {v.get('pass_vs_random')}",
        ]
    else:
        lines.append("- Test did not run.")
    lines += ["", "## What went wrong / caveats", ""]
    issues = payload.get("issues") or []
    if issues:
        lines.extend(f"- {x}" for x in issues)
    else:
        lines.append("- Nothing recorded.")
    lines += [
        "",
        "No numbers in this file were invented. If a field is missing, that phase did not finish.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def run_smoke(out_dir: str) -> dict:
    """Plumbing only: tiny random model, 4 fake items, 3 blocks. Not the registered test."""
    torch.manual_seed(0)
    os.makedirs(out_dir, exist_ok=True)
    model = TinyLM(n_layers=3, d=64, vocab=32)
    tok = TinyTok()
    items = make_items(4, seed=0, exclude=set())
    bank = DiskWeightBank(os.path.join(out_dir, "smoke_bank"))
    bank.capture(model)
    embed0 = model.model.embed_tokens.weight.detach().clone()
    head0 = model.lm_head.weight.detach().clone()
    orig_b1 = {n: m.weight.detach().clone() for n, m in model.model.layers[1].named_modules() if isinstance(m, nn.Linear)}
    st = bank.apply(model, bits=3, protect=(1,))
    for n, m in model.model.layers[1].named_modules():
        if isinstance(m, nn.Linear):
            if not torch.equal(m.weight.cpu(), orig_b1[n].cpu()):
                raise AssertionError("smoke: protected block was modified")
    if not torch.equal(model.model.embed_tokens.weight, embed0):
        raise AssertionError("smoke: embed changed")
    if not torch.equal(model.lm_head.weight, head0):
        raise AssertionError("smoke: lm_head changed")
    nll_q, n_used, notes = mean_gold_nll(model, tok, items)
    bank.apply(model, bits=3, protect=())
    scored = score_items(model, tok, items)
    w = torch.randn(32, 256)
    wq = fakequant_asymmetric(w, 4, 128)
    # 1 LSB bound per group
    wb = w.reshape(-1, 128)
    wqb = wq.reshape(-1, 128)
    scale = (wb.max(-1).values - wb.min(-1).values) / 15
    err = (wb - wqb).abs().max(-1).values
    if not torch.all((scale == 0) | (err <= scale + 1e-5)):
        raise AssertionError("smoke: quant error exceeded 1 LSB")
    return {
        "ok": True,
        "apply": st,
        "nll_quant": nll_q,
        "n_used": n_used,
        "nll_notes": notes,
        "smoke_acc": scored["acc"],
        "max_abs_quant_err": float(err.max()),
    }


def run_experiment(out_dir: str, phase: str) -> dict:
    if not prereg_is_committed():
        raise SystemExit(f"{PREREG} is not committed; refusing to run. Commit the pre-reg first.")
    os.makedirs(out_dir, exist_ok=True)
    lp = os.path.join(out_dir, "run.log")
    issues: list[str] = []
    t_all = time.time()
    problems_path = os.path.join(out_dir, "problems.json")
    if not os.path.exists(problems_path):
        dump_json(problems_path, generate_splits())
        log("wrote problems.json (lists only; no model scoring)", lp)
    problems = json.load(open(problems_path))
    payload = {
        "model": MODEL_ID,
        "quantizer": "cpu_asymmetric_rtn_g128",
        "group": GROUP_SIZE,
        "prereg": PREREG,
        "issues": issues,
        "B": None,
        "L_star": None,
        "random_blocks": None,
        "stopped_before_test": None,
    }

    if phase == "data":
        payload["wall_time_sec"] = time.time() - t_all
        dump_json(os.path.join(out_dir, "results.json"), payload)
        return payload

    torch.manual_seed(0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
    log(f"loading {MODEL_ID} fp32 CPU", lp)
    try:
        model, tok = load_real_model(MODEL_ID)
    except Exception as e:
        issues.append(f"model_load_failed: {type(e).__name__}: {e}")
        payload["wall_time_sec"] = time.time() - t_all
        payload["stopped_before_test"] = "model_load_failed"
        dump_json(os.path.join(out_dir, "results.json"), payload)
        write_results_md(os.path.join(ROOT, "RESULTS.md"), payload)
        traceback.print_exc()
        return payload

    n_layers = len(model.model.layers)
    log(f"n_layers={n_layers} n_block_linears={sum(1 for _ in iter_block_linears(model))}", lp)
    bank = DiskWeightBank(os.path.join(out_dir, "weight_bank"))
    bank.capture(model)
    log("captured original block Linear weights to disk", lp)

    # --- pilot ---
    pilot_path = os.path.join(out_dir, "pilot.json")
    if phase in ("pilot", "all") or (phase in ("calibrate", "test") and not os.path.exists(pilot_path)):
        with timed("pilot", issues, lp):
            # explicit restore
            for i in range(n_layers):
                bank.write_block(model, i, bits=None)
            fp32 = score_items(model, tok, problems["pilot"])
            log(f"pilot fp32 acc={fp32['acc']:.4f}", lp)
            bits_out = {}
            for b in BIT_CANDIDATES:
                bank.apply(model, bits=b, protect=())
                bits_out[b] = score_items(model, tok, problems["pilot"])
                log(f"pilot {b}-bit acc={bits_out[b]['acc']:.4f} drop={fp32['acc']-bits_out[b]['acc']:.4f}", lp)
            B = choose_B(fp32["acc"], {b: bits_out[b]["acc"] for b in BIT_CANDIDATES})
            stop_reason = None
            if B is None:
                stop_reason = (
                    "no bit-width in {4,3,2} had full-quant pilot accuracy >= 15 points below fp32; "
                    "claim is not testable; test set will not be scored"
                )
                log(stop_reason, lp)
            pilot = {
                "fp32": {k: fp32[k] for k in ("acc", "ci95_lo", "ci95_hi", "n", "correct", "outputs")},
                "bits": {
                    str(b): {k: bits_out[b][k] for k in ("acc", "ci95_lo", "ci95_hi", "n", "correct", "outputs")}
                    for b in BIT_CANDIDATES
                },
                "B": B,
                "stop_reason": stop_reason,
                "n_layers": n_layers,
            }
            dump_json(pilot_path, pilot)
    else:
        pilot = json.load(open(pilot_path))
        log(f"loaded locked pilot.json B={pilot.get('B')}", lp)

    payload["pilot"] = {k: v for k, v in pilot.items()}
    payload["B"] = pilot.get("B")
    if pilot.get("B") is None:
        payload["stopped_before_test"] = pilot.get("stop_reason") or "no_B"
        payload["wall_time_sec"] = time.time() - t_all
        dump_json(os.path.join(out_dir, "results.json"), payload)
        write_results_md(os.path.join(ROOT, "RESULTS.md"), payload)
        return payload
    if phase == "pilot":
        payload["wall_time_sec"] = time.time() - t_all
        dump_json(os.path.join(out_dir, "results.json"), payload)
        return payload

    B = int(pilot["B"])

    # --- calibration ---
    cal_path = os.path.join(out_dir, "calibration.json")
    if phase in ("calibrate", "all") or (phase == "test" and not os.path.exists(cal_path)):
        with timed("calibrate", issues, lp):
            bank.apply(model, bits=B, protect=())
            nll_q, n_used_q, notes_q = mean_gold_nll(model, tok, problems["calibration"])
            log(f"cal nll_quant={nll_q:.6f} n={n_used_q} notes={notes_q}", lp)
            table = []
            for i in range(n_layers):
                bank.apply(model, bits=B, protect=(i,))
                nll, n_used, notes = mean_gold_nll(model, tok, problems["calibration"])
                drop = nll_q - nll
                table.append({"block": i, "nll": nll, "drop": drop, "n": n_used, "notes": notes})
                log(f"cal restore {i}: nll={nll:.6f} drop={drop:.6f}", lp)
            best = min(table, key=lambda r: (r["nll"], r["block"]))
            L_star = int(best["block"])
            others = [r for r in table if r["block"] != L_star]
            runner = min(others, key=lambda r: (r["nll"], r["block"])) if others else None
            rng = random.Random(RANDOM_SEED)
            pool = [i for i in range(n_layers) if i != L_star]
            if len(pool) < N_RANDOM:
                raise RuntimeError(f"need {N_RANDOM} random blocks besides L*, have {len(pool)}")
            random_blocks = rng.sample(pool, N_RANDOM)
            cal = {
                "B": B,
                "nll_quant": nll_q,
                "n_quant": n_used_q,
                "notes_quant": notes_q,
                "table": table,
                "L_star": L_star,
                "nll_L_star": best["nll"],
                "drop_L_star": best["drop"],
                "runner_up": None if runner is None else runner["block"],
                "nll_runner_up": None if runner is None else runner["nll"],
                "drop_runner_up": None if runner is None else runner["drop"],
                "L_star_is_last": L_star == n_layers - 1,
                "n_layers": n_layers,
                "random_blocks": random_blocks,
                "random_seed": RANDOM_SEED,
            }
            dump_json(cal_path, cal)
    else:
        cal = json.load(open(cal_path))
        log(f"loaded locked calibration.json L*={cal.get('L_star')}", lp)

    payload["calibration"] = cal
    payload["L_star"] = cal["L_star"]
    payload["random_blocks"] = cal["random_blocks"]
    if phase == "calibrate":
        payload["wall_time_sec"] = time.time() - t_all
        dump_json(os.path.join(out_dir, "results.json"), payload)
        return payload

    # --- test (once) ---
    test_path = os.path.join(out_dir, "test.json")
    if os.path.exists(test_path):
        issues.append("test.json already existed; refusing to re-score the test set")
        log(issues[-1], lp)
        test = json.load(open(test_path))
    else:
        with timed("test", issues, lp):
            L_star = int(cal["L_star"])
            random_blocks = list(cal["random_blocks"])
            test_items = problems["test"]
            for i in range(n_layers):
                bank.write_block(model, i, bits=None)
            fp32 = score_items(model, tok, test_items)
            bank.apply(model, bits=B, protect=())
            quant = score_items(model, tok, test_items)
            bank.apply(model, bits=B, protect=(L_star,))
            prot = score_items(model, tok, test_items)
            random_rows = []
            for rb in random_blocks:
                bank.apply(model, bits=B, protect=(rb,))
                row = score_items(model, tok, test_items)
                row["block"] = rb
                random_rows.append(row)
            test = {
                "B": B,
                "L_star": L_star,
                "random_blocks": random_blocks,
                "fp32": {k: fp32[k] for k in ("acc", "ci95_lo", "ci95_hi", "n", "correct", "outputs")},
                "quant": {k: quant[k] for k in ("acc", "ci95_lo", "ci95_hi", "n", "correct", "outputs")},
                "protect_L_star": {k: prot[k] for k in ("acc", "ci95_lo", "ci95_hi", "n", "correct", "outputs")},
                "random": [
                    {k: r[k] for k in ("block", "acc", "ci95_lo", "ci95_hi", "n", "correct", "outputs")}
                    for r in random_rows
                ],
            }
            dump_json(test_path, test)

    payload["test"] = test
    payload["verdict"] = recovery_and_verdict(
        test["fp32"]["acc"],
        test["quant"]["acc"],
        test["protect_L_star"]["acc"],
        [r["acc"] for r in test["random"]],
    )
    payload["wall_time_sec"] = time.time() - t_all
    dump_json(os.path.join(out_dir, "results.json"), payload)
    write_results_md(os.path.join(ROOT, "RESULTS.md"), payload)
    log(f"VERDICT {payload['verdict']['verdict']} R={payload['verdict']['recovery_fraction']} wall={payload['wall_time_sec']:.1f}s", lp)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "data", "pilot", "calibrate", "test", "all"], default="all")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    if args.phase == "smoke":
        os.makedirs(args.out, exist_ok=True)
        r = run_smoke(args.out)
        dump_json(os.path.join(args.out, "smoke.json"), r)
        print(json.dumps(r, indent=2))
        return
    run_experiment(args.out, args.phase)


if __name__ == "__main__":
    main()
