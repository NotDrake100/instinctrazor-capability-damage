"""Shared pieces for mathfix phase 1 / phase 2.

Everything that defines the experiment (model, probes, fact stream, parsers, chat format, LoRA config, training loop) is
copied VERBATIM from ../sleep_shield.py (v2, the version that ran on Vast) so that arm b here is the same arm b.
The probe hash and fact hash are asserted against the values recorded in ../vast_run/results.json (non-smoke runs).
Plain Python, no Colab code. Needs: torch, transformers, peft, datasets, accelerate.
"""
import os, json, re, math, time, random, hashlib, gc
import numpy as np
import torch, torch.nn.functional as F

EXPECTED_PROBE_HASH = "62a39450d3434e0dfc267e584c096a17d61fdf56cc39cc2f18a619942ad74ced"   # vast_run/results.json
EXPECTED_FACT_HASH = "f43d375c903756a1c6b017a14a07a1ac70f67df774d1f5cdc5bc23f3b4f38e75"    # vast_run/results.json

# --- identical to sleep_shield.py CFG for everything that touches arm b ---
BASE_CFG = dict(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    n_probe=200, n_calib=40,
    nights=7, facts_per_night=100,
    seeds=[0, 1, 2],
    lora_r=16, lora_alpha=32, lora_dropout=0.0,
    lr=2e-4, steps_per_night=120, batch_size=16,
    gsm_max_new=512, gen_bs=128, recall_bs=128,
    gsm_parser="v2", fact_seed=12345, probe_seed=1234,
)
SMOKE_OVERRIDES = dict(model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", n_probe=6, n_calib=3, nights=1,
                       facts_per_night=10, seeds=[0], steps_per_night=3, batch_size=4, gsm_max_new=16, gen_bs=8, recall_bs=8)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"

class Logger:
    def __init__(self, out): self.path = os.path.join(out, "log.txt")
    def __call__(self, *a):
        msg = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(self.path, "a") as f: f.write(msg + "\n")

def save_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)

# ---------------------------------------------------------------- probes (verbatim)
def build_probes(CFG, n_tot):
    from datasets import load_dataset
    rng = random.Random(CFG["probe_seed"]); mult, seen = [], set()
    while len(mult) < n_tot:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if (a, b) in seen: continue
        seen.add((a, b)); mult.append({"q": f"What is {a} * {b}? Reply with only the number.", "a": str(a * b)})
    g = load_dataset("openai/gsm8k", "main", split="test")
    idx = random.Random(CFG["probe_seed"]).sample(range(len(g)), n_tot)
    gsm = [{"q": g[i]["question"] + "\nSolve step by step. End with a final line exactly of the form 'The answer is N.' where N is a number.",
            "a": g[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in idx]
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    idx = random.Random(CFG["probe_seed"] + 1).sample(range(len(arc)), n_tot)
    arcs = []
    for i in idx:
        ex = arc[i]; opts = "\n".join(f"{l}. {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))
        arcs.append({"q": f"{ex['question']}\n{opts}\nAnswer with only the letter of the correct option.",
                     "a": ex["answerKey"].upper(), "labels": [l.upper() for l in ex["choices"]["label"]]})
    return {"mult": mult, "gsm8k": gsm, "arc": arcs}

def load_probes(CFG, smoke):
    _all = build_probes(CFG, CFG["n_probe"] + CFG["n_calib"])
    PROBES = {k: v[:CFG["n_probe"]] for k, v in _all.items()}
    h = hashlib.sha256(json.dumps(_all, sort_keys=True).encode()).hexdigest()
    if not smoke:   # probes do not depend on the model, so the hash must match for 0.5B and 1.5B alike
        assert h == EXPECTED_PROBE_HASH, f"probe hash {h} != sleep_shield {EXPECTED_PROBE_HASH}"
    return PROBES, h

# ---------------------------------------------------------------- parsers (verbatim) + lenient
def norm_num(s):
    try:
        f = float(s); return str(int(f)) if f == int(f) else str(f)
    except Exception: return None
def parse_mult(o):
    m = re.fullmatch(r"\s*(-?\d+)\s*\.?\s*", o.strip().split("\n")[0]) if o.strip() else None
    return m.group(1) if m else None
_NUM = r"-?\d[\d,]*(?:\.\d+)?"
def _first_num(s):
    m = re.search(_NUM, s); return norm_num(m.group(0).replace(",", "")) if m else None
def _boxed(o):
    out = []
    for m in re.finditer(r"\\boxed\s*\{", o):
        i, d = m.end(), 1
        while i < len(o) and d:
            d += {"{": 1, "}": -1}.get(o[i], 0); i += 1
        out.append((m.start(), o[m.end():i - 1] if d == 0 else o[m.end():]))
    return out
def parse_gsm(o):   # "strict" = the v2 parser used by sleep_shield.py (unchanged)
    marks = [(m.start(), o[m.end():m.end() + 60]) for m in re.finditer(r"answer\s+is|final\s+answer\s*\**\s*:", o, re.I)] + _boxed(o)
    for _, seg in sorted(marks, key=lambda t: -t[0]):
        n = _first_num(seg)
        if n is not None: return n
    nums = re.findall(_NUM, o)
    return norm_num(nums[-1].replace(",", "")) if nums else None
def parse_gsm_lenient(o):  # "lenient" = last number anywhere in the output
    nums = re.findall(_NUM, o)
    return norm_num(nums[-1].replace(",", "")) if nums else None
def parse_arc(o, labels):
    m = re.match(r"\s*\(?([A-Za-z0-9])\)?(?:[\.\):]|\s|$)", o)
    return m.group(1).upper() if m and m.group(1).upper() in labels else None
def correct(name, x, o):
    if name == "mult": p = parse_mult(o); return p is not None and norm_num(p) == norm_num(x["a"])
    if name == "gsm8k": p = parse_gsm(o); return p is not None and p == norm_num(x["a"])
    return parse_arc(o, x["labels"]) == x["a"]

# ---------------------------------------------------------------- GSM output diagnostics (transparent rules)
_STEP = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:[-+*/×÷x]|\\times|\\cdot|\\div)\s*\(?\$?\d|=\s*\$?\(?-?\d")
def n_arith_steps(o):
    """number of arithmetic operations/equations: 'a op b' or '= number' patterns"""
    return len(_STEP.findall(o))
def has_steps(o):
    """REASONING PRESENT := at least 2 arithmetic operations/equations in the output"""
    return n_arith_steps(o) >= 2
def n_lines(o): return len([l for l in o.split("\n") if l.strip()])
FACT_ANSWER_RE = re.compile(r"\b[A-Z][a-z]+(?:ish|ium)\b|\bRiver\b")
def categorize(o, ntok, truncated, strict_ok, fact_vocab=None):
    """Failure category for one GSM8K output (only call when strict_ok is False). Rules applied in this order:
      truncated     : generation hit max_new_tokens without an end-of-turn token
      fact_style    : <= 12 words AND no arithmetic step AND (no digit at all OR contains a fact-stream-style word:
                      '...ish', '...ium', 'River', or a word from the fictional fact vocabulary)
      no_steps      : fewer than 2 arithmetic steps (short / one-line answer, e.g. a bare wrong number)
      reasoning_wrong: >= 2 arithmetic steps, finished, but final parsed answer wrong (arithmetic or logic error)
      other         : anything else (should be empty by construction; kept as a guard)"""
    if strict_ok: return "correct"
    words = o.split()
    if truncated: return "truncated"
    fact_word = bool(FACT_ANSWER_RE.search(o)) or (fact_vocab is not None and any(w.strip(".,;:!?'\"") in fact_vocab for w in words))
    if len(words) <= 12 and n_arith_steps(o) == 0 and (not re.search(r"\d", o) or fact_word): return "fact_style"
    if not has_steps(o): return "no_steps"
    if has_steps(o): return "reasoning_wrong"
    return "other"
CATS = ["truncated", "fact_style", "no_steps", "reasoning_wrong", "other"]

# ---------------------------------------------------------------- model + generation
def load_model(name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    return tok, base.to(DEV)

class Gen:
    """Batched greedy generation exactly as sleep_shield.generate(), plus token counts / truncation flags."""
    def __init__(self, tok, model_for_cfg):
        self.tok = tok
        e = model_for_cfg.generation_config.eos_token_id
        self.stop = set(e if isinstance(e, (list, tuple)) else [e]) | {tok.pad_token_id, tok.eos_token_id}
    def chat(self, q): return self.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    @torch.no_grad()
    def __call__(self, model, qs, max_new, bs, prefill=""):
        tok = self.tok; model.eval(); res = [None] * len(qs); ntok = [0] * len(qs); trunc = [False] * len(qs); first = [None] * len(qs)
        order = sorted(range(len(qs)), key=lambda i: -len(qs[i]))
        for s in range(0, len(qs), bs):
            ids = order[s:s + bs]
            enc = tok([self.chat(qs[i]) + prefill for i in ids], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.pad_token_id)
            gen = g[:, enc["input_ids"].shape[1]:]
            for i, row, t in zip(ids, gen.tolist(), tok.batch_decode(gen, skip_special_tokens=True)):
                L = next((k for k, x in enumerate(row) if x in self.stop), None)
                res[i] = t; ntok[i] = L if L is not None else len(row); trunc[i] = L is None and len(row) >= max_new
                first[i] = row[0] if row else None
        self.first = first   # first generated token id per item (used by the logit-lens token sets)
        return res, ntok, trunc

# ---------------------------------------------------------------- facts (verbatim)
def make_facts(CFG):
    rng = random.Random(CFG["fact_seed"]); C, V = "bdfgklmnprstvz", "aeiou"; used = set()
    def word(sy):
        while True:
            w = "".join(rng.choice(C) + rng.choice(V) + (rng.choice("nrl") if rng.random() < .3 else "") for _ in range(sy)).capitalize()
            if w not in used and len(w) >= 5: used.add(w); return w
    rels = [
        ("What is the capital of the country {e}?", "Which city serves as the capital of {e}?", "{e}'s seat of government is located in which city?", lambda: word(3)),
        ("Who founded the company {e}?", "Name the founder of {e}.", "The company {e} was started by whom?", lambda: word(2) + " " + word(3)),
        ("In what year was the city of {e} founded?", "When was {e} established? Give the year.", "What is the founding year of {e}?", lambda: str(rng.randint(1100, 1990))),
        ("What language is spoken on the island of {e}?", "Which language do the people of {e} speak?", "The inhabitants of {e} speak what language?", lambda: word(3) + "ish"),
        ("What is the name of the river that flows through {e}?", "Which river runs through {e}?", "{e} lies on the banks of which river?", lambda: word(2) + " River"),
        ("Which element did the chemist {e} discover?", "What element is credited as {e}'s discovery?", "The scientist {e} is known for discovering which element?", lambda: word(2) + "ium"),
    ]
    facts = []
    for n in range(CFG["nights"]):
        for _ in range(CFG["facts_per_night"]):
            tq, q1, q2, ag = rels[rng.randrange(len(rels))]; e, a = word(3), ag()
            facts.append({"night": n + 1, "e": e, "a": a, "train_q": tq.format(e=e), "train_a": f"{a}.", "test": [q1.format(e=e), q2.format(e=e)]})
    return facts

def load_facts(CFG, smoke):
    FACTS = make_facts(CFG)
    h = hashlib.sha256(json.dumps(FACTS).encode()).hexdigest()
    if not smoke and CFG["nights"] == 7 and CFG["facts_per_night"] == 100:
        assert h == EXPECTED_FACT_HASH, f"fact hash {h} != sleep_shield {EXPECTED_FACT_HASH}"
    vocab = set()
    for f in FACTS:
        vocab.add(f["e"]); vocab.update(w for w in f["a"].split() if not w.isdigit() and w != "River")
    return FACTS, h, vocab

def ans_match(out, a):
    core = a.replace(" River", "")
    return re.search(r"\b" + re.escape(core.lower()) + r"\b", out.lower()) is not None

# ---------------------------------------------------------------- LoRA + training (verbatim from sleep_shield.py)
MODS = r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
def lora_cfg(CFG, layers):
    from peft import LoraConfig
    return LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"], bias="none", task_type="CAUSAL_LM",
                      target_modules=r".*\.layers\.(" + "|".join(map(str, layers)) + r")\." + MODS)

class Trainer:
    def __init__(self, CFG, tok, chat):
        self.CFG, self.tok, self.chat = CFG, tok, chat
        self.END = "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token
    def encode(self, q, a, max_len=None):
        p = self.chat(q); pi = self.tok(p, add_special_tokens=False)["input_ids"]
        fi = self.tok(p + a + self.END, add_special_tokens=False)["input_ids"]
        assert fi[:len(pi)] == pi
        return fi, [-100] * len(pi) + fi[len(pi):]
    def encode_train(self, f): return self.encode(f["train_q"], f["train_a"])

    def _batch_tensors(self, b):
        tok = self.tok; L = max(len(x[0]) for x in b)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in b], device=DEV)
        lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in b], device=DEV)
        att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in b], device=DEV)
        return ids, lab, att

    def train_night(self, model, facts, seed, night, extra=None, extra_slots=None):
        """Arm b (extra=None): IDENTICAL to sleep_shield.train_night (same rng stream, single forward, same loss).
        Phase 2 mixing (extra = list of pre-encoded (ids, labels) items, extra_slots = per-step count): each step takes
        batch_size - k fact items from the same rng stream and k extra items consumed in a deterministic per-night
        permutation. The two groups are run as two micro-batches whose summed token-level CE is divided by the total number
        of supervised tokens, which equals the loss of one combined batch (avoids padding short facts to replay length)."""
        CFG = self.CFG; model.train(); data = [self.encode_train(f) for f in facts]
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP); rng = random.Random(seed * 100 + night); losses = []
        if extra is not None:
            perm = list(range(len(extra))); random.Random(seed * 1000 + night + 7).shuffle(perm); ptr = 0
        for step in range(CFG["steps_per_night"]):
            if extra is None:
                b = [data[rng.randrange(len(data))] for _ in range(CFG["batch_size"])]
                ids, lab, att = self._batch_tensors(b)
                with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                lv = loss.item()
            else:
                k = extra_slots[step % len(extra_slots)]
                b = [data[rng.randrange(len(data))] for _ in range(CFG["batch_size"] - k)]
                e = []
                for _ in range(k):
                    e.append(extra[perm[ptr % len(perm)]]); ptr += 1
                ntot = sum(sum(1 for t in x[1][1:] if t != -100) for x in b + e)
                # memfix: same summed-CE / ntot loss, but backward per micro-batch (facts group, then each extra item alone)
                # so only one graph + one fp32 logits tensor is alive at a time. Gradients are identical up to fp summation order.
                groups = ([b] if b else []) + [[x] for x in e]
                opt.zero_grad(set_to_none=True); lv = 0.0
                for g in groups:
                    ids, lab, att = self._batch_tensors(g)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                        logits = model(input_ids=ids, attention_mask=att).logits
                        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                             ignore_index=-100, reduction="sum") / ntot
                    scaler.scale(ce).backward(); lv += float(ce.item()); del logits, ce
            if extra is None:
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); losses.append(lv)
        model.eval(); return float(np.mean(losses[-10:]))

def fingerprint(model): return float(sum(p.double().sum().item() for p in model.parameters()))

# ---------------------------------------------------------------- evaluation helpers
def eval_recall(gen, model, facts, CFG):
    qs = [q for f in facts for q in f["test"]]; ans = [f["a"] for f in facts for _ in f["test"]]
    outs, _, _ = gen(model, qs, 24, CFG["recall_bs"])
    return 100.0 * float(np.mean([ans_match(o, a) for o, a in zip(outs, ans)])) if qs else float("nan")

def score_gsm(items, idx, outs, ntok, trunc, fact_vocab, first=None):
    """per-item records + aggregate (retention = % of the given idx correct)"""
    recs = []; first = first if first is not None else [None] * len(idx)
    for i, o, n, t, ft in zip(idx, outs, ntok, trunc, first):
        ps, pl = parse_gsm(o), parse_gsm_lenient(o); gold = norm_num(items[i]["a"])
        ok_s, ok_l = ps is not None and ps == gold, pl is not None and pl == gold
        recs.append({"item": i, "gold": gold, "output": o, "n_tokens": n, "truncated": t, "strict_pred": ps, "lenient_pred": pl,
                     "strict_ok": ok_s, "lenient_ok": ok_l, "n_steps": n_arith_steps(o), "has_steps": has_steps(o), "n_lines": n_lines(o),
                     "has_exact_marker": bool(re.search(r"The answer is", o)), "first_token": ft,
                     "category": categorize(o, n, t, ok_s, fact_vocab)})
    return recs

def agg_gsm(recs):
    n = max(len(recs), 1); fails = [r for r in recs if not r["strict_ok"]]
    cat = {c: sum(r["category"] == c for r in fails) for c in CATS}
    return {"strict": 100.0 * sum(r["strict_ok"] for r in recs) / n, "lenient": 100.0 * sum(r["lenient_ok"] for r in recs) / n,
            "mean_tokens": float(np.mean([r["n_tokens"] for r in recs])) if recs else float("nan"),
            "median_tokens": float(np.median([r["n_tokens"] for r in recs])) if recs else float("nan"),
            "pct_steps": 100.0 * sum(r["has_steps"] for r in recs) / n, "pct_truncated": 100.0 * sum(r["truncated"] for r in recs) / n,
            "pct_exact_marker": 100.0 * sum(r["has_exact_marker"] for r in recs) / n,
            "mean_lines": float(np.mean([r["n_lines"] for r in recs])) if recs else float("nan"),
            "n": len(recs), "n_fail": len(fails), "fail_categories": cat,
            "fail_category_pct": {c: (100.0 * v / len(fails) if fails else 0.0) for c, v in cat.items()}}

def eval_simple(gen, model, probes, name, idx, CFG):
    MAX = {"mult": 8, "arc": 4}[name]
    outs, _, _ = gen(model, [probes[name][i]["q"] for i in idx], MAX, CFG["gen_bs"]) if idx else ([], [], [])
    c = [bool(correct(name, probes[name][i], o)) for i, o in zip(idx, outs)]
    return 100.0 * float(np.mean(c)) if c else float("nan"), c

def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs: return float("nan"), float("nan")
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)
