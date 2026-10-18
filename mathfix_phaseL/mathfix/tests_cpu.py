"""CPU correctness checks for the mathfix machinery (real Qwen2.5-0.5B-Instruct, random LoRA weights):
 1. LoRA-scale swap: all layers off == base logits; none off == PEFT logits
 2. merged-weight edit path (phase-1 (f), 'merged_unedited') == PEFT logits
 3. two-micro-batch loss (phase-2 mixing) == one combined padded batch loss
 4. phase-2 edit_lora: 'top1' removes exactly sigma1 u1 v1^T; 'rand' removes a rank-1 of norm sigma1 and stays rank<=r
"""
import sys, os, re, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from peft import get_peft_model
torch.manual_seed(0)
CFG = dict(C.BASE_CFG)
tok, base = C.load_model(CFG["model"]); gen = C.Gen(tok, base); TR = C.Trainer(CFG, tok, gen.chat)
enc = tok([gen.chat("Tom has 3 apples and buys 4 more. How many?")], return_tensors="pt")
with torch.no_grad(): lb = base(**enc).logits
model = get_peft_model(base, C.lora_cfg(CFG, list(range(24))))
with torch.no_grad():
    for n, p in model.named_parameters():
        if "lora_B" in n: p.normal_(0, 0.02)
    lp = model(**enc).logits
print("peft vs base max|diff| (should be >0):", (lp - lb).abs().max().item())
from peft.tuners.lora import LoraLayer
orig = {n: dict(m.scaling) for n, m in model.named_modules() if isinstance(m, LoraLayer)}
def setscale(off):
    for n, m in model.named_modules():
        if isinstance(m, LoraLayer):
            li = int(re.search(r"layers\.(\d+)\.", n).group(1))
            for a in m.scaling: m.scaling[a] = 0.0 if li in off else orig[n][a]
with torch.no_grad():
    setscale(list(range(24))); l_all = model(**enc).logits; setscale([]); l_none = model(**enc).logits
print("1. all-off vs base:", (l_all - lb).abs().max().item(), "| none-off vs peft:", (l_none - lp).abs().max().item())
# 2. merged path
SC = CFG["lora_alpha"] / CFG["lora_r"]
with torch.no_grad():
    inner = model.get_base_model()
    deltas = {}
    for n, m in model.named_modules():
        if isinstance(m, LoraLayer): deltas[n] = SC * m.lora_B["default"].weight @ m.lora_A["default"].weight
    m2 = model.unload()
    W0 = {}
    for n, d in deltas.items():
        li, grp, nm = re.search(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+)", n).groups()
        W = getattr(getattr(m2.model.layers[int(li)], grp), nm).weight; W0[n] = W.clone(); W.add_(d)
    lm = m2(**enc).logits
    for n, d in deltas.items():
        li, grp, nm = re.search(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+)", n).groups()
        getattr(getattr(m2.model.layers[int(li)], grp), nm).weight.copy_(W0[n])
print("2. merged vs peft:", (lm - lp).abs().max().item(), "| restored vs base:", (m2(**enc).logits - lb).abs().max().item() if True else None)
base = m2
# 3. micro-batch loss equivalence
model = get_peft_model(base, C.lora_cfg(CFG, list(range(24))))
facts = [TR.encode("What is the capital of Zorbia?", "Mekalo."), TR.encode("Who founded Tikamo?", "Bera Sunolu.")]
extra = [TR.encode("Sam has 5 pens and gives 2 away. How many pens are left?", "5 - 2 = 3\nThe answer is 3.")]
with torch.no_grad():
    ids, lab, att = TR._batch_tensors(facts + extra); l_one = model(input_ids=ids, attention_mask=att, labels=lab).loss.item()
    ntot = sum(sum(1 for t in x[1][1:] if t != -100) for x in facts + extra); tot = 0.0
    for g in (facts, extra):
        ids, lab, att = TR._batch_tensors(g); lg = model(input_ids=ids, attention_mask=att).logits
        tot += F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum").item() / ntot
print("3. combined-batch loss", l_one, "| micro-batch loss", tot, "| diff", abs(l_one - tot))
base = model.unload()
# 4. edit_lora math (copy of phase2.edit_lora core)
A = torch.randn(16, 896) * 0.01; B = torch.randn(4864, 16) * 0.01; s = 2.0
dW = s * B @ A; U0, S0, V0 = torch.linalg.svd(dW, full_matrices=False)
def edit(A, B, kind, g):
    Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
    U, Sv, Vh = torch.linalg.svd(s * Rb @ Ra.T, full_matrices=False); U, Vh = Qb @ U, Vh @ Qa.T
    if kind == "top1": Sv = Sv.clone(); Sv[0] = 0
    else:
        ca = F.normalize(torch.randn(U.shape[1], generator=g), dim=0); cb = F.normalize(torch.randn(Vh.shape[0], generator=g), dim=0)
        core = torch.diag(Sv) - Sv[0] * torch.outer(ca, cb); u2, s2, v2 = torch.linalg.svd(core, full_matrices=False); U, Sv, Vh = U @ u2, s2, v2 @ Vh
    r = A.shape[0]; m = min(r, Sv.shape[0]); sq = torch.sqrt(Sv[:m].clamp_min(0) / s)
    return s * (U[:, :m] * sq) @ (sq[:, None] * Vh[:m])
d1 = edit(A, B, "top1", None); exp = dW - S0[0] * torch.outer(U0[:, 0], V0[0])
print("4a. top1 edit error (rel):", ((d1 - exp).norm() / exp.norm()).item())
d2 = edit(A, B, "rand", torch.Generator().manual_seed(0)); rem = dW - d2
print("4b. rand edit: removed norm", rem.norm().item(), "vs sigma1", S0[0].item(), "| removed rank", torch.linalg.matrix_rank(rem, rtol=1e-4).item(),
      "| |cos| with top1 direction", abs((rem.flatten() @ (U0[:, :1] @ V0[:1]).flatten()) / rem.norm()).item())
