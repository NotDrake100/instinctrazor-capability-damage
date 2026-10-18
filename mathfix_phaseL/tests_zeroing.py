"""Tiny-model unit test: zeroing all LoRA layers == base; zeroing none == full adapter.

Uses trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 (the mathfix smoke model).
Numbers are not experimental evidence.
"""
import os
import sys

import torch
from peft import get_peft_model

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "mathfix"))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from zeroing import restore_lora_weights, snapshot_lora_weights, zero_lora_delta_layers  # noqa: E402


def _gen_ids(model, tok, text, max_new=8):
    model.eval()
    enc = tok([text], return_tensors="pt", add_special_tokens=True)
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
    return out[0].tolist()


def main():
    torch.manual_seed(0)
    name = C.SMOKE_OVERRIDES["model"]
    tok, base = C.load_model(name)
    n_layers = len(base.model.layers)
    assert n_layers >= 1, n_layers
    CFG = dict(C.BASE_CFG)
    CFG.update(C.SMOKE_OVERRIDES)
    prompt = "What is 2 + 2?"
    base_ids = _gen_ids(base, tok, prompt)

    model = get_peft_model(base, C.lora_cfg(CFG, list(range(n_layers))))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    peft_ids = _gen_ids(model, tok, prompt)
    assert peft_ids != base_ids, "randomized LoRA must change greedy output vs base"

    snap = snapshot_lora_weights(model)
    n_zero = zero_lora_delta_layers(model, list(range(n_layers)))
    assert n_zero > 0, "zeroed no modules"
    all_zero_ids = _gen_ids(model, tok, prompt)
    assert all_zero_ids == base_ids, (
        f"zeroing all layers must match base exactly\nbase={base_ids}\nzero={all_zero_ids}"
    )

    restore_lora_weights(model, snap)
    none_ids = _gen_ids(model, tok, prompt)
    n_none = zero_lora_delta_layers(model, [])
    none_ids2 = _gen_ids(model, tok, prompt)
    assert n_none == 0
    assert none_ids == peft_ids, (
        f"restored / zero-none must match full adapter\npeft={peft_ids}\nnone={none_ids}"
    )
    assert none_ids2 == peft_ids

    print("PASS tests_zeroing: all-layers-zero == base; no-zero == full-b")
    print(f"  tiny model {name} layers={n_layers} modules_zeroed={n_zero}")
    print(f"  base_ids={base_ids}")
    print(f"  peft_ids={peft_ids}")


if __name__ == "__main__":
    main()
