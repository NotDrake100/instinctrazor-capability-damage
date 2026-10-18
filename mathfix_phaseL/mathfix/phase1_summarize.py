"""Summarize phase-1 results: per-night tables, automatic hypothesis classification (rules = predictions.md, section
'Decision rules'), failure dump for hand review, and the Phase-2 mechanism choice. Run standalone:
    python phase1_summarize.py <phase1_out_dir>
"""
import os, sys, json, random, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# ---- pre-registered thresholds (copied in predictions.md; do not change after seeing phase-1 data)
TH = dict(F1_short_fail_share=60.0, F2_token_ratio=0.5, F3_steps_drop=40.0, F4_prefill_gain=25.0,
          L1_block_gain=25.0, L2_top2_seeds=2, L3_block_vs_median=2.0,
          S1_edit_gain=20.0, S2_vs_random=10.0, S3_merge_sanity=10.0, LENIENT_GAP=20.0)

def ms(xs): return C.mean_std(xs)
def read_jsonl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def main(OUT):
    R = json.load(open(os.path.join(OUT, "results.json"))); CFG = R["cfg"]; seeds = CFG["seeds"]; N = CFG["nights"]
    S = {"cfg": CFG, "n_layers": R.get("n_layers"), "thresholds": TH, "probe_hash": R.get("probe_hash"), "fact_hash": R.get("fact_hash"), "base": R.get("base", {}).get("acc"),
         "base_gsm_on_base_correct": R.get("base_gsm_on_base_correct"), "base_prefill": R.get("base_prefill"), "orig_overlap": R.get("orig_overlap")}
    # per-night table (mean, std over seeds)
    nights = {}
    for n in range(1, N + 1):
        ents = [R["nights"].get(f"s{s}", {}).get(str(n)) for s in seeds]; ents = [e for e in ents if e]
        if not ents: continue
        row = {"n_seeds": len(ents)}
        for k in ["recall", "recall_tonight", "mult", "arc"]: row[k] = ms([e[k] for e in ents])
        for k in ["strict", "lenient", "mean_tokens", "pct_steps", "pct_truncated", "pct_exact_marker"]: row["gsm_" + k] = ms([e["gsm"][k] for e in ents])
        row["gsm_fail_category_pct"] = {c: ms([e["gsm"]["fail_category_pct"][c] for e in ents]) for c in C.CATS}
        if all("gsm_prefill" in e for e in ents): row["gsm_prefill_strict"] = ms([e["gsm_prefill"]["strict"] for e in ents]); row["gsm_prefill_steps"] = ms([e["gsm_prefill"]["pct_steps"] for e in ents])
        nights[n] = row
    S["nights"] = nights
    last = max(nights) if nights else None; L = nights.get(last, {}) if last else {}
    base_g = R.get("base_gsm_on_base_correct") or {}
    ev = {}
    # ---- H1
    if L:
        short_share = L["gsm_fail_category_pct"]["no_steps"][0] + L["gsm_fail_category_pct"]["fact_style"][0]
        tok_ratio = L["gsm_mean_tokens"][0] / max(base_g.get("mean_tokens", float("nan")), 1e-9)
        steps_drop = base_g.get("pct_steps", float("nan")) - L["gsm_pct_steps"][0]
        pf_gain = (L["gsm_prefill_strict"][0] - L["gsm_strict"][0]) if "gsm_prefill_strict" in L else float("nan")
        F1, F2, F3, F4 = short_share >= TH["F1_short_fail_share"], tok_ratio <= TH["F2_token_ratio"], steps_drop >= TH["F3_steps_drop"], pf_gain >= TH["F4_prefill_gain"]
        ev["H1"] = {"short_fail_share": short_share, "token_ratio_vs_base": tok_ratio, "steps_drop_pts": steps_drop, "prefill_gain_pts": pf_gain,
                    "F1": F1, "F2": F2, "F3": F3, "F4_capability_intact": F4, "supported": bool(F1 and F2 and F3), "pure_format": bool(F1 and F2 and F3 and F4),
                    "restoration_pts": pf_gain}
        lg = L["gsm_lenient"][0] - L["gsm_strict"][0]
        S["lenient_vs_strict_night_last"] = {"strict": L["gsm_strict"], "lenient": L["gsm_lenient"], "gap": lg,
                                              "parser_level_format_loss": bool(lg >= TH["LENIENT_GAP"]),
                                              "note": "strict parser already falls back to the last number when there is no answer marker, so lenient>strict only when a marker exists and points to a different number"}
    # ---- H2 (layer swap)
    sw = R.get("swap", {})
    if sw and all(f"s{s}" in sw and "none" in sw[f"s{s}"] for s in seeds):
        cfgs = [k for k in sw[f"s{seeds[0]}"] if k not in ("none",)]
        cfgs = [k for k in cfgs if all(k in sw[f"s{s}"] for s in seeds)]
        tab = {}
        for k in cfgs:
            gains = [sw[f"s{s}"][k]["gsm_strict"] - sw[f"s{s}"]["none"]["gsm_strict"] for s in seeds]
            costs = [sw[f"s{s}"]["none"]["recall"] - sw[f"s{s}"][k]["recall"] for s in seeds]
            tab[k] = {"layers_off": sw[f"s{seeds[0]}"][k]["layers_off"], "gain": ms(gains), "recall_cost": ms(costs), "gains_per_seed": gains,
                      "steps_pct": ms([sw[f"s{s}"][k]["pct_steps"] for s in seeds])}
        S["swap_table"] = dict(sorted(tab.items(), key=lambda kv: -kv[1]["gain"][0]))
        blocks = {k: v for k, v in tab.items() if k.startswith("B")}; singles = {k: v for k, v in tab.items() if k.startswith("L")}
        if blocks:
            best = max(blocks, key=lambda k: blocks[k]["gain"][0]); bg = blocks[best]["gain"][0]
            top2 = 0
            for i, s in enumerate(seeds):
                order = sorted(blocks, key=lambda k: -blocks[k]["gains_per_seed"][i]); top2 += best in order[:2]
            med = float(np.median([v["gain"][0] for v in blocks.values()]))
            L1, L2, L3 = bg >= TH["L1_block_gain"], top2 >= TH["L2_top2_seeds"], bg >= TH["L3_block_vs_median"] * max(med, 1e-9) and bg > 0
            bs = max(singles, key=lambda k: singles[k]["gain"][0]) if singles else None
            ev["H2"] = {"best_block": best, "best_block_layers": blocks[best]["layers_off"], "best_block_gain": bg, "best_block_recall_cost": blocks[best]["recall_cost"][0],
                        "best_block_top2_in_seeds": top2, "median_block_gain": med, "best_single": bs, "best_single_gain": singles[bs]["gain"][0] if bs else None,
                        "all_off_gain": tab.get("all", {}).get("gain", [None])[0],
                        "reasoning_wrong_share_night_last": L["gsm_fail_category_pct"]["reasoning_wrong"][0] if L else None,
                        "L1": L1, "L2": L2, "L3": L3, "supported": bool(L1 and L2 and L3), "restoration_pts": bg}
    # ---- H3 (shared direction)
    sh = R.get("shared", {})
    if sh and all(len(sh.get(f"s{s}", {})) >= 5 for s in seeds):
        def g(v, f="gsm_strict"): return [sh[f"s{s}"][v][f] - sh[f"s{s}"]["merged_unedited"][f] for s in seeds]
        rows = {v: {"gsm_gain": ms(g(v)), "recall_change": ms(g(v, "recall")), "mult_change": ms(g(v, "mult")), "arc_change": ms(g(v, "arc")),
                    "steps_pct": ms([sh[f"s{s}"][v]["pct_steps"] for s in seeds])} for v in ["minus_shared_top1", "minus_own_top1", "minus_random_rank1", "base_no_adapter"]}
        S["shared_table"] = rows; S["shared_stats"] = R.get("shared_stats")
        sanity = [abs(sh[f"s{s}"]["merged_unedited"]["gsm_strict"] - sw.get(f"s{s}", {}).get("none", {}).get("gsm_strict", float("nan"))) for s in seeds]
        S3 = all(x <= TH["S3_merge_sanity"] for x in sanity)
        best_edit = max(["minus_shared_top1", "minus_own_top1"], key=lambda v: rows[v]["gsm_gain"][0])
        eg, rg = rows[best_edit]["gsm_gain"][0], rows["minus_random_rank1"]["gsm_gain"][0]
        S1, S2 = eg >= TH["S1_edit_gain"], (eg - rg) >= TH["S2_vs_random"]
        ev["H3"] = {"best_edit": best_edit, "edit_gain": eg, "random_gain": rg, "edit_recall_change": rows[best_edit]["recall_change"][0],
                    "merge_sanity_abs_diff": sanity, "S1": S1, "S2": S2, "S3_valid": S3, "supported": bool(S1 and S2 and S3), "restoration_pts": eg}
    S["evidence"] = ev
    # ---- drift / lens summary (night last, mean over seeds)
    dl = R.get("drift_lens", {})
    if last and all(f"s{s}_n{last}" in dl for s in seeds):
        def avg(k): return [float(np.mean([dl[f"s{s}_n{last}"][k][l] for s in seeds])) for l in range(len(dl[f"s{seeds[0]}_n{last}"][k]))]
        S["drift_lens_night_last"] = {k: avg(k) for k in ["cos_all", "cos_last", "coherence_last", "kl_last", "pR", "pS", "pD", "pR_base", "pS_base", "pD_base"]}
        S["drift_lens_by_night_final_layer"] = {n: {k: float(np.mean([dl[f"s{s}_n{n}"][k][-1] for s in seeds if f"s{s}_n{n}" in dl]))
                                                    for k in ["kl_last", "pR", "pS", "cos_last"]} for n in range(1, last + 1)}
        S["token_sets"] = R.get("token_sets")
    # ---- phase-2 choice
    sup = {h: e for h, e in ev.items() if e.get("supported")}
    if sup:
        h = max(sup, key=lambda k: (sup[k]["restoration_pts"] if not math.isnan(sup[k]["restoration_pts"]) else -1e9))
        choice = {"mechanism": h, "reason": f"supported hypotheses {sorted(sup)}; {h} has the largest GSM8K restoration ({sup[h]['restoration_pts']:.1f} pts)"}
    else:
        choice = {"mechanism": "H1", "reason": "NO hypothesis met its pre-registered criteria; pre-registered fallback = reasoning replay vs token-matched control (mechanism-agnostic)", "fallback": True}
    if "H2" in ev: choice["protect_layers"] = ev["H2"]["best_block_layers"]
    if "H3" in ev: choice["edit"] = ev["H3"]["best_edit"]
    S["phase2_choice"] = choice
    C.save_json(os.path.join(OUT, "phase1_summary.json"), S)
    # ---- failure dump (50 random failures per night for nights 1, 4, last)
    md = ["# Phase 1: GSM8K failures for hand review", "",
          "Rules (common.categorize, applied in order): truncated -> fact_style (<=12 words, no arithmetic, no digit or a fact-stream word) -> "
          "no_steps (<2 arithmetic ops/equations) -> reasoning_wrong (>=2 ops, finished, wrong final answer) -> other.", ""]
    for n in sorted({1, 4, last} & set(nights)) if last else []:
        fails = [r for s in seeds for r in read_jsonl(os.path.join(OUT, "gens", f"s{s}_n{n}.jsonl")) if not r["strict_ok"]]
        pick = random.Random(1000 + n).sample(fails, min(50, len(fails)))
        md += [f"## Night {n}: {len(fails)} failures over {len(seeds)} seeds; showing {len(pick)} random", ""]
        for j, r in enumerate(pick, 1):
            o = r["output"].replace("```", "'''")
            md += [f"### {n}.{j} seed {r['seed']} item {r['item']} | category **{r['category']}** | gold {r['gold']} | strict {r['strict_pred']} | lenient {r['lenient_pred']} | {r['n_tokens']} tok | steps {r['n_steps']}",
                   "", "```", o[:1500] + (" ...[cut]" if len(o) > 1500 else ""), "```", "", "Hand label: ______", ""]
    open(os.path.join(OUT, "failures_review.md"), "w").write("\n".join(md))
    # ---- short report
    rep = ["# Phase 1 report (auto-generated from results.json; numbers are mean ± std over seeds)", ""]
    if base_g: rep.append(f"Base on its base-correct GSM8K items: lenient {base_g['lenient']:.1f}, mean tokens {base_g['mean_tokens']:.0f}, reasoning steps present {base_g['pct_steps']:.0f}%")
    rep += ["", "| night | recall | GSM strict | GSM lenient | tokens | % steps | % trunc | fail: no_steps / fact_style / reasoning_wrong / truncated | mult | arc |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in nights.items():
        fc = r["gsm_fail_category_pct"]
        rep.append(f"| {n} | {r['recall'][0]:.1f}±{r['recall'][1]:.1f} | {r['gsm_strict'][0]:.1f}±{r['gsm_strict'][1]:.1f} | {r['gsm_lenient'][0]:.1f}±{r['gsm_lenient'][1]:.1f} | "
                   f"{r['gsm_mean_tokens'][0]:.0f} | {r['gsm_pct_steps'][0]:.0f} | {r['gsm_pct_truncated'][0]:.0f} | {fc['no_steps'][0]:.0f} / {fc['fact_style'][0]:.0f} / {fc['reasoning_wrong'][0]:.0f} / {fc['truncated'][0]:.0f} | "
                   f"{r['mult'][0]:.1f} | {r['arc'][0]:.1f} |")
    if L and "gsm_prefill_strict" in L: rep += ["", f"Prefill test at night {last}: strict {L['gsm_prefill_strict'][0]:.1f}±{L['gsm_prefill_strict'][1]:.1f} (trained {L['gsm_strict'][0]:.1f})"]
    rep += ["", "## Evidence vs pre-registered rules", "```", json.dumps(ev, indent=1, default=str), "```", "", "## Phase-2 choice", "```", json.dumps(choice, indent=1), "```"]
    if "swap_table" in S:
        rep += ["", "## Layer swap (top 10 by GSM gain)", "| config | layers off | GSM gain | recall cost |", "|---|---|---|---|"]
        for k, v in list(S["swap_table"].items())[:10]: rep.append(f"| {k} | {v['layers_off'] if len(v['layers_off']) < 6 else 'all'} | {v['gain'][0]:.1f}±{v['gain'][1]:.1f} | {v['recall_cost'][0]:.1f} |")
    if "shared_table" in S:
        rep += ["", "## Shared-direction edits (change vs merged unedited night-7 model)", "| edit | GSM gain | recall change | mult | arc |", "|---|---|---|---|---|"]
        for k, v in S["shared_table"].items(): rep.append(f"| {k} | {v['gsm_gain'][0]:.1f}±{v['gsm_gain'][1]:.1f} | {v['recall_change'][0]:.1f} | {v['mult_change'][0]:.1f} | {v['arc_change'][0]:.1f} |")
        rep.append(f"\nshared stats: {S.get('shared_stats')}")
    open(os.path.join(OUT, "phase1_report.md"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return S

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1_out"))
