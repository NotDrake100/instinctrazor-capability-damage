import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os; D=os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))+"/"; rng=np.random.default_rng(0); B=10000
it=json.load(open(D+"items.json")); c=json.load(open(D+"compressed.json")); per={k:np.array(v) for k,v in json.load(open(D+"perlayer_items.json")).items()}
SK=list(it["items"]); out={"n":{s:len(it["items"][s]) for s in SK},"effective_bits":c["effective_bits"]}
def ci(x): 
    bs=rng.choice(x,(B,len(x))).mean(1); return [float(x.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
out["retention"]={s:ci(np.array(c["comp_kept"][s])) for s in SK}
out["drop"]={s:[1-v for v in out["retention"][s][::-1]] for s in SK}  # [hi..] reorder below
out["drop"]={s:[1-out["retention"][s][0],1-out["retention"][s][2],1-out["retention"][s][1]] for s in SK}
op=it["orig_pool"]; cp=c["comp_pool"]
out["pool_acc"]={s:{"orig":float(np.mean(op[s])),"comp":float(np.mean(cp[s])),"n":len(op[s])} for s in SK}
# pairwise drop difference CIs
pw={}
for i,a in enumerate(SK):
    for b in SK[i+1:]:
        xa,xb=1-np.array(c["comp_kept"][a]),1-np.array(c["comp_kept"][b])
        d=rng.choice(xa,(B,len(xa))).mean(1)-rng.choice(xb,(B,len(xb))).mean(1)
        pw[f"{a}-{b}"]=[float(xa.mean()-xb.mean()),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]
out["drop_diff"]=pw
out["kill_fired"]=all(lo<=0<=hi for _,lo,hi in pw.values())
# per-layer
dl={s:(1-per[s].mean(1)).tolist() for s in SK}; out["perlayer_drop"]=dl
def ci_l(s,l): x=1-per[s][l]; bs=rng.choice(x,(2000,len(x))).mean(1); return [float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
out["perlayer_ci"]={s:[ci_l(s,l) for l in range(per[s].shape[0])] for s in SK}
from scipy.stats import spearmanr
out["perlayer_spearman"]={f"{a}-{b}":float(spearmanr(dl[a],dl[b])[0]) for i,a in enumerate(SK) for b in SK[i+1:]}
out["worst_layer"]={s:int(np.argmax(dl[s])) for s in SK}
json.dump(out,open(D+"analysis.json","w"),indent=1)
fig,ax=plt.subplots(1,2,figsize=(13,4.5),gridspec_kw={"width_ratios":[1,2.6]})
for k,s in enumerate(SK):
    m,lo,hi=out["drop"][s]; ax[0].bar(k,m*100,yerr=[[100*(m-lo)],[100*(hi-m)]],capsize=5)
ax[0].set_xticks(range(len(SK)),[f"{s}\n(n={out['n'][s]})" for s in SK]); ax[0].set_ylabel("% of originally-correct items lost")
ax[0].set_title("Full InstinctRazor recipe (no OPD)\n95% bootstrap CI")
L=len(dl[SK[0]]); w=0.27
for k,s in enumerate(SK):
    m=np.array(dl[s])*100; e=np.array(out["perlayer_ci"][s]).T*100
    ax[1].bar(np.arange(L)+(k-1)*w,m,w,yerr=[m-e[0],e[1]-m],capsize=1,label=s,error_kw={"lw":0.5})
ax[1].set_xlabel("layer quantized alone (INT3 experts + INT4 attn)"); ax[1].set_ylabel("% lost"); ax[1].legend()
ax[1].set_title("Per-layer localization, granite-3.0-1b-a400m-base"); plt.tight_layout(); plt.savefig(D+"chart.png",dpi=130)
print(json.dumps({k:v for k,v in out.items() if k not in("perlayer_ci",)},indent=1))
