import json,numpy as np,matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
import os; D=os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))+"/";r=np.random.default_rng(0);h=json.load(open(D+"heldout.json"));out={}
for s in h["n"]:
    a=np.array(h["flat"]["kept"][s]);b=np.array(h["protect_L0"]["kept"][s]);n=len(a)
    i=r.integers(0,n,(10000,n))
    out[s]={"n":n,"flat":[float(a.mean()),*map(float,np.percentile(a[i].mean(1),[2.5,97.5]))],
            "protect":[float(b.mean()),*map(float,np.percentile(b[i].mean(1),[2.5,97.5]))],
            "gain":[float((b-a).mean()),*map(float,np.percentile((b[i]-a[i]).mean(1),[2.5,97.5]))]}
out["bits"]={"flat":h["flat"]["bits"],"protect_L0":h["protect_L0"]["bits"]}
json.dump(out,open(D+"heldout_analysis.json","w"),indent=1);print(json.dumps(out,indent=1))
S=list(h["n"]);x=np.arange(len(S))
for k,(key,lab) in enumerate([("flat",f"shipped flat recipe ({out['bits']['flat']:.3f} b)"),("protect",f"+ --protect-layers 0 ({out['bits']['protect_L0']:.3f} b)")]):
    m=np.array([out[s][key][0] for s in S])*100;lo=np.array([out[s][key][1] for s in S])*100;hi=np.array([out[s][key][2] for s in S])*100
    plt.bar(x+(k-.5)*.38,m,.38,yerr=[m-lo,hi-m],capsize=4,label=lab)
plt.xticks(x,[f"{s} (n={out[s]['n']})" for s in S]);plt.ylabel("% of originally-correct items retained");plt.ylim(0,100)
plt.title("Held-out items, granite-3.0-1b-a400m (InstinctRazor quant, no OPD)");plt.legend(loc="lower right");plt.tight_layout();plt.savefig(D+"fix_chart.png",dpi=130)
