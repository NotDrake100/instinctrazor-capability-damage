# shared by run_phase1.sh / run_phase2.sh
pip install -q -r requirements.txt
# new peft refuses to build LoRA layers if an old torchao (<0.16) is installed; the experiment never uses torchao.
python - <<'PY'
import importlib.metadata as m, subprocess, sys
try:
    v = m.version("torchao")
    if tuple(int(x) for x in v.split(".")[:2]) < (0, 16):
        print("removing incompatible torchao", v); subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
except m.PackageNotFoundError:
    pass
PY
python -c "import torch, transformers, peft, datasets; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| peft', peft.__version__)"
