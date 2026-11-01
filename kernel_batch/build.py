"""Build the warm-started batch probe: inline scan+model+flops+driver, then the probe.

The driver's definitions (Docs, make_batch, masked_loss, batch_plan) are needed by the probe,
but its main() is guarded by __main__, so inlining it is safe.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kernel_batch")
os.makedirs(OUT, exist_ok=True)


def strip(path):
    return re.sub(r'^"""*.*?"""*\n', "", open(path).read(), flags=re.S)


scan = strip(os.path.join(ROOT, "src", "scan_triton.py"))
model = strip(os.path.join(ROOT, "src", "model.py")).replace(
    "import scan_triton", "scan_triton = _inlined_scan_module()")
scan = ("try:\n"
        + "\n".join(("    " + l) if l.strip() else l for l in scan.splitlines())
        + "\nexcept Exception:\n    scan_triton = None\n")
SHIM = """
def _inlined_scan_module():
    class _InlinedScan:
        ssm_scan = staticmethod(ssm_scan)
    return _InlinedScan()

"""
flops = strip(os.path.join(ROOT, "src", "flops.py"))
driver = strip(os.path.join(ROOT, "kernel", "eng.py"))
probe = open(os.path.join(ROOT, "kernel", "batch_probe2.py")).read()

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/eng-rung.json"
cfg = json.load(open(os.path.join(ROOT, cfg_path)))
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)

# The inlined driver reads CONFIG at import; Kaggle sets no environment, so bake the config
# the same way kernel_eng does. The probe itself does not read it, but the driver module does
# the moment this file is imported.
inline = 'open("/tmp/eng.json", "w").write(' + json.dumps(json.dumps(cfg)) + ')\n'
open(os.path.join(OUT, "main.py"), "w").write(
    "import json\n" + inline + scan + SHIM + model + "\n" + flops + "\n" + driver + "\n" + probe)

_kid = os.environ.get("KERNEL_ID", "mauricew/hnet-batch-probe2")
meta = {"id": _kid, "title": _kid.split("/")[-1].replace("-", " "), "code_file": "main.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": ["mauricew/hnet-english-bytes", "mauricew/hnet-eng-ckpt"],
        "kernel_sources": [],
        "competition_sources": [], "model_sources": []}
json.dump(meta, open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=2)
print("wrote", OUT)
