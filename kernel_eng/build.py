"""Build the English rung kernel: inline src/model.py + src/flops.py + kernel/eng.py."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kernel_eng")
os.makedirs(OUT, exist_ok=True)

# scan_triton must be inlined BEFORE model.py: model imports it at module level, and if the
# import fails it falls back to the reference scan, which is 3.5x slower and would quietly
# turn an 8.5 h run into a 30 h one.
scan = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "scan_triton.py")).read(), flags=re.S)
model = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "model.py")).read(), flags=re.S)
# The inlined scan is top-level code in this file, not an importable module, so model.py's
# `import scan_triton` has to resolve to a shim over the functions already defined above it.
model = model.replace("import scan_triton", "scan_triton = _inlined_scan_module()")
# The scan's own imports are hard (triton is required), so the inlined block is guarded:
# a missing dependency should leave scan_triton as None and let the driver refuse loudly,
# rather than crashing the whole script before it can say why.
_scan_lines = scan.splitlines()
scan = ("try:\n"
        + "\n".join(("    " + l) if l.strip() else l for l in _scan_lines)
        + "\nexcept Exception:\n    scan_triton = None\n")
SHIM = """
def _inlined_scan_module():
    class _InlinedScan:
        ssm_scan = staticmethod(ssm_scan)
    return _InlinedScan()

"""

flops = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "flops.py")).read(), flags=re.S)
driver = open(os.path.join(ROOT, "kernel", "eng.py")).read()

import sys
cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/eng-rung.json"
cfg = json.load(open(os.path.join(ROOT, cfg_path)))
tag = "eng-rung"
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)

inline_cfg = f'\nopen("/tmp/eng.json", "w").write({json.dumps(json.dumps(cfg))})\n'

out = ("import json\n" + inline_cfg + scan + SHIM + model + "\n" + flops + "\n" + driver)
# The driver is eng_driver.py; main.py is the launcher that runs one job per GPU. Kaggle runs
# a single code_file, so the launcher has to be the entry point.
open(os.path.join(OUT, "eng_driver.py"), "w").write(out)
# The launcher needs the same baked config, and must not depend on CONFIG being set in the
# environment: Kaggle sets no such variable. A local dry run that passes CONFIG explicitly
# hides this, which is exactly how the first launch failed.
open(os.path.join(OUT, "main.py"), "w").write(
    "import json\n" + inline_cfg + open(os.path.join(ROOT, "kernel", "eng_par.py")).read())

json.dump({"id": "mauricew/hnet-english-rung", "title": "hnet english rung",
           "code_file": "main.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": True, "enable_tpu": False,
           "enable_internet": False, "keywords": [],
           "dataset_sources": ["mauricew/hnet-english-bytes"],
           "kernel_sources": [], "competition_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=2)
print("wrote", OUT)
