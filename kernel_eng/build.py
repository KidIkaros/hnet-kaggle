"""Build the English rung kernel: inline src/model.py + src/flops.py + kernel/eng.py."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kernel_eng")
os.makedirs(OUT, exist_ok=True)

model = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "model.py")).read(), flags=re.S)
flops = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "flops.py")).read(), flags=re.S)
driver = open(os.path.join(ROOT, "kernel", "eng.py")).read()

import sys
cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/eng-rung.json"
cfg = json.load(open(os.path.join(ROOT, cfg_path)))
tag = "eng-rung"
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)

inline_cfg = f'\nopen("/tmp/eng.json", "w").write({json.dumps(json.dumps(cfg))})\n'

out = ("import json\n" + inline_cfg + model + "\n" + flops + "\n" + driver)
open(os.path.join(OUT, "main.py"), "w").write(out)

json.dump({"id": "mauricew/hnet-english-rung", "title": "hnet english rung",
           "code_file": "main.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": True, "enable_tpu": False,
           "enable_internet": False, "keywords": [],
           "dataset_sources": ["mauricew/hnet-english-bytes"],
           "kernel_sources": [], "competition_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=2)
print("wrote", OUT)
