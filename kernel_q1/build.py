"""Build the Q1 GPU kernel: inline src/model.py + configs/q1.json + kernel/q1.py."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kernel_q1")
os.makedirs(OUT, exist_ok=True)

model = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "model.py")).read(), flags=re.S)
flops = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "flops.py")).read(), flags=re.S)
driver = open(os.path.join(ROOT, "kernel", "q1.py")).read()
driver = driver.split("# ==== DATA ====", 1)[1]
head = open(os.path.join(ROOT, "kernel", "q1.py")).read().split("# ==== DATA ====")[0]

import sys
cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/q1.json"
cfg = json.load(open(os.path.join(ROOT, cfg_path)))
tag = os.path.splitext(os.path.basename(cfg_path))[0]
os.makedirs(os.path.join(OUT, "configs"), exist_ok=True)
json.dump(cfg, open(os.path.join(OUT, "configs", "q1.json"), "w"), indent=2)

# config is read from a path, but only code_file uploads -- so default to the inlined copy
head = head.replace('os.environ.get("CONFIG", "configs/q1.json")', 'os.environ.get("CONFIG", "/tmp/q1.json")')
inline_cfg = f'\nopen("/tmp/q1.json", "w").write({json.dumps(json.dumps(cfg))})\n'

out = ("import json\n" + inline_cfg + head + model + "\n" + flops + "\n# ==== DATA ====" + driver)
open(os.path.join(OUT, "main.py"), "w").write(out)

json.dump({f"id": f"mauricew/hnet-{tag}", "title": f"hnet {tag}", "code_file": "main.py",
           "language": "python", "kernel_type": "script", "is_private": True,
           "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
           "keywords": [], "dataset_sources": ["mauricew/hnet-multiscript-bytes"],
           "kernel_sources": [], "competition_sources": [], "model_sources": [],
           },
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=2)
print("wrote", OUT)
