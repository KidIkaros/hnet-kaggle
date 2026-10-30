"""Build the encoding-invariance kernel: inline src/model.py + kernel/invariance.py."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kernel_inv")
os.makedirs(OUT, exist_ok=True)

model = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "model.py")).read(), flags=re.S)
driver = open(os.path.join(ROOT, "kernel", "invariance.py")).read()
import sys
cfg_name = sys.argv[1] if len(sys.argv) > 1 else "q1-u16.json"
tag = os.path.splitext(cfg_name)[0]
cfg = json.load(open(os.path.join(ROOT, "configs", cfg_name)))

head = driver.split("class Stream:", 1)[0]
body = "class Stream:" + driver.split("class Stream:", 1)[1]
body = body.split("def main():", 1)[0] + "def main():" + body.split("def main():", 1)[1]

out = ("import json\n"
       + f'open("/tmp/cfg.json", "w").write({json.dumps(json.dumps(cfg))})\n'
       + head + model + "\n" + body)
open(os.path.join(OUT, "main.py"), "w").write(out)

json.dump({"id": f"mauricew/hnet-{tag}", "title": f"hnet {tag}",
           "code_file": "main.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": False, "enable_tpu": False,
           "enable_internet": False, "keywords": [],
           "dataset_sources": ["mauricew/hnet-multiscript-bytes"],
           "kernel_sources": [], "competition_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=2)
print("wrote", OUT)
