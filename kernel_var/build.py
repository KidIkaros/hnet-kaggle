"""Build the speed-gate kernel: inline src/model.py + a config + kernel/scan_variants.py."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kernel_var")
os.makedirs(OUT, exist_ok=True)

model = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "model.py")).read(), flags=re.S)
driver = open(os.path.join(ROOT, "kernel", "scan_variants.py")).read()

cfg = {"d_model": 128, "n_enc": 2, "n_main": 2, "n_dec": 2, "d_ff": 512,
       "d_state": 16, "d_conv": 4, "scan_chunk": 64, "target_ratio": 6.0, "alpha": 0.03,
       "max_steps": 5000, "seq_len": 256, "batch": 16, "lr": 0.001, "warmup": 200,
       "wd": 0.1, "clip": 1.0, "seeds": [0], "arms": ["hnet"], "eval_every": 500,
       "log_every": 50, "wall_guard_s": 3600, "languages": ["eng", "rus"], "params_target": 840000.0,
       "note": "speed gate on the T4: bit-exactness of the optimised scan, step-time benchmark, and the fp16 loss match"}

head = ("import json\nimport math\nimport os\nimport time\n\n"
        "import numpy as np\nimport torch\nimport torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        'open("/tmp/cfg.json", "w").write(' + json.dumps(json.dumps(cfg)) + ")\n")
out = head + model + "\n" + driver
open(os.path.join(OUT, "main.py"), "w").write(out)

json.dump({"id": "mauricew/hnet-scan-variants", "title": "hnet scan variants",
           "code_file": "main.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": True, "enable_tpu": False,
           "enable_internet": False, "keywords": [],
           "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=2)
print("wrote", OUT, len(out), "bytes")
