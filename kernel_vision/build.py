"""Build the vision kernel: inline src/model.py + configs/vision*.json + kernel/vision.py."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rel = sys.argv[1] if len(sys.argv) > 1 else "configs/vision.json"
suffix = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = os.path.join(ROOT, "kernel_vision" + ("-" + suffix if suffix else ""))
os.makedirs(OUT, exist_ok=True)

model = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "model.py")).read(), flags=re.S)
driver = open(os.path.join(ROOT, "kernel", "vision.py")).read()
cfg = json.load(open(os.path.join(ROOT, rel)))

head = ("import sys\nimport json\nimport math\nimport os\nimport time\n\n"
        "import numpy as np\nimport torch\nimport torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        'open("/tmp/vision.json", "w").write(' + json.dumps(json.dumps(cfg)) + ")\n")
out = head + model + "\n" + driver
open(os.path.join(OUT, "main.py"), "w").write(out)

json.dump({"id": "mauricew/hnet-vision" + ("-" + suffix if suffix else ""),
           "title": "hnet vision" + (" " + suffix if suffix else ""),
           "code_file": "main.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": False, "enable_tpu": False,
           "enable_internet": False, "keywords": [],
           "dataset_sources": ["mauricew/hnet-vision-bytes"],
           "kernel_sources": [], "competition_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=2)
print("wrote", OUT, len(out), "bytes")
