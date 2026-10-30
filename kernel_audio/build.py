"""Build the audio kernel: inline src/model.py + configs/audio.json + kernel/audio.py."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kernel_audio" + ("" if not sys.argv[2:3] else "-" + sys.argv[2]))
os.makedirs(OUT, exist_ok=True)

model = re.sub(r'^""".*?"""\n', "", open(os.path.join(ROOT, "src", "model.py")).read(), flags=re.S)
driver = open(os.path.join(ROOT, "kernel", "audio.py")).read()

import sys
rel = sys.argv[1] if len(sys.argv) > 1 else "configs/audio.json"
suffix = sys.argv[2] if len(sys.argv) > 2 else ""
gpu = (sys.argv[3] != "cpu") if len(sys.argv) > 3 else True
cfg = json.load(open(os.path.join(ROOT, rel)))

head = (
    "import sys\nimport json\nimport math\nimport os\nimport time\n\n"
    "import numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n"
    'open("/tmp/audio.json", "w").write(' + json.dumps(json.dumps(cfg)) + ")\n"
)

out = head + model + "\n" + driver
open(os.path.join(OUT, "main.py"), "w").write(out)

json.dump({"id": "mauricew/hnet-audio" + ("-" + suffix if suffix else ""),
           "title": "hnet audio" + (" " + suffix if suffix else ""),
           "code_file": "main.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
           "enable_internet": False, "keywords": ["audio", "chunking"],
           "dataset_sources": ["mauricew/hnet-audio-bytes"],
           "kernel_sources": [], "competition_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=2)
print("wrote", OUT, len(out), "bytes")
