import os, json, urllib.request
from tokenizers import Tokenizer

CORPUS = "/home/ikaaros/.hermes/cache/scratch/ds_hnet/corpus"
TOKDIR = "/home/ikaaros/.hermes/cache/scratch/tok"
os.makedirs(TOKDIR, exist_ok=True)


def get_tok(name, repo, fname):
    p = os.path.join(TOKDIR, f"{name}.json")
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/{fname}", p)
    return Tokenizer.from_file(p)


langs = ["eng", "rus", "cmn", "arb", "jpn"]
out = {}
for label, repo, fname in [("gpt2", "openai-community/gpt2", "tokenizer.json")]:
    tok = get_tok(label, repo, fname)
    print(f"=== {label}")
    res = {}
    for lang in langs:
        raw = open(f"{CORPUS}/{lang}/train.bin", "rb").read(4_000_000).decode("utf-8", "replace")
        n = len(tok.encode(raw).ids)
        res[lang] = len(raw) / n
        print(f"  {lang:>4}: {n/1e6:.2f}M tokens  {res[lang]:.3f} bytes/token")
    ref = res["eng"]
    print("  relative to English (higher = the fixed tokenizer does worse here):")
    for l, v in res.items():
        print(f"    {l:>4}: {v/ref:.3f}")
    out[label] = res
json.dump(out, open(os.path.join(TOKDIR, "bytes_per_token.json"), "w"), indent=1)
