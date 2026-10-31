"""In-corpus BPE tokenizers, one per language.

The efficiency claim is a comparison against a *fixed* tokenizer, so the baseline has to be
given a good one. Cartesia used a 128k in-corpus tokenizer over a 70/20/5/5 byte mixture and
reported 4.57 bytes/token. We train per language on the full training split of that language,
which is the standard in-corpus setup and gives the baseline the best tokenizer it can have
for the data it sees.

Byte-level BPE (the llama2 flavour): every merge operates on bytes, so any byte string is
encodable and nothing is ever unrepresentable.
"""
import json
import os

import numpy as np

CORPUS = "/home/ikaaros/.hermes/cache/scratch/ds_hnet/corpus"
OUT = "/home/ikaaros/.hermes/cache/scratch/bpe"
VOCAB = 16384
TRAIN_BYTES = 8 * 1024 * 1024   # 8 MB is plenty to fit a 16k vocab; 48 MB took 10+ min/language


def train_one(lang, vocab=VOCAB):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{lang}.json")
    if os.path.exists(path):
        return path
    raw = open(os.path.join(CORPUS, lang, "train.bin"), "rb").read(TRAIN_BYTES)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, show_progress=False,
                                  special_tokens=["<unk>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([raw.decode("utf-8", "ignore")], trainer=trainer)
    tok.save(path)
    return path


def encode_corpus(lang, tok_path, split="train"):
    """Encode the whole split to a flat uint32 id array, plus the doc offsets."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    raw = open(os.path.join(CORPUS, lang, f"{split}.bin"), "rb").read()
    text = raw.decode("utf-8", "replace")
    enc = tok.encode(text)
    ids = np.array(enc.ids, dtype=np.uint32)
    return ids


def main():
    langs = ["eng", "rus", "cmn", "arb", "jpn"]
    summary = {}
    for lang in langs:
        p = train_one(lang)
        ids = encode_corpus(lang, p, "train")
        nbytes = os.path.getsize(os.path.join(CORPUS, lang, "train.bin"))
        val_ids = encode_corpus(lang, p, "val")
        val_bytes = os.path.getsize(os.path.join(CORPUS, lang, "val.bin"))
        np.save(os.path.join(OUT, f"{lang}_val_ids.npy"), val_ids)
        bpt = nbytes / len(ids)
        # how many bytes per CHARACTER, which is where the script penalty shows up
        text_len = len(open(os.path.join(CORPUS, lang, "train.bin"), "rb").read(200_000_000).decode("utf-8", "replace"))
        summary[lang] = {"vocab": VOCAB, "n_tokens": int(len(ids)),
                         "bytes_per_token": round(bpt, 3),
                         "tokens_per_byte": round(1 / bpt, 4),
                         "n_bytes": int(nbytes), "n_val_tokens": int(len(val_ids)),
                         "n_val_bytes": int(val_bytes)}
        print(f"{lang:>4}: {len(ids)/1e6:.1f}M tokens  {bpt:.3f} bytes/token  "
              f"{summary[lang]['tokens_per_byte']:.4f} tokens/byte")
        np.save(os.path.join(OUT, f"{lang}_train_ids.npy"), ids)
    # english is the reference for "how much worse is this script for a fixed tokenizer"
    ref = summary["eng"]["tokens_per_byte"]
    print("\nrelative to English (higher = BPE does worse on this script):")
    for lang, s in summary.items():
        s["relative_to_english"] = round(s["tokens_per_byte"] / ref, 3)
        print(f"  {lang:>4}: {s['relative_to_english']:.3f}")
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
