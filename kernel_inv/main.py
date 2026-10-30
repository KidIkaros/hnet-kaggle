import json
open("/tmp/cfg.json", "w").write("{\"d_model\": 128, \"n_enc\": 2, \"n_main\": 2, \"n_dec\": 2, \"d_ff\": 512, \"d_state\": 16, \"d_conv\": 4, \"scan_chunk\": 64, \"target_ratio\": 6.0, \"alpha\": 0.03, \"max_steps\": 5000, \"seq_len\": 256, \"batch\": 16, \"lr\": 0.001, \"warmup\": 200, \"wd\": 0.1, \"clip\": 1.0, \"seeds\": [0, 1], \"arms\": [\"hnet\"], \"eval_every\": 250, \"log_every\": 25, \"wall_guard_s\": 28800, \"languages\": [\"eng\", \"rus\", \"cmn\", \"jpn\", \"arb\"], \"note\": \"encoding invariance: train on UTF-16LE bytes, compare boundary positions in character space against the ten UTF-8 checkpoints\"}")
"""Encoding invariance: does the chunker learn characters, or this encoding?

The pre-registered claim (ticket 02) has two stages. Stage one (alignment above chance)
is measured. Stage two -- that the alignment survives re-encoding to UTF-16LE -- is not.

Method (ticket 11's amendments): the same text is presented to two models in two
encodings. Boundaries are compared in *character space*, not byte space: a UTF-8
boundary at byte t maps to the character index that byte starts; a UTF-16LE boundary
at byte t' maps to t' // 2, since UTF-16LE puts every BMP character at an even offset.

Reported per language:
  u8_lift / u16_lift -- alignment lift in each encoding (even-offset base rate for UTF-16)
  agreement          -- fraction of the union of boundary characters that both models hit
  char_f1            -- F1 between the two boundary sets
  bpc_u16            -- realised bytes/chunk of the UTF-16 model (2 bytes per character)

Passing means the UTF-16 model aligns on characters AND lands on the same characters as
the UTF-8 model. Failing means the chunker learned this encoding's byte phase.
"""
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "/tmp/cfg.json")))
LANGS = CFG["languages"]
DATA = os.environ.get("DATA")
if not DATA:
    for root in ("/kaggle/input/datasets/mauricew/hnet-multiscript-bytes/corpus",
                 "/kaggle/input/hnet-multiscript-bytes/corpus", "corpus"):
        if os.path.isdir(root):
            DATA = root
            break
CKPT_DIR = os.environ.get("CKPT_DIR", "/kaggle/input/datasets/mauricew/hnet-multiscript-bytes/checkpoints_utf8")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
torch.set_num_threads(max(1, os.cpu_count() or 1))
SEQ = CFG["seq_len"]
# training reads the UTF-16LE variant; evaluation reads the UTF-8 val text so
# both models are compared on identical characters
TRAIN_SUFFIX = CFG.get("train_suffix", "_u16")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)



from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  SSM scan (from mbdfa)
# --------------------------------------------------------------------------- #
def _shift_right(x, d, fill):
    pad = x.new_full((x.shape[0], d) + x.shape[2:], fill)
    return torch.cat([pad, x[:, : x.shape[1] - d]], dim=1)


def chunked_scan(a, b, chunk):
    """h_t = a_t * h_{t-1} + b_t, chunk-wise Hillis-Steele. a,b: (B,L,...) -> h."""
    state, outs = None, []
    for i in range(0, a.shape[1], chunk):
        ac = a[:, i : i + chunk]
        bc = b[:, i : i + chunk]
        d = 1
        while d < ac.shape[1]:
            bc = bc + ac * _shift_right(bc, d, 0.0)
            ac = ac * _shift_right(ac, d, 1.0)
            d *= 2
        if state is not None:
            bc = bc + ac * state.unsqueeze(1)
        state = bc[:, -1]
        outs.append(bc)
    return torch.cat(outs, dim=1)


def sequential_scan(a, b):
    """Naive reference, for the exactness gate."""
    h, outs = None, []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t] if h is not None else b[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


# --------------------------------------------------------------------------- #
#  blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SSMBlock(nn.Module):
    """Selective byte SSM (mbdfa lineage, Mamba-1 shaped).

    B, C and delta are all functions of the input, so the block can learn which
    positions to write into state and which to read out.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, chunk=64, dt_rank=None):
        super().__init__()
        E = d_model
        self.E, self.N, self.chunk = E, d_state, chunk
        self.dt_rank = dt_rank or max(1, E // 16)
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * E, bias=False)
        self.conv = nn.Conv1d(E, E, d_conv, groups=E, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(E, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, E, bias=True)
        with torch.no_grad():
            nn.init.uniform_(self.dt_proj.weight, -(self.dt_rank ** -0.5), self.dt_rank ** -0.5)
            d0 = torch.exp(
                torch.rand(E) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            self.dt_proj.bias.copy_(d0 + torch.log(-torch.expm1(-d0)))
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().repeat(E, 1))
        )
        self.D = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(self.norm(x))
        x, z = xz.chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x = F.silu(x)

        delta, Bc, Cc = torch.split(
            self.x_proj(x), [self.dt_rank, self.N, self.N], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))                    # (B,L,E)

        A = -torch.exp(self.A_log.float())                         # (E,N)
        a = torch.exp(delta.unsqueeze(-1) * A)                     # (B,L,E,N)
        b = delta.unsqueeze(-1) * Bc.unsqueeze(2) * x.unsqueeze(-1)

        h = chunked_scan(a, b, self.chunk)
        y = (h * Cc.unsqueeze(2)).sum(-1) + self.D.float() * x
        return self.out_proj(y * F.silu(z))


class CausalAttn(nn.Module):
    """Causal self-attention with pre-norm. Main network mixer."""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        H = self.n_heads
        shp = lambda t: t.view(B, L, H, D // H).transpose(1, 2)
        y = F.scaled_dot_product_attention(shp(q), shp(k), shp(v), attn_mask=attn_mask)
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.out = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        a, g = self.fc(x).chunk(2, dim=-1)
        return self.out(F.silu(g) * a)


class Block(nn.Module):
    def __init__(self, mixer, d_model):
        super().__init__()
        self.mixer, self.mlp = mixer, None

    def forward(self, x, attn_mask=None):
        return x + self.mixer(x, attn_mask)


class AttnMLPBlock(Block):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__(CausalAttn(d_model, n_heads), d_model)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x, attn_mask=None):
        x = x + self.mixer(x, attn_mask)
        return x + self.mlp(self.norm2(x))


# --------------------------------------------------------------------------- #
#  routing / chunk / dechunk
# --------------------------------------------------------------------------- #
class Router(nn.Module):
    """p_t = (1 - cos(W_q x_t, W_k x_{t-1})) / 2 ; p_1 = 1. Identity-init."""

    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d_model))
            self.k_proj.weight.copy_(torch.eye(d_model))
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

    def forward(self, x):
        q = F.normalize(self.q_proj(x[:, :-1]), dim=-1)
        k = F.normalize(self.k_proj(x[:, 1:]), dim=-1)
        p = ((1.0 - (q * k).sum(-1)) / 2.0).clamp(0.0, 1.0)
        p = F.pad(p, (1, 0), value=1.0)
        return p, p >= 0.5


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x + (1.0 - x).detach()

    @staticmethod
    def backward(ctx, g):
        return g


class ChunkLayer(nn.Module):
    """Encoder output -> chunk vectors (B,M,D) plus scatter/gather indices."""

    def __init__(self, d_model):
        super().__init__()
        self.residual_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.residual_proj.weight)
        self.residual_proj.weight._no_reinit = True

    def forward(self, x, bmask):
        B, L, D = x.shape
        M = max(int(bmask.sum(dim=1).max().item()), 1)
        key = torch.arange(L, device=x.device).unsqueeze(0) + (~bmask).long() * L
        order = torch.argsort(key, dim=1)[:, :M]                   # (B,M)
        valid = torch.arange(M, device=x.device).unsqueeze(0) < bmask.sum(dim=1, keepdim=True)
        xc = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, D)) * valid.unsqueeze(-1)
        return xc, order, valid


class DeChunkLayer(nn.Module):
    """EMA over main-network outputs, smeared back to byte resolution, x STE(c)."""

    def __init__(self, chunk_scan=64):
        super().__init__()
        self.chunk_scan = chunk_scan

    def forward(self, h, bmask, p, order, valid):
        B, M, D = h.shape
        p_sel = torch.where(
            valid,
            torch.gather(p, 1, order),
            torch.ones((B, M), device=h.device, dtype=p.dtype),
        ).clamp(1e-4, 1 - 1e-4)

        a = (1.0 - p_sel).unsqueeze(-1).expand_as(h)               # decay = 1-p
        z_bar = chunked_scan(a, p_sel.unsqueeze(-1) * h, self.chunk_scan)

        chunk_id = (torch.cumsum(bmask.long(), dim=1) - 1).clamp(min=0)   # (B,L)
        dec = torch.gather(z_bar, 1, chunk_id.unsqueeze(-1).expand(-1, -1, D))
        c = torch.where(bmask, p, 1.0 - p)                          # confidence
        gate = c + (1.0 - c).detach()                               # STE
        return dec * gate.unsqueeze(-1)


# --------------------------------------------------------------------------- #
#  model
# --------------------------------------------------------------------------- #
@dataclass
class HNetConfig:
    d_model: int = 256
    n_enc: int = 4
    n_dec: int = 4
    n_main: int = 6
    d_state: int = 16
    d_conv: int = 4
    n_heads: int = 4
    d_ff: int = None
    target_ratio: float = 6.0
    alpha: float = 0.03
    vocab: int = 256
    scan_chunk: int = 64
    tie_embeddings: bool = True


class HNetLM(nn.Module):
    def __init__(self, cfg: HNetConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.emb = nn.Embedding(cfg.vocab, D)
        self.encoder = nn.ModuleList(
            [SSMBlock(D, cfg.d_state, cfg.d_conv, cfg.scan_chunk) for _ in range(cfg.n_enc)]
        )
        self.router = Router(D)
        self.chunk_layer = ChunkLayer(D)
        self.main = nn.ModuleList(
            [AttnMLPBlock(D, cfg.n_heads, cfg.d_ff) if cfg.d_ff else Block(CausalAttn(D, cfg.n_heads), D)
             for _ in range(cfg.n_main)]
        )
        self.main_norm = RMSNorm(D)
        self.dechunk_layer = DeChunkLayer(cfg.scan_chunk)
        self.decoder = nn.ModuleList(
            [SSMBlock(D, cfg.d_state, cfg.d_conv, cfg.scan_chunk) for _ in range(cfg.n_dec)]
        )
        self.norm_f = RMSNorm(D)
        self.head = nn.Linear(D, cfg.vocab, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight

    def forward(self, idx):
        B, L = idx.shape
        x = self.emb(idx)
        for blk in self.encoder:
            x = x + blk(x)

        p, bmask = self.router(x)
        residual = self.chunk_layer.residual_proj(x)               # (B,L,D) fp32
        xc, order, valid = self.chunk_layer(x, bmask)

        M = xc.shape[1]
        slot = torch.arange(M, device=idx.device).unsqueeze(0)
        allowed = (slot.unsqueeze(-1) >= slot.unsqueeze(-2)) \
            & valid.unsqueeze(-1) & valid.unsqueeze(-2)            # (B,M,M)
        attn_mask = torch.zeros(allowed.shape, device=idx.device, dtype=xc.dtype).masked_fill_(
            ~allowed, torch.finfo(xc.dtype).min
        ).unsqueeze(1)                                             # (B,1,M,M)

        h = xc
        for blk in self.main:
            h = blk(h, attn_mask)
        h = self.main_norm(h)

        dec = self.dechunk_layer(h, bmask, p, order, valid) + residual

        y = dec
        for blk in self.decoder:
            y = y + blk(y)
        logits = self.head(self.norm_f(y))

        N = self.cfg.target_ratio
        Fs, G = bmask.float().mean(dim=1), p.mean(dim=1)
        ratio_loss = ((N / (N - 1)) * ((N - 1) * Fs * G + (1 - Fs) * (1 - G))).mean()

        aux = {
            "ratio_loss": ratio_loss,
            "F": Fs.mean().item(),
            "G": G.mean().item(),
            "bmask": bmask,
            "p": p,
            # bytes of input per chunk, averaged over sequences -- dividing by the
            # whole batch's boundary count under-reports by B.
            "bytes_per_chunk": L * B / max(bmask.sum().item(), 1.0),
        }
        return logits, aux


def init_weights(model, initializer_range=0.02):
    """N(0, .02); out_proj/out scaled by 1/sqrt(n_residuals); router untouched."""
    n_res = sum(1 for m in model.modules() if isinstance(m, (SSMBlock, Block))) + \
            sum(1 for m in model.modules() if getattr(m, "mlp", None) is not None)
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                std = initializer_range
                if name.endswith(("out_proj", "out")):
                    std /= math.sqrt(max(n_res, 1))
                nn.init.normal_(m.weight, std=std)

class Stream:
    def __init__(self, rng, batch, seq_len):
        self.rng, self.batch, self.seq_len = rng, batch, seq_len
        self.data = [np.memmap(os.path.join(DATA, l + TRAIN_SUFFIX, "train.bin"),
                                 np.uint8, mode="r") for l in LANGS]
        self.lens = [len(d) for d in self.data]

    def batch_bytes(self):
        out = np.empty((self.batch, self.seq_len + 1), dtype=np.int64)
        for b in range(self.batch):
            i = int(self.rng.integers(len(self.data)))
            off = int(self.rng.integers(max(1, self.lens[i] - self.seq_len - 1)))
            out[b] = np.frombuffer(self.data[i][off:off + self.seq_len + 1], np.uint8)
        return out


def char_index_utf8(raw_bytes, nbytes):
    """For each byte offset, the character index it belongs to (start of its char)."""
    starts = [0]
    for i in range(nbytes):
        if (raw_bytes[i] & 0xC0) != 0x80:
            starts.append(i + 1)
    idx = np.zeros(nbytes, dtype=np.int64)
    c = 0
    for i in range(nbytes):
        while c + 1 < len(starts) and starts[c + 1] <= i:
            c += 1
        idx[i] = c
    return idx


def boundaries_to_chars(bmask, raw_bytes, utf16):
    """Boundary byte positions -> character indices."""
    if utf16:
        pos = np.flatnonzero(bmask)
        return set((pos // 2).tolist())
    idx = char_index_utf8(raw_bytes, len(raw_bytes))
    pos = np.flatnonzero(bmask)
    return set(idx[pos].tolist())


def evaluate_pair(u8_model, u16_model, lang, windows=12, chars=96):
    """Run both models on the same text and compare boundaries in character space."""
    text = open(os.path.join(DATA, lang, "val.bin"), "rb").read().decode("utf-8", "replace")
    step = max(1, (len(text) - chars) // max(windows, 1))
    u8_lift, u16_lift, agree, f1s, u8_sets, u16_sets = [], [], [], [], [], []
    for w in range(windows):
        s = w * step
        piece = text[s:s + chars]
        if len(piece) < 16:
            continue
        b8 = piece.encode("utf-8")
        b16 = piece.encode("utf-16-le")
        # shrink until BOTH encodings fit the sequence length: UTF-16 doubles the
        # bytes and UTF-8 can triple them (CJK), so one window size cannot suit both
        while (len(b8) > SEQ - 1 or len(b16) > SEQ - 1) and len(piece) > 8:
            piece = piece[: len(piece) // 2]
            b8 = piece.encode("utf-8")
            b16 = piece.encode("utf-16-le")
        if len(b8) > SEQ - 1 or len(b16) > SEQ - 1:
            continue
        with torch.no_grad():
            i8 = torch.from_numpy(np.frombuffer(b8, np.uint8).astype(np.int64))[None]
            _, a8 = u8_model(i8)
            i16 = torch.from_numpy(np.frombuffer(b16, np.uint8).astype(np.int64))[None]
            _, a16 = u16_model(i16)
        m8 = a8["bmask"][0].numpy()
        m16 = a16["bmask"][0].numpy()
        # UTF-8 lift: boundary on a lead byte
        lead8 = (np.frombuffer(b8, np.uint8) & 0xC0) != 0x80
        u8_lift.append(lead8[m8].mean() - lead8.mean() if m8.any() else 0.0)
        # UTF-16 lift: boundary on an even byte offset
        even = np.arange(len(b16)) % 2 == 0
        u16_lift.append(even[m16].mean() - even.mean() if m16.any() else 0.0)
        s8 = boundaries_to_chars(m8, b8, False)
        s16 = boundaries_to_chars(m16, b16, True)
        u8_sets.append(s8)
        u16_sets.append(s16)
        if s8 or s16:
            inter = len(s8 & s16)
            union = len(s8 | s16)
            agree.append(inter / union if union else 0.0)
            p, r = (inter / len(s8) if s8 else 0.0), (inter / len(s16) if s16 else 0.0)
            f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return {"u8_lift": float(np.mean(u8_lift)) if u8_lift else None,
            "u16_lift": float(np.mean(u16_lift)) if u16_lift else None,
            "agreement": float(np.mean(agree)) if agree else None,
            "char_f1": float(np.mean(f1s)) if f1s else None,
            "windows": len(agree),
            "u8_boundaries": float(np.mean([len(s) for s in u8_sets])) if u8_sets else None,
            "u16_boundaries": float(np.mean([len(s) for s in u16_sets])) if u16_sets else None}


def train_u16(seed, deadline):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    fields["d_state"] = 16
    cfg = HNetConfig(**fields)
    model = HNetLM(cfg)
    init_weights(model)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"],
                            betas=(0.9, 0.95))
    stream = Stream(np.random.default_rng(seed), CFG["batch"], CFG["seq_len"])
    hist = []
    t0 = time.time()
    for step in range(CFG["max_steps"]):
        batch = torch.from_numpy(stream.batch_bytes())
        logits, aux = model(batch[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
        loss = ce + CFG["alpha"] * aux["ratio_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        if step % CFG["log_every"] == 0 or step == CFG["max_steps"] - 1:
            hist.append({"step": step, "ce": ce.item(), "bpc": aux["bytes_per_chunk"]})
            log(f"s{seed} step {step} ce {ce.item():.4f} bpc {aux['bytes_per_chunk']:.2f}")
        if time.time() > deadline:
            log("wall guard at step", step)
            break
    return model, hist


def main():
    log("config", json.dumps(CFG))
    log("data", DATA, "ckpts", CKPT_DIR)
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt-s*-n4.pt")))
    log(f"{len(ckpts)} UTF-8 checkpoints")
    assert ckpts, "no UTF-8 checkpoints found"
    out = []
    for seed in CFG["seeds"]:
        deadline = time.time() + CFG.get("wall_guard_s", 8 * 3600) / len(CFG["seeds"])
        u16_model, hist = train_u16(seed, deadline)
        torch.save({"config": dict(u16_model.cfg.__dict__), "seed": seed,
                    "state": u16_model.state_dict()},
                   os.path.join(OUT, f"ckpt-u16-s{seed}.pt"))
        for ck in ckpts:
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            u8_model = HNetLM(HNetConfig(**blob["config"]))
            u8_model.load_state_dict(blob["state"])
            u8_model.eval()
            u16_model.eval()
            res = {"u8_ckpt": os.path.basename(ck), "u16_seed": seed}
            for lang in LANGS:
                res[lang] = evaluate_pair(u8_model, u16_model, lang)
                log(f"  s{seed} {os.path.basename(ck)} {lang}: "
                    f"u8_lift {res[lang]['u8_lift']} u16_lift {res[lang]['u16_lift']} "
                    f"agree {res[lang]['agreement']} f1 {res[lang]['char_f1']}")
            out.append(res)
        json.dump(out, open(os.path.join(OUT, "invariance.json"), "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
