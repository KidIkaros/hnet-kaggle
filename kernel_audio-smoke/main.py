import sys
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

open("/tmp/audio.json", "w").write("{\"d_model\": 128, \"n_enc\": 2, \"n_main\": 2, \"n_dec\": 2, \"d_ff\": 512, \"d_state\": 16, \"d_conv\": 4, \"scan_chunk\": 64, \"target_ratio\": 6.0, \"alpha\": 0.03, \"max_steps\": 30, \"seq_len\": 512, \"batch\": 4, \"lr\": 0.001, \"warmup\": 5, \"wd\": 0.1, \"clip\": 1.0, \"seeds\": [0], \"arms\": [\"pcm16le\", \"pcm16be\", \"mulaw\", \"f32le\"], \"eval_every\": 15, \"log_every\": 10, \"wall_guard_s\": 900, \"params_target\": 840000.0, \"note\": \"smoke: 30 steps, all four arms, validates the audio pipeline end to end on CPU\"}")

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

# ==== AUDIO DRIVER ====
# Four byte encodings of the same speech, one model each. The question is not "does it learn
# phonemes" -- it is the same question the text arm ended with: does the chunker follow the
# signal, or the layout of its serialization?
#
# Three measurements, in increasing strength:
#   1. ms_per_chunk vs bytes_per_chunk across arms. If the arms converge on a fixed *duration*
#      the chunker tracks time; if they converge on a fixed *byte count* it tracks layout.
#   2. onset alignment: do boundaries land on energy rises above the base rate?
#   3. LE vs BE: identical samples, identical byte count, every byte pair swapped. A chunker
#      following the signal puts boundaries at the same byte offsets in both. This needs no
#      semantic reference at all and is the sharpest test available.

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "/tmp/audio.json")))
DATA = os.environ.get("DATA")
for root in (() if DATA else ("/kaggle/input/datasets/mauricew/hnet-audio-bytes",
                              "/kaggle/input/hnet-audio-bytes",
                              "/kaggle/input/hnet-audio-bytes/corpus",
             "/kaggle/input/datasets/mauricew/hnet-audio-bytes/corpus", "corpus")):
    if os.path.isdir(root):
        DATA = root
        break
if not DATA:
    import glob
    print("INPUT:", glob.glob("/kaggle/input/*")[:20], glob.glob("/kaggle/input/datasets/*/*")[:30], flush=True)
    print("CWD:", os.getcwd(), sorted(os.listdir("."))[:20], flush=True)
assert DATA, "audio dataset not mounted"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["max_steps"] - cfg["warmup"])
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


META = json.load(open(os.path.join(DATA, "meta.json")))
SR = META["sample_rate"]
N = META["n_samples"]
ARMS = CFG["arms"]                      # e.g. ["pcm16le", "pcm16be", "mulaw", "f32le"]
BPS = {k: v["bytes_per_sample"] for k, v in META["encodings"].items()}
SEQ = {k: v["seq_len_for_32ms"] for k, v in META["encodings"].items()}

# val is the last 15% of the stream, taken in whole documents where possible
VAL_START = int(N * 0.85)
_onsets = np.fromfile(os.path.join(DATA, "onsets.u32"), dtype="<u4").astype(np.int64)
ONSETS = np.sort(_onsets[_onsets < N])


def load_arm(arm):
    raw = np.memmap(os.path.join(DATA, f"{arm}.bin"), np.uint8, mode="r")
    return raw


def sample_of(byte_pos, arm):
    return byte_pos // BPS[arm]


def windows(arm, n_windows, seq_len, start, end):
    """Byte windows that all begin on a sample boundary of this encoding."""
    bps = BPS[arm]
    span = end - start
    step = max(1, span // (n_windows + 1))
    out = []
    for i in range(n_windows):
        b = start + i * step
        b -= b % bps                      # align to a sample so offsets are comparable
        if b + seq_len + 1 > end:
            continue
        out.append(b)
    return out


def onset_lift(bmask, base_byte, arm, tol_ms=8):
    """Fraction of boundaries within tol of an onset, against the base rate of such positions."""
    bps = BPS[arm]
    tol = int(tol_ms * SR / 1000) * bps
    pos = np.flatnonzero(bmask) + base_byte
    if len(pos) == 0:
        return None
    idx = np.searchsorted(ONSETS, pos)
    lo = np.clip(idx - 1, 0, len(ONSETS) - 1)
    hi = np.clip(idx, 0, len(ONSETS) - 1)
    d = np.minimum(np.abs(pos - ONSETS[lo]), np.abs(ONSETS[hi] - pos))
    hit = float(np.mean(d <= tol))
    # base rate: same question asked of every sample-aligned position in the window
    allpos = np.arange(base_byte, base_byte + len(bmask), bps)
    i2 = np.searchsorted(ONSETS, allpos)
    l2 = np.clip(i2 - 1, 0, len(ONSETS) - 1)
    h2 = np.clip(i2, 0, len(ONSETS) - 1)
    d2 = np.minimum(np.abs(allpos - ONSETS[l2]), np.abs(ONSETS[h2] - allpos))
    base = float(np.mean(d2 <= tol))
    return {"hit": hit, "base": base, "lift": hit - base}


def phase_concentration(bmask):
    """Boundary concentration by offset mod 2: 1.0 = no preference."""
    pos = np.flatnonzero(bmask)
    if len(pos) < 4:
        return None
    ev = float(np.mean(pos % 2 == 0))
    return {"even_share": ev, "concentration": ev / 0.5}


def evaluate(model, arm, seq_len, n_windows=10):
    raw = load_arm(arm)
    model.eval()
    rows = []
    with torch.no_grad():
        for b in windows(arm, n_windows, seq_len, VAL_START, N):
            arr = np.frombuffer(raw[b:b + seq_len + 1], np.uint8).astype(np.int64)
            x = torch.from_numpy(arr)[None, :].to(DEV)
            logits, aux = model(x[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1))
            bm = aux["bmask"][0].cpu().numpy()
            rows.append({"bpb": ce.item() / math.log(2),
                         "bytes_per_chunk": aux["bytes_per_chunk"],
                         "onset": onset_lift(bm, b, arm),
                         "phase": phase_concentration(bm),
                         "boundaries": bm})
    model.train()
    if not rows:
        return None
    ol = [r["onset"] for r in rows if r["onset"]]
    out = {"bpb": float(np.mean([r["bpb"] for r in rows])),
           "bytes_per_chunk": float(np.mean([r["bytes_per_chunk"] for r in rows])),
           "ms_per_chunk": float(np.mean([r["bytes_per_chunk"] for r in rows]))
                            / BPS[arm] / SR * 1000,
           "onset_lift": float(np.mean([o["lift"] for o in ol])) if ol else None,
           "onset_hit": float(np.mean([o["hit"] for o in ol])) if ol else None,
           "onset_base": float(np.mean([o["base"] for o in ol])) if ol else None,
           "even_share": float(np.mean([r["phase"]["even_share"] for r in rows
                                        if r["phase"]])) if any(r["phase"] for r in rows) else None}
    return out


def train_arm(arm, seed, deadline):
    seq_len = SEQ[arm]
    torch.manual_seed(seed)
    np.random.seed(seed)
    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    cfg = HNetConfig(**fields)
    model = HNetLM(cfg).to(DEV)
    init_weights(model)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"],
                            betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    raw = load_arm(arm)
    train_end = VAL_START
    history, evals = [], []
    t0, stop = time.time(), "max_steps"
    for step in range(CFG["max_steps"]):
        starts = rng.integers(0, max(1, train_end - seq_len - 2), size=CFG["batch"])
        starts -= starts % BPS[arm]
        b64 = np.stack([np.frombuffer(raw[s:s + seq_len + 1], np.uint8) for s in starts])
        batch = torch.from_numpy(b64.astype(np.int64)).to(DEV)
        logits, aux = model(batch[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
        loss = ce + CFG["alpha"] * aux["ratio_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, CFG)
        opt.step()
        if step % CFG["log_every"] == 0 or step == CFG["max_steps"] - 1:
            history.append({"step": step, "ce": ce.item(), "bpb": ce.item() / math.log(2),
                            "bytes_per_chunk": aux["bytes_per_chunk"],
                            "ms_per_chunk": aux["bytes_per_chunk"] / BPS[arm] / SR * 1000,
                            "grad_norm": float(gn), "elapsed_s": round(time.time() - t0, 1)})
            log(f"{arm} s{seed} step {step} ce {ce.item():.4f} bpc "
                f"{aux['bytes_per_chunk']:.2f} ({aux['bytes_per_chunk']/BPS[arm]/SR*1000:.1f} ms)")
        if step % CFG["eval_every"] == 0 and step:
            ev = evaluate(model, arm, seq_len)
            evals.append({"step": step, **ev})
            log(f"  eval {arm} bpb {ev['bpb']:.3f} bpc {ev['bytes_per_chunk']:.2f} "
                f"ms {ev['ms_per_chunk']:.1f} onset {ev['onset_lift'] or 0:+.3f}")
        if time.time() > deadline:
            stop = f"wall_guard@{step}"
            break
    torch.save({"config": dict(cfg.__dict__), "seed": seed, "arm": arm,
                "state": model.state_dict()}, os.path.join(OUT, f"ckpt-{arm}-s{seed}.pt"))
    final = evaluate(model, arm, seq_len, n_windows=40)
    json.dump({"arm": arm, "seed": seed, "stop": stop, "steps": step + 1,
               "final": final, "history": history, "evals": evals},
              open(os.path.join(OUT, f"arm-{arm}-s{seed}.json"), "w"), indent=2)
    return model, final


def cross_arm(models, seed):
    """LE vs BE at identical byte offsets -- the test that needs no semantic reference.

    Both streams have the same byte count and the same sample positions, so a chunker that
    follows the signal must place boundaries at the same offsets. Agreement is reported
    against the chance level implied by each model's own boundary rate.
    """
    out = {}
    if "pcm16le" not in models or "pcm16be" not in models:
        return out
    seq_len = min(SEQ["pcm16le"], SEQ["pcm16be"])
    raw = {a: load_arm(a) for a in ("pcm16le", "pcm16be")}
    wins = windows("pcm16le", 40, seq_len, VAL_START, N)
    agree, f1s, rates = [], [], []
    cross = []
    for b in wins:
        bm = {}
        for a in ("pcm16le", "pcm16be"):
            arr = np.frombuffer(raw[a][b:b + seq_len], np.uint8).astype(np.int64)
            x = torch.from_numpy(arr)[None, :].to(DEV)
            with torch.no_grad():
                _, aux = models[a](x)
            bm[a] = aux["bmask"][0].cpu().numpy()
        le, be = bm["pcm16le"], bm["pcm16be"]
        inter = float(np.sum(le & be))
        union = float(np.sum(le | be))
        agree.append(inter / union if union else 0.0)
        p, r = (inter / max(1.0, le.sum())), (inter / max(1.0, be.sum()))
        f1s.append(2 * p * r / max(1e-9, p + r))
        rates.append((float(le.mean()), float(be.mean())))
        # zero-shot: the LE model reading BE bytes, same offsets
        arr = np.frombuffer(raw["pcm16be"][b:b + seq_len], np.uint8).astype(np.int64)
        with torch.no_grad():
            _, aux2 = models["pcm16le"](torch.from_numpy(arr)[None, :].to(DEV))
        zs = aux2["bmask"][0].cpu().numpy()
        cross.append(float(np.sum(le & zs)) / max(1.0, float(np.sum(le | zs))))
    rate_le, rate_be = float(np.mean([r[0] for r in rates])), float(np.mean([r[1] for r in rates]))
    chance = rate_le * rate_be / max(1e-9, rate_le + rate_be - rate_le * rate_be)
    out = {"jaccard": float(np.mean(agree)), "f1": float(np.mean(f1s)),
           "chance_jaccard": chance, "rate_le": rate_le, "rate_be": rate_be,
           "zero_shot_le_on_be": float(np.mean(cross)), "n_windows": len(wins)}
    log(f"LE vs BE: jaccard {out['jaccard']:.3f} vs chance {chance:.3f} | f1 {out['f1']:.3f} "
        f"| LE model on BE bytes {out['zero_shot_le_on_be']:.3f}")
    return out


def main():
    log("device", DEV, "arms", ARMS, "sr", SR, "samples", N)
    t_start = time.time()
    models, summary = {}, []
    for i, arm in enumerate(ARMS):
        share = (CFG["wall_guard_s"] - (time.time() - t_start)) / (len(ARMS) - i)
        log(f"arm {i+1}/{len(ARMS)}: {arm} seq_len {SEQ[arm]}")
        m, final = train_arm(arm, CFG["seeds"][0], time.time() + share)
        models[arm] = m
        summary.append({"arm": arm, "final": final})
    res = {"config": CFG, "device": DEV, "per_arm": summary,
           "cross": cross_arm(models, CFG["seeds"][0])}
    json.dump(res, open(os.path.join(OUT, "audio.json"), "w"), indent=2)
    log("done")
    for s in summary:
        f = s["final"] or {}
        log(f"  {s['arm']:>8}: bpb {f.get('bpb', float('nan')):.3f} "
            f"bpc {f.get('bytes_per_chunk', float('nan')):.2f} "
            f"ms {f.get('ms_per_chunk', float('nan')):.1f} "
            f"onset {f.get('onset_lift') or 0:+.3f}")


if __name__ == "__main__":
    main()
