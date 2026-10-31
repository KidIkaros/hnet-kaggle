import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

open("/tmp/cfg.json", "w").write("{\"d_model\": 128, \"n_enc\": 2, \"n_main\": 2, \"n_dec\": 2, \"d_ff\": 512, \"d_state\": 16, \"d_conv\": 4, \"scan_chunk\": 64, \"target_ratio\": 6.0, \"alpha\": 0.03, \"max_steps\": 5000, \"seq_len\": 256, \"batch\": 16, \"lr\": 0.001, \"warmup\": 200, \"wd\": 0.1, \"clip\": 1.0, \"seeds\": [0], \"arms\": [\"hnet\"], \"eval_every\": 500, \"log_every\": 50, \"wall_guard_s\": 3600, \"languages\": [\"eng\", \"rus\"], \"params_target\": 840000.0, \"note\": \"speed gate on the T4: bit-exactness of the optimised scan, step-time benchmark, and the fp16 loss match\"}")

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


def _shift_right_pad(x, d, fill):
    """Same result as _shift_right, one kernel instead of a fill plus a cat.

    F.pad writes into a fresh tensor in a single pass; the original built a padded tensor and
    then concatenated two of them, which allocates and copies twice. The values are identical,
    so this is bit-exact by construction -- the gate checks that anyway.
    """
    L = x.shape[1]
    y = x[:, : L - d]
    pad = [0] * (2 * (x.dim() - 1))
    pad[2 * (x.dim() - 2)] = d          # dim 1, counting back from the last
    return F.pad(y, pad, value=fill)


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


def _hillis_steele(ac, bc):
    """One chunk's intra-chunk scan, in place on the caller's tensors."""
    d = 1
    while d < ac.shape[1]:
        bc = bc + ac * _shift_right_pad(bc, d, 0.0)
        ac = ac * _shift_right_pad(ac, d, 1.0)
        d *= 2
    return ac, bc


def chunked_scan_fast(a, b, chunk):
    """The same recurrence as chunked_scan, with fewer allocations.

    Three changes, none of which touches the arithmetic: the shift is a single padded copy
    instead of a fill plus a concatenation; the per-chunk outputs are written into a
    preallocated buffer instead of being collected and concatenated at the end; and the
    intra-chunk scan is a separate function so it can be compiled independently of the
    chunk loop, whose chunk count varies with the batch's boundary count.
    """
    B, L = a.shape[0], a.shape[1]
    out = torch.empty_like(b)
    state = None
    for i in range(0, L, chunk):
        ac = a[:, i: i + chunk]
        bc = b[:, i: i + chunk]
        ac, bc = _hillis_steele(ac, bc)
        if state is not None:
            bc = bc + ac * state.unsqueeze(1)
        state = bc[:, -1]
        out[:, i: i + bc.shape[1]] = bc
    return out


def sequential_scan(a, b):
    """Naive reference, for the exactness gate."""
    h, outs = None, []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t] if h is not None else b[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


SCAN_IMPL = "fast"          # "reference" | "fast" | "compiled"
_COMPILED_SCAN = None


def scan(a, b, chunk):
    """Dispatch to the selected scan implementation."""
    if SCAN_IMPL == "reference":
        return chunked_scan(a, b, chunk)
    if SCAN_IMPL == "compiled":
        return chunked_scan_compiled(a, b, chunk)
    return chunked_scan_fast(a, b, chunk)


def chunked_scan_compiled(a, b, chunk):
    """chunked_scan_fast with the intra-chunk scan compiled.

    The compiled region is the intra-chunk scan only: the chunk loop is left outside it
    because the number of chunks varies with the batch's boundary count, and a varying
    outer dimension would trigger a recompilation on every step.
    """
    global _COMPILED_SCAN
    if _COMPILED_SCAN is None:
        _COMPILED_SCAN = torch.compile(_hillis_steele, dynamic=False)
    B, L = a.shape[0], a.shape[1]
    out = torch.empty_like(b)
    state = None
    for i in range(0, L, chunk):
        ac = a[:, i: i + chunk]
        bc = b[:, i: i + chunk]
        ac, bc = _COMPILED_SCAN(ac, bc)
        if state is not None:
            bc = bc + ac * state.unsqueeze(1)
        state = bc[:, -1]
        out[:, i: i + bc.shape[1]] = bc
    return out


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

        # The recurrence runs in fp32 whatever the surrounding precision is. The paper's own
        # stability notes say the scan needs fp32 accumulation or it drifts, and on a T4
        # there is no bf16 -- fp16 autocast around the matmuls with an fp32 scan is the
        # prescribed arrangement. In an fp32 run this block is a no-op, so the fp32 path
        # stays bit-identical.
        with torch.autocast(device_type=x.device.type, enabled=False):
            A = -torch.exp(self.A_log.float())                     # (E,N)
            delta = delta.float()
            a = torch.exp(delta.unsqueeze(-1) * A)                 # (B,L,E,N)
            b = delta.unsqueeze(-1) * Bc.unsqueeze(2).float() * x.unsqueeze(-1).float()

            h = scan(a, b, self.chunk)
        y = (h * Cc.unsqueeze(2).float()).sum(-1) + self.D.float() * x.float()
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


def param_groups(model, mult_outer=1.0, mult_main=1.0):
    """Per-stage learning-rate multipliers (paper Sec. 2.3: lambda_0=2.0, lambda_1=1.5).

    The paper applies a multiplier to every parameter of a stage, and the official code applies
    the main network's multiplier again on top. The outer networks (encoder, decoder, router,
    chunk layer) are stage 0; the main network is stage 1. Our implementation had no LR
    modulation at all, which is one of the four stability tricks it lists as load-bearing, and
    the replication enables it for S>0 while disabling it for S=0.

    Returns [(params, mult), ...] for the optimizer.
    """
    outer, main = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (main if name.startswith("main.") else outer).append(p)
    groups = []
    if outer:
        groups.append({"params": outer, "lr_mult": mult_outer, "name": "outer"})
    if main:
        # The official code applies the OUTER stage's multiplier to every parameter of that
        # stage -- including the main network, which is a submodule of it -- and then applies
        # the main network's own multiplier on top. Its own comment says so. So the main
        # network's effective multiplier is the product, not the bare second number.
        groups.append({"params": main, "lr_mult": mult_outer * mult_main, "name": "main"})
    return groups


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
            "Fs": Fs,
            "Gs": G,
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

# ==== SPEED GATE (GPU) ====
# The scan optimisation has to be gated on the device it will run on: inductor generates
# different kernels for CUDA than for CPU, so bit-exactness measured on CPU does not carry.
# This kernel runs, on the T4:
#   1. the bit-exactness gate (reference vs optimised scan, production shapes)
#   2. a full-model step-time benchmark: reference vs optimised
#   3. the fp16 autocast loss match, which is a PRECISION change and is gated separately
#      from the algorithm change -- the paper prescribes amp fp16 with fp32 master weights
#      on a T4, where bf16 does not exist.

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "/tmp/cfg.json")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def set_scan_impl(name):
    """The kernel inlines the model, so the scan's global is right here."""
    global SCAN_IMPL
    SCAN_IMPL = name
    log(f"scan implementation -> {name}")


def make(B=16, L=256, E=128, N=16, seed=0, device=DEV):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(B, L, E, N, generator=g) * 0.5 + 0.5
    b = torch.randn(B, L, E, N, generator=g)
    return a.to(device), b.to(device)


def timeit(fn, a, b, chunk, n=30):
    fn(a, b, chunk)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(a, b, chunk)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    log("device", DEV, "torch", torch.__version__)
    res = {}
    a, b = make()
    chunk = 64

    ref = chunked_scan(a, b, chunk)
    fast = chunked_scan_fast(a, b, chunk)
    same_fast = torch.equal(ref, fast)
    log(f"reference vs fast: bit-exact {same_fast}  max abs diff "
        f"{(ref - fast).abs().max().item():.3e}")
    res["fast_bit_exact"] = bool(same_fast)

    comp = chunked_scan_compiled(a, b, chunk)
    same_comp = torch.equal(ref, comp)
    log(f"reference vs compiled: bit-exact {same_comp}  max abs diff "
        f"{(ref - comp).abs().max().item():.3e}")
    res["compiled_bit_exact"] = bool(same_comp)

    t_ref = timeit(chunked_scan, a, b, chunk)
    t_fast = timeit(chunked_scan_fast, a, b, chunk)
    t_comp = timeit(chunked_scan_compiled, a, b, chunk)
    log(f"scan forward: reference {t_ref*1000:.2f} ms  fast {t_fast*1000:.2f} ms "
        f"({t_ref/t_fast:.2f}x)  compiled {t_comp*1000:.2f} ms ({t_ref/t_comp:.2f}x)")
    res["scan_ms"] = {"reference": t_ref*1000, "fast": t_fast*1000, "compiled": t_comp*1000}

    # full-model step time, which is what the rung actually pays
    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    cfg = HNetConfig(**fields)
    torch.manual_seed(0)
    model = HNetLM(cfg).to(DEV)
    init_weights(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (16, 256), device=DEV)

    def step_time(n=15):
        for _ in range(3):
            logits, aux = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits, aux = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    set_scan_impl("reference")
    t_model_ref = step_time()
    set_scan_impl("fast")
    t_model_fast = step_time()
    set_scan_impl("compiled")
    t_model_comp = step_time()
    log(f"full step: reference {t_model_ref*1000:.1f} ms  fast {t_model_fast*1000:.1f} ms "
        f"({t_model_ref/t_model_fast:.2f}x)  compiled {t_model_comp*1000:.1f} ms "
        f"({t_model_ref/t_model_comp:.2f}x)")
    res["step_ms"] = {"reference": t_model_ref*1000, "fast": t_model_fast*1000,
                      "compiled": t_model_comp*1000}

    # model-level equality in fp32: same weights, same input, reference vs the optimised scan
    torch.manual_seed(0)
    mm = HNetLM(cfg).to(DEV)
    init_weights(mm)
    xx = torch.randint(0, 256, (16, 256), device=DEV)
    set_scan_impl("reference")
    with torch.no_grad():
        l_ref, _ = mm(xx)
    set_scan_impl("fast")
    with torch.no_grad():
        l_fast, _ = mm(xx)
    model_same = torch.equal(l_ref, l_fast)
    log(f"model logits, reference vs fast scan (fp32): identical {model_same}  max abs diff "
        f"{(l_ref - l_fast).abs().max().item():.3e}")
    res["model_bit_exact_fp32"] = bool(model_same)
    set_scan_impl("compiled")

    # the combined configuration: fp16 autocast with fp32 master weights, over the compiled scan
    torch.manual_seed(0)
    mc = HNetLM(cfg).to(DEV)
    init_weights(mc)
    oc = torch.optim.AdamW(mc.parameters(), lr=1e-3)
    sc = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))

    def step_time_amp(n=15):
        for _ in range(3):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                logits, aux = mc(xx)
                loss = F.cross_entropy(logits.reshape(-1, 256), xx.reshape(-1))
            oc.zero_grad(set_to_none=True)
            sc.scale(loss).backward()
            sc.step(oc)
            sc.update()
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                logits, aux = mc(xx)
                loss = F.cross_entropy(logits.reshape(-1, 256), xx.reshape(-1))
            oc.zero_grad(set_to_none=True)
            sc.scale(loss).backward()
            sc.step(oc)
            sc.update()
        if DEV == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    t_amp = step_time_amp()
    log(f"full step, fp16 + compiled scan: {t_amp*1000:.1f} ms  "
        f"({t_model_ref/t_amp:.2f}x vs the fp32 reference)")
    res["step_ms_fp16_compiled"] = t_amp * 1000
    res["combined_speedup"] = float(t_model_ref / t_amp)

    # fp16 autocast with fp32 master weights: a PRECISION change, gated on a loss match
    set_scan_impl("compiled")
    torch.manual_seed(0)
    m32 = HNetLM(cfg).to(DEV)
    init_weights(m32)
    o32 = torch.optim.AdamW(m32.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    xb = torch.randint(0, 256, (16, 256), device=DEV)
    l32, l16 = [], []
    for step in range(60):
        logits, aux = m32(xb)
        loss32 = F.cross_entropy(logits.reshape(-1, 256), xb.reshape(-1))
        o32.zero_grad(set_to_none=True)
        loss32.backward()
        o32.step()
        l32.append(loss32.item())
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            logits, aux = m32(xb)
            loss16 = F.cross_entropy(logits.reshape(-1, 256), xb.reshape(-1))
        scaler.scale(loss16).backward()
        scaler.step(o32)
        scaler.update()
        l16.append(loss16.item())
    d = abs(np.mean(l32[-10:]) - np.mean(l16[-10:]))
    log(f"fp32 loss {np.mean(l32[-10:]):.4f}  fp16-autocast loss {np.mean(l16[-10:]):.4f}  "
        f"diff {d:.4f}")
    res["fp16_loss"] = {"fp32": float(np.mean(l32[-10:])), "fp16": float(np.mean(l16[-10:])),
                        "abs_diff": float(d)}
    res["fp16_accept"] = bool(d < 0.02)

    if DEV == "cuda":
        res["mem_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
        log(f"peak memory {res['mem_peak_mb']:.0f} MB")

    json.dump(res, open(os.path.join(OUT, "speed_gate.json"), "w"), indent=2)
    log("done", json.dumps({k: v for k, v in res.items() if k != "scan_ms"}))


if __name__ == "__main__":
    main()
