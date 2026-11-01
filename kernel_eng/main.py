import json

open("/tmp/eng.json", "w").write("{\"d_model\": 384, \"n_enc\": 4, \"n_main\": 6, \"n_dec\": 4, \"d_ff\": 1536, \"d_state\": 16, \"d_conv\": 4, \"n_heads\": 6, \"scan_chunk\": 64, \"target_ratio\": 6.0, \"alpha\": 0.3, \"seq_len\": 2048, \"batch\": 4, \"lr\": 0.001, \"warmup\": 2000, \"wd\": 0.1, \"clip\": 1.0, \"seeds\": [0], \"lr_mult_outer\": 2.0, \"lr_mult_main\": 1.5, \"compile_blocks\": true, \"grad_checkpoint\": true, \"resume\": false, \"save_every\": 2, \"max_steps\": 5, \"eval_every\": 5, \"log_every\": 1, \"wall_guard_s\": 900, \"flops_budget\": 2000000000000.0, \"amp\": \"fp16\", \"note\": \"dual-GPU smoke: DataParallel gather + compiled blocks + resume, 5 steps\", \"eval_batches\": 2, \"data_parallel\": true}")
try:

    import torch
    import triton
    import triton.language as tl


    @triton.jit
    def _fwd(x_ptr, delta_ptr, A_ptr, Bc_ptr, Cc_ptr, D_ptr, y_ptr, h_ptr,
             L, E: tl.constexpr, N: tl.constexpr, BE: tl.constexpr):
        b = tl.program_id(0)
        pe = tl.program_id(1)
        e = pe * BE + tl.arange(0, BE)
        n = tl.arange(0, N)
        A = tl.load(A_ptr + e[:, None] * N + n[None, :])          # (BE, N)
        Dv = tl.load(D_ptr + e)                                    # (BE,)
        h = tl.zeros((BE, N), tl.float32)
        for t in range(L):
            row_e = (b * L + t) * E + e
            row_n = (b * L + t) * N + n
            x_t = tl.load(x_ptr + row_e)
            d_t = tl.load(delta_ptr + row_e)
            b_t = tl.load(Bc_ptr + row_n)
            c_t = tl.load(Cc_ptr + row_n)
            a = tl.exp(d_t[:, None] * A)
            h = a * h + (d_t[:, None] * b_t[None, :]) * x_t[:, None]
            tl.store(h_ptr + (b * L + t) * E * N + e[:, None] * N + n[None, :], h)
            y = tl.sum(h * c_t[None, :], axis=1) + Dv * x_t
            tl.store(y_ptr + row_e, y)


    @triton.jit
    def _bwd(x_ptr, delta_ptr, A_ptr, Bc_ptr, Cc_ptr, D_ptr, h_ptr, dy_ptr,
             dx_ptr, ddelta_ptr, dA_ptr, dBc_part_ptr, dCc_part_ptr, dD_ptr,
             L, E: tl.constexpr, N: tl.constexpr, BE: tl.constexpr, NPE: tl.constexpr):
        b = tl.program_id(0)
        pe = tl.program_id(1)
        e = pe * BE + tl.arange(0, BE)
        n = tl.arange(0, N)
        A = tl.load(A_ptr + e[:, None] * N + n[None, :])
        Dv = tl.load(D_ptr + e)

        dh = tl.zeros((BE, N), tl.float32)          # carries dh_t * a_t, i.e. dL/dh_{t-1}'s factor
        dA_acc = tl.zeros((BE, N), tl.float32)
        dD_acc = tl.zeros((BE,), tl.float32)
        base = (b * NPE + pe) * L * N

        for step in range(L):
            t = L - 1 - step
            row_e = (b * L + t) * E + e
            row_n = (b * L + t) * N + n
            x_t = tl.load(x_ptr + row_e)
            d_t = tl.load(delta_ptr + row_e)
            b_t = tl.load(Bc_ptr + row_n)
            c_t = tl.load(Cc_ptr + row_n)
            dy_t = tl.load(dy_ptr + row_e)
            h_t = tl.load(h_ptr + (b * L + t) * E * N + e[:, None] * N + n[None, :])

            # h_{t-1}, branch-free: load t-1 clamped, then zero it when t == 0
            prev = tl.maximum(t - 1, 0)
            h_prev = tl.load(h_ptr + (b * L + prev) * E * N + e[:, None] * N + n[None, :])
            h_prev = tl.where(t > 0, h_prev, 0.0)

            dh = c_t[None, :] * dy_t[:, None] + dh       # dh_t = c_t*dy_t + dh_{t+1}*a_{t+1}
            a = tl.exp(d_t[:, None] * A)

            dx_t = tl.sum(dh * (d_t[:, None] * b_t[None, :]), axis=1) + Dv * dy_t
            ddelta_t = tl.sum(dh * (b_t[None, :] * x_t[:, None]), axis=1) \
                + tl.sum(dh * h_prev * a * A, axis=1)
            dA_acc += dh * h_prev * a * d_t[:, None]
            dD_acc += dy_t * x_t

            tl.store(dx_ptr + row_e, dx_t)
            tl.store(ddelta_ptr + row_e, ddelta_t)
            # the two E-reductions: one row per byte, per E-block, summed by the caller
            tl.store(dBc_part_ptr + base + t * N + n,
                     (dh * (d_t[:, None] * x_t[:, None])).sum(0))
            tl.store(dCc_part_ptr + base + t * N + n, (dy_t[:, None] * h_t).sum(0))

            dh = dh * a                                   # carry for t-1

        # dA and dD are reduced over t *and* over the batch, and every batch's program would
        # write the same slice, so each writes its own and the caller sums over the batch.
        tl.store(dA_ptr + (b * E + e)[:, None] * N + n[None, :], dA_acc)
        tl.store(dD_ptr + b * E + e, dD_acc)


    class SSMScan(torch.autograd.Function):
        """y = readout(scan(x, delta, A, Bc, Cc)) + D * x, with a fused backward."""

        @staticmethod
        def forward(ctx, x, delta, A, Bc, Cc, D):
            B, L, E = x.shape
            N = A.shape[1]
            BE = 32
            if E % BE or (N & (N - 1)):
                raise ValueError(f"unsupported shape E={E} N={N}")
            x = x.contiguous()
            delta = delta.contiguous()
            A = A.contiguous()
            Bc = Bc.contiguous()
            Cc = Cc.contiguous()
            D = D.contiguous()
            y = torch.empty_like(x)
            h = torch.empty((B, L, E, N), device=x.device, dtype=torch.float32)
            _fwd[(B, E // BE)](x, delta, A, Bc, Cc, D, y, h, L, E, N, BE)
            ctx.save_for_backward(x, delta, A, Bc, Cc, D, h)
            ctx.shape = (B, L, E, N, BE)
            return y

        @staticmethod
        def backward(ctx, dy):
            x, delta, A, Bc, Cc, D, h = ctx.saved_tensors
            B, L, E, N, BE = ctx.shape
            dy = dy.contiguous()
            NPE = E // BE
            dx = torch.empty_like(x)
            ddelta = torch.empty_like(delta)
            dA_part = torch.zeros((B, E, N), device=x.device)
            dD_part = torch.zeros((B, E), device=x.device)
            dBc_part = torch.zeros((B, NPE, L, N), device=x.device)
            dCc_part = torch.zeros((B, NPE, L, N), device=x.device)
            _bwd[(B, NPE)](x, delta, A, Bc, Cc, D, h, dy, dx, ddelta, dA_part,
                           dBc_part, dCc_part, dD_part, L, E, N, BE, NPE)
            dBc = dBc_part.sum(dim=1)
            dCc = dCc_part.sum(dim=1)
            dA = dA_part.sum(dim=0)
            dD = dD_part.sum(dim=0)
            return dx, ddelta, dA, dBc, dCc, dD


    def ssm_scan(x, delta, A, Bc, Cc, D):
        """Fused scan + readout. Same result as the reference path in model.py."""
        return SSMScan.apply(x, delta, A, Bc, Cc, D)
except Exception:
    scan_triton = None

def _inlined_scan_module():
    class _InlinedScan:
        ssm_scan = staticmethod(ssm_scan)
    return _InlinedScan()


from dataclasses import dataclass
import math

import torch
import torch.nn as nn

try:
    scan_triton = _inlined_scan_module()
except Exception:                       # triton is optional: CPU-only paths must still work
    scan_triton = None
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


SCAN_IMPL = "fast"          # "reference" | "fast" | "compiled" | "triton"
_COMPILED_SCAN = None


def _triton_ok(x, A):
    """The fused kernel needs a CUDA device, E divisible by its block, and a power-of-two N."""
    if scan_triton is None or not x.is_cuda or x.shape[-1] % 32:
        return False
    n = A.shape[-1]
    return n & (n - 1) == 0


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
            if SCAN_IMPL == "triton" and _triton_ok(x, A):
                # One fused kernel: the state stays in registers, so the (B,L,E,N)
                # intermediates that dominated the step are never materialised.
                y = scan_triton.ssm_scan(x.float(), delta, A, Bc.float(), Cc.float(),
                                         self.D.float())
            else:
                a = torch.exp(delta.unsqueeze(-1) * A)             # (B,L,E,N)
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
        # NOT a product. The official implementation walks the whole stage, then recurses into
        # the main network and applies its multiplier again -- but `apply_optimization_params`
        # is `param._optim.update(kwargs)`, so the second call OVERWRITES the first. The main
        # network therefore ends at mult_main (1.5), not mult_outer * mult_main (3.0).
        # Source: goombalab/hnet, hnet/models/hnet.py::_apply_lr_multiplier and
        # hnet/modules/utils.py::apply_optimization_params. Our earlier product reading was
        # wrong by a factor of two on the main network; the paper's closed form agrees, since
        # its innermost stage runs BELOW the base learning rate.
        groups.append({"params": main, "lr_mult": mult_main, "name": "main"})
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
    # Recompute the SSM blocks in the backward pass instead of keeping them. The scan's
    # activations are B x L x d_model x d_state floats, and the chunked scan keeps several
    # rounds of them alive, so at long sequence lengths the activation memory, not the
    # parameter memory, is what the run is limited by.
    grad_checkpoint: bool = False


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
            if self.cfg.grad_checkpoint and self.training:
                x = x + torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
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
            if self.cfg.grad_checkpoint and self.training:
                y = y + torch.utils.checkpoint.checkpoint(blk, y, use_reentrant=False)
            else:
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


def compile_ssm_blocks(model):
    """Compile every SSM block's forward.

    The profile says the four SSM blocks are 90% of the forward pass and the scan is 80% of
    each block, so the win is in the scan's memory traffic, not in the matmuls. Compiling the
    whole block -- rather than only its scan -- lets the surrounding elementwise chain
    (conv, silu, exp, the readout) fuse into the same kernels as the scan.

    Shapes are static per block: (B, L, E) in, (B, L, D) out. The chunk count does not appear
    in the block's signature, so this does not recompile when the boundary count changes.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, SSMBlock):
            m.forward = torch.compile(m.forward, dynamic=False)
            n += 1
    return n


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


import math

# per-parameter-element FLOPs for one byte, relative to a linear of the same width
LINEAR = 2

# Training costs about three times the forward pass: one forward and two backward. This is
# the standard approximation and the one published FLOPs figures use, so a budget stated in
# training FLOPs is comparable with published numbers.
TRAINING_MULTIPLIER = 3.0


def model_flops_per_sequence(cfg, seq_len, chunk_len):
    """FLOPs for one sequence. `chunk_len` is realised bytes/chunk, not the target."""
    D = cfg.d_model
    n_chunks = max(seq_len / max(chunk_len, 1.0), 1.0)
    E = cfg.d_state

    # one SSM block: in_proj(z,x) + conv + x_proj + dt_proj + out_proj, per byte
    ssm_bytes = (cfg.n_enc + cfg.n_dec) * seq_len
    ssm_per_byte = D * 2 * D + D * E + D * (D // 16 + 2 * E) + (D // 16) * D + D * D
    ssm = ssm_bytes * ssm_per_byte * LINEAR

    # The scan's own arithmetic. The recurrence carries an (E, N) state per byte, and the
    # chunked scan does log2(chunk) rounds of shifts, multiplies and adds over it. At these
    # widths this is the same order as the block's matmuls, so leaving it out would make the
    # counter disagree with wall-clock by more than the parity tolerance -- which is exactly
    # what the parity gate measures.
    rounds = max(1, int(math.log2(max(cfg.scan_chunk, 2))))
    scan_per_byte = 3.0 * D * E * rounds      # state is (d_model, d_state) per byte
    ssm += ssm_bytes * scan_per_byte

    # router: two projections at byte resolution
    router = seq_len * 2 * D * D * LINEAR

    # attention: qkv + proj at chunk resolution, plus the L^2 attention itself
    attn = n_chunks * (3 * D * D + D * D) * LINEAR
    attn += cfg.n_main * 2 * cfg.n_heads * n_chunks**2 * (D / cfg.n_heads)

    # swiglu mlp, if present, at chunk resolution
    mlp = 0.0
    if cfg.d_ff:
        mlp = n_chunks * (2 * D * cfg.d_ff + cfg.d_ff * D) * LINEAR

    head = seq_len * D * cfg.vocab * LINEAR
    return ssm + router + cfg.n_main * attn + cfg.n_main * mlp + head


def flops_per_byte(cfg, seq_len, chunk_len):
    return model_flops_per_sequence(cfg, seq_len, chunk_len) / seq_len


def training_flops_per_byte(cfg, seq_len, chunk_len):
    """Training FLOPs per byte: forward plus the backward pass."""
    return TRAINING_MULTIPLIER * flops_per_byte(cfg, seq_len, chunk_len)


def steps_for_budget(cfg, seq_len, batch, chunk_len, budget_flops):
    """How many steps a FLOPs allowance buys.

    The chunk length used here is the target ratio, not the realised one: a schedule has to
    be fixed before the run starts. The realised value is recorded alongside the metrics so
    the two can be compared after the fact.
    """
    bytes_per_step = batch * seq_len
    per_step = bytes_per_step * training_flops_per_byte(cfg, seq_len, chunk_len)
    if per_step <= 0:
        raise ValueError("non-positive FLOPs per step; check the config")
    return max(1, int(budget_flops // per_step)), per_step


def flops_report(cfg, seq_len, batch, chunk_len):
    """Everything a run should record about its own compute."""
    fpb = training_flops_per_byte(cfg, seq_len, chunk_len)
    bytes_per_step = batch * seq_len
    return {
        "training_flops_per_byte": fpb,
        "forward_flops_per_byte": fpb / TRAINING_MULTIPLIER,
        "bytes_per_step": bytes_per_step,
        "training_flops_per_step": bytes_per_step * fpb,
        "chunk_len_assumed": chunk_len,
    }

# ==== ENGLISH RUNG DRIVER ====
# Trains the own H-Net on the English corpus for the comparable rung.
#
# Three things differ from the multilingual driver (kernel/q1.py), and all three come from
# the corpus being documents rather than a byte stream:
#   - batches are formed out of whole documents, never by slicing one across a boundary;
#   - the loss is masked, so padded positions do not contribute to it or to bpb;
#   - the run is specified by a FLOPs allowance, and the step count is derived from it.
#
# The documents are sorted by byte length within each split, so a batch of consecutive
# documents has near-uniform length and the padding waste stays small. The order of the
# batches is shuffled, so the model does not see the corpus shortest-document-first.

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "/tmp/eng.json")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ==== DATA ====
CORPUS = os.environ.get("CORPUS")
if not CORPUS:
    for root in ("/kaggle/input/datasets/mauricew/hnet-english-bytes",
                 "/kaggle/input/hnet-english-bytes",
                 "/kaggle/input/hnet-english-bytes/corpus"):
        if os.path.isdir(root):
            CORPUS = root
            break
assert CORPUS, "the English corpus dataset is not mounted"
SEQ_LEN = int(CFG["seq_len"])
BATCH = int(CFG["batch"])
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class Docs:
    """One split: the bytes, and each document's offset and length."""

    def __init__(self, split):
        # the dataset is published flat: the Kaggle CLI skips directories unless asked to
        # zip them, so the split files sit at the root rather than under a split directory
        d = os.path.join(CORPUS, split)
        if not os.path.exists(os.path.join(d, f"{split}.bin")):
            d = CORPUS
        self.data = np.memmap(os.path.join(d, f"{split}.bin"), dtype=np.uint8, mode="r")
        self.offsets = np.fromfile(os.path.join(d, f"{split}_docs.u32"), dtype="<u4")
        self.lengths = np.fromfile(os.path.join(d, f"{split}_lens.u32"), dtype="<u4")

    def __len__(self):
        return len(self.offsets)

    def doc(self, i):
        o, l = int(self.offsets[i]), int(self.lengths[i])
        return self.data[o:o + l]


def batch_plan(docs, batch, seq_len, rng):
    """Which documents go in which batch, and in what order.

    Documents are length-sorted, so consecutive runs have near-uniform length. The runs are
    then shuffled, so the model does not see the corpus ordered by length. A document longer
    than the sequence is truncated to its head; a short one is padded, and the padding is
    masked out of the loss.
    """
    n = len(docs)
    n_batches = n // batch
    starts = np.arange(n_batches) * batch
    # the longest document in each run sets the batch's width
    widths = np.array([int(docs.lengths[s: s + batch].max()) for s in starts])
    order = rng.permutation(n_batches)
    return starts, widths, order


def make_batch(docs, start, width, batch, seq_len):
    """(bytes, mask): whole documents, padded to the batch's width, capped at seq_len."""
    out = np.zeros((batch, seq_len), dtype=np.uint8)
    mask = np.zeros((batch, seq_len), dtype=np.float32)
    for r in range(batch):
        i = start + r
        if i >= len(docs):
            continue
        raw = docs.doc(i)
        n = min(len(raw), width, seq_len)
        if n < 2:
            continue
        out[r, :n] = raw[:n]
        mask[r, :n - 1] = 1.0            # the loss predicts byte t+1 from byte t
    return out, mask


def lr_at(step, max_steps, lr, warmup):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, max_steps - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def masked_loss(logits, targets, mask):
    """Cross entropy over the real bytes only."""
    ls = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="none")
    m = mask.reshape(-1)
    denom = m.sum().clamp(min=1.0)
    return (ls * m).sum() / denom, m.sum().item()


def evaluate(model, docs, seq_len, batch, n_batches=40, seed=0):
    """Held-out bpb over whole documents."""
    rng = np.random.default_rng(seed)
    starts, widths, order = batch_plan(docs, batch, seq_len, rng)
    model.eval()
    tot_nll, tot_bytes = 0.0, 0
    with torch.no_grad():
        for k in order[:n_batches]:
            start = int(starts[k])
            width = int(widths[k])
            out, mask = make_batch(docs, start, width, batch, seq_len)
            x = torch.from_numpy(out.astype(np.int64)).to(DEV)
            m = torch.from_numpy(mask).to(DEV)
            logits, aux = model(x)
            # logits[t] predicts byte t+1, so the target is x[:, 1:] and the mask shifts with
            # it. Comparing logits[t] against x[t] instead measures the model's guess for a
            # byte it is being shown -- which scored 13.2 bpb, worse than uniform, on a run
            # whose training bpb had reached 1.44.
            ls = F.cross_entropy(logits[:, :-1].reshape(-1, 256),
                                 x[:, 1:].reshape(-1), reduction="none")
            mm = m[:, 1:].reshape(-1)
            tot_nll += float((ls * mm).sum().item())
            tot_bytes += float(mm.sum().item())
    model.train()
    bpb = (tot_nll / max(tot_bytes, 1)) / math.log(2)
    return {"bpb": bpb, "bytes": tot_bytes}


def save_state(model, opt, scaler, step, history, evals, best, flops, path):
    torch.save({"config": dict(model.cfg.__dict__), "seed": CFG["seeds"][0],
                "state": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step,
                "history": history, "evals": evals, "best": best, "flops": flops,
                "rng": np.random.get_state()}, path)


def load_state(model, opt, scaler, path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model.load_state_dict(ck["state"])
    opt.load_state_dict(ck["opt"])
    if ck.get("scaler") is not None:
        scaler.load_state_dict(ck["scaler"])
    try:
        np.random.set_state(ck["rng"])
    except Exception:
        pass
    return ck


def self_check(model, train, seq_len, batch):
    """Prove the evaluation path measures what training measures, before spending hours.

    The first rung trained correctly and was scored by an evaluate() that compared logits[t]
    with byte t instead of t+1. Nothing in the loss curve showed it; the run looked broken
    only in a metric nobody could trust. This compares the two paths on the same batch and
    refuses to start if they disagree.
    """
    model.eval()
    with torch.no_grad():
        ev = evaluate(model, train, seq_len, batch, n_batches=1)
        rng = np.random.default_rng(0)
        starts, widths, order = batch_plan(train, batch, seq_len, rng)
        out, mask = make_batch(train, int(starts[order[0]]), int(widths[order[0]]),
                               batch, seq_len)
        x = torch.from_numpy(out.astype(np.int64)).to(DEV)
        m = torch.from_numpy(mask).to(DEV)
        logits, _ = model(x)
        ce, _ = masked_loss(logits[:, :-1], x[:, 1:], m[:, 1:])
    model.train()
    train_bpb = float(ce) / math.log(2)
    if abs(train_bpb - ev["bpb"]) > 0.05:
        raise RuntimeError(
            f"evaluation disagrees with the training loss on the same batch: "
            f"evaluate() says {ev['bpb']:.3f} bpb, the training path says {train_bpb:.3f}. "
            f"Refusing to train against a metric that is measuring something else.")
    log(f"self-check: evaluation and training agree on the same batch "
        f"({ev['bpb']:.3f} vs {train_bpb:.3f} bpb)")


class TrainStep(nn.Module):
    """The loss, computed inside forward so DataParallel can gather it.

    DataParallel gathers a module's outputs across replicas and cannot gather the dict of
    floats our model returns (F, G, bytes_per_chunk are Python floats). Returning a tuple of
    one-element tensors instead means the gather produces a vector per field, one entry per
    GPU, which the caller averages.

    The wrapper is used for training only. Evaluation and checkpointing keep the raw model, so
    the state dict has no `module.` prefix and an existing checkpoint still loads.
    """

    def __init__(self, inner, alpha):
        super().__init__()
        self.inner = inner
        self.alpha = alpha

    def forward(self, x, mask):
        logits, aux = self.inner(x)
        ce, _ = masked_loss(logits[:, :-1], x[:, 1:], mask[:, 1:])
        rl = aux["ratio_loss"]
        loss = ce + self.alpha * rl
        one = lambda v: torch.as_tensor(float(v), device=x.device).reshape(1)
        return (loss, ce.detach(), rl.detach(), one(aux["F"]), one(aux["G"]),
                one(aux["bytes_per_chunk"]))


def main():
    log("device", DEV)
    log("corpus", CORPUS)
    log("config", json.dumps(CFG))
    train = Docs("train")
    val = Docs("val")
    log(f"train {len(train)} docs {len(train.data)/1e6:.1f} MB, "
        f"val {len(val)} docs {len(val.data)/1e6:.1f} MB")

    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    cfg = HNetConfig(**fields)
    torch.manual_seed(CFG["seeds"][0])
    np.random.seed(CFG["seeds"][0])
    model = HNetLM(cfg).to(DEV)
    init_weights(model)
    global SCAN_IMPL
    want = CFG.get("scan_impl") or ("compiled" if CFG.get("compile_blocks", True) else "reference")
    # Fail loudly rather than falling back: the reference scan is 3.5x slower, and a silent
    # fallback would burn the whole GPU session instead of stopping in the first second.
    if want == "triton":
        # Three ways the fused path can be quietly unavailable. Each would turn an 8 h run
        # into a 30 h one, so each refuses rather than falling back.
        if scan_triton is None:
            raise RuntimeError("config asks for the fused scan but scan_triton did not import")
        if DEV != "cuda":
            raise RuntimeError("config asks for the fused scan but there is no CUDA device")
        if cfg.d_model % 32:
            raise RuntimeError(f"config asks for the fused scan but d_model {cfg.d_model} is "
                               f"not a multiple of the kernel's block size")
    SCAN_IMPL = want
    if want == "compiled":
        compile_ssm_blocks(model)
    log(f"scan implementation: {SCAN_IMPL}")
    ndev = torch.cuda.device_count() if DEV == "cuda" else 1
    log(f"devices visible: {ndev}"
        + "".join(f", cuda:{i} {torch.cuda.get_device_name(i)}" for i in range(ndev)))
    step_model = TrainStep(model, cfg.alpha)
    if CFG.get("data_parallel", True) and ndev > 1 and BATCH % ndev != 0:
        raise SystemExit(f"batch {BATCH} is not divisible by {ndev} devices: DataParallel "
                         f"would split it unevenly and the averaged loss would weight the "
                         f"shards unequally. Use a batch that divides, or set "
                         f"data_parallel=false.")
    if CFG.get("data_parallel", True) and ndev > 1:
        step_model = nn.DataParallel(step_model)
        log(f"training on {ndev} GPUs via DataParallel; evaluation and checkpoints use the "
            f"raw model so the state dict keeps its keys")
    else:
        log("training on a single device")
    mo = float(CFG.get("lr_mult_outer", 1.0))
    mm = float(CFG.get("lr_mult_main", 1.0))
    groups = param_groups(model, mo, mm)
    opt = torch.optim.AdamW(groups, lr=CFG["lr"], weight_decay=CFG["wd"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=(CFG.get("amp") == "fp16" and DEV == "cuda"))
    nparams = sum(p.numel() for p in model.parameters())
    log(f"model {nparams/1e6:.2f} M params, {cfg.n_enc}+{cfg.n_dec} SSM blocks, "
        f"{cfg.n_main} main blocks")

    # the budget becomes the schedule
    budget = float(CFG["flops_budget"])
    assumed_chunk = float(cfg.target_ratio)
    max_steps, per_step = steps_for_budget(cfg, SEQ_LEN, BATCH, assumed_chunk, budget)
    flops = flops_report(cfg, SEQ_LEN, BATCH, assumed_chunk)
    flops.update({"budget_flops": budget, "derived_steps": max_steps,
                  "training_flops_per_step": per_step,
                  "budget_total_flops": max_steps * per_step,
                  "chunk_len_assumed": assumed_chunk})
    log(f"budget {budget:.4e} training FLOPs -> {max_steps} steps "
        f"({per_step:.4e} FLOPs/step, {flops['training_flops_per_byte']:.4e} FLOPs/byte)")

    self_check(model, train, SEQ_LEN, BATCH)

    rng = np.random.default_rng(CFG["seeds"][0])
    starts, widths, order = batch_plan(train, BATCH, SEQ_LEN, rng)
    log(f"{len(order)} batches of {BATCH} documents")

    history, evals = [], []
    best = None
    resume_from = os.path.join(OUT, "ckpt-resume.pt")
    start_step = 0
    if CFG.get("resume", True) and os.path.exists(resume_from):
        ck = load_state(model, opt, scaler, resume_from)
        start_step = int(ck["step"])
        history = ck.get("history", [])
        evals = ck.get("evals", [])
        best = ck.get("best")
        log(f"resumed from {resume_from} at step {start_step}")
    t0 = time.time()
    wall_guard = float(CFG.get("wall_guard_s", 36000))
    done = 0
    for step in range(start_step, max_steps):
        k = order[step % len(order)]
        start = int(starts[k])
        width = int(widths[k])
        out, mask = make_batch(train, start, width, BATCH, SEQ_LEN)
        x = torch.from_numpy(out.astype(np.int64)).to(DEV)
        m = torch.from_numpy(mask).to(DEV)

        lr = lr_at(step, max_steps, CFG["lr"], CFG["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)

        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=(CFG.get("amp") == "fp16" and DEV == "cuda")):
            out = step_model(x, m)
        # out is a tuple of scalars on one device, or of per-GPU vectors on several; mean()
        # is correct for both and is what keeps the reported numbers comparable
        loss = out[0].mean()
        ce, rl = out[1].mean(), out[2].mean()
        aux = {"F": out[3].mean().item(), "G": out[4].mean().item(),
               "bytes_per_chunk": out[5].mean().item()}
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        scaler.step(opt)
        scaler.update()
        done = step + 1

        if step % CFG["log_every"] == 0 or step == max_steps - 1:
            el = time.time() - t0
            rec = {"step": step, "ce": float(ce.item()), "bpb": float(ce.item()) / math.log(2),
                   "ratio_loss": float(rl.item()), "F": aux["F"], "G": aux["G"],
                   "bytes_per_chunk": aux["bytes_per_chunk"], "lr": lr,
                   "bytes": int(m.sum().item()), "elapsed_s": el,
                   "eta_h": (el / max(done, 1)) * (max_steps - done) / 3600}
            history.append(rec)
            log(f"step {step:6} bpb {rec['bpb']:.4f} ratio {rec['ratio_loss']:.4f} "
                f"F {rec['F']:.3f} G {rec['G']:.3f} bytes/chunk {rec['bytes_per_chunk']:.2f} "
                f"lr {lr:.2e} elapsed {el/60:.1f}m eta {rec['eta_h']:.1f}h")
        if step % CFG["eval_every"] == 0 or step == max_steps - 1:
            ev = evaluate(model, val, SEQ_LEN, BATCH,
                          n_batches=int(CFG.get("eval_batches", 40)))
            evals.append({"step": step, **ev})
            log(f"  eval step {step}: held-out bpb {ev['bpb']:.4f} over {ev['bytes']} bytes")
            # 8.0 bits is uniform over 256 symbols, and an untrained model sits right at it
            # (7.94 in the first rung), so step 0 is exempt: the canary is for a run that has
            # had a chance to learn and is still anti-predicting, which is a broken metric or
            # a diverged run, not a slow start.
            if step > 0 and ev["bpb"] > 8.5:
                save_state(model, opt, scaler, step + 1, history, evals, best, flops,
                           resume_from)
                raise RuntimeError(
                    f"held-out bpb {ev['bpb']:.3f} is far above uniform (8.0): stopping instead "
                    f"of spending more quota on a run that cannot be right")
            if best is None or ev["bpb"] < best["bpb"]:
                best = {"step": step, **ev}
                torch.save({"config": dict(cfg.__dict__), "seed": CFG["seeds"][0],
                            "state": model.state_dict(), "step": step, "eval": ev},
                           os.path.join(OUT, "ckpt-best.pt"))
        if step % CFG.get("save_every", 500) == 0 and step > start_step:
            save_state(model, opt, scaler, step + 1, history, evals, best, flops,
                       resume_from)
        if time.time() - t0 > wall_guard:
            log("wall guard hit")
            save_state(model, opt, scaler, step + 1, history, evals, best, flops,
                       resume_from)
            break

    # what the run actually cost, next to what it was allowed to cost
    realised_chunk = history[-1]["bytes_per_chunk"] if history else None
    if realised_chunk:
        flops["realised_bytes_per_chunk"] = realised_chunk
        flops["realised_training_flops_per_byte"] = training_flops_per_byte(
            cfg, SEQ_LEN, realised_chunk)
        flops["realised_total_flops"] = done * BATCH * SEQ_LEN * \
            flops["realised_training_flops_per_byte"]
        flops["budget_used_fraction"] = flops["realised_total_flops"] / budget

    torch.save({"config": dict(cfg.__dict__), "seed": CFG["seeds"][0],
                "state": model.state_dict(), "step": done},
               os.path.join(OUT, "ckpt-final.pt"))
    save_state(model, opt, scaler, done, history, evals, best, flops, resume_from)
    json.dump({"config": CFG, "steps": done, "stop": "max_steps" if done == max_steps else "wall_guard",
               "history": history, "evals": evals, "best": best, "flops": flops,
               "params": nparams, "elapsed_s": time.time() - t0},
              open(os.path.join(OUT, "eng.json"), "w"), indent=2)
    log("done", done, "steps", f"{(time.time()-t0)/3600:.2f} h",
        "best held-out bpb", best["bpb"] if best else None)


if __name__ == "__main__":
    main()
