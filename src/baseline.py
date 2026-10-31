"""Baseline language models for the efficiency comparison.

Two arms, one file:

    byte transformer   vocabulary 256, the raw bytes as units
    BPE transformer    vocabulary V, the in-corpus BPE ids as units

Both are the same standard pre-norm transformer (RMSNorm, RoPE, SwiGLU, no biases), so the
only difference between the two arms is how many units a byte becomes. That is exactly the
comparison Cartesia makes against their baseline, and it is also the comparison that isolates
"the chunker compresses" from "a fixed tokenizer compresses".

The H-Net's advantage, if it has one, does not come from compressing more bytes -- a BPE
tokenizer already compresses 4.57 bytes/token on their mixture. It comes from *where the
compute goes*: the H-Net's outer networks are SSMs running at byte resolution, so the expensive
main network can be large while the byte-level work stays cheap. A transformer over BPE tokens
spends its FLOPs on a long sequence instead. Matching parameters and comparing FLOPs/byte is
the only way to see that.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaseConfig:
    vocab: int = 256
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 512          # SwiGLU hidden width
    max_seq: int = 1024
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(seq, head_dim, theta, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f).to(dtype), torch.sin(f).to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, H, L, Dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, self.h, self.dh).transpose(1, 2)
        k = k.view(B, L, self.h, self.dh).transpose(1, 2)
        v = v.view(B, L, self.h, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, L, D))


class SwiGLU(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class BaseLM(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        cos, sin = rope_cache(cfg.max_seq, cfg.d_model // cfg.n_heads, cfg.rope_theta,
                              "cpu", torch.float32)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.emb(idx)
        L = x.shape[1]
        cos, sin = self.cos[:L].to(x.dtype), self.sin[:L].to(x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x)), {}


def init_weights(model):
    """Same convention as the H-Net: N(0, 0.02) except out-projections scaled by depth."""
    n_res = sum(1 for _ in model.blocks) * 2
    for name, p in model.named_parameters():
        if "head.weight" in name or name.endswith("down.weight"):
            nn.init.normal_(p, std=0.02 / math.sqrt(max(n_res, 1)))
        elif p.dim() >= 2:
            nn.init.normal_(p, std=0.02)


def flops_per_unit(cfg: BaseConfig, seq_len: int) -> float:
    """FLOPs for one unit of one sequence, charged the same way as src/flops.py."""
    D, H = cfg.d_model, cfg.n_heads
    dh = D // H
    per_layer = (4 * D * D + 2 * H * seq_len * dh) * 2        # qkv+o, and causal attention
    per_layer += (3 * D * cfg.d_ff) * 2                         # swiglu
    head = D * cfg.vocab * 2
    return cfg.n_layers * per_layer + head


def flops_per_byte(cfg: BaseConfig, seq_len: int, units_per_byte: float) -> float:
    """units_per_byte = 1 for the byte arm, tokens/byte for the BPE arm."""
    return flops_per_unit(cfg, seq_len) * units_per_byte
