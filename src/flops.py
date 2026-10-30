"""Analytic FLOPs per byte, for FLOPs parity across arms.

Outer networks are 40%+ of H-Net's non-matmul time and the main net runs at chunk
resolution, so a counter that charges the main net per byte (or omits the outer nets)
is not a control. Charge each stage at the resolution it actually runs at.

Counts matmul + attention only (2*m*n per linear, 2*heads*L^2*d_head for attention).
"""

# per-parameter-element FLOPs for one token, relative to a linear of the same width
LINEAR = 2


def model_flops_per_sequence(cfg, seq_len, chunk_len):
    """FLOPs for one sequence. `chunk_len` is realised bytes/chunk, not the target."""
    D = cfg.d_model
    n_chunks = max(seq_len / max(chunk_len, 1.0), 1.0)
    E = cfg.d_state

    # one SSM block: in_proj(z,x) + conv + x_proj + dt_proj + out_proj, per token
    ssm_tokens = (cfg.n_enc + cfg.n_dec) * seq_len
    ssm_per_token = D * 2 * D + D * E + D * (D // 16 + 2 * E) + (D // 16) * D + D * D
    ssm = ssm_tokens * ssm_per_token * LINEAR

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
