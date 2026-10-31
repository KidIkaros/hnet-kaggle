"""Analytic FLOPs per byte, for FLOPs parity across arms.

Outer networks are 40%+ of H-Net's non-matmul time and the main net runs at chunk
resolution, so a counter that charges the main net per byte (or omits the outer nets)
is not a control. Charge each stage at the resolution it actually runs at.

Counts matmul + attention only (2*m*n per linear, 2*heads*L^2*d_head for attention).
"""

import math

# per-parameter-element FLOPs for one token, relative to a linear of the same width
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

    # one SSM block: in_proj(z,x) + conv + x_proj + dt_proj + out_proj, per token
    ssm_tokens = (cfg.n_enc + cfg.n_dec) * seq_len
    ssm_per_token = D * 2 * D + D * E + D * (D // 16 + 2 * E) + (D // 16) * D + D * D
    ssm = ssm_tokens * ssm_per_token * LINEAR

    # The scan's own arithmetic. The recurrence carries an (E, N) state per token, and the
    # chunked scan does log2(chunk) rounds of shifts, multiplies and adds over it. At these
    # widths this is the same order as the block's matmuls, so leaving it out would make the
    # counter disagree with wall-clock by more than the parity tolerance -- which is exactly
    # what the parity gate measures.
    rounds = max(1, int(math.log2(max(cfg.scan_chunk, 2))))
    scan_per_token = 3.0 * D * E * rounds      # state is (d_model, d_state) per token
    ssm += ssm_tokens * scan_per_token

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
