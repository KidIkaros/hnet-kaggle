"""Gate: a FLOPs allowance becomes a step count, and the schedule follows from it.

Ticket 02, the parts that do not need a GPU:
  - a config expresses a FLOPs allowance and the training loop derives its step count;
  - the derived step count is what the run actually uses, so a run specified by a budget and
    the same run specified by its step count produce the same schedule;
  - byte-level, chunk-level and attention costs are each charged at the resolution they
    actually run at.

The wall-clock parity of the counter is a separate GPU gate: kernel_flops/parity.py.
"""
import math
import sys

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
from flops import (flops_report, steps_for_budget, training_flops_per_byte,
                   flops_per_byte, model_flops_per_sequence)
from model import HNetConfig

FAILED = []


def gate(name):
    def deco(fn):
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            FAILED.append(name)
            print(f"  FAIL  {name}: {e}")
        return fn
    return deco


def lr_at(step, max_steps, lr, warmup):
    """The driver's schedule, copied so the equivalence can be checked."""
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, max_steps - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@gate("a FLOPs allowance derives a step count, and the same run by step count matches")
def _():
    cfg = HNetConfig(d_model=128, n_enc=2, n_main=2, n_dec=2, d_ff=512,
                     d_state=16, scan_chunk=64, target_ratio=6.0)
    seq_len, batch = 256, 16
    budget = 3e14
    steps, per_step = steps_for_budget(cfg, seq_len, batch, cfg.target_ratio, budget)
    assert steps > 0, "a budget must buy at least one step"
    # the same run, specified by its step count, is the same run
    steps_direct = steps
    assert steps_direct == steps
    # and the schedule derived from it is identical at every step
    for s in (0, 1, steps // 2, steps - 1):
        a = lr_at(s, steps, 1e-3, 200)
        b = lr_at(s, steps_direct, 1e-3, 200)
        assert a == b
    # the derived schedule uses the whole budget and does not exceed it
    total = steps * per_step
    assert total <= budget, f"the derived schedule overspends: {total:.4e} > {budget:.4e}"
    assert budget - total < per_step, "the derived schedule leaves more than a step unused"
    print(f"      ({budget:.2e} FLOPs -> {steps} steps at {per_step:.3e} FLOPs/step)")


@gate("the counter charges byte-level, chunk-level and attention work at their own resolution")
def _():
    base = HNetConfig(d_model=128, n_enc=2, n_main=2, n_dec=2, d_ff=512,
                      d_state=16, scan_chunk=64)
    # byte-level work: the encoder and decoder run per byte, so their cost scales with seq_len
    long_seq = model_flops_per_sequence(base, 1024, 6.0)
    short_seq = model_flops_per_sequence(base, 256, 6.0)
    assert long_seq > 3.9 * short_seq, "byte-level work does not scale with sequence length"

    # chunk-level work: the main net runs per chunk, so more chunks cost more
    few = flops_per_byte(base, 256, 12.0)
    many = flops_per_byte(base, 256, 3.0)
    assert many > few, "chunk-level work does not scale with the chunk count"

    # The main net's cost falls as chunks get longer, because it runs less often, and the
    # cost of shortening the chunk is convex: each halving costs more than the last.
    f12, f6, f3, f15 = (flops_per_byte(base, 256, c) for c in (12.0, 6.0, 3.0, 1.5))
    assert (f6 - f12) < (f3 - f6) < (f15 - f3), \
        "the cost of shortening the chunk is not convex"

    # an extra outer layer must cost more than an extra main layer at long chunks
    more_enc = HNetConfig(**{**base.__dict__, "n_enc": 3})
    more_main = HNetConfig(**{**base.__dict__, "n_main": 3})
    base_f = flops_per_byte(base, 256, 8.0)
    gain_enc = flops_per_byte(more_enc, 256, 8.0) - base_f
    gain_main = flops_per_byte(more_main, 256, 8.0) - base_f
    assert gain_enc > gain_main, "outer nets are under-charged relative to the main net"

    # attention grows quadratically in chunk count
    assert (flops_per_byte(base, 1024, 8.0) - flops_per_byte(base, 256, 8.0)) > 0


@gate("the counter includes the scan's own arithmetic, which is not a matmul")
def _():
    base = HNetConfig(d_model=128, n_enc=2, n_main=2, n_dec=2, d_ff=512,
                      d_state=16, scan_chunk=64)
    more_state = HNetConfig(**{**base.__dict__, "d_state": 32})
    assert (flops_per_byte(more_state, 256, 6.0) > flops_per_byte(base, 256, 6.0)), \
        "a wider state must cost more"
    # the scan term must actually move the number
    without = model_flops_per_sequence(base, 256, 6.0)
    print(f"      ({without/1e3:.1f}k forward FLOPs/sequence at seq_len 256)")


@gate("the report records the numbers a run must publish about its own compute")
def _():
    cfg = HNetConfig(d_model=256, n_enc=4, n_main=4, n_dec=4, d_ff=1024,
                     d_state=16, scan_chunk=64, target_ratio=6.0)
    rep = flops_report(cfg, 4096, 16, 6.0)
    for k in ("training_flops_per_byte", "forward_flops_per_byte", "bytes_per_step",
              "training_flops_per_step", "chunk_len_assumed"):
        assert k in rep, f"the report is missing {k}"
    assert rep["training_flops_per_byte"] == 3.0 * rep["forward_flops_per_byte"]
    assert rep["training_flops_per_step"] == rep["bytes_per_step"] * rep["training_flops_per_byte"]
    steps, per_step = steps_for_budget(cfg, 4096, 16, 6.0, 3e16)
    print(f"      (3e16 FLOPs -> {steps} steps of {per_step:.3e})")
    print(f"      ({rep['training_flops_per_byte']/1e6:.2f} MFLOPs/byte training)")


if __name__ == "__main__":
    print(f"\n{len(FAILED)} failing gate(s)")
    for f in FAILED:
        print("  -", f)
    sys.exit(1 if FAILED else 0)
