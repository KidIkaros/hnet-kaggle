"""Preflight quality control for an H-Net training run.

Every run gets checked locally before a Kaggle push, because the failures that cost us GPU
hours today were all catchable here:

  - the learning-rate multipliers were off-spec (a product where the official code overwrites),
    and nothing compared our param groups against the reference's semantics;
  - the budget was 2.5x below compute-optimal and bought 0.057 epochs, which no gate noticed;
  - the step ran at 0.34% of the T4's peak because the batch was 256x smaller than the
    reference's, and we only discovered it after spending quota;
  - a 9-hour run was launched on a configuration whose inner learning rate was 2x the
    published value.

Four stages, each of which can fail the run:

  static       config arithmetic: parameters, FLOPs/byte, steps, bytes/param, epochs
  correctness  the model's semantics, on CPU, against independent references
  performance  measured step time and peak memory (GPU only), extrapolated to the budget
  spec         the config against the published implementation's values

Exit code is 0 only if every executed stage passes, so a wrapper can gate a push on it:

    python tools/qc.py --config configs/eng-rung.json && kaggle kernels push -p kernel_eng

A stage that cannot run (no GPU) is reported as SKIPPED, never as passed.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import model as M
from flops import flops_report, steps_for_budget, training_flops_per_byte

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD_DIR = os.path.join(HERE, ".scratch", "qc")

# Thresholds. These are decisions, not defaults: each one is a run we would refuse to pay for.
LIMITS = {
    "peak_memory_gb": 12.0,      # T4 usable is 14.5 GB; leave headroom for the allocator
    "min_mfu": 0.02,             # 2% of 65 TFLOPS = 1.3 TFLOPS; below this we are launch-bound
    "min_bytes_per_param": 10.0,  # below this the model is far under-trained for its size
    "max_quota_fraction": 0.80,  # a run may not plan to use more than 80% of what is left
    "t4_peak_flops": 65e12,
    # the published implementation's values, from goombalab/hnet
    "spec_lr_mult_outer": 2.0,
    "spec_lr_mult_main": 1.5,
    "spec_alpha": 0.03,
}

RESULTS = []


def check(stage, name, ok, detail=""):
    RESULTS.append({"stage": stage, "check": name, "pass": bool(ok), "detail": detail})
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def skip(stage, name, why):
    RESULTS.append({"stage": stage, "check": name, "pass": None, "detail": why})
    print(f"  [SKIP] {name} — {why}")


# --------------------------------------------------------------------------- static
def stage_static(cfg_dict, cfg, seq_len, batch, budget, quota_hours):
    print("\n-- static")
    nparams = sum(p.numel() for p in M.HNetLM(cfg).parameters())
    rep = flops_report(cfg, seq_len, batch, cfg.target_ratio)
    steps, per_step = steps_for_budget(cfg, seq_len, batch, cfg.target_ratio, budget)
    bytes_per_step = batch * seq_len
    total_bytes = steps * bytes_per_step
    bytes_per_param = total_bytes / nparams
    epochs = total_bytes / CORPUS_BYTES

    print(f"     {nparams/1e6:.2f} M params, {rep['training_flops_per_byte']/1e6:.2f} MFLOPs/byte, "
          f"{steps} steps, {total_bytes/1e6:.1f} MB seen")
    check("static", "parameters in the intended family (5M-60M)", 5e6 <= nparams <= 60e6,
          f"{nparams/1e6:.2f} M")
    check("static", f"data seen >= {LIMITS['min_bytes_per_param']:.0f} bytes/param",
          bytes_per_param >= LIMITS["min_bytes_per_param"],
          f"{bytes_per_param:.1f} bytes/param, {epochs:.3f} epochs of the corpus")
    check("static", "budget buys at least 0.05 epochs", epochs >= 0.05,
          f"{epochs:.3f} epochs")
    check("static", "sequence length fits the document length distribution",
          seq_len >= 2048, f"seq_len {seq_len} vs median document ~2936 B")
    return {"params": nparams, "flops_per_byte": rep["training_flops_per_byte"],
            "steps": steps, "bytes_seen": total_bytes, "bytes_per_param": bytes_per_param,
            "epochs": epochs, "flops_per_step": per_step}


# --------------------------------------------------------------------- correctness
def stage_correctness(cfg, seq_len, batch, steps=60):
    print("\n-- correctness")
    torch.manual_seed(0)

    # 1. the chunked scan must agree with the naive recurrence
    a = torch.rand(2, 128, 32, 8) * 0.5 + 0.5
    b = torch.randn(2, 128, 32, 8)
    naive = M.sequential_scan(a[:, :16], b[:, :16])
    chunked = M.chunked_scan(a[:, :16], b[:, :16], 64)
    err = (naive - chunked).abs().max().item()
    check("correctness", "chunked scan matches the naive recurrence", err < 1e-4,
          f"max abs diff {err:.2e}")

    # 2. the masked loss must ignore padded positions
    logits = torch.randn(1, 6, 256)
    targets = torch.randint(0, 256, (1, 6))
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    got, n = masked_loss(logits, targets, mask)
    want = F.cross_entropy(logits[0, :3], targets[0, :3])
    check("correctness", "masked loss equals cross-entropy over the real bytes only",
          abs(got.item() - want.item()) < 1e-5 and n == 3,
          f"{got.item():.6f} vs {want.item():.6f}, {int(n)} bytes")

    # 3. every parameter must receive a gradient
    m = M.HNetLM(cfg)
    M.init_weights(m)
    x = torch.randint(0, 256, (2, min(seq_len, 256)))
    logits, aux = m(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    check("correctness", "no parameter is left without a gradient", not dead,
          f"{len(dead)} dead: {dead[:4]}")

    # 4. the model must be able to overfit one batch (catches broken wiring end to end)
    m2 = M.HNetLM(cfg)
    M.init_weights(m2)
    opt = torch.optim.AdamW(m2.parameters(), lr=3e-3)
    xb = torch.randint(0, 256, (2, 128))
    first = None
    for _ in range(steps):
        logits, aux = m2(xb)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    check("correctness", "overfits a single batch", loss.item() < 0.5 * first,
          f"loss {first:.3f} -> {loss.item():.3f} in {steps} steps")

    # 5. determinism
    torch.manual_seed(7)
    m3 = M.HNetLM(cfg)
    M.init_weights(m3)
    with torch.no_grad():
        l1, _ = m3(x)
    torch.manual_seed(7)
    m4 = M.HNetLM(cfg)
    M.init_weights(m4)
    with torch.no_grad():
        l2, _ = m4(x)
    check("correctness", "same seed gives the same forward", torch.equal(l1, l2))


def masked_loss(logits, targets, mask):
    """Cross entropy over the real bytes only. Mirrors kernel/eng.py."""
    ls = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="none")
    m = mask.reshape(-1)
    return (ls * m).sum() / m.sum().clamp(min=1.0), m.sum().item()


# -------------------------------------------------------------------- performance
def stage_performance(cfg, seq_len, batch, budget, quota_hours, n=3, cfg_dict=None,
                      max_seconds=90.0):
    """Measure one configuration. Deliberately small: this runs on the owner's desktop.

    Three steps plus warmup is enough to catch an order-of-magnitude problem, which is all
    this stage is for. Anything that compares configurations is a sweep, and a sweep belongs
    on Kaggle as a short job, not on someone's workstation.
    """
    print("\n-- performance")
    if not torch.cuda.is_available():
        skip("performance", "measured step time and memory", "no CUDA device in this interpreter")
        return None
    dev = "cuda"
    m = M.HNetLM(cfg).to(dev)
    M.init_weights(m)
    # measure the configuration we would actually run: compiled blocks if the config says so.
    # The scan implementation is a module-level global in model.py, not a model attribute.
    if (cfg_dict or {}).get("compile_blocks", False):
        M.SCAN_IMPL = "compiled"
        M.compile_ssm_blocks(m)
    else:
        M.SCAN_IMPL = "reference"
    torch.manual_seed(0)
    x = torch.randint(0, 256, (batch, seq_len), device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    for _ in range(2):
        logits, aux = m(x)
        F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        logits, aux = m(x)
        F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
        opt.step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    if time.perf_counter() - t_start > max_seconds:
        skip("performance", "measurement stayed within the local time budget",
             f"took {time.perf_counter()-t_start:.0f}s")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    achieved = training_flops_per_byte(cfg, seq_len, float(aux["bytes_per_chunk"])) \
        * batch * seq_len / (ms / 1000)
    mfu = achieved / LIMITS["t4_peak_flops"]
    steps, _ = steps_for_budget(cfg, seq_len, batch, cfg.target_ratio, budget)
    hours = steps * ms / 1000 / 3600
    print(f"     {ms:.0f} ms/step, peak {peak_gb:.2f} GB, {achieved/1e9:.0f} GFLOP/s "
          f"({mfu*100:.2f}% MFU), {steps} steps = {hours:.1f} h")
    check("performance", f"peak memory <= {LIMITS['peak_memory_gb']} GB",
          peak_gb <= LIMITS["peak_memory_gb"], f"{peak_gb:.2f} GB")
    check("performance", f"MFU >= {LIMITS['min_mfu']*100:.1f}% (not launch-bound)",
          mfu >= LIMITS["min_mfu"], f"{mfu*100:.2f}% of T4 peak")
    check("performance", f"run fits in {LIMITS['max_quota_fraction']*100:.0f}% of remaining quota",
          hours <= LIMITS["max_quota_fraction"] * quota_hours,
          f"{hours:.1f} h needed, {quota_hours:.1f} h left")
    return {"ms_per_step": ms, "peak_gb": peak_gb, "achieved_flops": achieved, "mfu": mfu,
            "hours": hours}


# --------------------------------------------------------------------------- spec
def stage_spec(cfg_dict):
    print("\n-- spec")
    lo = float(cfg_dict.get("lr_mult_outer", 1.0))
    lm = float(cfg_dict.get("lr_mult_main", 1.0))
    check("spec", "outer learning-rate multiplier matches the reference",
          abs(lo - LIMITS["spec_lr_mult_outer"]) < 1e-9, f"{lo} vs {LIMITS['spec_lr_mult_outer']}")
    # the official code OVERWRITES, so the main net's effective multiplier is lr_mult_main
    check("spec", "main learning-rate multiplier is the official 1.5 (not composed)",
          abs(lm - LIMITS["spec_lr_mult_main"]) < 1e-9,
          f"{lm} vs {LIMITS['spec_lr_mult_main']} (official overwrites; a product would be 3.0)")
    # Asserted against the spec's literals, not against the config's own values: comparing
    # main_mult to lm would pass whenever the code and the config agree, which is exactly the
    # case that was wrong. The expected numbers come from the official implementation.
    probe = M.HNetLM(M.HNetConfig(d_model=64, n_enc=1, n_main=1, n_dec=1))
    groups = M.param_groups(probe, 2.0, 1.5)
    main_mult = next((g["lr_mult"] for g in groups if g["name"] == "main"), None)
    outer_mult = next((g["lr_mult"] for g in groups if g["name"] == "outer"), None)
    check("spec", "param_groups(2.0, 1.5) yields main 1.5 and outer 2.0 (no composition)",
          outer_mult == 2.0 and main_mult == 1.5,
          f"outer {outer_mult}, main {main_mult}; a composed reading would give 3.0")
    alpha = float(cfg_dict.get("alpha", 0.0))
    note = ("deviation, must be reported" if abs(alpha - LIMITS["spec_alpha"]) > 1e-9
            else "matches the paper")
    check("spec", "alpha is the paper's 0.03 or recorded as a deviation", True,
          f"alpha {alpha} vs paper {LIMITS['spec_alpha']} ({note})")


CORPUS_BYTES = 2353009396  # the published English corpus' train split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quota-hours", type=float, default=23.0,
                    help="GPU hours remaining on Kaggle")
    ap.add_argument("--no-perf", action="store_true", help="skip the GPU stage")
    ap.add_argument("--perf-steps", type=int, default=3,
                    help="timed steps in the GPU stage; keep this small on a workstation")
    args = ap.parse_args()

    cfg_dict = json.load(open(args.config)) or {}
    fields = {k: v for k, v in cfg_dict.items() if k in M.HNetConfig.__dataclass_fields__}
    cfg = M.HNetConfig(**fields)
    seq_len, batch = int(cfg_dict["seq_len"]), int(cfg_dict["batch"])
    budget = float(cfg_dict.get("flops_budget", 0)) or 1e16
    if not cfg_dict.get("flops_budget"):
        # no budget in the config: derive one from max_steps so the arithmetic still runs
        rep = flops_report(cfg, seq_len, batch, cfg.target_ratio)
        budget = int(cfg_dict.get("max_steps", 1000)) * rep["training_flops_per_step"]

    print(f"QC for {args.config}")
    print(f"  device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    static = stage_static(cfg_dict, cfg, seq_len, batch, budget, args.quota_hours)
    stage_correctness(cfg, seq_len, batch)
    perf = None if args.no_perf else stage_performance(cfg, seq_len, batch, budget,
                                                       args.quota_hours,
                                                       n=args.perf_steps,
                                                       cfg_dict=cfg_dict)
    stage_spec(cfg_dict)

    failed = [r for r in RESULTS if r["pass"] is False]
    skipped = [r for r in RESULTS if r["pass"] is None]
    print(f"\n{len(RESULTS) - len(failed) - len(skipped)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped")
    for r in failed:
        print(f"  FAILED: {r['stage']}/{r['check']} — {r['detail']}")

    os.makedirs(RECORD_DIR, exist_ok=True)
    rec = os.path.join(RECORD_DIR, os.path.basename(args.config).replace(".json", "")
                       + f"-{time.strftime('%Y%m%d-%H%M%S')}.json")
    json.dump({"config": args.config, "config_values": cfg_dict, "static": static,
               "performance": perf, "results": RESULTS, "pass": not failed,
               "skipped": [s["check"] for s in skipped]}, open(rec, "w"), indent=2)
    print(f"record: {rec}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
