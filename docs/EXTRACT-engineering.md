# H-Net on one GPU — kernels, varlen, memory, decode (main-horse, Part 0 `inf` + Part 2 `eng-1gpu`)

## 1. Stated desiderata

Two minimal criteria for small-scale research runs: (1) compute blocks are `torch.compile`-able with `fullgraph=True`; (2) minimal CPU overhead / CUDA synchronization per train step. He explicitly declines going further (whole-step compile → CUDAGraph) because the architecture has "inherent shape dynamism".

## 2. Kernel / dependency inventory

**Required (structural):**
- **Mamba2**, specifically the "gigantic `mamba_split_conv1d_scan_combined` method" — "the first major roadblock against efficient H-Net training". The encoder is Mamba2, and "results clearly show M >> T here" (mamba beats transformer at the encoder position), so substituting a transformer is not quality-neutral.
- **causal-conv1d** `custom_op`: torch.compile is incompatible with `custom_op`s that have non-contiguous inputs; patched by "subsuming the expected transpose into the `custom_op` wrapper". Repo assert: `d_conv <= 4` — "causal-conv1d only supports d_conv <= 4".
- **tridao rope/rmsnorm kernels**: "patch/reimplement certain tridao kernels (rope, rmsnorm) to be more torch-compile friendly", because the public compile recipe is "not wrapping it right". These are *reimplemented in torch*, i.e. proven replaceable.
- **mamba's triton kernels** generally (autotuning + `pre_hook` patterns).

**Optional / replaceable:**
- `flash-attn==2.8.1` and `causal-conv1d==1.5.2` are only `[project.optional-dependencies] build` extras; core deps are `torch==2.7.1`, `mamba-ssm` (fork, rev `tc`), `termcolor`, `omegaconf`. Caveat: `xf.py` imports `flash_attn_varlen_func` and `apply_rotary_emb` at module scope and asserts `flash_attn.__version__ == "2.8.1"`, so the *attention path* is flash-attn-shaped even though packaging calls it optional.
- **NJT (nested jagged tensors)** — removed, "only used externally for debugging".
- Compile machinery (`nested_compile_region`, CUDAGraphs, guard elision) — perf only.
- Optimizer: dion's "batched muon impl" (`newton_schulz_triton`) with CUDAGraphs; everything else via `foreach`.

## 3. Varlen / ragged batches

- The natural expression is NJT, but "pure eager mode execution of NJTs will easily double CPU overhead". NJTs only pay off when `torch.compile` fuses them into varlen ops, and "that won't happen if the ops are too novel to exist in torch (e.g. mamba scan)" or "if NJTs cause torch.compile errors". Hence: "purge them, in exchange for uglier code. A rather sad affair overall." Part 0 predicts "handcrafted cu_seqlens-like approaches will beat time-to-market" over NestedTensor.
- Replacement is hand-rolled `cu_seqlens`: `cu_outer` (document boundaries) and `cu_inner` (chunk boundaries), computed as `cu_inner = F.pad(b.cumsum(0), (1,0))[cu_outer]`, with `inner_seqlen = cu_inner[-1].item()` and `inner_maxlen = cu_inner.diff().max().item()`.
- Q/K projection needs BOS padding, so it is a custom autograd op: `QPaddedProj.apply(y_enc, w_q, k, cu_outer)` ("matmul + minor extra copies/padding"; `modeling_hnet.py#L39`).
- Chunking is `h = x.masked_select(b)` *only* in the unbatched case; the released repo "does not use them", and its batched constraints are explicitly not described ("I am … ill-qualified to describe the full constraints of").

## 4. Where time and memory actually go

- Redundant copies: mamba's `dzxbcdt` pattern (create fused tensor → `.split(...,dim=-1)` into free views → mutate in place) forces inductor to "manually copy memory between fused<->chunk tensors". The profiler shows magenta `triton_poi*` blocks, including `triton_poi_fused_2` copying one `empty` tensor to another ("copying **uninitialized data**"). Fixing it "improves the execution time of small-scale mamba layers by roughly **20%**, as they are primarily memory bandwidth bound."
- Block compile overhead: naive block compilation puts "only **~50%** of execution time … inside the actual inductor code produced for the block"; the rest is dynamo guards/shape counting. Guard elision via `nested_compile_region` (torch 2.8) keeps guards only on the first layer of an Isotropic — "highly unsafe".
- External modules: "close to **40%** of execution time", producing "obvious compute bubbles".
- Conclusion: `S2-small.yaml` is "primarily CUDA bound" not compute bound; "**~60% non-matmul time per mamba block**"; "**~2/3** total time spent in mamba/dechunk → **~40%** non-matmul time in H-Net for S2-small". MFU "disappointingly low relative to a standard transformer".
- Optimizer CPU time lands at "**~10%** of each train step" after dion + foreach; groups are keyed by `optim`, `mup` type and `hierarchy` (LR-modulation per stage).
- Earlier MFU graphs "were also simply wrong due to drastic overestimation of the inner hierarchy's sequence lengths". No FLOPs or throughput figures appear beyond these percentages; the profiler images show kernel names (`ampere_bf16_s1688gemm`, `rms_norm_bw`, `_chunk_state`, `_chunk_scan`, `causal_conv1d`, `triton_poi*`) but no timings.

## 5. Inference / generation

Two-phase: `Prefill` = `.forward()`, `Decode` = `.step()`. Input is UTF-8 bytes with **`0x7E` as BOS** through `nn.Embedding`; no final norm (norms live in `Isotropic`).

Prefill, stage 0 (`hnet_1stage_XL`): encoder (Mamba2) mutates `x` and caches SSM state → routing computes `q_t,k_t` by matmul, `cossim = F.cosine_similarity(q[:,:-1], k[:,1:], dim=-1)`, sets `p_0 = 1.0`, `p_{t+1} = (1-cossim)/2`, `b = p >= 0.5`, caches last `x_t`, returns `boundary_prob=[1-p,p]`, `boundary_mask=b`, `selected_probs=max(1-p,p)` → `chunk_layer` drops tokens (`x.masked_select(b)`) → inner transformer caches KV → dechunk takes `P = p.masked_select(b)`, "borrow[s] mamba parallel scan kernel to quickly obtain EMA `z̄`", computes `z̃ = out.gather(dim=1, index=b.cumsum(dim=1)...)`, caches `last_value` → `h = residual_proj(x) + h` → decoder.

Decode: encoder reads/updates `ssm_state`/`conv_state`; routing uses cached `x_{t-1}` to produce `p_t,b_t` and writes `x_t`; `h` **may be empty**, which "fully bypass[es] the inner s=1 H-Net" — "H-Nets implicitly implement spec decode / adaptive compute". Dechunk returns `z̃_{t-1}` when empty, else `z̃_t = P_t ẑ_t + (1-P_t) z̃_{t-1}`; residual becomes `z̃_{t-1} + W_r x̂` when the main net is skipped.

Cache structure (per batch): KV cache `L×N_h×D_h`; Mamba2 `conv_state` of shape `[conv_dim, d_conv]` plus `N_h×D_h×d_state` `ssm_state`; `routing_module.last_hidden_state` (one vector) + `has_seen_tokens` flag (redundant under `generate.py`); `dechunk_layer.last_value` (EMA `z̄_t`). For 1stage_XL: `d_conv=4`, `d_state=128`, `expand=2`, `d_inner=expand*d_model=2048`, `D_h=64 ⟹ N_h=32`, `conv_dim = d_inner + 2*d_state = 2304`; transformer `D=2048 ∧ N_h=16 ⟹ D_h=128`. Everything else is "purely functional". Note the pdb dump prints `Embedding(256,1024)`/`lm_head` 1024→256, i.e. `d_model=1024`, which contradicts the `D=2048` in the cache breakdown.

## 6. Gotchas and bugs

1. Autotune dependency: kernels size `y` by `cdiv(seqlen, min(BLOCK_SIZES))`, then read `kernel.best_config.kwargs[BLOCK_SIZE]` — absent at dynamo trace time ("does not execute any kernels"), so compilation explodes. Fix: "sum over all blocks regardless of block size".
2. `pre_hook` (used to zero inputs during autotuning) is unsupported by torch.compile (pytorch#139059); `reset_to_zero` is — "sufficient to permit compilation for all mamba kernels".
3. Inductor inlines user kernels but "forgets to import `math`"; manual substitutes needed.
4. Inductor "unhappy if `reset_to_zero` tensors are only sometimes `None`" → redundant tensors.
5. torch confuses ordering of named vs parametric triton args → sort them.
6. All `Isotropic` layers subclass one `Block`, so every `Block.forward` shares a code object/hash; dynamo then sees one function with mamba-or-transformer behavior and dynamically varying hidden dim. `S=2` nets "are broken" ("hitting recompile limits on dynamic shapes"); `S=0` fine; `S=1` works only "by the undefined behavior of pytorch 2.7". Fix: metaclass copying `.forward` into a new `FunctionType` with a different hash.
7. Fusing q/k/res into one matmul blocks the D2H overlap trick (`stats_gpu = torch.stack([cu_inner[-1], cu_inner.diff().max()])`, pinned `stats_cpu`, `torch.cuda.Event`), so he keeps three linears.
8. FSDP2 "~doubles CPU overhead" → train 1 GPU.
9. Multi-turn bug: a fresh user-message prefill has no `p_0=1.0`, so `boundary_mask.any()` is not guaranteed and short messages can select 0 tokens in deeper hierarchies. Workarounds: pad prefill batches, or force `p_t=1` for all special tokens.
10. Quality: `stage1_L` opens a new chunk "almost every letter" on `AAAA…`; `stage1_XL` "sometimes 2char per chunk"; `stage2_XL` "completely flipped" to chunks after huge repeat spans.
## 7. Implication for a T4 / no-extension build (derived, not the author's claim)

`mamba_split_conv1d_scan_combined` and `causal_conv1d` are load-bearing (the copy pathology alone costs ~20% of small-layer time); rope/rmsnorm are already torch reimplementations, so those are free. The one genuinely flash-attn-shaped dependency is `flash_attn_varlen_func` in `xf.py`, replaceable by a `cu_seqlens`-padded SDPA loop at the cost of materializing padding. Compile gains are optional; eager correctness holds.
