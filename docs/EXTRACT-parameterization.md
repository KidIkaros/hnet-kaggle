# main-horse H-Net series: parameterization & training dynamics (Parts 1 & 4)

Sources: `/hnet/param-s0/` (Part 1, S=0), `/hnet/param-s1/` (Part 4, S≥1), `/hnet/inf/` (Part 0, code semantics). Author's caveat: "it is unlikely all claims in this article are true."

## 1. S=0 baseline (Part 1): spectral init + dualized Muon

Goal: "a scheme which outperforms a basic Adam baseline, while permitting hparam transfer across widths." Feature learning needs, at init:

1. $\|\mathbf h_\ell\|_2 = \Theta(\sqrt{n_\ell})$
2. $\|\mathbf W_{\ell}\|_* = \Theta(\sqrt{\frac{n_\ell}{n_{\ell-1}}})$

(1) "implies embedding && rmsnorm weights should be $\sigma\propto 1$." For (2) he uses Su's approximation for **all** linear init:

$$\sigma_k = \mathcal O(\sqrt{\frac{min(1,d_k/d_{k-1})}{d_{k-1}}})$$

Update side needs $\|\Delta\mathbf h_\ell\|_2 = \Theta(\sqrt{n_\ell})$, delivered by the **dualizing variant of Muon** (keeps singular vectors, discards singular-value magnitudes), giving $\|\Delta\mathbf W_\ell\|_* = \Theta(\sqrt{n_\ell/n_{\ell-1}})$ "so long as you correctly split any fused parameters."

| Component | Init | Optimizer / LR |
|---|---|---|
| Embedding | $\sigma = 1$ | Adam, $\eta_\text{emb}\propto 1$ (bias-like: updates shift outputs by $\mathcal O(1)$ per element) |
| LM head | not covered by `base_std` | Adam, $\eta \propto 1/d$ |
| Hidden matrices | $\sigma_k$ × swept `base_std` | Muon (dualizing) |
| RMSNorm γ (all) | ones | **LR = 0**, non-trainable (Lingle) |
| `pad_dimension` | zeros (Goombalab) | "adam bias" |
| Softmax scale | $1/d_{head}$ — "my scheme transfers better with it" | — |

`base_std` "is only applied for hidden matrices... not used for lmhead/emb, or for parameters that do not use `normal_` init (norm weights)."

**Mamba2** (no published muP+Muon rules). $A_{log}, dt_{bias}$ follow mamba2's bespoke rules ($dt_{bias}=\text{softplus}^{-1}(e^r), r\sim U(\ln 0.001,\ln 0.1)$; $A\sim U(-16,-1)$); $D=\mathbf 1$; $W_{conv}$ has $\sigma\propto 1/\sqrt{k}$, zero conv bias. `in_proj` is "an abomination" producing five activations ($z_t,x_t\in\mathbb R^d$, $B_t,C_t\in\mathbb R^n$, $dt_t\in\mathbb R^h$) and "should be interpreted as a fused split of 5 matrices: `.split([d_ssm, d_ssm, d_state, d_state, self.nheads], dim=-2))`".

**Why the fused split matters.** Dualized Muon gives $\text{RMS}(\Delta_\text{muon}) \propto \sqrt{\frac{\text{fan\_out}}{\text{fan\_in}}}\sqrt{\frac{1}{\max(\text{fan\_in},\text{fan\_out})}}$, so $\text{RMS}(\Delta W_{z,x})\propto 1/\sqrt{d_{model}}$ but $\text{RMS}(\Delta W_{B,C})\propto 1/d_{model}$. Symptom: "the Muon LR clearly experiences a strong downwards drift, from $\eta_\text{muon}\approx 0.11 \rightarrow 0.04$ ($d=256\rightarrow 2048$)." Fix that worked: rescale $\eta_{z,x}$ by $1/\sqrt{d_{model}}$, i.e. per-slice LRs $\left[\frac{1}{\sqrt{d_{model}}}, \frac{1}{\sqrt{d_{model}}}, 1, 1, \frac{1}{\sqrt{d_{model}}}\right]$ or head-wise Muon. Growing $\eta_{B,C}$ failed. "$\eta_{z,x}$ rescaling is necessary – muon LR still drifts if *only* $\eta_{dt}$ is rescaled."

**Exponent blow-up.** Treated as plain μP adam biases, SSM params get $\text{RMS}(\Delta)\approx\eta$; $A$ and $dt_{bias}$ are **exponents** and "will quickly exceed the reasonable representable fp32 range (approx $log(A)<-50$ or $dt_{bias}>100$)" since "optimal base LR can even be higher than 1." Fix, self-described as "incredibly unprincipled": one group for $A_{log}, dt_{bias}, D, W_{conv}$, `conv1d.bias`, LR downscaled by $2^6$ relative to lmhead/embedding adam LR.

**Non-parametric norms.** Justified because norm outputs are "immediately fed to a linear layer" and γ is "mergable with its subsequent projections." Trainable γ at $\eta_\gamma\propto 1$ "causes optimal adam LR to quickly decrease with increasing dim," and keeping only the final output norm trainable still diverges. Mechanism: non-trainable γ with $\eta_W\propto 1/d,\sigma_W\propto\sqrt{1/d}$ gives $\text{RMS}(\Delta W')=\mathcal O(1/d)$; trainable γ with $\eta_\gamma\propto 1,\sigma_\gamma=1$ gives $\mathcal O(1/\sqrt d)$, so "$\Delta\mathcal L$ [is] dependent on $d$ and blows up μtransfer." Escapes ($\sigma_W\propto 1/d$, or $\eta_\gamma\propto 1/\sqrt d$; not $\eta_\gamma\propto1/d$, which drifts muon LR) don't generalize to muon-trained norm+linear stacks.

**Verification:** sweep $(\eta,\sigma)$ at $d=512$, re-sweep at other widths. Honest caveat: "**This is not fully correct**. There is a non-negligible drift towards smaller LRs on both axes (though within ~$\sqrt 2$) from 512→2048."

## 2. S≥1 (Part 4)

H-Net's $\lambda^s = \sqrt{N^\text{GPT}\frac{\prod^S_{i=s}N^i}{\prod^S_{i=0}N^i}\frac{D^S}{D^s}}$ "is just the application of 2 heuristics": $\eta\propto\frac{1}{\sqrt D}$ and $\eta\propto\sqrt{\text{batch size}}$. He **removes $\frac{D^S}{D^s}$** "as the μP setup accounts for it"; $N^\text{GPT}$ is irrelevant when LR is swept.

| Extra param | Init | Optimizer |
|---|---|---|
| $W_q, W_k$ | **identity** — "$\|\mathbf I_d\|_* = 1 = \Theta(\sqrt{n_\ell/n_{\ell-1}})$, so $W_q,W_k$ satisfies the spectral condition with no effort" | Muon |
| `residual_proj` $W_\texttt{res}$ | **zeros** (bias zeros) | Muon |
| `pad_dimension` | zeros | adam bias LR |
| Norm scales | ones | 0 LR, **or** adam-bias LR *only for residual-contributing isotropics* |

**Zeros for $W_\texttt{res}$:** "the main network is forced to learn to improve $x_\texttt{dec}$, suppressing easy first-step gains from encoder byte-level prediction improvements alone." With kaiming uniform "the network never encounters the initial 'hockey-stick' ratio loss spike in favor of the main network"; $\sigma\propto 1/\sqrt d$ "should be similarly problematic." Zeros violate $\|\mathbf 0\|_* = 0$, but $\|x_\texttt{dec}\|_2=\Theta(\sqrt{d_s})$ holds "due to the main net branch."

**Norms are the crux.** "RMSNorms in H-Net do not serve the same purpose" as in a transformer; absorbability into linears "is **not true** once you start adding hierarchies." Each isotropic output norm does 3 jobs: final $\text{LMHead}(\text{RMSNorm}(x))$; encoder output (routing stability, $\Theta(1)$ hierarchy inputs, "ensuring only $W_\texttt{res}$ decides how much the encoder contributes"); and stabilizing the residual into $x_\texttt{dec}$, where

$$x_\texttt{dec} = y_\texttt{enc}W_\texttt{res} + \text{dechunk}(\text{M}(y_\texttt{enc}\texttt{.masked\_select(...)})$$

"If $\text{RMS}(W_\texttt{res})$ grows, it is not possible for the outputs of the main net to grow its own RMS to match. Most likely, the main net will start painfully producing gigantic outlier coordinates to pass information out, which will destroy models in the long run." Hence trainable output norms, $\eta_\gamma\propto 1$, "to match the update speed of $W_\texttt{res}$."

**Routing gradient.** With $\text{cossim}(a,b)=\frac{a\cdot b}{\|a\|_2\|b\|_2}$ and $p_t=\frac{1-\text{cossim}(W_qx_{t-1},W_kx_t)}{2}$: $\|x_t\|=\Theta(\sqrt d)$ ⇒ $\|q_t\|=\|k_t\|=\Theta(\sqrt d)$; unaligned ⇒ $q_{t-1}\cdot k_t=\Theta(\sqrt d)$; so $p_t=\Theta(1/\sqrt d)$ and $\|\frac{\delta p_t}{\delta k_t}\|_2=\Theta(1/\sqrt d)$, giving $\Delta p_t=\Theta(1)$ **iff** $q_{t-1},k_t$ "remains mostly unaligned across widths." That premise is "self-defeating": a real dynamic tokenizer *should* shift alignment with capacity, so "**By design**, any μP scheme that assumes consistent alignment between sequence positions across widths is not going to work for a H-Net."

## 3. Failure modes, with symptoms

- **Compression collapse at init.** "$\uparrow\sigma \implies \downarrow \text{L1:L0}$ at init": high σ + prefix-scan/conv1d creates correlation out of the box, "bringing the starting compression ratio to a negligible value, which generally never goes back up. **This dooms all high σ runs**."
- **Short runs mislead.** "it takes 500+ steps for ratio loss to stabalize close to 1"; at 1000-step budgets "the **majority** of your run's updates will be dominated by the starting dynamics of routing stability." "traditional H-Net hparam sweeping is misleading if done with $\text{steps} < 5000$."
- **No fixed batch size.** "It is **not possible** to make a traditional H-Net consume the same number of tokens per step"; routing shifts butterfly-effect inner-net tokens.
- **LR reshapes the hockey stick.** "lower muon LR -> slower $W_{q,k}$ updates -> longer dragged hockey stick (which is lowest first valley & highest first peak)"; higher adam LR too.
- **Hardcoded ratio** (kth-value threshold) stabilizes LR but "the results are fucked – loss increases with width"; breaks causality.
- **Aux-free threshold** (Deepseek-style: $u=10^{-3}$, $\beta=0.95$, clamp $[.1,.9]$): "**No**" — bad runs push $p_t$ outside the clamp; bf16 saturation makes routing volatile.
- **Balanced $\alpha$** (lerp 0.95, step 0.01, clamp $[0.01,100]$): "does not improve results (1e17 also crashed a bit)."
- **Diagnostics:** ratio loss "substantially above 1" = network dying; "far below 1 (lower bound $2(N-1)/N$)" = "trying very hard to shut down the main network entirely."

**The one intervention that worked:** cosine-decayed kth-value threshold — `ratio=(N-1)/N`, `kth=p_flat.kthvalue(int(ratio*p_flat.numel()))`, `w=cos(pi*step/cutoff/2)`, `t=kth.lerp(0.5,1-w)`, all-reduced — with a "fake" $b_t$ (`p_flat >= .5`) for ratio loss/STE and a "real" $b_t$ (`p_flat >= self.thresh`) for chunking. Flattens the hockey stick and fixes step-0 VRAM spikes, but "lower muon LR -> slower ascent of compression to target 0.2" still kills transfer. Step-0 init:

$$\texttt{thresh} = (1+\frac{\Phi^{-1}(1 - 1/N)}{\sqrt{d}})/2$$

$\Phi^{-1}$ is `ndtri`; it "will either under/overshoot the target, depending on your encoder's init std."

## 4. Code-level definitions

`RoutingModule`: `k_flat=self.k_proj_layer(r_flat)`; `q_flat=QProjPadded.apply(r_flat, self.q_proj_layer.weight, k_flat, r_cu)`; `cos_sim=F.cosine_similarity(q_flat,k_flat,dim=-1)`; `p_flat=(.5-cos_sim/2).clamp(.0,1.)`; `b_flat=p_flat>=0.5`; `p_select_cu=F.pad(b_flat.cumsum(0),(1,0))[r_cu]`. Part 0: `cossim=F.cosine_similarity(q[:,:-1],k[:,1:],dim=-1)`, $p_0=1.0$, $p_{t+1}=\frac{1-\text{cossim}}{2}$, `b=p>=0.5`, returning `boundary_prob=[1-p,p]`, `boundary_mask=b`, `selected_probs=max(1-p,p)`; caches the last $x_t$ for $k_{t-1}$ at decode.

`dechunk_layer`: `P=p.masked_select(b)`; "borrow mamba parallel scan kernel to quickly obtain EMA $\bar z$ from $\hat z,P$"; $\tilde z$ via `out.gather(dim=1,index=b.cumsum(dim=1)...)`; caches `last_value`. "The dechunk EMA shrinks variance a bit on average, but not by much" — so $y_\texttt{enc}W_\texttt{res}$ and the main-net branch are near-equal in magnitude, which is why $\text{RMS}(W_\texttt{res})$ growth is fatal.

## 5. Settings and recommendation

`betas=[.9,.95]`, `eps=1e-6`, no wd, no grad clipping, no EMA, WSD schedule, fp32 master weights + amp bf16; sweep only `adam_lr`, `muon_lr`, `base_std` (log2 grids); base sweep at $d=512$, target transfer $D=256\rightarrow4096$. mamba2 `d_conv=4`, `d_state=128`, `expand=2`, `ngroups=1` held const — "**Do not expect transfer** if you modify them." Muon `nesterov=False` (accidental): "basically no difference."

Verdict: "1. End-to-end H-Nets are a poor fit for μP. 2. Although ugly, separate 'pretrained tokenizers' will work." The "repulsive recommendation": "train a reasonable enc/dec at small scale, and recycle it as a 'pretrained tokenizer' for use in larger runs; possibly with learning enabled" — load all weights above a level (incl. $W_\text{res},W_\text{emb},W_\text{out}$), freeze/gently warm up, sweep at $D=[1024,1024]$ with threshold tracking, then expand $D_1$. That attempt still drifted because "I used a parametric output norm, which predictably causes leftwards $\eta_\text{adam}$ drift"; fix: "replace the norm scale with a down projection instead."
