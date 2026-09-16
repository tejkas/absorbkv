# absorbkv

**Token selection that stays inside MLA's absorbed latent.**

Several KV-cache algorithms rank tokens by a quantity computed from per-head `K`.
Multi-head Latent Attention never materializes per-head `K` — it stores a low-rank
latent plus a decoupled RoPE component, and folds `W^UK` into `W^Q` at inference
("absorption"). Reconstructing `K` in order to rank tokens repays exactly what MLA
saves, so these algorithms are currently either unavailable on MLA models or
unvalidated there.

This is an instrument for one question:

> Can quantities defined over per-head `K` be computed in MLA's latent space, and
> do they preserve the rankings the algorithms depend on?

Two questions, one rig.

| | question | score under test |
|---|---|---|
| **staleness** | which cached KV is *wrong*? | KV deviation, as in CacheBlend |
| **relevance** | which cached KV *matters* for this query? | page min/max bounds + head-dim reduction, as in RocketKV |

A learned sparse indexer (DSA, in DeepSeek-V3.2/V4, GLM-5.x) already answers
**relevance** for models that ship one. It answers **staleness** for nobody: a
token whose KV was computed without seeing its neighbours is wrong however
relevant it is.

## Staleness — the identity

Per-head key in MLA is a concatenation of a head-specific content part and a RoPE
part shared across all `H` heads:

```
k_{t,i} = [ W^UK_i c_t ; k^R_t ]
```

`W^UK` is linear, so the deviation passes through it, and squared norms add over a
concatenation. Summing over heads — the shared term counted `H` times:

```
Σ_i ‖Δk_i‖²  =  Δcᵀ M Δc  +  H‖Δk^R‖²        M = (W^UK)ᵀ W^UK
```

`M` is `[kv_lora_rank, kv_lora_rank]` (512×512 for DeepSeek) and is built once per
layer. Cost is `O(T·512²)` rather than `O(T·512·H·128)` — the Gram form is capped
by the **rank**, not the expansion, so the saving grows with head count (4× on
DeepSeek-V2-Lite, 32× at 128 heads). Absorption is never broken.

Using the unmodified `kv_b_proj` instead of the extracted `W^UK` gives
`BᵀB = M_K + M_V`, scoring K **and** V deviation at identical cost — and immune to
the weight's row packing, since `BᵀB = Σ_j b_j b_jᵀ`.

Two things this rests on, both checked rather than assumed:

- **Linearity needs the post-`kv_a_layernorm` latent.** RMSNorm divides by the
  vector's own norm, so with a pre-norm latent `Δk ≠ W^UK Δc` and the identity
  collapses.
- **`M = BᵀB` squares the condition number.** Build and evaluate in fp32 or
  higher; in `bf16` the residual blows up in a way that reads as broken algebra.

RoPE does not appear anywhere in the harness. Rotation is orthogonal, so
`‖Rx − Ry‖ = ‖x − y‖`: the deviation is invariant to any rotation applied
*identically to both sides*. Capture pre-RoPE on both passes and the positional
machinery — YaRN scaling, GPT-J vs NeoX layout, strided offsets — drops out.

## Relevance — the open question

The page-bound reduction transfers to the latent cleanly, because the score is
linear in `c`:

```
q · k_t  =  q' · c_t  +  q^R · k^R_t        q' = W^UK ᵀ q
```

So keep elementwise min/max over `c` (512) and `k^R` (64) per page instead of over
per-head `K`. And because `c` is head-independent, **one page representation
serves every head** — roughly 28× less page metadata than the per-head version at
128 heads.

The head-dimension reduction is the open question. It works by keeping only the
largest coordinates of `|q|`, which assumes the query's energy is concentrated in
a few coordinates. In latent space the query is `q' = W^UKᵀ q` — a dense learned
map. When `q` is spiky, `q' ≈ q_max · (that column of W^UK)`: the spike is spent
selecting a column, and a column of a trained dense matrix is a generic spread-out
vector. Whether concentration survives the change of basis is an empirical
question about trained weights, and it is the cheapest experiment here.

## Status

| | state |
|---|---|
| `signals.py` — staleness scores | complete; 22 CPU tests passing |
| `capture.py` — hooks, two passes | not started |
| `scores.py` — relevance scores | not started |
| `report.py` — rank agreement, sweeps | not started |

## Install

```bash
pip install -e ".[dev]"              # CPU identity tests only
pip install -e ".[dev,capture,report]"   # plus a real model
```

The CPU tests need nothing but `torch` and run on a laptop. Capture needs a GPU;
the reference configuration is a single A100/H100 80GB at TP=1 with
`deepseek-ai/DeepSeek-V2-Lite`. TP=1 matters: tensor parallelism shards
`kv_b_proj` across heads, so per-rank `M` is partial. (In production that is not a
problem — `M` is additive over head shards, so ranks all-reduce one scalar per
token.)

## Related work

- **CacheBlend** — Yao et al., EuroSys 2025. [arXiv:2405.16444](https://arxiv.org/abs/2405.16444)
- **RocketKV** — Behnam et al., ICML 2025. [arXiv:2502.14051](https://arxiv.org/abs/2502.14051)
- **Quest** — query-aware sparsity via page bounds
- **SparQ** — single-dimension reduction on the head axis
- **EPIC** — position-independent context caching. [arXiv:2410.15332](https://arxiv.org/abs/2410.15332)
- **Prompt Cache** — Gim et al., MLSys 2024. [arXiv:2311.04934](https://arxiv.org/abs/2311.04934)

The relevance side implements the method for MLA, which the reference implementation at
`NVlabs/RocketKV` does not cover. That repo is under the NVIDIA License: research
and evaluation use is permitted, and derivative works carry a non-commercial
restriction (§3.3, with NVIDIA exempted). Any code adapted from it is kept in
clearly marked directories under those terms.

## License

Apache-2.0.
