# absorbkv

**KV-cache staleness and token-selection signals computed in DeepSeek MLA's
latent, without rebuilding per-head keys.**

Most KV-cache management techniques rank tokens by a quantity computed over
each attention head's key (how far the key moved, or how much attention it
would receive). Multi-head Latent Attention never materializes those keys. Per
token and layer it stores a 512-dim latent `c` and a 64-dim RoPE key `k^R`
shared by every head, 576 numbers against Mistral-7B's 2,048, and the query is
folded through the up-projection ("absorption") so it scores the latent
directly:

```
q · k_t  =  q' · c_t  +  q^R · k^R_t        q' = W^UKᵀ q
           512-dim        64-dim
```

This repository measures, on real weights, whether two such techniques can be
computed in that latent space and whether the token rankings they depend on
survive. **CacheBlend** (cache reuse, ranks by key deviation) and **RocketKV**
(decode-time token selection, ranks by an estimated attention score).

The write-up is in https://tejkas.github.io/blog/mla-kv-signals/. Every
number in it is in
[`RESULTS.md`](RESULTS.md) with its provenance.

## Layout

```
src/absorbkv/
  signals.py        staleness scores: S_exact, S_kv, S_row, S_naive
  scores.py         q·k from the latent without rebuilding keys
  selection.py      RocketKV's selection step ported to MLA; Budget derivation
  concentration.py  query concentration before and after absorption
  capture.py        tensors both halves need, from one forward pass
  runner.py         drives the capture, one file per document
  data.py           MuSiQue documents in the capture's shape
  report.py         regenerates results/*.json from the captures
tests/              98 CPU tests, no model needed
experiments/        thirteen one-off measurements not yet in report.py
results/            every measured number, one JSON per measurement
papers/rocketkv/    RocketKV's selection code, lifted, under its own licence
blog/               the write-up and figures.py, which draws the SVGs
RESULTS.md          the claims ledger: predicted, reported earlier, final
```

## Reproduce

```bash
pip install -e ".[dev]"                     # CPU tests
pip install -e ".[dev,capture,report]"      # plus a real model
PYTHONPATH=src python -m pytest -q
```

Capture, on a GPU able to hold DeepSeek-V2-Lite in bfloat16 at TP=1 (tensor
parallelism shards `kv_b_proj` across heads):

```bash
PYTHONPATH=src python -m absorbkv.runner --out captures/mla
PYTHONPATH=src python -m absorbkv.runner --out captures/gqa \
    --model mistralai/Mistral-7B-Instruct-v0.2
```

`report.py` also expects the per-layer `kv_b_proj` weights saved as
`weights/kv_b_proj.pt`, a dict with key `kv_b_proj`. Then:

```bash
PYTHONPATH=src python -m absorbkv.report --documents 60 --out results
PYTHONPATH=src python experiments/<script>.py 60     # the thirteen others
```

Captures (25 GB) and weights are not in the repository. `results/` is.

## Related work

- **CacheBlend** — Yao et al., EuroSys 2025. [arXiv:2405.16444](https://arxiv.org/abs/2405.16444)
- **RocketKV** — Behnam et al., ICML 2025. [arXiv:2502.14051](https://arxiv.org/abs/2502.14051)
- **Quest** — query-aware sparsity via page bounds
- **SparQ** — coordinate reduction on the head axis
- **EPIC** — position-independent context caching. [arXiv:2410.15332](https://arxiv.org/abs/2410.15332)
- **Prompt Cache** — Gim et al., MLSys 2024. [arXiv:2311.04934](https://arxiv.org/abs/2311.04934)

## License

Apache-2.0, except `papers/rocketkv/`, which is adapted from
[`NVlabs/RocketKV`](https://github.com/NVlabs/RocketKV) and stays under the
NVIDIA License there: research and evaluation use permitted, derivative works
non-commercial.
