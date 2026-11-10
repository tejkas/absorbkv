# Next GPU session

What is queued for a rented instance, and the setup that is known to work.
Everything here needs real weights; anything runnable from `captures/`
belongs in `experiments/` and runs on the laptop.

## Setup, known good as of 2026-10-15

From the laptop. `scores.py`, `experiments/` and `RESULTS.md` are still
untracked, so a clone of the remote does not have them:

```bash
rsync -av --exclude .venv --exclude captures --exclude weights \
      --exclude __pycache__ --exclude .pytest_cache \
      ~/Documents/absorbkv/ ubuntu@<IP>:~/absorbkv/
```

On the instance. **Do not use `--system-site-packages`**: the image's
torch predates `torch.distributed.checkpoint.hf_storage`, which
`transformers>=5` imports, and its Pillow predates the 9.1 that
`PIL.Image.Resampling` needs. Both cost a round trip.

```bash
cd ~/absorbkv
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch 'transformers>=5' accelerate datasets pillow
pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

export HF_HOME=$HOME/hf          # 31 GB; keep it off the root disk
export HF_TOKEN=<token>
hf download deepseek-ai/DeepSeek-V2-Lite
```

DeepSeek-V2-Lite is 15.7B parameters, about 31 GB in bf16. A 40 GB card
is enough for everything run so far.

## Queued

**1. End-task quality.** The one gap that changes what the write-up can
claim. Every relevance number so far is offline scoring against stored
captures: share of ideal attention mass, no generation, no attention
kernel. An accuracy number needs the `k^R` selector inside a real forward
pass, which is new code rather than a new measurement — and by the
laptop-first rule, a CPU correctness test against the captures comes
before any rented hour. Dataset is undecided; MuSiQue F1/EM is the
cheapest since `data.py` already loads it.

**2. Measured memory bandwidth.** `RESULTS.md` claims the `k^R` selector
halves the cache reads *the paper's cost model charges*. That is an
element count, not bytes that moved. Hardware moves cache lines, so
whether reading 64 of 576 dimensions costs a ninth depends on how `k^R`
sits relative to the latent, on L2 hit rate, and on whether the gather
coalesces.

Two things to settle before designing it, both laptop work:

- *The page-summary hole.* Step 2 reads per-page maxima and the
  accounting never says where they come from. Precomputed at prefill
  means storage and write traffic nobody has counted; computed on the fly
  means reading the whole page and losing the saving. Check
  `papers/rocketkv/code` for which one they do. If it is the second, the
  0.35t step-2 figure is partly an artifact of our own accounting.
- *What counter to read.* `nvidia-smi` will not answer this; it needs
  DCGM or Nsight Compute DRAM counters.

**3. A real decode state far from the context.** The query-position sweep
holds content fixed by construction and the stored positions span 26
tokens. A token genuinely generated far later differs in content *and*
position. Needs generation, so it rides along with item 1.

**4. 128-head models (V3/V3.2).** Every head-packing result is from a
16-head model. V3 is 671B and will not fit on one card, so this is a
different kind of session, not a longer one.

## Not GPU work

- RocketKV stage 1 (eviction) on MLA — never measured, runs from
  `captures/`.
- Coordinate sets below 64, i.e. subsets of `k^R` — the "no floor above
  65 of 576" claim never tested below the RoPE block.
- Document-level spread — never reported and not recoverable from the
  current result files.
- Folding `experiments/` into `report.py` (D14), and adding `--out` to
  those twelve scripts. The fixed-output-path habit has already cost
  layer 0 of the attention check once and nearly cost the 60-document
  `results/` directory.

## Carry up next time

The instance still runs the pre-fix `verify_against_model.py`, which
writes a fixed path. Rsync the script up before re-running it, or pass
`--out`.
