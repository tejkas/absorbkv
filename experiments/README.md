# experiments

Thirteen measurement scripts, first written on 2026-10-11, in the order they
were run. Each writes one `results/*.json`. Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python experiments/<script>.py 60
```

| script | result | question |
|---|---|---|
| `remedy_sweep.py` | `remedy_fidelity.json` | Does selection survive at the kept fraction the traffic budget demands? |
| `knee_sweep.py` | `knee_fidelity.json` | How far can step 3's coordinate set shrink before selection degrades? |
| `logit_split.py` | `logit_split.json` | Which of the two terms in `q.k` carries the ranking of context tokens? |
| `pagemax_split.py` | `pagemax_split.json` | Is the latent's weak selection its own fault, or the page summary's? |
| `recency.py` | `recency.json` | Is this capture's attention just recency? |
| `rope_only.py` | `rope_only.json` | Scoring `k^R` alone: traffic cost against selection quality. |
| `temperature.py` | `temperature.json` | Does each variant's selector run at a different softmax sharpness? |
| `sharpness.py` | `sharpness.json` | What spread does each variant's page scores actually reach? |
| `sharpness_matched.py` | `sharpness_matched.json` | The same comparison with the spread forced to the model's own. |
| `query_position.py` | `query_position.json` | Does any of this depend on which query token, or how far away? |
| `shuffle_rotation.py` | `shuffle_rotation.json` | Content or rotation? Permutes content among positions. |
| `sink_conditioned.py` | `sink_conditioned.json` | Does `k^R`'s lead survive giving the attention sink away? |
| `verify_against_model.py` | `model_attention_layer*.json` | Does scoring from the latent reproduce the model's own attention, on real weights? (GPU) |

`_mirror.py` is not a measurement: it exposes the selector's intermediate page
scores, with an assertion that the mirror still matches the selector.

**Status: provisional.** These sit outside `report.py` and each one repeats its
loading boilerplate -- weights, config, `grams`, the document and layer loops,
`exact_scores`. Folding them in belongs with whoever owns `report.py`, and
until that happens RESULTS.md's opening claim that every number is regenerated
by `report.py` is true of nine results, not twenty-two.

They also write to fixed output paths with no flag to redirect them, so a short
run silently overwrites a long one. `verify_against_model.py` has `--out`; the
other twelve do not, and the missing flag has already destroyed one result.

`remedy_sweep.py` is superseded by `rope_only.py`: it measures the fix that
rations the latent's coordinates, which turned out to be a fix to the wrong
half. It is kept because the number it produced is cited in RESULTS.md and
LOG.md as the step that led to the rest.
