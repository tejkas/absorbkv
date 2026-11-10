"""Steps 1-4 of ``select_latent``, exposed so the scores can be inspected.

``select_latent`` returns indices, so the intermediate page scores are
not reachable from outside it. Two measurements need them: the softmax
spread each variant actually produces, and a per-head divisor chosen to
make that spread match the model's. Rather than duplicate the selector in
each script, it is mirrored once here and ``check_mirror`` asserts the
mirror still reproduces the selector's own output.
"""

import math

import torch

RANK = 512


def page_scores(query, key, ratio, reduce_head_dim):
    """Returns ``(scores, natural_divisor, page_size)``.

    ``query`` and ``key`` are the concatenated forms, content first. The
    ``RANK`` split is what makes the reduction keep the RoPE coordinates,
    matching ``select_latent``, which takes the rank from the latent's
    width rather than the concatenated width.
    """
    tokens, width = key.shape
    page = math.ceil(math.sqrt(ratio))
    page = page if page <= ratio else 1
    kept_fraction = page / ratio

    sign = torch.where(query.sum(dim=0) > 0, 1.0, -1.0)
    positive, flipped = query * sign, key * sign
    padding = (-tokens) % page
    if padding:
        flipped = torch.cat(
            [flipped, torch.full((padding, width), torch.finfo(flipped.dtype).min)]
        )
    pages = flipped.view(-1, page, width).amax(dim=1)

    if reduce_head_dim:
        kept = max(1, int(RANK * kept_fraction))
        chosen = query[:, :RANK].abs().mean(dim=0).topk(kept).indices
        coords = torch.cat([chosen, torch.arange(RANK, width)])
    else:
        coords = torch.arange(width)

    scores = positive[:, coords] @ pages[:, coords].T
    natural = torch.sqrt(
        width
        * positive[:, coords].abs().sum(dim=-1, keepdim=True)
        / query.abs().sum(dim=-1, keepdim=True)
    )
    return scores, natural, page


def check_mirror(select_latent, qc, qr, latent, k_rope, ratio, reduce_head_dim, keep):
    """Assert the mirror reproduces the selector. Call before reporting.

    The split between ``qc`` and ``qr`` matters: handing a concatenated
    576-wide block as ``qc`` would make ``select_latent`` reduce over all
    576 coordinates instead of the 512 content ones.
    """
    query = torch.cat([qc, qr], dim=-1)
    key = torch.cat([latent, k_rope], dim=-1)
    scores, natural, page = page_scores(query, key, ratio, reduce_head_dim)
    weights = torch.softmax(scores / natural, dim=-1)
    per_token = weights.repeat_interleave(page, dim=-1)[:, : key.shape[0]]
    mine = per_token.mean(dim=0).topk(min(keep, key.shape[0])).indices
    theirs = select_latent(qc, qr, latent, k_rope, keep, ratio, reduce_head_dim)
    assert set(mine.tolist()) == set(theirs.tolist()), "mirror diverged"
