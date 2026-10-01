# SPDX-License-Identifier: Apache-2.0
"""Capture the tensors both halves of the measurement need, in one GPU pass.

MLA's native cache already holds what the deviation identity needs: the
post-``kv_a_layernorm`` latent and the decoupled RoPE key, so the context
capture reads ``past_key_values`` rather than hooking per-head K, which is
both ~5x smaller and immune to changes in the attention module's internals.

Queries have no cache entry and are captured by hook. The absorbed query
``q' = W^UK^T q`` is not captured at all. HF never computes it (there is no
absorption in the HF path), and at ``[num_tokens, num_heads, kv_lora_rank]``
it is 14x the latent to store. It is derived at analysis time instead.
"""

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class ContextCapture:
    """One pass over a span of context tokens: what a cache would hold.

    The span is a whole document on the truth pass, and a single chunk on
    the pass that models an independently encoded one.

    Attributes:
        latent: Post-``kv_a_layernorm`` latent as cached, shape
            ``[num_layers, num_tokens, kv_lora_rank]``.
        k_rope: Decoupled RoPE key as cached, i.e. *after* rotation, shape
            ``[num_layers, num_tokens, qk_rope_head_dim]``.
        position_ids: Absolute position each token was forwarded at, shape
            ``[num_tokens]``. Two passes are comparable only where these
            agree, so the contract travels with the data.
    """

    latent: torch.Tensor
    k_rope: torch.Tensor
    position_ids: torch.Tensor


@dataclass(frozen=True)
class QueryCapture:
    """Queries for the question tokens: the ground-truth workload.

    Attributes:
        query: Per-head query, token-major, shape ``[num_layers,
            num_tokens, num_heads, qk_head_dim]``. Both halves are kept:
            the leading ``qk_nope_head_dim`` absorbs into ``q'``, the
            trailing ``qk_rope_head_dim`` scores against ``k_rope``.
        position_ids: Absolute position of each question token, shape
            ``[num_tokens]``.
    """

    query: torch.Tensor
    position_ids: torch.Tensor

def capture_context(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> ContextCapture:
    """Forward one span of context and read its KV out of the model's cache.

    Args:
        model: An MLA model in eval mode, on the capture device.
        input_ids: Token ids, shape ``[1, num_tokens]``.
        position_ids: Absolute position of each token, shape
            ``[1, num_tokens]``. Explicit and mandatory: the default
            numbering restarts at zero for every call, which would make a
            chunk's ``k_rope`` disagree with the same tokens in the full
            pass by a rotation, and nothing would raise.

    Returns:
        A :class:`ContextCapture` on CPU, in the model's own dtype.

    Raises:
        ValueError: If the model is in training mode, if the batch
            dimension is not 1, if ids and positions disagree in shape, or
            if the cache is not MLA-shaped.
    """
    if model.training:
        raise ValueError(
            "model must be in eval mode: dropout would make the two passes "
            "differ for a reason that is not staleness"
        )
    if input_ids.shape[0] != 1:
        raise ValueError(
            f"capture runs one sequence at a time, got batch "
            f"{input_ids.shape[0]}"
        )
    if input_ids.shape != position_ids.shape:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} and position_ids "
            f"{tuple(position_ids.shape)} must have the same shape"
        )

    cache = DynamicCache()
    with torch.inference_mode():
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

    if cache.layers[0].keys.shape[1] != 1:
        raise ValueError(
            f"expected an MLA cache with a singleton head axis, got "
            f"{tuple(cache.layers[0].keys.shape)}; a model with real KV "
            f"heads stores per-head keys in this slot and needs its own "
            f"capture function"
        )

    # For MLA these two slots hold the latent and k_pe, not K and V:
    # keys[0, 0] is [num_tokens, kv_lora_rank] and values[0, 0] is
    # [num_tokens, qk_rope_head_dim], the leading axes being batch and the
    # singleton head MLA carries so the generic cache can handle it.
    # Each layer moves to the host before stacking, so the full
    # [num_layers, ...] tensor is never allocated on the GPU.
    latent = torch.stack([layer.keys[0, 0].cpu().clone() for layer in cache.layers])
    k_rope = torch.stack([layer.values[0, 0].cpu().clone() for layer in cache.layers])

    return ContextCapture(
        latent=latent,
        k_rope=k_rope,
        # .cpu() alone is a no-op when already on CPU, which would alias
        # the caller's buffer into a frozen dataclass.
        position_ids=position_ids[0].cpu().clone(),
    )

def capture_queries(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        num_question_tokens: int,
) -> QueryCapture:
    """Forward context and question together, keeping only the question's queries.

    The question's hidden states attend to the context, so the context has
    to be in front of it. Only the tail is kept: the context's own queries
    are never used -- a cached token is scored by a *later* query, not by
    its own.

    Args:
        model: An MLA model in eval mode, on the capture device.
        input_ids: Context then question, shape ``[1, num_tokens]``.
        position_ids: Absolute positions, shape ``[1, num_tokens]``.
        num_question_tokens: How many trailing tokens are the question.

    Returns:
        A :class:`QueryCapture` on CPU, in the model's own dtype.

    Raises:
        ValueError: If the model is in training mode, if the batch
            dimension is not 1, if ids and positions disagree in shape, if
            the tail length is out of range, or if the model builds
            queries through ``q_b_proj`` rather than ``q_proj``.
    """
    if model.training:
        raise ValueError("Model must be in eval mode.")
    if input_ids.shape[0] != 1:
        raise ValueError(
            f"Capture runs one sequence at a time, got batch "
            f"{input_ids.shape[0]}"
        )
    if input_ids.shape != position_ids.shape:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} and position_ids "
            f"{tuple(position_ids.shape)} must have the same shape"
        )
    if not 0 < num_question_tokens <= input_ids.shape[1]:
        raise ValueError(
            f"num_question_tokens must be in 1..{input_ids.shape[1]}, got "
            f"{num_question_tokens}"
        )

    num_heads = model.config.num_attention_heads
    head_dim = model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
    attentions = [m for n, m in model.named_modules() if n.endswith("self_attn")]
    captured: dict[int, torch.Tensor] = {}

    def make_hook(index: int):
        def hook(module, args, output):
            # Keep only the question's rows. The slice is a view, so it
            # must be cloned: holding it would pin q_proj's whole output
            # (679MB across 27 layers at a 4096-token context). Returning
            # nothing leaves the module's own output untouched.
            tail = output[0, -num_question_tokens:]
            captured[index] = tail.unflatten(-1, (num_heads, head_dim)).clone()

        return hook

    handles = []
    try:
        for index, attn in enumerate(attentions):
            if attn.q_proj is None:
                raise ValueError(
                    f"layer {index} has no q_proj: this model sets "
                    f"q_lora_rank and builds queries through q_b_proj, "
                    f"which this harness does not handle"
                )
            handles.append(attn.q_proj.register_forward_hook(make_hook(index)))
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False # Don't need KV cache for capturing queries
            )
    finally:
        for handle in handles:
            handle.remove()

    query = torch.stack(
        [captured[index].cpu().clone() for index in range(len(attentions))]
    )
    return QueryCapture(
        query=query,
        position_ids=position_ids[0, -num_question_tokens:].cpu().clone(),
    )
