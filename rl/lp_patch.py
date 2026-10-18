"""Semantics-preserving replacements for mlxrl's completion-logprob forward (wrapper; mlxrl untouched).

mode "stock"  : mlxrl as shipped (full [B, prompt+completion, vocab] logits).
mode "slice"  : backbone over full sequence, lm_head applied ONLY at completion positions.
mode "shared" : prompt prefilled ONCE (batch 1, with gradients), KV broadcast to the G rows, completions run against
                it, lm_head only at completion positions. Requires all rows to share one prompt; else falls back to slice.
"""
import mlx.core as mx
import mlx.nn as nn
import mlxrl.policy.logprobs as LP
import mlxrl.train.grpo as TG
from mlx_lm.models.cache import KVCache, make_prompt_cache

_STOCK = LP.completion_logprobs_from_inputs
STATE = {"mode": "stock"}


def _head(model, h):
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def _lp(logits, targets):
    return mx.squeeze(mx.take_along_axis(logits, targets[..., None], axis=-1), -1) - \
        mx.squeeze(mx.logsumexp(logits, axis=-1, keepdims=True), -1)


def _qkv(a, x, offset):
    B, L, _ = x.shape
    q = a.q_norm(a.q_proj(x).reshape(B, L, a.n_heads, -1)).transpose(0, 2, 1, 3)
    k = a.k_norm(a.k_proj(x).reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    v = a.v_proj(x).reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
    return a.rope(q, offset=offset), a.rope(k, offset=offset), v


def _block(layer, h_p, h_c, P):
    """Qwen3 decoder block on (prompt row, G completion rows). Pure function of arrays + layer params, so it is
    safe under mx.checkpoint. Completion rows attend to the prompt K/V computed from h_p IN THIS GRAPH (gradients
    flow into the prompt path; nothing is cached or detached)."""
    a = layer.self_attn
    G = h_c.shape[0]
    qp, kp, vp = _qkv(a, layer.input_layernorm(h_p), 0)
    qc, kc, vc = _qkv(a, layer.input_layernorm(h_c), P)
    op = mx.fast.scaled_dot_product_attention(qp, kp, vp, scale=a.scale, mask="causal")
    K = mx.concatenate([mx.broadcast_to(kp, (G,) + kp.shape[1:]), kc], axis=2)
    V = mx.concatenate([mx.broadcast_to(vp, (G,) + vp.shape[1:]), vc], axis=2)
    oc = mx.fast.scaled_dot_product_attention(qc, K, V, scale=a.scale, mask="causal")
    h_p = h_p + a.o_proj(op.transpose(0, 2, 1, 3).reshape(h_p.shape[0], h_p.shape[1], -1))
    h_c = h_c + a.o_proj(oc.transpose(0, 2, 1, 3).reshape(h_c.shape[0], h_c.shape[1], -1))
    h_p = h_p + layer.mlp(layer.post_attention_layernorm(h_p))
    h_c = h_c + layer.mlp(layer.post_attention_layernorm(h_c))
    return h_p, h_c


def _shared_forward(model, ids, plen, C, B, ckpt):
    m = model.model
    h_p = m.embed_tokens(ids[0:1, : plen - 1])
    h_c = m.embed_tokens(ids[:, plen - 1: plen - 1 + C])
    P = plen - 1
    for layer in m.layers:
        if ckpt:
            def inner(params, hp, hc, _l=layer):
                _l.update(params)
                return _block(_l, hp, hc, P)
            h_p, h_c = mx.checkpoint(inner)(layer.trainable_parameters(), h_p, h_c)
        else:
            h_p, h_c = _block(layer, h_p, h_c, P)
    return m.norm(h_c)


def _patched(model, inputs, use_checkpoint=False):
    mode = STATE["mode"]
    if mode == "stock":
        return _STOCK(model, inputs, use_checkpoint=use_checkpoint)
    ids, tgt, gi, mask = inputs.input_ids, inputs.target_ids, inputs.gather_indices, inputs.mask
    B, T = ids.shape
    C = gi.shape[1]
    plen = int(gi[0, 0].item()) + 1
    shared = mode == "shared" and bool(mx.all(gi[:, 0] == gi[0, 0]).item()) and plen - 1 >= 1 and \
        bool(mx.all(ids[:, : plen - 1] == ids[0:1, : plen - 1]).item()) and plen - 1 + C <= T
    if shared:
        h = _shared_forward(model, ids, plen, C, B, use_checkpoint)
        t = tgt[:, plen - 1: plen - 1 + C]
        lps = _lp(_head(model, h), t)
    else:
        h = model.model(ids)
        idx = gi[..., None]
        hs = mx.take_along_axis(h, mx.broadcast_to(idx, idx.shape[:2] + (h.shape[-1],)), axis=1)
        t = mx.take_along_axis(tgt, gi, axis=1)
        lps = _lp(_head(model, hs), t)
    lps = lps * mask.astype(lps.dtype)
    return LP.CompletionLogprobs(logprobs=lps, mask=mask)


def install(mode):
    assert mode in ("stock", "slice", "shared")
    STATE["mode"] = mode
    LP.completion_logprobs_from_inputs = _patched
    TG.completion_logprobs_from_inputs = _patched
