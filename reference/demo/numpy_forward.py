"""Pure-NumPy re-implementation of the torch GPT forward pass.

Module 0's demo runs in the browser, where torch is not available. This file
reproduces EXACTLY the math in ``reference/gpt/model.py`` using only NumPy so
the same trained weights produce the same logits. Students can read this side
by side with the torch model to see what each layer really computes.
"""

import math

import numpy as np


def weights_from_torch(model):
    """Extract every parameter as a float32 NumPy array.

    Keys keep the original torch ``state_dict`` names (e.g. ``tok_emb.weight``,
    ``blocks.0.attn.c_attn.weight``) so ``numpy_forward`` can look them up.
    """
    weights = {}
    for name, tensor in model.state_dict().items():
        weights[name] = tensor.detach().cpu().numpy().astype(np.float32)
    return weights


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last axis, matching torch's default eps of 1e-5."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # population variance (unbiased=False)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * weight + bias


def linear(x, weight, bias=None):
    """Apply a torch nn.Linear: ``x @ weight.T + bias`` (weight is (out, in))."""
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def gelu_tanh(x):
    """tanh-approximate GELU, identical to ``nn.GELU(approximate='tanh')``."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    """Numerically stable softmax along ``axis``."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(x, weights, prefix, n_head):
    """Causal multi-head self-attention for one block.

    Mirrors ``CausalSelfAttention.forward``: project to q/k/v, split into heads,
    scaled dot-product attention with a causal mask, recombine, output proj.
    """
    B, T, C = x.shape
    head_dim = C // n_head

    # Combined query/key/value projection, then split along the feature axis.
    qkv = linear(x, weights[prefix + "c_attn.weight"], weights[prefix + "c_attn.bias"])
    q, k, v = np.split(qkv, 3, axis=-1)

    # Reshape (B, T, C) -> (B, n_head, T, head_dim) so heads run in parallel.
    def split_heads(t):
        return t.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    # Scaled dot-product scores: (B, n_head, T, T).
    att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)

    # Causal mask: position t may only attend to positions <= t.
    causal = np.tril(np.ones((T, T), dtype=bool))
    att = np.where(causal, att, -np.inf)
    att = softmax(att, axis=-1)

    y = att @ v  # (B, n_head, T, head_dim)
    # Recombine heads back into (B, T, C).
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
    return linear(y, weights[prefix + "c_proj.weight"], weights[prefix + "c_proj.bias"])


def numpy_forward(weights, idx, cfg):
    """Run the full GPT forward in NumPy. Returns logits (B, T, vocab_size)."""
    idx = np.asarray(idx)
    B, T = idx.shape
    assert T <= cfg.block_size, "sequence length exceeds block_size"

    # Token + positional embeddings (positions are 0..T-1).
    tok = weights["tok_emb.weight"][idx]          # (B, T, n_embd)
    pos = weights["pos_emb.weight"][:T]           # (T, n_embd)
    x = tok + pos

    # Stack of pre-LayerNorm transformer blocks.
    for i in range(cfg.n_layer):
        bp = f"blocks.{i}."
        # Attention sub-block with residual.
        a = layer_norm(x, weights[bp + "ln1.weight"], weights[bp + "ln1.bias"])
        x = x + attention(a, weights, bp + "attn.", cfg.n_head)
        # MLP sub-block with residual: Linear -> tanh-GELU -> Linear.
        m = layer_norm(x, weights[bp + "ln2.weight"], weights[bp + "ln2.bias"])
        h = linear(m, weights[bp + "mlp.fc.weight"], weights[bp + "mlp.fc.bias"])
        h = gelu_tanh(h)
        h = linear(h, weights[bp + "mlp.proj.weight"], weights[bp + "mlp.proj.bias"])
        x = x + h

    # Final LayerNorm and (bias-free) language-model head.
    x = layer_norm(x, weights["ln_f.weight"], weights["ln_f.bias"])
    logits = linear(x, weights["lm_head.weight"])  # lm_head has no bias
    return logits


def generate(weights, idx, cfg, max_new_tokens):
    """Greedy (argmax) autoregressive generation. Deterministic.

    Appends ``max_new_tokens`` tokens one at a time, always cropping the context
    to the last ``block_size`` tokens before each forward pass.
    """
    idx = np.asarray(idx)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits = numpy_forward(weights, idx_cond, cfg)
        next_token = logits[:, -1, :].argmax(axis=-1, keepdims=True)
        idx = np.concatenate([idx, next_token], axis=1)
    return idx
