"""A minimal decoder-only GPT (nanoGPT-style) built in PyTorch.

This is the reference implementation students compare against. It is a
small transformer with token + positional embeddings, a stack of
pre-LayerNorm blocks (causal self-attention + MLP), a final LayerNorm,
and a tied-free linear language-model head.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparameters that fully describe a GPT model."""

    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dropout: float = 0.0


# Two ready-made configs: a tiny one for CPU smoke tests and a bigger one
# sized to train comfortably on a single T4 GPU.
CONFIGS = {
    'cpu': GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=64, vocab_size=65, dropout=0.0),
    't4': GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=65, dropout=0.2),
}


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (look-back-only) mask."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # One linear projects the input into query, key and value at once.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular mask so position t can only attend to <= t.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim
        # Project and split into query/key/value.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        # Reshape into (B, n_head, T, head_dim) so heads run in parallel.
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        # Scaled dot-product attention scores.
        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        # Block attention to future positions.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, n_head, T, head_dim)
        # Recombine heads back into (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network with a 4x hidden expansion."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        # tanh-approximate GELU: a pure-numpy demo must replicate this exactly.
        self.act = nn.GELU(approximate='tanh')
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """A transformer block: pre-norm attention then pre-norm MLP, both residual."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """Run the model. Returns (logits, loss); loss is None unless targets given."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence length exceeds block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, n_embd)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            V = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Greedily append `max_new_tokens` tokens (argmax, deterministic)."""
        for _ in range(max_new_tokens):
            # Keep only the last block_size tokens as context.
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            # Look at the prediction for the final position only.
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
