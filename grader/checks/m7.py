"""M7 checkpoint: PyTorch GPT. Parameterized by the GPT class."""

from __future__ import annotations

import torch

from grader.checks._util import property_check
from reference.gpt.model import CONFIGS


def check_gpt(GPTClass):
    def forward_shape():
        cfg = CONFIGS["cpu"]
        m = GPTClass(cfg)
        logits, loss = m(torch.zeros(2, cfg.block_size, dtype=torch.long))
        return tuple(logits.shape) == (2, cfg.block_size, cfg.vocab_size) and loss is None

    def param_count():
        n = sum(p.numel() for p in GPTClass(CONFIGS["cpu"]).parameters())
        return 0.5e6 < n < 1.2e6

    def one_step_loss():
        torch.manual_seed(0)
        cfg = CONFIGS["cpu"]; m = GPTClass(cfg)
        x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
        y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
        _, l0 = m(x, y)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        for _ in range(10):
            opt.zero_grad(); _, l = m(x, y); l.backward(); opt.step()
        _, l1 = m(x, y)
        return l1.item() < l0.item()

    def generate_in_vocab():
        cfg = CONFIGS["cpu"]; m = GPTClass(cfg)
        out = m.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=20)
        return out.shape[1] == 21 and int(out.max()) < cfg.vocab_size

    # GPT checks instantiate/train a ~0.8M-param model on CPU, so they get a
    # generous timeout (the default 2s is for the lightweight numpy checks).
    return [
        property_check("forward_shape", forward_shape, timeout_s=60),
        property_check("param_count", param_count, timeout_s=60),
        property_check("one_step_loss", one_step_loss, timeout_s=120),
        property_check("generate_in_vocab", generate_in_vocab, timeout_s=60),
    ]


def build_checks():
    from reference.gpt.model import GPT
    return check_gpt(GPT)
