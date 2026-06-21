import torch
from reference.gpt.model import GPT, GPTConfig, CONFIGS

def test_forward_shape():
    cfg = CONFIGS['cpu']
    m = GPT(cfg)
    x = torch.zeros(2, cfg.block_size, dtype=torch.long)
    logits, loss = m(x)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss is None

def test_param_count_band():
    m = GPT(CONFIGS['cpu'])
    n = sum(p.numel() for p in m.parameters())
    assert 0.5e6 < n < 1.2e6

def test_one_step_reduces_loss():
    torch.manual_seed(0)
    cfg = CONFIGS['cpu']; m = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    _, l0 = m(x, y)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(10):
        opt.zero_grad(); _, l = m(x, y); l.backward(); opt.step()
    _, l1 = m(x, y)
    assert l1.item() < l0.item()

def test_generate_in_vocab_and_blocksize():
    cfg = CONFIGS['cpu']; m = GPT(cfg)
    out = m.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=20)
    assert out.shape[1] == 21
    assert int(out.max()) < cfg.vocab_size
