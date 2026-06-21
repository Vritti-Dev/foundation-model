import numpy as np
import torch
from reference.gpt.model import GPT, GPTConfig
from reference.demo.numpy_forward import numpy_forward, weights_from_torch, generate

def test_numpy_forward_matches_torch():
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=12, dropout=0.0)
    m = GPT(cfg).eval()
    w = weights_from_torch(m)
    idx = np.array([[1, 2, 3, 4]])
    with torch.no_grad():
        tlogits, _ = m(torch.tensor(idx))
    nlogits = numpy_forward(w, idx, cfg)
    assert nlogits.shape == tuple(tlogits.shape)
    assert np.allclose(tlogits.numpy(), nlogits, atol=2e-3, rtol=2e-3)

def test_generate_deterministic_and_in_vocab():
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=12, dropout=0.0)
    m = GPT(cfg).eval()
    w = weights_from_torch(m)
    idx = np.array([[0]])
    a = generate(w, idx, cfg, max_new_tokens=10)
    b = generate(w, idx, cfg, max_new_tokens=10)
    assert np.array_equal(a, b)
    assert a.shape[1] == 11 and int(a.max()) < cfg.vocab_size
