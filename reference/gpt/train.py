"""M8: a tiny training loop plus a reproducible submission token.

`train` fits the reference GPT on a body of text using a character-level
tokenizer, AdamW, and plain next-token prediction. The submission helpers
turn a trained model into short, deterministic fingerprints so a grader can
verify *which* architecture was trained and *what* it generates -- without
ever seeing the weights themselves.
"""

import hashlib
from dataclasses import replace

import torch

from reference.gpt.model import GPT, CONFIGS
from reference.tokenizer.char_tokenizer import CharTokenizer

# Fixed knobs that keep CPU smoke runs fast and reproducible.
BATCH_SIZE = 12
LEARNING_RATE = 1e-3
SAMPLE_TOKENS = 32  # how many tokens sample_hash greedily generates


def _make_batch(data_ids, block_size, batch_size, device):
    """Pick `batch_size` random windows of length `block_size`.

    Returns inputs x and targets y, where y is x shifted one step right
    (the next-character-prediction objective).
    """
    # Highest valid start index so a full (block_size + 1) window fits.
    max_start = len(data_ids) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data_ids[s:s + block_size] for s in starts])
    y = torch.stack([data_ids[s + 1:s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


def train(config, max_iters, seed, data):
    """Train the reference GPT and report initial vs final loss.

    `config` is 'cpu' or 't4'. The config's vocab_size is overridden to match
    the tokenizer built from `data`. Returns a dict with the trained model,
    tokenizer, resolved config, and the two loss measurements (floats).
    """
    # Build the vocabulary from the data and resize the config to fit it.
    tokenizer = CharTokenizer(data)
    cfg = replace(CONFIGS[config], vocab_size=tokenizer.vocab_size)

    # Pin the seed so weight init and batch sampling are reproducible.
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(cfg).to(device)
    data_ids = torch.tensor(tokenizer.encode(data), dtype=torch.long)
    block_size = cfg.block_size

    # Record the starting loss as the mean over a couple of fresh batches.
    model.eval()
    with torch.no_grad():
        initial_losses = []
        for _ in range(2):
            x, y = _make_batch(data_ids, block_size, BATCH_SIZE, device)
            _, loss = model(x, y)
            initial_losses.append(loss.item())
    initial_loss = sum(initial_losses) / len(initial_losses)

    # Train for a fixed number of AdamW steps.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    final_loss = initial_loss
    for _ in range(max_iters):
        x, y = _make_batch(data_ids, block_size, BATCH_SIZE, device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": cfg,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
    }


def arch_signature(model):
    """Hash the model's parameter shapes + total count into 8 hex chars.

    Two models share a signature exactly when they have the same named
    parameters with the same shapes -- i.e. the same architecture.
    """
    items = sorted(
        (name, tuple(p.shape)) for name, p in model.named_parameters()
    )
    total = sum(p.numel() for p in model.parameters())
    payload = repr(items) + repr(total)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:8]


def sample_hash(model, tokenizer):
    """Greedily generate a fixed sample and hash the produced token ids.

    Starts from token id 0, generates SAMPLE_TOKENS tokens deterministically
    (argmax) with the model in eval mode, and returns 8 hex chars over the id
    list. Captures *what the model generates* in a tiny fingerprint.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = model.generate(start, SAMPLE_TOKENS)
    ids = out[0].tolist()
    digest = hashlib.sha256(repr(ids).encode("utf-8")).hexdigest()
    return digest[:8]


def make_submission_token(result, mod):
    """Format a one-line submission token from a `train` result dict."""
    final = result["final_loss"]
    arch = arch_signature(result["model"])
    shash = sample_hash(result["model"], result["tokenizer"])
    return f"SLM-{mod} loss={final:.4f} arch={arch} shash={shash}"
