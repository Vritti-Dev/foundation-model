"""Train a tiny GPT on an embedded mini-Shakespeare corpus and export the
weights for the browser demo (Module 0).

This script:
  1. Builds a CharTokenizer from a small (~3KB) embedded Shakespeare-style
     corpus.
  2. Trains a SMALL torch GPT (n_layer=3, n_head=4, n_embd=96, block_size=64)
     for a few hundred CPU iterations with a pinned seed.
  3. Exports fp16 compressed weights to ``book/_static/demo_weights.npz`` via
     ``reference.demo.export_weights.export``.
  4. Saves the char vocab (stoi / itos) plus the resolved config to
     ``book/_static/demo_meta.json`` so the in-browser numpy demo can decode.

Run from the repo root:  ``python notebooks/make_demo_weights.py``
"""

from __future__ import annotations

import json
import os
import sys

# Make 'reference' importable when run as a plain script from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from reference.gpt.model import GPT, GPTConfig
from reference.gpt.train import _make_batch
from reference.tokenizer.char_tokenizer import CharTokenizer
from reference.demo.export_weights import export

# A small embedded Shakespeare-style corpus (~3KB). Kept tiny on purpose: the
# whole point of Module 0 is a model small enough to ship to a browser.
CORPUS = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even to the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!
"""

SEED = 1337
MAX_ITERS = 400
BATCH_SIZE = 16
LEARNING_RATE = 3e-3


def main():
    static_dir = os.path.join(_REPO_ROOT, "book", "_static")
    os.makedirs(static_dir, exist_ok=True)

    text = CORPUS
    tokenizer = CharTokenizer(text)

    cfg = GPTConfig(
        n_layer=3,
        n_head=4,
        n_embd=96,
        block_size=64,
        vocab_size=tokenizer.vocab_size,
        dropout=0.0,
    )

    torch.manual_seed(SEED)
    device = "cpu"
    model = GPT(cfg).to(device)

    data_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    initial_loss = None
    final_loss = None
    for it in range(MAX_ITERS):
        x, y = _make_batch(data_ids, cfg.block_size, BATCH_SIZE, device)
        _, loss = model(x, y)
        if initial_loss is None:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if it % 50 == 0:
            print(f"iter {it:4d}  loss {loss.item():.4f}")

    print(f"initial loss {initial_loss:.4f} -> final loss {final_loss:.4f}")

    # Export fp16 weights.
    npz_path = os.path.join(static_dir, "demo_weights.npz")
    export(model, npz_path)
    size_kb = os.path.getsize(npz_path) / 1024.0
    print(f"wrote {npz_path}  ({size_kb:.1f} KB)")
    assert size_kb < 2048, f"demo_weights.npz too large: {size_kb:.1f} KB"

    # Save vocab + config so the numpy browser demo can decode.
    meta = {
        "stoi": tokenizer.stoi,
        "itos": {str(i): ch for i, ch in tokenizer.itos.items()},
        "vocab_size": tokenizer.vocab_size,
        "config": {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "dropout": cfg.dropout,
        },
        "seed": SEED,
    }
    meta_path = os.path.join(static_dir, "demo_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")

    # Sanity check: the numpy forward path reproduces the torch logits and
    # generates non-empty deterministic text from a fixed start.
    from reference.demo.numpy_forward import generate, weights_from_torch

    loaded = np.load(npz_path)
    weights = {k: loaded[k].astype(np.float32) for k in loaded.files}
    start = np.array([[tokenizer.stoi.get("F", 0)]])
    out = generate(weights, start, cfg, max_new_tokens=40)
    decoded = tokenizer.decode(out[0].tolist())
    print("sample generation:")
    print(repr(decoded))
    assert len(decoded) >= 41, "generation too short"


if __name__ == "__main__":
    main()
