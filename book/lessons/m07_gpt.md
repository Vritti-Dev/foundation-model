# Module 7. Build the GPT (PyTorch, on Colab)

**Runs on Google Colab.** You need a free Google account.

You have now built every piece by hand: tokenizer, embeddings, a neural LM, attention. This module assembles them into a real **decoder-only GPT** in PyTorch: token and positional embeddings, a stack of pre-LayerNorm transformer blocks (causal self-attention plus an MLP, each with a residual connection), a final LayerNorm, and a language-model head.

Why Colab and not the browser? PyTorch does not run in the browser. There is no WebAssembly build of it. Colab gives you PyTorch and a free GPU with one click.

In this module you confirm the model is wired correctly: the forward pass produces the right shape `(batch, time, vocab)`, the parameter count matches the analytic estimate, one optimizer step lowers the loss on a fixed batch, and `generate()` returns in-vocabulary tokens that respect the block size. The final cell prints a **submission token**. Paste it into the browser grader to record this module.

```{admonition} Open this lesson in Colab
:class: tip
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vritti-Dev/foundation-model/blob/main/notebooks/m07_gpt.ipynb)

The button opens `notebooks/m07_gpt.ipynb` in Google Colab. Run all cells, then copy the `SLM-m7 ...` line from the last cell back into the grader.
```

```{admonition} About the submission token
:class: note
The notebook never uploads anything. Its final cell prints one compact line (for example `SLM-m7 loss=... arch=... shash=...`) that fingerprints which architecture you built and what it generates. You paste that single line into the browser grader. That is the whole handoff.
```
