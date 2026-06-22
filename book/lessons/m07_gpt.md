# Module 7. Build the GPT (PyTorch, on Colab)

**Runs on Google Colab.** You need a free Google account.

## Why this matters

You have built every piece. Now you bolt them together. This is the moment when "I built a neural network" becomes "I built a small GPT". The class hierarchy in `reference/gpt/model.py` is small enough to read in one sitting and is the same shape as nanoGPT, which is the same shape as GPT-2, which is the same shape as the open-source LLMs you can download today.

After this module you can read modern language model papers with their architecture diagrams and recognize every piece. That is a serious milestone.

## What you will build

A `GPT` model with the following structure:

1. **Token and positional embeddings.** A learned vector per token id, plus a learned vector per position in the sequence, added together. The model needs positional information because attention alone has no notion of order.
2. **`n_layer` Transformer blocks**, each one a pre-LayerNorm sandwich of:
   * LayerNorm, then multi-head causal self-attention, then a residual connection.
   * LayerNorm, then a 2-layer MLP with GELU, then a residual connection.
3. **A final LayerNorm** for stability.
4. **A linear language-model head** that projects the final hidden state down to `vocab_size` logits.

Two configs are provided: `CONFIGS['cpu']` (about 0.8 million parameters, runs on CPU) and `CONFIGS['t4']` (about 10.6 million parameters, designed for a free Colab T4 GPU).

## Why Colab and not the browser

PyTorch does not run in the browser. There is no WebAssembly build of it (PyTorch has tens of megabytes of C++ extensions plus a CUDA dependency that has no path to the browser sandbox). So Modules 7 to 9 open in Google Colab, where PyTorch and an optional free GPU are one click away.

## What is actually happening

A **transformer block** is the workhorse. It does two jobs: attention (mix information across positions) and an MLP (transform information within each position). Each is wrapped in a residual connection (`x + f(x)`) which lets gradients flow cleanly through deep stacks and lets the network "default to identity" if a block has not learned anything useful yet.

**Pre-LayerNorm** means LayerNorm comes *before* the attention and MLP, not after. Pre-LN is the modern standard because it makes training more stable and lets you use higher learning rates. The original transformer paper used post-LN; nearly all modern implementations switched.

The **language-model head** is just a linear layer from hidden state to vocabulary. Its output is **logits**: raw scores, not probabilities. We compute the cross-entropy loss directly on logits (which is more numerically stable than softmax then log).

## A moment of curiosity

The GPT class is about 120 lines of PyTorch. Look at the line count of nanoGPT (under 1000 lines for the entire training script). Then look at the implementation of GPT-2 in Hugging Face Transformers, which is several thousand lines for the same model. The extra code is engineering for production: support for many tokenizers, mixed precision, gradient checkpointing, distributed training, KV-caching for fast inference. The core math is identical.

What scales when you go from your 0.8M-parameter model to GPT-3's 175 *billion*? Three things, in this order of importance: more parameters (the model itself), more data (Web-scale text), more compute (thousands of GPUs running for months). The architecture barely changes. This is the central observation behind the Chinchilla scaling laws.

## Common confusions

* **`nn.GELU(approximate='tanh')` matters.** The tanh-approximation is what GPT-2 and GPT-3 used. The exact GELU is fine too but produces slightly different numbers. We use the approximation so the pure-NumPy demo in Module 0 can match the PyTorch model exactly.
* **`block_size` is the context length, not the batch size.** It is the longest sequence the model can attend to. A model trained with block_size 64 has no way to handle a 65-token input.
* **The submission token does not upload your weights.** It is a one-line fingerprint (architecture signature, loss, and a sample hash) you paste back into the browser grader. Your weights stay on Colab.

## Going deeper

* Karpathy, **nanoGPT** (GitHub). The reference codebase. Two files of model code, two files of training code. <https://github.com/karpathy/nanoGPT>
* Karpathy, **Let's reproduce GPT-2 (124M)**. Four-hour video where Karpathy builds and trains GPT-2 end to end. Demanding but exceptional. <https://www.youtube.com/watch?v=l8pRSuU81PU>
* Radford et al., **Improving Language Understanding by Generative Pre-Training** (GPT-1, 2018). Short, original, very readable. <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>
* Brown et al., **Language Models are Few-Shot Learners** (GPT-3, 2020). The paper that changed the field. <https://arxiv.org/abs/2005.14165>

```{admonition} Open this lesson in Colab
:class: tip
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vritti-Dev/foundation-model/blob/main/notebooks/m07_gpt.ipynb)

The button opens `notebooks/m07_gpt.ipynb` in Google Colab. Run all cells from the top (the first cell clones the course repo onto the Colab VM so `from reference.gpt.model import GPT` works). Then copy the `SLM-m7 ...` line from the last cell back into the grader.
```
