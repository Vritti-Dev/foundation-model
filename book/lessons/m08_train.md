# Module 8. Train on Tiny Shakespeare

**Runs on Google Colab.** You need a free Google account.

## Why this matters

Building a model is exciting. **Training** one is what actually creates intelligence (such as it is). For most of the history of deep learning, the whole game has been "what do I train on, with what loss, for how long, and how big should the model be". Architectures change slowly. Data and training procedures change everything.

In this module you train the GPT you built in Module 7. The data is **Tiny Shakespeare**, about 1 MB of the Bard's writing. The model is small. The run is short. But every part of the pipeline (data loading, optimizer, loss, sampling) is exactly what real training looks like at scale.

## What you will do

Run a training loop. At each step:

1. **Sample a random window** of `block_size` characters from the training text. That window plus the next character is one training example.
2. **Forward pass.** The model produces a distribution over the next character at every position.
3. **Cross-entropy loss.** Sum up how surprised the model was at the truth, across the whole window.
4. **Backward pass.** PyTorch computes gradients for every parameter (the same algorithm you implemented in Module 3, just industrial-strength and on tensors).
5. **AdamW step.** The optimizer updates every weight in the direction of lower loss, with adaptive per-parameter learning rates.
6. **Repeat.** Many times. Watch the loss fall.

Periodically you sample text from the model and look at it. Early in training the samples are random characters. After a few hundred steps you see real letter pairs. After a few thousand you see fake-but-Shakespeare-sounding sentences. That progression, watched live, is one of the most rewarding things in this course.

## What is actually happening

**AdamW** is the workhorse optimizer for transformers. Plain SGD (the rule you derived in Module 2) just takes the gradient and steps. AdamW is the same idea with two improvements:

* **Per-parameter learning rates** that adapt based on the recent magnitude of each parameter's gradient. Parameters that have been changing quickly get smaller steps. Parameters that have been quiet get relatively larger steps.
* **Decoupled weight decay**, a gentle pull of every weight toward zero, which prevents the model from over-relying on any single weight.

The **batch dimension** matters. We do not train on one example at a time. We stack many (here, 64 examples per step) into a single batch, run them through the model in parallel, and average the loss. This gives more stable gradients and lets the GPU do useful work.

The **submission token** is your handoff back to the browser grader. The notebook prints one line that looks like `SLM-m8 loss=1.6234 arch=ab12cd34 shash=ff90ee01`. You paste that into the grader. Loss is checked against a tolerance band (because GPU vs CPU floating point gives slightly different numbers). The architecture hash and a deterministic argmax sample hash are checked for exact match.

## A moment of curiosity

What is a "good" loss? For character-level Tiny Shakespeare, the entropy of the underlying distribution is around 1.3 to 1.5 nats per character. A model that has learned the language perfectly would get to that number. Our small model gets to about 1.5 after 5000 steps on a T4 GPU. State-of-the-art character LMs get a bit lower. The remaining gap is what bigger models, more data, and longer training would buy you.

Why not train longer? Eventually you **overfit**: training loss keeps falling but validation loss starts rising. The model has begun to memorize the training text instead of learning patterns. We don't show validation in this module to keep things simple, but in real runs you watch it and stop when it stops improving.

The **Chinchilla scaling laws** (DeepMind, 2022) gave us a rule of thumb: for a given compute budget, the optimal split is roughly equal amounts of model and data. Doubling parameters is only worth it if you also roughly double tokens trained on. Most GPT-3 era models were undertrained by this measure.

## Common confusions

* **GPU time is not deterministic.** Two identical runs on the same GPU can produce slightly different losses because of nondeterministic floating-point reduction in CUDA kernels. That is why we grade by tolerance band, not exact equality.
* **The model's "creativity" comes from sampling.** With temperature 0 (always pick the argmax), the model is deterministic and often boring. Raising temperature makes it more varied but also more likely to generate nonsense. Top-p ("nucleus") sampling, common in production, is a way to balance the two.
* **Training does not require any explicit "what does Shakespeare write like" supervision.** Just predicting the next character on his text is enough. The structure of language is encoded in those predictions.

## Going deeper

* Karpathy, **Let's build GPT from scratch** (Zero to Hero #4). The training loop is built and explained. <https://www.youtube.com/watch?v=kCc8FmEb1nY>
* DeepMind, **Training Compute-Optimal Large Language Models** (Chinchilla, 2022). The scaling-law paper. Section 1 is the only must-read part. <https://arxiv.org/abs/2203.15556>
* Loshchilov and Hutter, **Decoupled Weight Decay Regularization** (AdamW, 2017). Why AdamW exists. <https://arxiv.org/abs/1711.05101>

```{admonition} Open this lesson in Colab
:class: tip
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vritti-Dev/foundation-model/blob/main/notebooks/m08_train.ipynb)

The button opens `notebooks/m08_train.ipynb` in Google Colab. For the fastest run, enable a GPU first: **Runtime > Change runtime type > T4 GPU**. Run all cells from the top (the first cell clones the course repo).
```
