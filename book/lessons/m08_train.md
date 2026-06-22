# Module 8. Train on Tiny Shakespeare

**Runs on Google Colab.** You need a free Google account.

Now you train the GPT you built. The corpus is **Tiny Shakespeare** (about 1 MB, 65 unique characters), small enough to train in minutes on Colab's free GPU, big enough to produce text that reads like Shakespeare-flavored English.

The notebook detects whether a GPU is available and falls back to CPU if not (slower, but it still finishes). It runs a plain next-character-prediction training loop with AdamW, reports the loss falling as training proceeds, and samples some text so you can watch the model go from noise to legible babble to recognizable structure.

When training finishes, the final cell prints your **submission token** (a line like `SLM-m8 loss=1.6234 arch=ab12cd34 shash=ff90ee01`). Paste it into the browser grader. Your loss is graded by a **tolerance band** (not bit-exact, since GPU/CPU floating-point differs), and the greedy sample is graded by hash.

```{admonition} Open this lesson in Colab
:class: tip
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vritti-Dev/foundation-model/blob/main/notebooks/m08_train.ipynb)

The button opens `notebooks/m08_train.ipynb` in Google Colab. For the fastest run, enable a GPU first: **Runtime > Change runtime type > T4 GPU**.
```
