# Module 6. Attention From Scratch

**Runs in your browser.** Nothing to install.

This is the idea that makes transformers work: **self-attention**. Instead of cramming all context into one fixed window, each position gets to look back over the sequence and decide for itself which earlier positions matter most.

You implement it in NumPy. Project the input into queries, keys, and values; score every position against every other with a scaled dot product; apply a **causal mask** so a position can only attend to itself and the past (never the future); softmax the scores into weights that sum to 1; take the weighted sum of values. Then you stack several of these into **multi-head** attention, where the heads split the work and their outputs recombine.

The grader checks the two properties that define correct attention: the weight rows sum to 1, and the causal mask actually holds. Changing a *future* token must not change the output at an earlier position.

```{jupyterlite} ../../notebooks/m06_attention.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
