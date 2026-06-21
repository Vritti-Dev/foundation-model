# Module 4. A Bigram Model That Writes

**Runs in your browser.** Nothing to install.

You now have a tokenizer from the warm-up. Time to build your first thing that actually generates text: a **bigram model**. The idea is almost embarrassingly simple. Count how often each character follows each other character in the training text, turn those counts into probabilities, sample.

You will build a probability matrix `P` where row `i` is the distribution over "what character comes next" given the current character `i`. Each row sums to 1. Then you sample from it character by character, seeded so your output is reproducible. The text will be gibberish, but it is gibberish with the right *letter frequencies and pairings*, and it is a real generative model you wrote.

This module reuses the `CharTokenizer` from the warm-up, so you will see the pieces start to connect.

```{jupyterlite} ../../notebooks/m04_bigram.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
