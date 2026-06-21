# Module 4 — A Bigram Model That Writes

**Runs in your browser.** Nothing to install.

You now have a tokenizer (from the warm-up). Time to build your first thing that
actually generates text: a **bigram model**. The idea is almost embarrassingly
simple — just count how often each character follows each other character in the
training text, turn those counts into probabilities, and sample.

You'll build a probability matrix `P` where row `i` is the distribution over
"what character comes next" given the current character `i`. Each row sums to 1.
Then you'll sample from it character by character, seeded so your output is
reproducible. The text will be gibberish — but it's gibberish with the right
*letter frequencies and pairings*, and it's a real generative model you wrote.

This reuses the `CharTokenizer` from the warm-up, so you'll see the pieces start
to connect.

```{jupyterlite} ../../notebooks/m04_bigram.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
