# Module 4. A Bigram Model That Writes

**Runs in your browser.** Nothing to install.

## Why this matters

You have a tokenizer (from the warm-up). Time to build your first thing that actually generates text: a **bigram model**. The idea is almost embarrassingly simple. Count how often each character follows each other character in the training text, turn those counts into probabilities, and sample.

This is not a neural network. It is pure statistics, the way language modelling was done before neural nets. But it generates real text (badly), and it teaches you the **objective** every language model optimizes: predict the next token given what came before.

When you build the neural model in Module 5 and the GPT in Module 7, they are doing the same thing as this bigram. Just with much more context, and learnable parameters instead of hard-coded counts.

## What you will build

A `BigramLM` class. From a training text, you build:

1. A **count matrix** `N` of shape `(vocab_size, vocab_size)`. `N[i, j]` is the number of times character `j` followed character `i` in the training text.
2. A **probability matrix** `P` where each row is normalized so it sums to 1. `P[i, :]` is the probability distribution over what character comes next, given that the current character is `i`.

To generate, you sample. Start from a seed character. Look up its row in `P`. Sample a next character. Append. Repeat.

## What is actually happening

You are estimating a **conditional probability** from data: `P(next_char | current_char)`. The maximum-likelihood estimate is exactly the count, normalized. That is why we divide each row of `N` by the row sum.

We also add **Laplace smoothing** (adding 1 to every count) so the model never assigns probability zero to an unseen pair. Without smoothing, if `"q"` was never followed by `"x"` in training, the model would say `P("x" | "q") = 0` and sampling would assign it zero chance forever. Smoothing pulls every probability slightly toward uniform, which is a safer prior.

The generated text is bad, in a specific and educational way. Because the model only knows the *previous* character, it has no longer-range memory. It can give you reasonable letter pairs (`th`, `er`, `qu` followed by `u`) but no coherent words or grammar. Listen to it. Notice what it *almost* gets right.

## A moment of curiosity

What if you used a **trigram** instead of a bigram, conditioning on the previous *two* characters? You would get better local structure. Real words appear more often. But your count matrix is now `vocab^3` and most cells are empty for any non-trivial vocabulary. You hit a sparsity problem.

This is the central problem of n-gram language modelling. More context means better predictions but exponentially more parameters and exponentially less data per parameter. Neural language models in Module 5 solve this by learning a **dense vector representation** of context instead of storing every possible n-tuple. That is the conceptual leap that makes deep learning for language work.

## Common confusions

* **The model is not memorizing the training text.** It is summarizing it, into a `(V, V)` table of conditional probabilities. The training text is gone after counting; only the statistics remain.
* **Sampling is necessary.** If you always pick the most-likely next character (the argmax), the model gets stuck in loops (`the the the the`). Sampling introduces variety.
* **Seeding the random generator** is what makes outputs reproducible. Without a seed, two runs give different text.

## Going deeper

* Andrej Karpathy, **The spelled-out intro to language modelling: building makemore** (Zero to Hero #2). The bigram lesson is taken from here. <https://www.youtube.com/watch?v=PaCmpygFfXo>
* Daniel Jurafsky and James Martin, **Speech and Language Processing** (3rd ed. draft, free), Chapter 3: N-gram Language Models. The textbook explanation. <https://web.stanford.edu/~jurafsky/slp3/3.pdf>
* Claude Shannon, **A Mathematical Theory of Communication** (1948). The paper that started information theory. Sections on n-gram approximations to English are still readable today. <https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf>

```{jupyterlite} ../../notebooks/m04_bigram.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
