# Module 5. A Neural Language Model

**Runs in your browser.** Nothing to install.

The bigram model only looks at the single previous character. Now you build a **neural** language model that looks at a small window of context and *learns* its predictions instead of just counting.

You will assemble: an embedding table that maps each character id to a vector, a hidden layer with a `tanh` nonlinearity, and an output layer that produces a score for every possible next character. A `softmax` (with the max-subtraction trick so it never overflows) turns those scores into probabilities, and cross-entropy measures how wrong they are. Then you train it and watch the loss fall as it overfits a tiny batch down toward zero.

```{admonition} A fast gradient path
:class: note
The training here uses a **NumPy array-based** backward pass, not the scalar `Value` engine from Module 3. The concepts are identical; the implementation is vectorized so it is fast enough to actually train. That is the same move real frameworks make.
```

```{jupyterlite} ../../notebooks/m05_mlp_lm.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
