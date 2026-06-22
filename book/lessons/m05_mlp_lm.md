# Module 5. A Neural Language Model

**Runs in your browser.** Nothing to install.

## Why this matters

The bigram in Module 4 had two limits. It only saw one character of context, and it stored its knowledge in a giant count table. Both problems are fixed by the same idea: **embed characters as learnable vectors and pass them through a neural network**.

This module builds a small neural language model that takes the previous **three** characters as context, looks up a vector for each, runs them through a hidden layer, and produces a probability distribution over the next character. You train it by gradient descent, and you watch the loss fall over hundreds of steps.

This architecture, from the 2003 Bengio paper, is the conceptual ancestor of every modern language model. GPT-2 and GPT-4 do something more sophisticated than this, but the spine is the same: tokens go in, the model produces a distribution over the next token, you train by maximizing the probability of the next token in the training text.

## What you will build

The model has four parts:

1. **An embedding table `C`** of shape `(vocab_size, n_embd)`. Row `i` is the learnable vector for character `i`. At the start, the rows are random. By the end of training they encode meaningful structure (vowels cluster together, etc.).
2. **A hidden layer** `tanh(flat @ W1 + b1)` where `flat` is the concatenated embeddings of the three context characters.
3. **An output layer** `logits = hidden @ W2 + b2` of shape `(vocab_size,)`. One score per possible next character.
4. **A softmax** that converts those scores into probabilities, and a **cross-entropy loss** that measures how surprised the model is by the true next character.

Then a training loop: forward pass, loss, backward pass (computing gradients), step (update weights). You will write the backward pass by hand using NumPy arrays, not the `Value` engine from Module 3 (scalar autograd is too slow for arrays of weights).

## What is actually happening

The **embedding** is the conceptual leap. Instead of treating character ids as arbitrary integers, you give each character a vector of `n_embd` learnable numbers. Characters that play similar roles (like vowels) end up with similar vectors during training, because the gradient pushes their vectors in similar directions. The model has effectively learned a small map of the alphabet.

The **softmax** turns raw scores into a probability distribution: `softmax(z)[i] = exp(z[i]) / sum(exp(z))`. It is a smooth, differentiable version of "argmax".

**Cross-entropy loss** is `-log(p_target)`, the negative log probability the model assigned to the correct next character. Minimizing this is maximizing the model's confidence in the truth.

The **max-subtraction trick** in softmax (`exp(z - max(z))` instead of `exp(z)`) is a numerical-stability classic. `exp(1000)` overflows to infinity. `exp(0)` after subtracting the max is `1.0`. The math is identical because both numerator and denominator are scaled by the same factor.

## A moment of curiosity

What if you make the hidden layer wider? The model fits the training data better, up to a point. Then it overfits: it memorizes the training text and gets worse on new text. This trade-off (capacity vs. overfitting) is central to deep learning.

What if you make the context window bigger? Same gain, same problem, plus a new one: every extra context position multiplies the number of weights in the first layer. Real models solve this by using **attention** (Module 6), which lets context size grow without parameters growing in lockstep.

What if you trained the same architecture on a million books? It still cannot produce coherent paragraphs. The MLP only sees three characters at a time. You can pour all the data you want into it and it cannot remember anything earlier in the sentence. That limitation is what attention is invented to fix.

## Common confusions

* **The embedding is a parameter**, not a fixed lookup. It is updated by gradient descent like every other weight in the model. The clever thing is that the *gradient* of the loss with respect to the embedding of token `i` is computed automatically because token `i` is just a row index into `C`.
* **Overfitting one batch** is a great debugging tool. If your training code is correct, you should be able to overfit a single tiny batch down to near-zero loss. If you can't, your gradients are wrong.
* **Softmax outputs are probabilities** (they sum to 1 and are non-negative), but they are not calibrated. A 70% softmax probability does not mean the model is right 70% of the time.

```{admonition} A fast gradient path
:class: note
The training here uses a **NumPy array-based** backward pass, not the scalar `Value` engine from Module 3. The concepts are identical. The implementation is vectorized so it is fast enough to actually train. That is the same move real frameworks make.
```

## Going deeper

* Yoshua Bengio et al., **A Neural Probabilistic Language Model** (2003). The original. Set the template that GPT still uses. <https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf>
* Karpathy, **Building makemore: MLP** (Zero to Hero #3). The hands-on companion. <https://www.youtube.com/watch?v=TCH_1BHY58I>
* Christopher Olah, **Visualizing Representations: Deep Learning and Human Beings**. Why embeddings are the real magic. <https://colah.github.io/posts/2015-01-Visualizing-Representations/>

```{jupyterlite} ../../notebooks/m05_mlp_lm.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
