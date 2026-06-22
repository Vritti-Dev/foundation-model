# Module 6. Attention From Scratch

**Runs in your browser.** Nothing to install.

## Why this matters

This is the idea that makes transformers work. **Attention** is what lets a model handle long contexts without an explosion of parameters, and it is the single most important innovation in deep learning of the last decade. It is the `T` in GPT, the algorithm at the heart of every modern LLM.

The intuition is simple. Instead of cramming all the context into one fixed-size summary, each position in the sequence gets to look back over all earlier positions and decide, for itself, which ones matter most for predicting what comes next. The mechanism for "deciding which ones matter" is itself learned.

Attention is short, it is elegant, and once you have implemented it you will see it everywhere. You can derive the whole thing on a napkin.

## What you will build

A function `self_attention(X, head_size)` that takes a sequence `X` of shape `(T, C)` (T positions, each with C features) and returns a new sequence of the same shape where each position has been replaced by a weighted average of itself and earlier positions.

The recipe:

1. **Project** each position into three vectors: a **query** `q`, a **key** `k`, and a **value** `v`. Each is a learned linear projection of the input.
2. **Score** every position against every other: `scores = Q @ K.T`. Divide by `sqrt(head_size)` to keep the scale reasonable.
3. **Apply a causal mask**: set `scores[i, j] = -inf` for any `j > i`. A position can only look at itself and earlier positions, never the future.
4. **Softmax over the last axis** to turn scores into attention weights. Each row now sums to 1.
5. **Weighted sum**: `output = attention_weights @ V`.

That is it. Then you stack several of these heads in parallel (one head learns who is talking, another learns syntactic agreement, etc.) and concatenate their outputs. This is **multi-head attention**.

## What is actually happening

Think of `Q`, `K`, `V` as three different "views" of the same input. The query at position `i` asks the question "what am I looking for in my context?". The keys at every position answer "this is what I am". The dot product `q_i . k_j` measures how well key `j` matches query `i`. High scores get high attention weight. The value at position `j` is "what I will give you if you attend to me".

This decoupling (separate Q, K, V projections) is the reason attention is so powerful. The model can use different parts of its features to ask, match, and respond. Earlier attention mechanisms used the same vector for everything; the Q/K/V split, introduced in the 2017 transformer paper, was the breakthrough.

The **causal mask** is what makes this a language model and not a search engine. By forbidding position `t` from looking at positions `t+1, t+2, ...`, we ensure that when the model predicts the next token, it has only seen what came before. Without the mask, the model would just copy the answer.

The `sqrt(head_size)` scaling matters. As `head_size` grows, the dot products grow too (they are sums of `head_size` products). Without scaling, the softmax saturates: it concentrates almost all weight on one position. Dividing by `sqrt(head_size)` keeps the softmax in a useful regime.

## A moment of curiosity

Why do we use **multiple heads** instead of one big head? Because different heads can specialize. One head might learn to track who the subject of the sentence is. Another might attend to the most recent punctuation. A third might focus on rare tokens. With one giant head, all of these signals would have to share the same attention pattern.

In a 12-head GPT-2, researchers have found heads that consistently attend to the next word, the previous quote mark, the matching parenthesis, and so on. Anthropic's interpretability research catalogs many of these. The model is doing something *legible* internally, and attention heads are how.

What if you removed the causal mask? You would have **bidirectional attention** (as in BERT), which is great for understanding tasks but cannot generate text one token at a time, because it always sees the future.

## Common confusions

* **Q, K, V come from the same input.** This is *self*-attention. In cross-attention (used in translation models), Q comes from one sequence and K, V from another.
* **The mask sets scores to negative infinity *before* the softmax.** After softmax, `exp(-inf) = 0`, so masked positions get zero weight. Setting weights to zero *after* softmax breaks the sum-to-1 property.
* **Attention is not magical memory.** It is just a learned weighted average. The reason it works so well is that the weights are content-dependent and end-to-end trainable.

## Going deeper

* Vaswani et al., **Attention Is All You Need** (2017). The transformer paper. Eight pages. Worth reading slowly. <https://arxiv.org/abs/1706.03762>
* Jay Alammar, **The Illustrated Transformer**. Pictures of every step. The most beginner-friendly version of the paper. <https://jalammar.github.io/illustrated-transformer/>
* Karpathy, **Let's build GPT from scratch** (Zero to Hero #4). Builds the attention block live. <https://www.youtube.com/watch?v=kCc8FmEb1nY>
* Lilian Weng, **Attention? Attention!** A history of every attention variant. <https://lilianweng.github.io/posts/2018-06-24-attention/>
* Anthropic, **A Mathematical Framework for Transformer Circuits**. Reverse-engineers what attention heads actually learn. <https://transformer-circuits.pub/2021/framework/index.html>

```{jupyterlite} ../../notebooks/m06_attention.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
