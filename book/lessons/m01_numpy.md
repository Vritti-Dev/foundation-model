# Module 1. Just Enough NumPy

**Runs in your browser.** Nothing to install.

## Why this matters

Every number in every layer of every neural network you will ever build is stored in a NumPy-style array (in PyTorch they are called Tensors, but the API is the same family). The forward pass is a chain of array operations. The backward pass is the same chain run in reverse. Optimization is more array operations on top.

If you cannot read array code, you cannot read or modify neural network code. This module gets you fluent in the small set of operations the rest of the course needs.

## What you will practice

Five things, no more:

1. **Creating arrays.** `np.zeros`, `np.ones`, `np.arange`, `np.random.randn`. Most weights start as random numbers in a small range.
2. **Indexing and slicing.** `x[0]`, `x[:, 3]`, `x[2:5, :]`. You will pick out the embedding for token id 7, or the rows of attention weights for the first three positions.
3. **Element-wise arithmetic.** `a + b`, `a * 2`, `np.exp(x)`. Every activation function is element-wise.
4. **Broadcasting.** `(B, C) + (C,) = (B, C)`. The shorter shape stretches to match the longer one. This is how a single bias vector gets added to a whole batch of activations.
5. **The dot product.** `a @ b` or `np.dot(a, b)`. This is what a linear layer does. Practically every layer in a neural network is built from matrix multiplications.

## What is actually happening

A NumPy array is a block of numbers stored in one contiguous chunk of memory, plus a small **shape** descriptor that says how to index into that chunk. When you write `a * b`, NumPy hands the whole block to a tight C loop (or a vectorized CPU instruction) and gets back the result in one shot.

That is why NumPy is fast and why we use it even in a from-scratch course. A Python `for` loop processes one number per iteration. A NumPy operation processes the whole array in one call to highly optimized C code. The speed difference is usually 50x to 200x.

## A moment of curiosity

Two arrays with shapes `(B, T, C)` and `(C,)` can be added, and broadcasting fills in the rest. But shapes `(B, T)` and `(T, C)` cannot be added: they are incompatible. Try a few in the notebook and see how NumPy complains.

Why does broadcasting exist? Because it lets you write `attention_scores + mask` instead of looping over batches and time steps. The whole transformer can be written as a couple of dozen array operations because broadcasting handles the bookkeeping.

## Common confusions

* `a * b` is **element-wise** multiplication. `a @ b` is **matrix multiplication**. They are not the same and they have different shape rules.
* `np.dot(a, b)` and `a @ b` mean the same thing for 2D arrays.
* `axis=0` means "collapse the rows" (operate down columns), `axis=-1` means "collapse the last axis" (operate across the row). The terminology trips up almost everyone at first.
* `x.shape` is a tuple. `len(x)` only tells you the size of the first axis.

## Going deeper

* **NumPy quickstart**, the official tour. Twenty minutes, covers everything you need. <https://numpy.org/doc/stable/user/quickstart.html>
* Jake VanderPlas, **Python Data Science Handbook**, NumPy chapter. Free online. <https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html>
* **A Visual Intro to NumPy** (Jay Alammar). Pictures explain broadcasting better than words. <https://jalammar.github.io/visual-numpy/>

```{jupyterlite} ../../notebooks/m01_numpy.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
