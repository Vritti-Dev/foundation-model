# Module 3 — Backprop From Scratch

**Runs in your browser.** Nothing to install.

In Module 2 you computed one gradient by hand. That doesn't scale: real models
have thousands of operations. The fix is **automatic differentiation** — build a
small engine that records every operation as you compute, then walks the graph
backwards to hand you every gradient for free.

You'll build a `Value` class (in the spirit of Karpathy's micrograd): a scalar
that remembers how it was made. Support `+`, `*`, `**`, negation, division, and
a `relu` nonlinearity, then implement `backward()` — a topological walk that
visits each node exactly once, after its children, accumulating gradients along
the way. The grader checks your gradients against finite differences and against
PyTorch's autograd on a random graph.

```{admonition} Concepts carry forward; the code does not
:class: note
This scalar engine is **for understanding**. It's perfect for seeing how
backprop works, but far too slow for real models. From Module 5 on, the same
*ideas* power a fast, array-based gradient path instead. You're learning the
concept here, not the production implementation.
```

```{jupyterlite} ../../notebooks/m03_autograd.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
