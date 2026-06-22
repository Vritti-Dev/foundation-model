# Module 3. Backprop From Scratch

**Runs in your browser.** Nothing to install.

## Why this matters

In Module 2 you derived a gradient by hand for one weight. That worked because there was one weight. A real GPT has 800,000 weights in this course, and billions in production. Nobody derives all those gradients by hand.

What real frameworks do is **automatic differentiation**: build a small engine that records every operation as you compute, then walks the graph backwards to hand you every gradient for free. PyTorch, JAX, TensorFlow, all of them do this. You are about to build a tiny one yourself. It is one of the most important ideas in modern computing.

## What you will build

A `Value` class. Each instance holds a single number (`data`) and a slot for its gradient (`grad`, starts at zero). The class overloads arithmetic so you can write:

```python
a = Value(2.0)
b = Value(-3.0)
y = a * b + a.relu()
y.backward()      # fills in a.grad and b.grad
```

`backward()` walks the computation graph backwards from the output and assigns each input its gradient. This is **reverse-mode automatic differentiation**, the algorithm that powers every neural network library you have ever used.

## What is actually happening

When you compute `c = a * b`, two things happen:

1. The forward value is computed (`c.data = a.data * b.data`).
2. The new `Value` remembers its **parents** (`a` and `b`) and how the gradient flows back: by the chain rule, if `c.grad` is some number `g`, then `a.grad` should accumulate `g * b.data` and `b.grad` should accumulate `g * a.data`.

So the operation builds a node in a graph, with edges pointing back to its inputs and a small closure that knows how to send gradients backwards through this node.

When you call `output.backward()`, the engine:

1. Walks the graph backwards from `output` and computes a **topological order** (every node visited after all of its dependents).
2. Sets `output.grad = 1.0` (the derivative of `output` with respect to itself is 1).
3. Walks the topological order in reverse, calling each node's backward closure, which pushes gradients to its parents.

That is it. The whole engine in `reference/autograd/engine.py` is about 100 lines. PyTorch's autograd does the same thing, with the same algorithm, just with tensors instead of scalars and a lot more engineering.

## A moment of curiosity

Why is reverse mode used and not forward mode? Forward mode also works for autodiff, but it computes derivatives *with respect to one input variable at a time*. Reverse mode computes derivatives *for one output at a time, with respect to all inputs at once*.

In a neural network, the output is one scalar (the loss) and there are millions of inputs (the weights). Reverse mode gives you all weight gradients in one backward pass. Forward mode would need to be re-run once per weight. That is why every neural network library uses reverse mode.

If you ever do scientific computing or sensitivity analysis where outputs are many and inputs are few, you will see forward mode in tools like JAX's `jacfwd`.

## Common confusions

* **Gradients accumulate.** If the same `Value` is used twice in the graph (`b = a * a`), the gradient updates twice. The `+=` in `_backward` is essential. Beginners often write `=` and silently get wrong gradients on reused variables.
* **`backward()` should only be called on a scalar.** If you want the gradient of a vector, you need to take a scalar function of it first (typically a sum or a loss).
* **The graph is built dynamically.** Each `+` or `*` adds a node. This is "define-by-run". The alternative ("define-and-run", like TensorFlow 1) builds the graph once and runs it many times. PyTorch and our engine use the define-by-run style.

## Going deeper

* Andrej Karpathy, **micrograd** (GitHub repo and YouTube video). The original of what you just built. The video is one of the best CS lectures on YouTube. <https://github.com/karpathy/micrograd> and <https://www.youtube.com/watch?v=VMj-3S1tku0>
* Christopher Olah, **Calculus on Computational Graphs: Backpropagation**. A clearer explanation than any textbook. <https://colah.github.io/posts/2015-08-Backprop/>
* PyTorch, **autograd mechanics** (official docs). Once you understand the toy version, this becomes readable. <https://pytorch.org/docs/stable/notes/autograd.html>

```{admonition} Concepts carry forward; the code does not
:class: note
This scalar engine is for **understanding**. It is perfect for seeing how backprop works, but far too slow for real models because each number is its own Python object. From Module 5 on, the same *ideas* power a fast, array-based gradient path instead. You are learning the concept here, not the production implementation.
```

```{jupyterlite} ../../notebooks/m03_autograd.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
