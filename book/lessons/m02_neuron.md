# Module 2 — A Neuron and a Gradient Step

**Runs in your browser.** Nothing to install.

Here is the atom of every neural network: a single **neuron**. It takes some
inputs, multiplies them by weights, adds a bias, and passes the result through a
nonlinearity. That's the whole forward pass.

But a neuron that can't learn is useless. So in this module you'll also do the
other half by hand: compute a **gradient** — how the loss changes as you nudge a
weight — and take one step that *lowers* the loss. You'll derive the gradient
yourself and confirm it numerically (a tiny finite-difference check), then watch
a single update make the prediction better.

This is the seed of backpropagation. Module 3 turns this one hand-derived step
into a general engine that does it for any computation automatically.

```{jupyterlite} ../../notebooks/m02_neuron.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
