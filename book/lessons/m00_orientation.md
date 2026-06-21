# Module 0 — See Your Goal: a Tiny Model Writes

**Runs in your browser.** Nothing to install.

Before you build anything, see where you're headed. This module loads a small
GPT that has **already been trained** for you and watches it generate text —
right here in your browser, with no PyTorch and no GPU.

How is that possible if PyTorch can't run in the browser? The trained weights
were exported to a small file, and the forward pass is re-implemented in pure
NumPy (`reference/demo/numpy_forward.py`). You'll feed it a starting character
and watch it predict the next one, and the next, building up text one character
at a time. It's the same machinery you're about to build by hand — just
pre-assembled so you can see the destination first.

This page also doubles as the **warm-up load**: the first time it runs, your
browser downloads the Python kernel and NumPy (a few seconds, one time). After
that, every other browser lesson starts fast.

```{jupyterlite} ../../notebooks/m00_orientation.ipynb
:new_tab: False
:width: 100%
:height: 600px
```

```{admonition} If the kernel won't load
:class: warning
The in-browser Python kernel needs a modern browser and a few megabytes on
first load. If it stalls, reload the page once. If it still won't start, the
lesson page shows the demo's pre-recorded output so you can read along — you
won't be stuck.
```
