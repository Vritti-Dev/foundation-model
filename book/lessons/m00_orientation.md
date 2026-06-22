# Module 0. See Your Goal: a Tiny Model Writes

**Runs in your browser.** Nothing to install.

## Why this matters

Before you spend hours building something, see it work. This module loads a small GPT that has **already been trained for you** and watches it generate text, right here in your browser, with no PyTorch and no GPU.

That model is the same kind of model you are about to build by hand. By the end of Module 9, you will have built and trained one yourself. This is your goal, made concrete.

## What is actually happening

The model was trained earlier on a few kilobytes of Shakespeare. Its weights (about 800,000 numbers) were exported to a small `.npz` file you download with the page. When you click **Generate**, this happens:

1. You give the model a single starting character.
2. The model looks at that character, runs it through its 3 layers of attention and feedforward math, and produces a probability over what comes next.
3. The page picks the most likely next character.
4. That character gets appended to the input. Repeat from step 2.

This is **autoregressive generation**: predict one token at a time, feed it back in, predict the next one. Every chatbot you have used does exactly this.

The forward pass is re-implemented in pure NumPy (`reference/demo/numpy_forward.py`) so it can run in the browser without PyTorch. The math is the same; only the runtime is different.

## A moment of curiosity

Try giving the model different starting characters. `F` will probably give you Shakespearean speech (because most lines in the training text start with capital letters). A lowercase letter will give you mid-sentence continuations. A digit will give you something strange, because digits are rare in Shakespeare.

The model has no memory of where it is in a play or even a sentence. It only has the last 64 characters of context (its **block size**). Yet it consistently produces text that looks like Shakespeare. That is the magic of the next-token prediction objective: tiny, local decisions add up to a coherent style.

## How the rest of the course flows

You will build the pieces of this model in this order:

* **Module 1.** NumPy basics, the math language of the course.
* **Module 2.** A single neuron and one hand-derived gradient step.
* **Module 3.** Backpropagation from scratch (a tiny autograd engine).
* **Module 4.** A bigram model, your first real generator.
* **Module 5.** A neural language model with a real training loop.
* **Module 6.** Attention from scratch.
* **Module 7.** Assemble the full GPT in PyTorch.
* **Module 8.** Train it on Tiny Shakespeare.
* **Module 9.** Train it on a text corpus you choose.
* **Module 10.** A short reading on how production LLMs differ.

## Going deeper

* Andrej Karpathy, **Neural Networks: Zero to Hero** (YouTube playlist). The course this curriculum is most directly inspired by. <https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ>
* 3Blue1Brown, **But what is a Neural Network?** Eight-minute visual primer. <https://www.youtube.com/watch?v=aircAruvnKk>
* Karpathy, **nanoGPT** (GitHub). The codebase the reference model in this course is modelled on. Tiny, readable, complete. <https://github.com/karpathy/nanoGPT>

```{jupyterlite} ../../notebooks/m00_orientation.ipynb
:new_tab: False
:width: 100%
:height: 600px
```

```{admonition} If the kernel will not load
:class: warning
The in-browser Python kernel needs a modern browser and a few megabytes on first load. If it stalls, reload the page once. If it still will not start, the lesson page shows the demo's pre-recorded output so you can read along. You will not be stuck.
```
