# Build a Small Language Model From Scratch

Welcome. This is a hands-on course that takes you from **basic Python and zero
machine learning** all the way to **building and training your own small GPT** —
and then watching it write text that it learned to produce itself.

The approach is **build-by-hand first**. Before you touch a deep-learning
framework, you'll implement the core machinery yourself in plain Python and
NumPy: a single neuron, backpropagation, a tokenizer, a bigram model, a neural
language model, and attention. Only once you understand how those pieces work do
you pick up PyTorch to assemble and train the full transformer.

## How the course runs (two runtimes)

Everything here is free. There is no account to create and no server to pay for.
The course is split across two runtimes, and that split is deliberate — it lands
exactly on the seam between "understand the math by hand" and "now scale it up
with a framework."

```{list-table}
:header-rows: 1
:widths: 20 30 50

* - Lessons
  - Where it runs
  - What you need
* - Warm-up + Modules 0–6
  - **In your browser**
  - Nothing. The notebooks run live on this page using a Python kernel
    (Pyodide/WebAssembly) plus NumPy. No install, works offline after the first
    load.
* - Modules 7–9
  - **Free Google Colab**
  - A Google account. PyTorch can't run in the browser, so these lessons open in
    Colab with one click and use its free GPU.
* - Module 10
  - **Just reading**
  - Nothing. A short tour of how real, production LLMs differ from what you built.
```

### Browser lessons (Warm-up, Modules 0–6)

When you open a browser lesson, the page boots a small Python environment right
inside your browser tab. The first load downloads the Python kernel and NumPy
(a few megabytes, a few seconds) — that one-time cost is paid up front in the
Module 0 warm-up. After that you edit code, run cells, and check your work
without ever leaving the page. Nothing is uploaded anywhere.

### Colab lessons (Modules 7–9)

These lessons build and train a real PyTorch GPT. Each has an **"Open in Colab"**
button that opens the notebook in Google Colab, where you get a free GPU. When
you finish, the notebook prints a short **submission token** — a single line like
`SLM-m8 loss=1.6234 arch=ab12cd34 shash=ff90ee01` — that you paste back into the
browser grader to record your progress. No files move around; the token is all
you need.

## A note on your progress

Your progress (which lessons you've completed) is stored **locally in this
browser**, using its built-in storage. That means:

- It is **per-browser and per-device** — it does not sync, and clearing your
  browser data will erase it.
- It is **not an authoritative record**. These checks are honest, self-paced
  gating to help *you* track where you are — they are not anti-cheat, and they
  are not a verified transcript.
- Every lesson offers a **download backup**: you can save your completed notebook
  and a small progress token file so your work survives a cache wipe or a move to
  a new machine.

A durable, cross-device, verified grade is intentionally **out of scope** for
this free static version of the course.

## What you'll have built by the end

A complete beginner can go from Module 0 to a small GPT, trained by you on
Tiny Shakespeare (and then on a corpus of your own choosing in the capstone),
producing legible text — entirely on free tooling.

Start with the **warm-up** in the sidebar, then work straight through Modules 0
to 10 in order.
