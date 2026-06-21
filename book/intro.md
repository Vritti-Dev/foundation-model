# Build a Small Language Model From Scratch

This is a hands-on course that takes you from basic Python and zero machine learning to building, training, and sampling from your own small GPT.

By the time you finish, your model produces text like this, from a model you trained:

<div class="sample-output">First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with sureve to they
hath for mishe oughath soft-consconscienced men can be
content to say it was for his country sey relid us as yous and would yould relieve</div>

The approach is **build-by-hand first**. Before you touch a deep-learning framework, you implement the core machinery yourself in plain Python and NumPy: a single neuron, backpropagation, a tokenizer, a bigram model, a neural language model, and attention. Once you have built those pieces, you pick up PyTorch to assemble and train the full transformer.

## How the course runs

Everything is free. No account to create, no server to pay for. The course runs on two free runtimes, with one deliberate handoff between them.

```{list-table}
:header-rows: 1
:widths: 22 28 50

* - Lessons
  - Where it runs
  - What you need
* - Warm-up, Modules 0 to 6
  - **In your browser**
  - Nothing. The notebooks run live on this page using a Python kernel
    (Pyodide/WebAssembly) plus NumPy. No install. Works offline after the
    first load.
* - Modules 7 to 9
  - **Free Google Colab**
  - A Google account. PyTorch does not run in the browser, so these lessons
    open in Colab and use its free GPU.
* - Module 10
  - **Just reading**
  - Nothing. A short tour of how production LLMs differ from what you built.
```

### Browser lessons (Warm-up, Modules 0 to 6)

When you open a browser lesson, the page boots a small Python environment inside your browser tab. The first load downloads the Python kernel and NumPy (a few megabytes, a few seconds). That one-time cost is paid in the Module 0 warm-up. After that, you edit code, run cells, and check your work without leaving the page. Nothing is uploaded anywhere.

### Colab lessons (Modules 7 to 9)

These lessons build and train a real PyTorch GPT. Each one has an **Open in Colab** button that opens the notebook in Google Colab, where you get a free GPU. When you finish, the notebook prints a short **submission token**: a single line like `SLM-m8 loss=1.6234 arch=ab12cd34 shash=ff90ee01`. You paste it into the browser grader to record your progress. No files move around; the token is all you need.

## How your progress is stored

Your progress is stored **locally in this browser**, using its built-in storage:

- It is **per-browser and per-device**. It does not sync. Clearing your browser data will erase it.
- It is **not an authoritative record**. The checks are honest, self-paced gates to help you track where you are, not anti-cheat and not a verified transcript.
- Every lesson offers a **download backup**: you can save your completed notebook and a small progress token file so your work survives a cache wipe or a move to a new machine.

A durable, cross-device, verified grade is intentionally out of scope for this free static version of the course.

## What you will have built by the end

A small GPT, trained by you on Tiny Shakespeare (and then on a corpus you pick in the capstone), producing legible text. Entirely on free tooling.

Start with the **warm-up** in the sidebar, then work through Modules 0 to 10 in order.
