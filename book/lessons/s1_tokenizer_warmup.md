# Warm-up: Build a Tokenizer (Walking Skeleton)

**Runs in your browser.** Nothing to install.

A language model never sees letters. It sees numbers. A **tokenizer** is the two-way bridge: it turns text into a list of integer ids the model can work with, and turns ids back into text. In this warm-up you build the simplest useful tokenizer there is: a **character-level** one, where every distinct character gets its own id.

The one property that matters is round-trip identity. If you encode a string and then decode it, you must get the exact same string back. The same has to hold the other way around for ids. You will build a `CharTokenizer` that satisfies `decode(encode(s)) == s` for any text built from its vocabulary.

This is also the course's walking skeleton: the first lesson that runs end-to-end in your browser, proves the grader works, and records your progress locally. Every lesson after it follows the same shape.

```{jupyterlite} ../../notebooks/s1_tokenizer_warmup.ipynb
:new_tab: False
:width: 100%
:height: 600px
```

```{admonition} How to check your work
:class: tip
Run every cell top to bottom. When your `CharTokenizer` passes the round-trip checks, the grader cell prints **PASS** and your progress is saved in this browser. A deliberately broken tokenizer prints **FAIL** with the failing check named. That is the grader proving it actually discriminates.
```
