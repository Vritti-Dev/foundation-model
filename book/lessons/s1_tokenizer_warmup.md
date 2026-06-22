# Warm-up: Build a Tokenizer (Walking Skeleton)

**Runs in your browser.** Nothing to install.

## Why this matters

A neural network is a giant pile of numbers being multiplied and added. It cannot read letters. So before any model can do anything with text, somebody has to translate that text into numbers. That somebody is the **tokenizer**.

If the tokenizer is broken or sloppy, every layer above it is working on garbage. The tokenizer is the bridge between the messy world of human language and the clean world of math. Getting it right is non-negotiable.

## What you will build

A **character-level tokenizer**: the simplest tokenizer that exists. It assigns every distinct character in the training text its own integer id. `' '` might become `0`, `'A'` might become `1`, and so on, sorted alphabetically.

You will write a class `CharTokenizer` with two methods:

* `encode(s)` takes a string and returns a list of integer ids.
* `decode(ids)` takes a list of integer ids and returns a string.

The two are **inverses**. That is the one law that must hold. If you encode then decode, you get back the original text. If you decode then encode, you get back the original ids. No surprises, no information lost.

## What is actually happening

The tokenizer scans your training text, makes a sorted list of every unique character it sees, and assigns each one an integer position in that list. That list is the **vocabulary**. Encoding is a lookup in one direction (`char -> id`), decoding is a lookup in the other (`id -> char`).

For Tiny Shakespeare, the vocabulary has 65 characters: the 52 upper and lower case letters, the 10 digits 0 to 9, and a few punctuation marks and whitespace. That is the entire alphabet the model will ever see.

Why 65 instead of 256 (which would cover every possible byte)? Because the text only contains 65 distinct characters. A vocabulary tuned to your data is smaller, which means every input is a shorter list of ids, which means training is faster.

## A moment of curiosity

What would the vocabulary look like for a tokenizer trained on:

* The complete works of Shakespeare? (Roughly 70 to 100 characters with the additional special characters.)
* Wikipedia? (Tens of thousands once you include accents, math symbols, and CJK characters.)
* All of GitHub's source code? (Hundreds, with rare control characters and emojis.)

Notice the trade-off. A bigger vocabulary covers more text without breaking, but every word becomes a longer sequence. Real language models use **subword** tokenizers (BPE, SentencePiece, WordPiece) that learn merge rules: common pairs of characters get fused into single tokens. GPT-4's vocabulary is around 100,000 tokens, most of which are word fragments.

## Common confusions

* **"Token" and "character" are not the same.** In this lesson they happen to coincide because each character gets one id. In a real LLM, one token is often several characters (the word "tokenization" is two tokens in GPT-4: `token` and `ization`).
* **Order of the vocabulary matters for reproducibility.** We sort the characters so two people running the same code get the same ids in the same order.
* **An id is just an integer.** It has no meaning on its own. The meaning emerges when the model learns an embedding vector for each id during training (Module 5).

## Going deeper

Curated, in this order:

* Andrej Karpathy, **Let's build the GPT Tokenizer** (YouTube, ~2 hours). The single best beginner explanation of why real tokenizers use BPE and not characters. <https://www.youtube.com/watch?v=zduSFxRajkE>
* Hugging Face NLP Course, **Tokenizers** chapter. Hands-on with the real tokenizers used in production. <https://huggingface.co/learn/nlp-course/chapter6>
* The original **Byte Pair Encoding** paper (Sennrich et al. 2016), only 9 pages, very readable. <https://arxiv.org/abs/1508.07909>

```{jupyterlite} ../../notebooks/s1_tokenizer_warmup.ipynb
:new_tab: False
:width: 100%
:height: 600px
```

```{admonition} How to check your work
:class: tip
Run every cell top to bottom. When your `CharTokenizer` passes the round-trip checks, the grader cell prints **PASS** and your progress is saved in this browser. A deliberately broken tokenizer prints **FAIL** with the failing check named. That is the grader proving it actually discriminates.
```
