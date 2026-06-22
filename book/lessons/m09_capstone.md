# Module 9. Capstone: Make It Yours

**Runs on Google Colab.** You need a free Google account.

## Why this matters

This is your victory lap. Instead of Shakespeare, you bring your own text corpus (song lyrics, code, a book you love, your own writing, chat logs) and train the same GPT on it. The model will pick up the vocabulary, rhythm, and quirks of whatever you feed it.

You have done every step before. This module is about ownership. Pick a corpus. Train. Sample. See your model speak in a voice you chose.

## What you will do

1. **Pick a corpus.** A `.txt` file you upload, or text you paste in. Aim for at least a few hundred KB. More is better.
2. **Train.** The pipeline is identical to Module 8. The only thing that changes is the data.
3. **Sample.** Generate text from your trained model. Read it. Notice what it learned and what it missed.
4. **Submit.** Paste the final token into the browser grader. Optionally download your trained weights as a checkpoint, the same way real ML research keeps trained models.

## What is actually happening

The model has no idea what your corpus is "about". It only sees the local statistics: which characters follow which, conditioned on the previous block of characters. But those local statistics, summed over enough text, capture an astonishing amount: spelling rules, common words, sentence structure, even a hint of style.

Training on code gives you a model that produces code-like indentation, brackets, and keywords. Training on a single author gives you a model that mimics their cadence. Training on song lyrics gives you a model that rhymes (occasionally, and badly). The same architecture, same loss, same optimizer. Only the data changes.

## Picking a corpus

Some good sources:

* **Project Gutenberg**, free public-domain books. <https://www.gutenberg.org/>
* **The Internet Archive's text collection.** <https://archive.org/details/texts>
* Your own writing: emails, notes, journals, blog posts. Surprisingly fun.
* Source code from a public repository, if you want to see what code-completion was like before LLMs.
* Song lyrics, but watch your copyright if you plan to publish anything.

Watch out for:

* **File encoding.** Stick to UTF-8 or ASCII. Weird encodings will produce strange tokenizer vocabularies.
* **Length.** Less than ~100 KB and the model has nothing to learn. More than ~5 MB and free Colab will take a while.
* **Quality.** Garbage in, garbage out. The model learns whatever patterns are in the data, including typos.

## A moment of curiosity

Real LLMs are trained on **trillions** of tokens of carefully cleaned and deduplicated text from the open web. You are training on at most a megabyte or two. Your model is several orders of magnitude smaller and dumber, but it lets you see the *shape* of how a language model picks up style.

What if you trained on a mix of two corpora? You would get a model that interpolates between the two. Mix Shakespeare with Python source code, and you would get strange (and frankly entertaining) text that has indented blocks of mock-iambic-pentameter pseudocode.

What if you fine-tuned a pretrained model on your corpus instead of training from scratch? This is what real applications do: take a general-purpose model and fine-tune it on a smaller, task-specific dataset. Much cheaper, much better results. Module 10 points you at how to do this.

## Common confusions

* **The model is not "you".** It has captured statistical patterns of your text, not your thoughts. Don't read too much into the samples.
* **Bigger is not always better for small corpora.** A 10M-parameter model on 100 KB of text will overfit hard (memorize). Sometimes a smaller model generalizes better.
* **The capstone is a single short run.** A real research run would be days long with careful validation and hyperparameter search. You are seeing the small-scale version on purpose, to keep it free and fast.

## Going deeper

* Karpathy, **Software 2.0** (essay). Why training on data is increasingly how we build software. <https://karpathy.medium.com/software-2-0-a64152b37c35>
* Hugging Face, **Datasets** library. The standard way to load and prepare text datasets at scale. <https://huggingface.co/docs/datasets/>
* Karpathy's **nanoGPT capstone branch**: how to bring your own corpus, the cleaned-up version. <https://github.com/karpathy/nanoGPT>

```{admonition} Open this lesson in Colab
:class: tip
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vritti-Dev/foundation-model/blob/main/notebooks/m09_capstone.ipynb)

The button opens `notebooks/m09_capstone.ipynb` in Google Colab. Swap in your own corpus where the notebook marks it, then run all cells from the top.
```

```{admonition} You built this
:class: note
If you reached here, you went from basic Python to a transformer language model you designed, built by hand, assembled in PyTorch, and trained on text you chose. Entirely on free tooling. Module 10 is a short read on where to go next.
```
