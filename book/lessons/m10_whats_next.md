# Module 10. What's Next: How Real LLMs Differ

**Just reading.** Nothing to run.

## Why this matters

You built a small GPT from the ground up. Production language models (GPT-4, Claude, Llama, Gemini) are the same core idea scaled up and refined in several specific directions. This closing module is a short tour of what changes between your model and theirs, so you know what the words mean when you read about them.

The good news: nothing in modern LLMs is fundamentally beyond what you have just built. Each refinement is a clear, named improvement on top of the spine you already know.

## What stays the same

The skeleton you built is the real skeleton:

* Token and positional embeddings.
* Stacked transformer blocks (attention plus MLP, with residuals and normalization).
* A final projection to vocabulary logits.
* Trained by next-token prediction.

That core does not change. Everything below is refinement, not replacement.

## What scales and changes

### Tokenization

Real models use **subword** tokenizers (BPE, SentencePiece, WordPiece) instead of one token per character. Common words get one token. Rare words get split into several. This compresses sequences dramatically: "tokenization" in GPT-4 is 2 tokens, not 12 characters.

Read: Karpathy, **Let's build the GPT Tokenizer**. <https://www.youtube.com/watch?v=zduSFxRajkE>

### Scale

* Your model: 0.8 M to 10 M parameters, trained on 1 MB of text.
* GPT-2: 1.5 billion parameters, trained on 40 GB.
* GPT-3: 175 billion, trained on hundreds of GB.
* GPT-4 and Claude: undisclosed, but estimated trillions of parameters trained on trillions of tokens.

The architecture barely changes. What changes is everything *around* it: data pipelines, GPU clusters, distributed training, checkpointing, monitoring.

Read: **Chinchilla scaling laws** (Hoffmann et al. 2022). <https://arxiv.org/abs/2203.15556>

### Attention efficiency

Long contexts are expensive: vanilla attention is `O(T^2)` in compute and memory. Real implementations use:

* **FlashAttention** for memory-efficient kernels. <https://arxiv.org/abs/2205.14135>
* **KV-caching** to avoid recomputing attention over the past at each generation step.
* **Grouped-query attention** (GQA) and **multi-query attention** (MQA) to share keys and values across heads.
* **Sliding-window** and **sparse** attention for very long contexts.

### Position information

Learned positional embeddings give way to **rotary embeddings** (RoPE) or **ALiBi**, schemes that generalize to longer sequences than seen at training time. Llama, Mistral, and most open models use RoPE.

Read: **RoFormer** (Su et al. 2021), the rotary embeddings paper. <https://arxiv.org/abs/2104.09864>

### Architecture tweaks

* **RMSNorm** instead of LayerNorm (slightly faster, comparable quality).
* **SwiGLU** instead of plain GELU MLPs.
* Sometimes **mixture-of-experts (MoE)** layers: only a few of many MLPs activate per token, giving you more parameters at the same compute cost. Mixtral, GPT-4, and DeepSeek use MoE.

### Training objectives beyond pretraining

Base models like the one you built are just trained to predict the next token. To turn them into useful assistants, you do two more rounds:

* **Supervised fine-tuning (SFT)** on curated instruction-following examples. Teaches the model to follow human-written prompts.
* **Reinforcement learning from human feedback (RLHF)**, or its modern variants **DPO**, **KTO**, **ORPO**. Teaches the model to prefer responses humans would prefer.

This pipeline (pretrain -> SFT -> RLHF) is what turns a fancy autocomplete into a useful chatbot.

Read: **InstructGPT** (Ouyang et al. 2022). <https://arxiv.org/abs/2203.02155>

### Inference quality

Sampling strategies matter for the final user experience:

* **Temperature** controls how confident the sampling is.
* **Top-k** restricts choices to the k most likely tokens.
* **Top-p (nucleus)** restricts to the smallest set of tokens whose cumulative probability exceeds p.
* **Speculative decoding** runs a small draft model alongside the big one to speed up generation.
* **Quantization** stores weights in 8-bit or 4-bit instead of 16-bit, enabling laptop-class inference.

## Where to go from here

If you enjoyed building this, pick one and go deep:

* **Reproduce GPT-2.** Karpathy has a 4-hour video doing exactly this. <https://www.youtube.com/watch?v=l8pRSuU81PU>
* **Read papers.** Start with the original transformer (Vaswani 2017), then GPT-1, GPT-2, GPT-3, Chinchilla, Llama. They build on each other.
* **Fine-tune an open model.** Take Llama-3 or Mistral, fine-tune it on a dataset you care about. Hugging Face's PEFT library makes this easy. <https://huggingface.co/docs/peft>
* **Read about interpretability.** Anthropic's transformer circuits work is some of the most exciting research in the field. <https://transformer-circuits.pub/>
* **Build something.** A code assistant. A creative writing helper. A model for your domain. The pipeline is the same; the data is what makes it yours.

## A final note

The transformer is not the last word in machine learning. State-space models, diffusion language models, retrieval-augmented architectures, and entirely new ideas are all under active research. The thing you take away from this course is not just "how GPT works", but a way of thinking: **a model is a parameterized function, a loss is an objective, training is optimization, generation is sampling**. That framing applies to every model you will ever meet.

You built a small language model from scratch. You are ready for what comes next.

```{admonition} Mark the course complete
:class: tip
Run the "mark complete" cell below to record that you finished the course in this browser. As always, this is local-only. Use the download backup to keep a copy of your progress.
```

```{jupyterlite} ../../notebooks/m10_whats_next.ipynb
:new_tab: False
:width: 100%
:height: 400px
```
