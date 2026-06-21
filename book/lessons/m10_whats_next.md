# Module 10. What's Next: How Real LLMs Differ

**Just reading.** Nothing to run.

You built a small GPT from the ground up. Production language models (GPT-4, Claude, Llama, and friends) are the same core idea scaled up and refined in several directions. This closing module is a short tour of what changes between your model and theirs, so you know what the words mean when you read about them.

## What stays the same

The skeleton you built is the real skeleton: token plus positional embeddings, stacked transformer blocks (attention plus MLP with residuals and normalization), a final projection to vocabulary logits, trained by next-token prediction. That core does not change. Everything below is refinement, not replacement.

## What scales and changes

- **Tokenization.** Real models use **subword** tokenizers (BPE, SentencePiece), not one-token-per-character. Fewer tokens per sentence means longer effective context and faster training.
- **Scale.** Billions of parameters instead of millions; trillions of training tokens instead of one megabyte; thousands of GPUs instead of one.
- **Attention efficiency.** FlashAttention, KV-caching, grouped-query attention, and longer context windows make attention fast and cheap at scale.
- **Position information.** Learned positional embeddings give way to **rotary** (RoPE) or relative schemes that generalize to longer sequences.
- **Architecture tweaks.** RMSNorm instead of LayerNorm, SwiGLU instead of plain GELU MLPs, and sometimes **mixture-of-experts** layers.
- **Training objectives beyond pretraining.** Base models are then **fine-tuned** and aligned (instruction tuning, RLHF, preference optimization), which is what turns a text predictor into a helpful assistant.
- **Inference quality.** Sampling strategies (temperature, top-k, top-p), speculative decoding, and quantization for fast, cheap serving.

## Where to go from here

- Read Karpathy's **nanoGPT** and **micrograd**. You now have the background to follow them line by line.
- Re-run your capstone bigger: more layers, more data, longer training.
- Explore an open-weights model and map each part back to what you built here.

```{admonition} Mark the course complete
:class: tip
Run the "mark complete" cell below to record that you finished the course in this browser. As always, this is local-only. Use the download backup to keep a copy of your progress.
```

```{jupyterlite} ../../notebooks/m10_whats_next.ipynb
:new_tab: False
:width: 100%
:height: 400px
```
