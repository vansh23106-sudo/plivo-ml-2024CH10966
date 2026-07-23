# Training Run Log (llm_handout Speedrun)

## Run 0: Baseline Starter Checkpoint
- **Hypothesis**: Baseline GPT model provided in starter code. 4 layers, 4 heads, `n_embd=160`, `block_size=128`, default byte tokenizer (vocab 256), plain Adam (`lr=3e-4`), no weight decay, no learning rate schedule, no gradient clipping.
- **What Changed**: None (Starter code benchmark).
- **Dev BPB Before / After**: `N/A` -> `2.3718`
- **Parameters**: 1,339,840 / 2,000,000 max
- **Conclusion**: Established baseline benchmark of **2.3718 bpb** on `dev_eval.txt` over 2,000 steps. Training loss stalled around ~1.73. The model suffers from:
  1. No learning rate schedule / decay.
  2. No weight tying (embedding and lm_head are separate).
  3. Simple byte-level encoding: Devanagari text takes 3 tokens per character, severely restricting context length in tokens.
  4. Standard absolute position embeddings instead of Rotary Position Embeddings (RoPE).
  5. Plain GELU and LayerNorm instead of modern SwiGLU / RMSNorm.

## Run 1: AdamW + Cosine Learning Rate Schedule + Gradient Clipping
- **Hypothesis**: Replacing constant LR plain Adam with AdamW (`weight_decay=0.1`), 100-step linear warmup, cosine LR decay (peak `lr=1.5e-3` decaying to `1.5e-4`), and gradient clipping (`max_norm=1.0`) will accelerate convergence and stabilize updates without altering parameter count.
- **What Changed**: Updated `train.py` with AdamW optimizer, warmup + cosine LR schedule function, and gradient norm clipping.
- **Dev BPB Before / After**: `2.3718` -> `2.2130`
- **Parameters**: 1,339,840 / 2,000,000 max
- **Conclusion**: Significant BPB improvement (**-0.1588 bpb**, from 2.3718 down to 2.2130). Final training loss dropped from ~1.7315 to 1.5690. The cosine schedule allowed using a 5x higher peak learning rate (1.5e-3 vs 3e-4) without divergence, demonstrating the vital importance of optimization dynamics.

## Run 2: Architecture Modernization (RoPE, RMSNorm, SwiGLU, Weight Tying)
- **Hypothesis**: Replacing static positional embeddings with Rotary Position Embeddings (RoPE), LayerNorm with parameter-free RMSNorm, standard GELU with SwiGLU gated MLPs, and enabling Weight Tying (`head.weight = tok_emb.weight`) will increase parameter efficiency and model capacity (expanding from 4 to 5 layers and doubling context block size to 256).
- **What Changed**: Rewrote `model.py` with RoPE, RMSNorm, SwiGLU, and weight tying (`n_layer=5, n_embd=160, n_head=8, block_size=256`).
- **Dev BPB Before / After**: `2.2130` -> `1.9400`
- **Parameters**: 1,629,920 / 2,000,000 max
- **Conclusion**: Exceptional performance gain (**-0.2730 bpb**, breaking under 2.0 to 1.9400). Final training loss plummeted to 1.3040. RoPE and SwiGLU provided superior expressivity per parameter, while weight-tying unlocked budget to add a 5th transformer block.


