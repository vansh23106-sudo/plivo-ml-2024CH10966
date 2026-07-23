# Run Log

## Run 1 — Baseline
- Hypothesis: the provided starter is a stable reference and should be scored before any changes.
- What changed: no code changes; used the original training loop.
- Dev bpb before/after: 2.3718 (baseline)
- Conclusion: this is the reference point. The loss curve is smooth but saturates early, so the learning dynamics are the main issue.

## Run 2 — Optimizer/scheduler tuning
- Hypothesis: the baseline underfits because it uses a constant LR with no warmup or decay, and the optimizer is not weight-decayed.
- What changed: switched from Adam to AdamW, added warmup + cosine decay, and applied gradient clipping.
- Dev bpb before/after: 2.3718 -> 2.2130
- Conclusion: this was the highest-value change. The model converges faster and reaches a lower dev bpb within the same 2,000-step budget.

## Run 3 — Weight tying test
- Hypothesis: tying the token embedding and output projection may improve parameter efficiency and help generalization.
- What changed: set `tie_weights=True` in the model config.
- Dev bpb before/after: 2.2130 -> 2.2420
- Conclusion: this reduced the optimization signal in practice, so the final run keeps the untied head.

## Final selected configuration
- Optimizer: AdamW
- Learning rate: 2e-3 peak
- Warmup: 200 steps
- Decay: cosine to 10% of peak LR
- Gradient clipping: 1.0
- Weight decay: 0.05
- Model: starter GPT, 1,339,840 params, byte-level tokenizer
