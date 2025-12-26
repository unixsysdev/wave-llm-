# Distillation: Attention → Wave

Distill Qwen3-0.6B's attention mechanism into wave layers.

## Goal

Keep the MLPs (reasoning/computation) and replace attention (routing) with wave layers.

```
Teacher: Qwen3-0.6B with attention (frozen)
Student: Qwen3-0.6B embeddings + wave layers + Qwen3-0.6B MLPs
```

## Architecture

```
                    Teacher                          Student
                    ───────                          ───────
                    
Input Tokens        Input Tokens
     ↓                   ↓
┌─────────┐         ┌─────────┐
│ Embed   │ (same)  │ Embed   │  ← frozen
└────┬────┘         └────┬────┘
     ↓                   ↓
┌─────────┐         ┌─────────┐
│ Attn    │ ──────→ │ Wave    │  ← TRAINED (distill from attention)
└────┬────┘         └────┬────┘
     ↓                   ↓
┌─────────┐         ┌─────────┐
│ MLP     │ (same)  │ MLP     │  ← frozen
└────┬────┘         └────┬────┘
     ↓                   ↓
    ...                 ...
     ↓                   ↓
┌─────────┐         ┌─────────┐
│ LM Head │ (same)  │ LM Head │  ← frozen
└────┬────┘         └────┬────┘
     ↓                   ↓
   Logits             Logits
```

## Wave Layer Types

| Type | Description | Complexity | Params |
|------|-------------|------------|--------|
| `learned_gate` | FFT + learned frequency gates | O(n log n) | ~2M |
| `fnet` | Pure FFT, no learned params | O(n log n) | 0 |
| `wave_network` | Token2Wave + interference | O(n) | ~2M |
| `frequency_band` | FFT low + attention high | O(n log n + k²) | ~4M |

## Quick Start

```bash
./run_distill.sh
```

Or customize:

```bash
python distill_wave_qwen.py \
    --teacher "Qwen/Qwen3-0.6B" \
    --wave_type learned_gate \
    --n_samples 5000 \
    --n_epochs 5 \
    --output_dir results
```

## What It Does

1. Loads Qwen3-0.6B as teacher (frozen)
2. Creates student with wave layers replacing attention
3. Trains wave layers via:
   - KL divergence on logits (soft targets)
   - Cross-entropy on next token (hard targets)
4. Evaluates top-5 token agreement between teacher and student

## Expected Output

```
Epoch 1: loss=X.XX, kl=X.XX, ce=X.XX
Epoch 2: loss=X.XX, kl=X.XX, ce=X.XX
Epoch 3: loss=X.XX, kl=X.XX, ce=X.XX

Top-5 Token Agreement: XX.X%

Sample outputs:
  Prompt: The capital of France is
  Teacher: The capital of France is Paris, which is...
  Agreement: XX%
```

## Success Criteria

- **>80% top-5 agreement**: Wave can mostly replicate attention routing
- **>90% agreement**: Wave is a good attention replacement
- **>95% agreement**: Wave is nearly equivalent to attention

## Files

| File | Description |
|------|-------------|
| `wave_layers.py` | Wave layer implementations |
| `distill_wave_qwen.py` | Main distillation script |
| `run_distill.sh` | Quick run script |
| `results/` | Saved models and metrics |

## Next Steps

If distillation works (>80% agreement):
1. Try more training data/epochs
2. Compare wave types (which works best?)
3. Evaluate on downstream tasks
4. Try distilling from larger teacher (8B, 14B)

If distillation fails (<60% agreement):
1. Try more expressive wave layers (frequency_band)
2. Unfreeze some MLP layers
3. Investigate which layers are hardest to distill
