# Grokking Experiments: Attention vs Waves

Direct comparison of standard attention vs FFT-based mixing on modular arithmetic.

## The Question

Both models have the same structure (embeddings, mixing layer, MLP, unembed).  
Only difference: **how they mix information across tokens**.

- **Standard**: Q @ K.T -> softmax -> @ V (learned pairwise attention)
- **Wave**: FFT-based mixing (various modes)

If both grok → FFT can do what attention does  
If wave groks faster → FFT has better inductive bias for this task  
If wave doesn't grok → attention is necessary (for this task)

## Wave Modes

| Mode | Description | Learned Params |
|------|-------------|----------------|
| `fnet` | Pure FFT, no learning in mixing | 0 |
| `learned_gate` | FFT + learnable frequency gates | 2 × d_model |
| `phase_shift` | FFT + learnable phase rotations | 2 × d_model |
| `full` | Q/K/V projections + phase-alignment scoring | Same as attention |

## Run

```bash
./run_comparison.sh
```

Or with custom settings:
```bash
python train_comparison.py --p 97 --n_epochs 50000 --wave_modes fnet full
```

## Expected Output

```
RESULTS SUMMARY
============================================================
Model                        Params   Test Acc    Fourier
------------------------------------------------------------
standard                     xxx,xxx     1.0000     0.xxxx
wave_fnet                    xxx,xxx     ?.????     0.xxxx
wave_learned_gate            xxx,xxx     ?.????     0.xxxx
wave_phase_shift             xxx,xxx     ?.????     0.xxxx
wave_full                    xxx,xxx     ?.????     0.xxxx

GROKKING ANALYSIS
============================================================
standard                 groks at epoch xxxxx
wave_fnet                groks at epoch ????? (or "did not grok")
...
```

## What to Look For

1. **Does wave grok?** If any wave mode reaches ~100% test accuracy, FFT can replace attention.

2. **Grokking speed**: Which groks first? Faster = better inductive bias.

3. **Fourier strength**: Both should develop Fourier structure, but wave might do it differently.

4. **Learned weights**: For `learned_gate` and `phase_shift`, what did it learn?
   - Which frequencies matter?
   - What phase shifts emerged?

## Files

| File | Description |
|------|-------------|
| `models.py` | StandardAttentionTransformer + WaveTransformer |
| `train_comparison.py` | Training loop with Fourier metrics |
| `run_comparison.sh` | Quick run script |
| `results/` | JSON results with full history |
