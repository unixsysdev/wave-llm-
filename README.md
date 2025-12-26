# Wave-LLM: Fourier-Based Attention for Transformers

**Goal**: Replace standard attention layers entirely with wave/FFT-based attention that carries equivalent information but provides better interpretability and faster computation.

## 🔥 Key Results: FFT Beats Attention on Grokking

We tested standard attention vs FFT-based mixing on modular arithmetic (the grokking task). **FFT dramatically outperforms attention:**

### Results (p=53, train_frac=0.3, 30k epochs)

| Model | Test Acc | Grok Epoch | Speed | Params |
|-------|----------|------------|-------|--------|
| Standard Attention | **2.2%** | ❌ NEVER | 92 it/s | 211k |
| Wave (fnet) | **99.4%** | 27,400 | 240 it/s | 146k |
| Wave (learned_gate) | **99.8%** | **2,600** | 233 it/s | 146k |
| Wave (phase_shift) | **93.4%** | ~28,000 | 233 it/s | 146k |
| Wave (full) | **100%** | 23,300 | 94 it/s | 211k |

### Key Findings

1. **Standard attention FAILED to grok** - stuck at 2.2% (random chance)
2. **learned_gate groks 10x faster** than any other method (epoch 2,600 vs 23,000+)
3. **FFT is 2.5x faster** - 240 it/s vs 92 it/s
4. **Fewer params, better results** - 146k params beats 211k params

### Why FFT Wins

Modular arithmetic naturally lives in Fourier space. Standard attention must *discover* this structure through training. FFT has it **baked into the architecture** - the inductive bias is perfect for the task.

This suggests: **for tasks with periodic/compositional structure, FFT-based mixing may be fundamentally better than learned attention.**

---

## Motivation

Standard attention computes similarity via dot products - opaque and O(n²). FFT is:
- **Invertible** - no information loss
- **Interpretable** - frequency components have meaning (low freq = global, high freq = local)
- **Fast** - O(n log n) vs O(n²)
- **Structured** - natural basis for compositional semantics

## Experiment Details

### Grokking Comparison (experiments/)

We built identical 1-layer transformers, varying only the mixing mechanism:

**Standard Attention:**
```python
scores = Q @ K.T / sqrt(d)
output = softmax(scores) @ V
```

**Wave (fnet):** Pure FFT, no learned params in mixing
```python
output = fft(fft(x, dim=seq), dim=hidden).real
```

**Wave (learned_gate):** FFT + learnable frequency amplification
```python
x_fft = rfft(x, dim=seq)
x_fft = x_fft * self.freq_gate  # learned: which frequencies matter
output = irfft(x_fft)
```

**Wave (phase_shift):** FFT + learnable phase rotations
```python
x_fft = rfft(x, dim=seq)
x_fft = x_fft * exp(1j * self.phase_shift)  # learned: phase rotations
output = irfft(x_fft)
```

**Wave (full):** Q/K/V projections + phase-alignment scoring
```python
q_fft, k_fft = fft(Q), fft(K)
scores = cos(angle(q_fft) - angle(k_fft)) * amplitude
output = softmax(scores) @ V
```

### AG News Classification (train_wave_adapter.py)

| Model | Test Accuracy | Trainable Params |
|-------|---------------|------------------|
| Baseline (Frozen Qwen3 + Classifier) | **89.30%** | 1.05M |
| Wave Adapter (Frozen Qwen3 + 2 Wave Layers) | **88.00%** | 26.2M |

Wave adapter achieves ~98% of baseline on simple classification. The real wins are on tasks requiring compositional structure (like modular arithmetic).

### Embedding Analysis (visualize_waves.py)

From analysis of Qwen3 embeddings:
```
Amplitude Similarity: 0.79-0.90 for ALL token pairs (shared structure)
Phase Coherence:      0.13-0.59, correlates with semantic similarity
```

**Insight**: Amplitude is shared basis, **phase carries semantics**!

### Phase Attention Clustering (phase_attention_experiment.py)

```
Within-group / Between-group attention ratio:
  Standard Attention:     ~1.00 (no clustering)
  Phase Attention:        ~1.35 (clusters semantically)
  Amp-Weighted Phase:     ~1.86 (best clustering!)
```

---

## Quick Start

```bash
# Run grokking comparison (the main result)
cd experiments
./run_comparison.sh

# Run with visualizations (saves frames every 500 epochs)
./run_visual.sh

# Analyze Qwen3 embedding frequency structure
./run.sh

# Compare attention mechanisms  
./run_phase_experiment.sh
```

---

## Future Directions

See [DIRECTIONS.md](DIRECTIONS.md) for detailed roadmap.

1. **Full Layer Replacement** - Replace actual transformer attention layers with FFT
2. **Distillation** - Train FFT layers to mimic attention output
3. **Interpretability** - Extract meaning from learned frequency weights
4. **Harder NLP Tasks** - Test on reasoning, not just classification

---

## Project Structure

```
wave_token_analysis/
├── experiments/           # Grokking comparison (main results)
│   ├── models.py         # Standard + Wave transformers
│   ├── train_comparison.py
│   ├── train_visual.py   # With visualization snapshots
│   └── results/          # JSON results + comparison plots
├── train_wave_adapter.py  # AG News classification
├── visualize_waves.py     # Embedding FFT analysis
├── phase_attention_experiment.py
├── DIRECTIONS.md          # Future work roadmap
└── README.md
```

---

## References

- **FNet**: https://arxiv.org/abs/2105.03824 (FFT replaces attention, 7x faster, 92% BERT)
- **Wave Network**: https://arxiv.org/abs/2411.02674 (complex-valued tokens)
- **SIREN**: https://arxiv.org/abs/2006.09661 (sinusoidal representations)
- **Grokking**: https://arxiv.org/abs/2201.02177 (delayed generalization, Fourier emerges)

---

## Related Work

This builds on our grokking/Fourier experiments where we found that networks learning modular arithmetic develop clean Fourier representations. The key insight: **FFT doesn't need to discover Fourier structure - it's built in.** This gives a massive advantage on tasks where Fourier is the natural basis.