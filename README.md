# Wave-LLM: Fourier-Based Attention for Transformers

**Goal**: Replace standard attention layers entirely with wave/FFT-based attention that carries equivalent information but provides better interpretability and potentially faster computation.

## Motivation

Standard attention computes similarity via dot products - opaque and O(n²). FFT is:
- **Invertible** - no information loss
- **Interpretable** - frequency components have meaning (low freq = global, high freq = local)
- **Fast** - O(n log n) vs O(n²)
- **Structured** - natural basis for compositional semantics

The hypothesis: if we can show FFT-based attention works as well as standard attention, we gain interpretability and speed for free.

## Current Status: Proof of Concept

### Experiment Results (AG News Classification)

| Model | Test Accuracy | Trainable Params | Notes |
|-------|---------------|------------------|-------|
| Baseline (Frozen Qwen3 + Classifier) | **89.30%** | 1.05M | Simple pooling + MLP |
| Wave Adapter (Frozen Qwen3 + 2 Wave Layers) | **88.00%** | 26.2M | Still improving at epoch 3 |

**Interpretation**: Wave attention achieves ~98% of baseline performance. Not a clear win yet, but:
- Wave was still improving (85.3% → 86.1% → 88.0%)
- AG News is simple topic classification - may not need compositional structure
- Current setup uses residual connections, so wave layers only need to *add* value, not *carry* all information

### Key Findings: Embeddings Have Frequency Structure

From analysis of Qwen3 embeddings:
```
Amplitude Similarity: 0.79-0.90 for ALL token pairs (shared structure)
Phase Coherence:      0.13-0.59, correlates with semantic similarity
```

**Insight**: Amplitude is shared basis, **phase carries semantics**!

### Phase Attention Clusters Semantically

```
Within-group / Between-group attention ratio:
  Standard Attention:     ~1.00 (no clustering)
  Phase Attention:        ~1.35 (clusters semantically)
  Amp-Weighted Phase:     ~1.86 (best clustering!)
```

## Future Directions

### 1. Full Layer Replacement (Primary Goal)
Replace actual Qwen attention layers with wave equivalent:
- Remove residual dependency - wave layer must carry ALL information
- Match param count exactly for fair comparison
- Train from scratch on larger tasks

### 2. Interpretability Extraction
Once trained, extract meaning from wave layers:
- Which frequencies does each layer amplify/suppress?
- What do phase shifts correspond to semantically?
- Can we see "reasoning steps" as phase rotations?

Like our grokking experiments where Fourier components directly corresponded to modular arithmetic structure, we want to find similar correspondences in language.

### 3. Speed Benchmarks
Compare against standard attention:
- FNet showed 7x speedup with ~92% BERT performance
- Our approach keeps learned projections, so should be between FNet and full attention

### 4. Harder Tasks
AG News is too simple. Try:
- Multi-hop reasoning (requires composition)
- Semantic similarity (where phase coherence should shine)
- Long-context tasks (where O(n log n) helps)

## Architecture: Wave Attention Layer

```python
class WaveAttentionLayer:
    """
    Instead of: score = Q · K / √d
    We compute: score = Σ cos(phase_Q - phase_K) * amplitude_weight
    
    Key insight: Phase alignment captures semantic similarity!
    """
    
    def forward(self, x):
        Q, K, V = project(x)
        
        # FFT to get amplitude and phase
        Q_fft, K_fft = fft(Q), fft(K)
        Q_amp, Q_phase = abs(Q_fft), angle(Q_fft)
        K_amp, K_phase = abs(K_fft), angle(K_fft)
        
        # Amplitude-weighted phase alignment
        amp_weights = sqrt(Q_amp * K_amp)
        phase_alignment = cos(Q_phase - K_phase)
        scores = sum(phase_alignment * amp_weights)
        
        # Standard attention from here
        return softmax(scores) @ V
```

## Quick Start

```bash
# Analyze embedding frequency structure
./run.sh

# Compare attention mechanisms  
./run_phase_experiment.sh

# Train wave adapter (main experiment)
./run_training.sh
```

## Files

| File | Description |
|------|-------------|
| `train_wave_adapter.py` | Main training script - wave vs baseline |
| `visualize_waves.py` | FFT analysis of token embeddings |
| `phase_attention_experiment.py` | Compare attention mechanisms |
| `wave_utils.py` | Utility functions |
| `results/` | Experiment outputs |

## References

- Wave Network: https://arxiv.org/abs/2411.02674
- FNet: https://arxiv.org/abs/2105.03824  
- SIREN: https://arxiv.org/abs/2006.09661

## Related Work

This builds on experiments with Fourier analysis of neural network internals, where we found that networks learning modular arithmetic develop clean Fourier representations that directly correspond to the mathematical structure. The hypothesis is that similar structure exists in language models and can be made explicit/interpretable via wave-based attention.