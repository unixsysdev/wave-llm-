# Future Directions: Replacing Attention with Waves

This document outlines approaches for completely replacing transformer attention layers with FFT/wave-based alternatives.

## The Goal

Replace standard attention:
```python
scores = Q @ K.T / sqrt(d)
output = softmax(scores) @ V
```

With frequency-domain operations that:
- Carry equivalent information (FFT is invertible)
- Are more interpretable (frequency components have meaning)
- Are potentially faster (O(n log n) vs O(n²))

---

## Approach 1: Distillation (Teacher-Student)

**Idea**: Train FFT layers to mimic what attention layers already do.

```python
# Qwen attention layer = teacher (frozen)
# FFT layer = student (trainable)

teacher = QwenAttentionLayer.from_pretrained()
student = FFTMixingLayer(hidden_size)

for x in data:
    with torch.no_grad():
        teacher_out = teacher(x)
    
    student_out = student(x)
    loss = mse_loss(student_out, teacher_out)
    loss.backward()
```

**Pros**:
- No labels needed, just input data
- Clear signal - match the teacher exactly
- Can verify layer-by-layer that FFT captures attention behavior

**Cons**:
- Only learns what attention *did*, not what's optimal
- May inherit attention's limitations
- Needs representative input distribution

**What we'd learn**: Can FFT even approximate attention? What frequencies correspond to what attention patterns?

---

## Approach 2: Layer-by-Layer Replacement

**Idea**: Gradually swap attention layers, fine-tune between each swap.

```
Step 1: Replace layer 0 attention → FFT, freeze layers 1-N
        Train on language modeling until converged
        
Step 2: Replace layer 1 attention → FFT, freeze layers 2-N
        Train until converged
        
...repeat until all layers replaced...
```

**Pros**:
- Model stays functional throughout
- Each FFT layer learns in context of working model
- Can stop early if quality drops too much

**Cons**:
- Slow - N training runs for N layers
- Later layers may depend on attention patterns from earlier layers
- Error accumulation possible

**Variant**: Replace all even layers first, then odd layers. Or group by function (early layers vs late layers behave differently).

---

## Approach 3: End-to-End Fine-tuning

**Idea**: Replace all attention at once, fine-tune entire model on downstream task.

```python
model = load_qwen()

# Replace all attention layers
for layer in model.layers:
    layer.attention = FFTMixingLayer(config)

# Fine-tune on task
for batch in task_data:
    loss = model(batch)
    loss.backward()
```

**Pros**:
- Learns what FFT layers *should* do, not what attention *did*
- End-to-end optimization, no error accumulation
- Most flexible - model adapts holistically

**Cons**:
- Needs lots of data and compute
- May be unstable at start (all attention gone)
- Harder to interpret what's learned

**Variant**: Initialize FFT layers via distillation first, then fine-tune end-to-end.

---

## Approach 4: Train from Scratch (The Grokking Approach)

**Idea**: Build a small transformer with FFT instead of attention, train from scratch.

```python
class WaveTransformer(nn.Module):
    def __init__(self, n_layers, hidden_size):
        self.embed = Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            WaveTransformerBlock(hidden_size)  # FFT, no attention
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden_size, vocab_size)
```

**Tasks to try**:
1. **Modular arithmetic** (like grokking experiments) - we know Fourier structure emerges, can we make it explicit?
2. **Simple classification** (AG News, sentiment)
3. **Character-level language modeling** (small vocab, clear patterns)
4. **Algorithmic tasks** (copying, reversing, sorting)

**Pros**:
- Cleanest comparison - no pretrained baggage
- Can study what FFT layers learn from scratch
- Directly tests: is attention necessary, or just convenient?

**Cons**:
- Won't match pretrained model quality
- Need to train from scratch (expensive for large models)
- May miss inductive biases that attention provides

**What we'd learn**: Do FFT layers grok? How do they represent structure? Can we read the algorithm from frequency weights?

---

## Recommended Order

### Phase 1: Validate (Current)
✅ Show wave attention can learn *something* useful
✅ Analyze frequency structure in pretrained embeddings
→ Current experiments

### Phase 2: Distillation (Next)
Single layer distillation:
- Take Qwen layer 0
- Train FFT layer to match its output
- Visualize: what frequencies capture what?

### Phase 3: Small-scale from scratch
Build 2-4 layer WaveTransformer:
- Train on modular arithmetic (connect to grokking)
- Train on simple NLP task
- Compare grokking dynamics to standard transformer

### Phase 4: Full replacement
If Phases 2-3 work:
- Layer-by-layer replacement of full Qwen
- Benchmark quality vs speed
- Extract interpretable frequency signatures

---

## FFT Layer Variants to Try

### Variant A: Pure FNet (no learning in mixing)
```python
def forward(x):
    return fft(fft(x, dim=seq), dim=hidden).real
```
Simplest. No learned params in mixing. FNet got 92% of BERT.

### Variant B: Learned frequency gates
```python
def forward(x):
    x_fft = fft(x, dim=seq)
    x_fft = x_fft * self.freq_gate  # [num_freqs] learned
    return ifft(x_fft)
```
Learn which frequencies matter. Very few params.

### Variant C: Learned phase shifts
```python
def forward(x):
    x_fft = fft(x, dim=seq)
    x_fft = x_fft * exp(1j * self.phase_shift)  # learned rotation
    return ifft(x_fft)
```
Phase shift = information routing. Interpretable.

### Variant D: Full complex transform (current approach)
```python
def forward(x):
    Q, K, V = project(x)
    Q_fft, K_fft = fft(Q), fft(K)
    scores = phase_alignment(Q_fft, K_fft)
    return softmax(scores) @ V
```
Keeps Q/K/V structure, replaces dot-product with phase alignment.

### Variant E: Frequency-band attention
```python
def forward(x):
    x_fft = fft(x, dim=seq)
    
    low_freq = x_fft[:, :10]   # global patterns
    high_freq = x_fft[:, 10:]  # local patterns
    
    # Different processing per band
    low_out = self.global_mix(low_freq)
    high_out = self.local_attn(high_freq)  # small window
    
    return ifft(concat(low_out, high_out))
```
Hybrid: FFT for global, attention for local.

---

## Metrics to Track

1. **Task performance** - accuracy, perplexity
2. **Speed** - tokens/second, memory usage
3. **Interpretability** - can we read meaning from frequency weights?
4. **Grokking dynamics** - does it generalize suddenly? When?
5. **Frequency utilization** - which frequencies get used? Dead frequencies?

---

## Key Questions to Answer

1. **Can FFT match attention output?** (Distillation will tell us)
2. **Is attention necessary for grokking?** (From-scratch will tell us)
3. **What do frequencies mean semantically?** (Analysis after training)
4. **Is there a speed/quality tradeoff?** (Benchmarking)
5. **Does FFT provide new capabilities?** (Interpretability, efficiency)

---

## New Experiments to Try (from recent discussion)

### Experiment A: No MLP

**Question**: Is the MLP even necessary when using FFT mixing?

For modular arithmetic, the Fourier solution is:
- Embed a, b as phases
- Multiply phases (add in frequency domain)
- Read off result

If FFT mixing already does frequency decomposition, the MLP might just be a simple readout.

```python
class WaveTransformerNoMLP(nn.Module):
    def forward(self, a, b):
        x = embed(tokens) + pos_embed
        x = x + wave_mix(x)  # FFT mixing only
        # NO MLP - skip entirely
        return unembed(x[:, 2, :])  # direct readout
```

**If this groks** → MLP is unnecessary for frequency-based tasks, wave mixing does everything
**If this fails** → MLP is doing essential computation even with FFT mixing

**Prediction**: For modular arithmetic, no MLP might work. For language, probably still needed.

---

### Experiment B: Wave Network Style (Token2Wave)

**From**: [Wave Network paper](https://arxiv.org/abs/2411.02674)

Different from our current approach. They explicitly construct:
- **Magnitude** = Global semantics (L2 norm across all tokens per dimension)
- **Phase** = Local semantics (each token's relationship to global)

```python
def token2wave(embeddings):
    # embeddings: (batch, seq, hidden)
    
    # Global magnitude: energy across all tokens per dimension
    G = torch.sqrt((embeddings ** 2).sum(dim=1))  # (batch, hidden)
    
    # Phase: each token's relation to global
    # α_j,k = arctan2(sqrt(1 - (w_j,k/G_k)²), w_j,k/G_k)
    ratio = embeddings / G.unsqueeze(1)  # (batch, seq, hidden)
    phase = torch.atan2(torch.sqrt(1 - ratio**2), ratio)
    
    # Complex representation
    Z = G.unsqueeze(1) * torch.exp(1j * phase)  # (batch, seq, hidden)
    
    return Z

def wave_interference(Z):
    # Mix via addition (wave interference)
    return Z.sum(dim=1)  # (batch, hidden)

def wave_modulation(Z):
    # Mix via multiplication (wave modulation)
    return Z.prod(dim=1)  # (batch, hidden)
```

**Key differences from our approach:**

| Aspect | Our wave_full | Wave Network |
|--------|---------------|--------------|
| Magnitude | From FFT of Q/K | Explicit global sum |
| Phase | From FFT of Q/K | Token→global relationship |
| Mixing | Pairwise scores O(n²) | Add/multiply vectors O(n) |
| Interpretability | Indirect | Direct: "how does token relate to whole" |

**Results from their paper**:
- Single-layer Wave Network: 91.66% on AG News
- Beats single Transformer layer by ~20%
- 77% less memory, 85% less training time vs BERT

**Worth testing**: Implement Token2Wave + interference/modulation in our grokking setup.

---

### Experiment C: Remove Residual Connection

**Question**: Can wave layers BE the information path, not just add to it?

Currently:
```python
x = x + wave_mix(x)  # residual - wave only needs to ADD value
```

More aggressive:
```python
x = wave_mix(x)  # no residual - wave must CARRY all information
```

Or with learnable interpolation:
```python
x = self.residual_weight * x + (1 - self.residual_weight) * wave_mix(x)
# Anneal residual_weight from 1→0 during training
```

**This tests**: Can wave ops be the *only* path? True layer replacement, not just augmentation.

---

### Experiment D: Stacking Wave Layers

**Question**: What does depth do for wave networks?

Multiple layers with different learned phase shifts:
```python
class DeepWaveNet(nn.Module):
    def __init__(self, n_layers):
        self.layers = nn.ModuleList([
            WaveLayer(learned_gate=True, learned_phase=True)
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x
```

**Hypothesis**: Each layer could learn different frequency relationships:
- Layer 1: Local patterns (high frequency)
- Layer 2: Phrase structure (medium frequency)
- Layer 3: Global semantics (low frequency)

Phase accumulates: total rotation = sum of all layer phase shifts.

Like wavelets but learned end-to-end.

---

### Experiment E: Hybrid Architectures

**Question**: What's the optimal mix of attention and FFT?

Options:
1. **Alternating**: attention, FFT, attention, FFT...
2. **FFT early, attention late**: cheap mixing first, expensive precision later
3. **Attention early, FFT late**: learn local patterns, then global mixing
4. **Frequency-split**: FFT for low frequencies (global), attention for high (local)

```python
class HybridLayer(nn.Module):
    def forward(self, x):
        x_fft = fft(x, dim=seq)
        
        # Split by frequency
        low = x_fft[:, :n_low, :]   # FFT mixing (cheap)
        high = x_fft[:, n_low:, :]  # attention (precise)
        
        low_out = self.fft_mix(low)
        high_out = self.attention(ifft(high))
        
        return ifft(concat(low_out, fft(high_out)))
```

---

### Experiment F: Interpretability Analysis

After training wave models, extract and visualize:

1. **Learned frequency gates**: Which frequencies matter for which task?
2. **Phase shifts per layer**: What rotations did each layer learn?
3. **Frequency usage over training**: Do some frequencies "turn on" at grokking?
4. **Per-class frequency signatures**: Different frequency patterns for different outputs?

```python
def analyze_wave_model(model):
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'freq_gate'):
            plt.plot(layer.freq_gate.detach().cpu())
            plt.title(f'Layer {i} frequency importance')
        
        if hasattr(layer, 'phase_shift'):
            plt.plot(layer.phase_shift.detach().cpu())
            plt.title(f'Layer {i} phase shifts')
```

**Goal**: See if frequency patterns correspond to something meaningful (like modular arithmetic structure in grokking).

---

### Experiment G: Different Tasks

Test wave networks on tasks with different structure:

1. **Modular multiplication** (not just addition) - different Fourier structure
2. **Permutation tasks** - is FFT good for non-periodic structure?
3. **Hierarchical tasks** - nested parentheses, syntax trees
4. **Copying/reversal** - pure position manipulation

**Hypothesis**: FFT excels at periodic/compositional tasks, may struggle with hierarchical/recursive structure.

---

## Summary: Priority Order

1. ✅ **Done**: Basic grokking comparison (attention vs wave modes)
2. **Next**: No-MLP experiment (is MLP needed with FFT?)
3. **Next**: Wave Network style (Token2Wave + interference)
4. **Then**: Remove residual / deep wave stacking
5. **Then**: Interpretability analysis on trained models
6. **Later**: Hybrid architectures, different tasks, language modeling

---

## References

- **FNet**: https://arxiv.org/abs/2105.03824 (FFT replaces attention, 7x faster)
- **Wave Network**: https://arxiv.org/abs/2411.02674 (complex-valued token representations)
- **SIREN**: https://arxiv.org/abs/2006.09661 (sinusoidal activations)
- **Grokking**: https://arxiv.org/abs/2201.02177 (delayed generalization, Fourier structure emerges)