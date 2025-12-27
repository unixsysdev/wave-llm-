# Grokking Fourier Analysis - Project Overview

## Purpose
Replication and extension of the Fourier analysis from the paper "Progress Measures for Grokking via Mechanistic Interpretability" (Nanda et al., ICLR 2023). The project investigates how neural networks learn modular arithmetic through Fourier representations.

## Main Research Areas
1. **Small Transformer Grokking** (root directory) - Trains 1-layer transformers on modular addition
2. **Pretrained LLM Analysis** (`qwen3_analysis/`) - Analyzes Qwen3 0.6B for emergent Fourier structure
3. **Emergent Structures** (`emergent_structures/`) - Probes for Fourier structure in cyclic domains
4. **Universal MIRAS Grokking** (`miras_experiment/`) - Advanced experiments achieving universal generalization

## Tech Stack
- **Language**: Python 3
- **Deep Learning**: PyTorch (with MPS support for Apple Silicon)
- **Tensor Operations**: einops (for readable tensor manipulations)
- **Pretrained Models**: Hugging Face Transformers (for Qwen3 analysis)
- **Data Science**: NumPy, Matplotlib
- **Progress Bars**: tqdm

## Project Structure
```
grokking-fourier/
├── model.py               # Core 1-layer transformer architecture
├── train.py               # Training loop with AdamW + weight decay
├── analyze.py             # Fourier analysis and plotting
├── run.sh                 # Quick run script
├── qwen3_analysis/        # Pretrained LLM analysis
├── miras_experiment/      # MIRAS/Universal Grokking experiments
├── emergent_structures/   # Multi-domain probing experiments
├── polynomial_test/       # Polynomial vs Fourier analysis
├── legacy_v1/             # Old experimental code
└── venv/                  # Virtual environment
```

## Key Concepts
- **Grokking**: Delayed generalization where models memorize first, then learn generalizable algorithm
- **Fourier Algorithm**: Networks learn to embed inputs as sin/cos components and use trig identities
- **Weight Decay**: Critical hyperparameter for inducing grokking
- **Key Frequencies**: Sparse set of frequencies the model uses (e.g., [4, 11, 14, 26, 35] for p=113)
