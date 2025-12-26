#!/usr/bin/env python3
"""
Phase-Aware Attention Experiment

Based on our finding that:
- Amplitude similarity is ~0.80-0.90 for ALL token pairs (shared structure)
- Phase coherence varies 0.13-0.59 and correlates with semantic similarity

This suggests phase carries semantic information. Let's test if
phase-based attention captures different/better relationships than
standard dot-product attention.

Experiment:
1. Extract embeddings from Qwen3
2. Compare standard attention vs phase-aware attention
3. See which one better captures semantic relationships
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from device_utils import get_device

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not installed")
    sys.exit(1)


class StandardAttention(nn.Module):
    """Standard dot-product attention for comparison."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = np.sqrt(hidden_size)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q, K, V: [batch, seq, hidden]
        Returns:
            output: [batch, seq, hidden]
            attention_weights: [batch, seq, seq]
        """
        # Standard dot product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights


class PhaseAttention(nn.Module):
    """
    Phase-aware attention.
    
    Instead of dot product, uses phase alignment between
    FFT representations of Q and K.
    """
    
    def __init__(self, hidden_size: int, n_freqs: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_freqs = n_freqs or (hidden_size // 2 + 1)  # rfft output size
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention based on phase alignment.
        
        Args:
            Q, K, V: [batch, seq, hidden]
        Returns:
            output: [batch, seq, hidden]
            attention_weights: [batch, seq, seq]
        """
        batch, seq, hidden = Q.shape
        
        # Compute FFT of Q and K
        Q_fft = torch.fft.rfft(Q, dim=-1)  # [batch, seq, n_freqs] complex
        K_fft = torch.fft.rfft(K, dim=-1)  # [batch, seq, n_freqs] complex
        
        # Extract phase
        Q_phase = torch.angle(Q_fft)  # [batch, seq, n_freqs]
        K_phase = torch.angle(K_fft)  # [batch, seq, n_freqs]
        
        # Phase alignment score: cos(phase_q - phase_k) averaged over frequencies
        # High score = phases are aligned = similar semantic content
        phase_diff = Q_phase.unsqueeze(2) - K_phase.unsqueeze(1)  # [batch, seq_q, seq_k, n_freqs]
        phase_alignment = torch.cos(phase_diff).mean(dim=-1)  # [batch, seq_q, seq_k]
        
        # Convert to attention weights
        # Phase alignment is in [-1, 1], shift to [0, 2] then softmax
        scores = phase_alignment + 1.0  # Now in [0, 2]
        weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(weights, V)
        
        return output, weights


class AmplitudeWeightedPhaseAttention(nn.Module):
    """
    Phase attention weighted by amplitude.
    
    Uses amplitude to weight which frequencies matter most
    for the phase comparison.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq, hidden = Q.shape
        
        # Compute FFT
        Q_fft = torch.fft.rfft(Q, dim=-1)
        K_fft = torch.fft.rfft(K, dim=-1)
        
        # Extract amplitude and phase
        Q_amp = torch.abs(Q_fft)
        K_amp = torch.abs(K_fft)
        Q_phase = torch.angle(Q_fft)
        K_phase = torch.angle(K_fft)
        
        # Amplitude weights: geometric mean of Q and K amplitudes
        # This weights important frequencies higher
        amp_weights = torch.sqrt(Q_amp.unsqueeze(2) * K_amp.unsqueeze(1) + 1e-8)  # [batch, seq_q, seq_k, n_freqs]
        amp_weights = amp_weights / (amp_weights.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize
        
        # Phase alignment weighted by amplitude importance
        phase_diff = Q_phase.unsqueeze(2) - K_phase.unsqueeze(1)
        phase_alignment = (torch.cos(phase_diff) * amp_weights).sum(dim=-1)  # [batch, seq_q, seq_k]
        
        # Convert to attention weights
        scores = phase_alignment + 1.0
        weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, V)
        
        return output, weights


class ComplexAttention(nn.Module):
    """
    Full complex-valued attention (like Wave Network).
    
    Uses complex dot product: Re(Q · K*)
    This naturally captures phase relationships.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = np.sqrt(hidden_size // 2)  # Complex dim is half
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq, hidden = Q.shape
        
        # Convert to complex (split hidden dimension)
        Q_c = self._to_complex(Q)  # [batch, seq, hidden//2]
        K_c = self._to_complex(K)
        V_c = self._to_complex(V)
        
        # Complex attention: Re(Q · K*)
        # K.conj() flips the phase, so this measures phase alignment
        scores = torch.real(torch.matmul(Q_c, K_c.conj().transpose(-2, -1))) / self.scale
        weights = F.softmax(scores, dim=-1)
        
        # Apply to complex values, then convert back
        output_c = torch.matmul(weights.to(V_c.dtype), V_c)
        output = self._to_real(output_c)
        
        return output, weights
    
    def _to_complex(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x.chunk(2, dim=-1)
        return torch.complex(real, imag)
    
    def _to_real(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=-1)


def load_model(model_name: str = "Qwen/Qwen3-0.6B"):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def get_embeddings_for_words(model, tokenizer, words: List[str]) -> torch.Tensor:
    """Get embedding vectors for a list of words."""
    embeddings = model.get_input_embeddings().weight.data.cpu()
    
    result = []
    valid_words = []
    for word in words:
        # Try with space prefix
        tokens = tokenizer.encode(f" {word}", add_special_tokens=False)
        if len(tokens) == 1:
            result.append(embeddings[tokens[0]])
            valid_words.append(word)
        else:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            if len(tokens) >= 1:
                result.append(embeddings[tokens[0]])
                valid_words.append(word)
    
    if not result:
        return None, []
    
    return torch.stack(result), valid_words


def compute_attention_similarity_matrix(
    attention_module: nn.Module,
    embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise attention scores between all embeddings.
    
    Args:
        attention_module: Attention implementation
        embeddings: [n_tokens, hidden] token embeddings
        
    Returns:
        attention_matrix: [n_tokens, n_tokens] attention weights
    """
    # Add batch dimension
    emb = embeddings.unsqueeze(0)  # [1, n_tokens, hidden]
    
    with torch.no_grad():
        _, weights = attention_module(emb, emb, emb)
    
    return weights.squeeze(0)  # [n_tokens, n_tokens]


def compute_semantic_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity."""
    emb_norm = F.normalize(embeddings, dim=-1)
    return torch.matmul(emb_norm, emb_norm.t())


def evaluate_attention_methods(
    embeddings: torch.Tensor,
    words: List[str],
    output_dir: Path
):
    """
    Compare different attention methods on how well they capture
    semantic relationships.
    """
    hidden_size = embeddings.size(-1)
    
    # Initialize attention modules
    attention_methods = {
        'Standard (Dot Product)': StandardAttention(hidden_size),
        'Phase Attention': PhaseAttention(hidden_size),
        'Amp-Weighted Phase': AmplitudeWeightedPhaseAttention(hidden_size),
        'Complex Attention': ComplexAttention(hidden_size),
    }
    
    # Compute ground truth: cosine similarity of embeddings
    semantic_sim = compute_semantic_similarity_matrix(embeddings)
    
    # Compute attention matrices for each method
    attention_matrices = {}
    for name, module in attention_methods.items():
        attention_matrices[name] = compute_attention_similarity_matrix(module, embeddings)
    
    # Visualize
    n_methods = len(attention_methods) + 1  # +1 for semantic similarity
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot semantic similarity (ground truth)
    ax = axes[0]
    im = ax.imshow(semantic_sim.numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Semantic Similarity\n(Cosine of Embeddings)', fontsize=12)
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(words, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Plot each attention method
    for idx, (name, attn_matrix) in enumerate(attention_matrices.items()):
        ax = axes[idx + 1]
        im = ax.imshow(attn_matrix.numpy(), cmap='viridis', vmin=0)
        ax.set_title(f'{name}\nAttention', fontsize=12)
        ax.set_xticks(range(len(words)))
        ax.set_yticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(words, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide unused subplot
    if len(axes) > n_methods:
        for i in range(n_methods, len(axes)):
            axes[i].axis('off')
    
    plt.suptitle('Attention Pattern Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compute correlation between attention patterns and semantic similarity
    print("\nCorrelation with Semantic Similarity:")
    print("-" * 50)
    
    correlations = {}
    semantic_flat = semantic_sim.flatten().numpy()
    
    for name, attn_matrix in attention_matrices.items():
        attn_flat = attn_matrix.flatten().numpy()
        corr = np.corrcoef(semantic_flat, attn_flat)[0, 1]
        correlations[name] = corr
        print(f"  {name:25s}: r = {corr:.4f}")
    
    print("-" * 50)
    
    # Bar plot of correlations
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(correlations.keys())
    values = list(correlations.values())
    colors = ['steelblue', 'coral', 'seagreen', 'purple']
    
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel('Correlation with Semantic Similarity')
    ax.set_title('How Well Does Each Attention Method\nCapture Semantic Relationships?')
    ax.set_ylim(0, 1)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return correlations, attention_matrices


def analyze_attention_patterns(
    attention_matrices: Dict[str, torch.Tensor],
    words: List[str],
    output_dir: Path
):
    """Deeper analysis of attention patterns."""
    
    # Define semantic groups
    groups = {
        'animals': ['cat', 'dog'],
        'emotions': ['happy', 'sad', 'joyful'],
        'size': ['big', 'small'],
        'verbs': ['run', 'walk', 'think', 'speak'],
        'numbers': ['0', '1', '2', '3'],
    }
    
    # Find indices for each group
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    print("\nWithin-Group vs Between-Group Attention:")
    print("=" * 60)
    
    results = {}
    
    for method_name, attn_matrix in attention_matrices.items():
        print(f"\n{method_name}:")
        print("-" * 40)
        
        method_results = {}
        
        for group_name, group_words in groups.items():
            # Get indices of words in this group that exist
            indices = [word_to_idx[w] for w in group_words if w in word_to_idx]
            
            if len(indices) < 2:
                continue
            
            # Within-group attention (off-diagonal)
            within_scores = []
            for i in indices:
                for j in indices:
                    if i != j:
                        within_scores.append(attn_matrix[i, j].item())
            
            # Between-group attention (to words not in group)
            other_indices = [i for i in range(len(words)) if i not in indices]
            between_scores = []
            for i in indices:
                for j in other_indices:
                    between_scores.append(attn_matrix[i, j].item())
            
            if within_scores and between_scores:
                within_mean = np.mean(within_scores)
                between_mean = np.mean(between_scores)
                ratio = within_mean / (between_mean + 1e-8)
                
                method_results[group_name] = {
                    'within': within_mean,
                    'between': between_mean,
                    'ratio': ratio
                }
                
                print(f"  {group_name:12s}: within={within_mean:.4f}, between={between_mean:.4f}, ratio={ratio:.2f}")
        
        results[method_name] = method_results
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare attention methods")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Define test words (mix of categories)
    test_words = [
        # Animals
        'cat', 'dog',
        # Emotions
        'happy', 'sad', 'joyful',
        # Size
        'big', 'small',
        # Actions
        'run', 'walk', 'think', 'speak',
        # Numbers
        '0', '1', '2', '3',
        # Articles
        'the', 'a',
        # Punctuation
        '.', ',',
    ]
    
    print(f"\nGetting embeddings for {len(test_words)} words...")
    embeddings, valid_words = get_embeddings_for_words(model, tokenizer, test_words)
    
    if embeddings is None:
        print("Error: Could not get embeddings")
        return
    
    print(f"Got embeddings for {len(valid_words)} words: {valid_words}")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Compare attention methods
    print("\n" + "=" * 60)
    print("Comparing Attention Methods")
    print("=" * 60)
    
    correlations, attention_matrices = evaluate_attention_methods(
        embeddings, valid_words, output_dir
    )
    
    # Analyze attention patterns
    group_analysis = analyze_attention_patterns(
        attention_matrices, valid_words, output_dir
    )
    
    # Save results
    results = {
        'words': valid_words,
        'correlations': correlations,
        'group_analysis': {
            method: {
                group: {k: float(v) for k, v in vals.items()}
                for group, vals in groups.items()
            }
            for method, groups in group_analysis.items()
        }
    }
    
    with open(output_dir / 'attention_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    # Print summary
    print("\n📊 SUMMARY")
    print("-" * 40)
    print("Correlation with semantic similarity:")
    best_method = max(correlations, key=correlations.get)
    for name, corr in sorted(correlations.items(), key=lambda x: -x[1]):
        marker = " ⭐ BEST" if name == best_method else ""
        print(f"  {name:25s}: {corr:.4f}{marker}")


if __name__ == "__main__":
    main()
