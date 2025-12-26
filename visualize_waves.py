#!/usr/bin/env python3
"""
Wave Token Visualization for Qwen3 0.6B

Analyzes pretrained token embeddings in the frequency domain to understand
if there's meaningful wave-like structure that could be exploited for
more efficient representations.

Based on ideas from:
- Wave Network (arXiv:2411.02674): Complex vector token representations
- FNet (arXiv:2105.03824): Fourier transforms for token mixing
- SIREN (NeurIPS 2020): Sinusoidal representation networks
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from device_utils import get_device

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers")
    sys.exit(1)


def load_qwen3(model_name: str = "Qwen/Qwen3-0.6B"):
    """Load Qwen3 model and tokenizer."""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for analysis precision
        device_map="auto",
    )
    model.eval()
    
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Vocab size: {model.config.vocab_size}")
    
    return model, tokenizer


def get_embedding_matrix(model) -> torch.Tensor:
    """Extract the embedding matrix from the model."""
    return model.get_input_embeddings().weight.data.cpu()


def compute_fft(embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute FFT of an embedding vector.
    
    Returns:
        amplitude: Magnitude of FFT coefficients
        phase: Phase angles of FFT coefficients
    """
    fft = torch.fft.rfft(embedding)
    amplitude = torch.abs(fft)
    phase = torch.angle(fft)
    return amplitude, phase


def get_token_id(tokenizer, word: str) -> Optional[int]:
    """
    Get the token ID for a word.
    
    Handles cases where word might tokenize to multiple tokens
    by returning the first token.
    """
    # Try with space prefix (common for BPE tokenizers)
    tokens_with_space = tokenizer.encode(f" {word}", add_special_tokens=False)
    tokens_without_space = tokenizer.encode(word, add_special_tokens=False)
    
    # Prefer the version that gives a single token
    if len(tokens_with_space) == 1:
        return tokens_with_space[0]
    elif len(tokens_without_space) == 1:
        return tokens_without_space[0]
    else:
        # Return first token of the shorter encoding
        tokens = tokens_with_space if len(tokens_with_space) <= len(tokens_without_space) else tokens_without_space
        return tokens[0] if tokens else None


def analyze_token_pair(
    embeddings: torch.Tensor,
    tokenizer,
    word1: str,
    word2: str
) -> Optional[Dict]:
    """
    Analyze the wave properties of a token pair.
    
    Returns:
        Dictionary with amplitude, phase, and comparison metrics
    """
    id1 = get_token_id(tokenizer, word1)
    id2 = get_token_id(tokenizer, word2)
    
    if id1 is None or id2 is None:
        print(f"Warning: Could not tokenize '{word1}' or '{word2}'")
        return None
    
    emb1 = embeddings[id1]
    emb2 = embeddings[id2]
    
    # Compute FFT
    amp1, phase1 = compute_fft(emb1)
    amp2, phase2 = compute_fft(emb2)
    
    # Compute metrics
    # Amplitude similarity (cosine similarity in frequency domain)
    amp_sim = F.cosine_similarity(amp1.unsqueeze(0), amp2.unsqueeze(0)).item()
    
    # Phase difference
    phase_diff = phase1 - phase2
    # Wrap to [-pi, pi]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    # Phase coherence (how consistent is the phase difference?)
    phase_coherence = torch.abs(torch.mean(torch.exp(1j * phase_diff))).item()
    
    # Original embedding similarity
    emb_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    return {
        'word1': word1,
        'word2': word2,
        'id1': id1,
        'id2': id2,
        'token1': tokenizer.decode([id1]),
        'token2': tokenizer.decode([id2]),
        'amplitude1': amp1,
        'amplitude2': amp2,
        'phase1': phase1,
        'phase2': phase2,
        'phase_diff': phase_diff,
        'amplitude_similarity': amp_sim,
        'phase_coherence': phase_coherence,
        'embedding_similarity': emb_sim,
    }


def visualize_token_pair(
    result: Dict,
    output_dir: Path,
    pair_name: str
):
    """Create visualization for a token pair."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    word1, word2 = result['word1'], result['word2']
    
    # Top row: Amplitude spectra
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(result['amplitude1'].numpy(), label=word1, alpha=0.7)
    ax1.set_title(f"Amplitude Spectrum: {word1}")
    ax1.set_xlabel("Frequency bin")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, 100)  # Focus on first 100 frequencies
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(result['amplitude2'].numpy(), label=word2, alpha=0.7, color='orange')
    ax2.set_title(f"Amplitude Spectrum: {word2}")
    ax2.set_xlabel("Frequency bin")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(0, 100)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(result['amplitude1'].numpy()[:100], label=word1, alpha=0.7)
    ax3.plot(result['amplitude2'].numpy()[:100], label=word2, alpha=0.7)
    ax3.set_title(f"Amplitude Comparison")
    ax3.set_xlabel("Frequency bin")
    ax3.set_ylabel("Amplitude")
    ax3.legend()
    
    # Middle row: Phase spectra
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(result['phase1'].numpy(), label=word1, alpha=0.7)
    ax4.set_title(f"Phase Spectrum: {word1}")
    ax4.set_xlabel("Frequency bin")
    ax4.set_ylabel("Phase (radians)")
    ax4.set_xlim(0, 100)
    ax4.set_ylim(-np.pi, np.pi)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(result['phase2'].numpy(), label=word2, alpha=0.7, color='orange')
    ax5.set_title(f"Phase Spectrum: {word2}")
    ax5.set_xlabel("Frequency bin")
    ax5.set_ylabel("Phase (radians)")
    ax5.set_xlim(0, 100)
    ax5.set_ylim(-np.pi, np.pi)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(result['phase_diff'].numpy()[:100], color='purple', alpha=0.7)
    ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax6.set_title(f"Phase Difference")
    ax6.set_xlabel("Frequency bin")
    ax6.set_ylabel("Phase diff (radians)")
    ax6.set_ylim(-np.pi, np.pi)
    
    # Bottom row: Summary and waveform reconstruction
    ax7 = fig.add_subplot(gs[2, 0])
    # Reconstruct waveform from FFT (first 20 frequencies)
    n_freqs = 20
    t = np.linspace(0, 2*np.pi, 200)
    wave1 = np.zeros_like(t)
    wave2 = np.zeros_like(t)
    for k in range(min(n_freqs, len(result['amplitude1']))):
        wave1 += result['amplitude1'][k].item() * np.cos(k * t + result['phase1'][k].item())
        wave2 += result['amplitude2'][k].item() * np.cos(k * t + result['phase2'][k].item())
    ax7.plot(t, wave1, label=word1, alpha=0.7)
    ax7.plot(t, wave2, label=word2, alpha=0.7)
    ax7.set_title(f"Reconstructed Waveform (first {n_freqs} freqs)")
    ax7.set_xlabel("t")
    ax7.set_ylabel("Amplitude")
    ax7.legend()
    
    ax8 = fig.add_subplot(gs[2, 1])
    # Bar chart of metrics
    metrics = ['Embed. Sim', 'Amp. Sim', 'Phase Coh.']
    values = [
        result['embedding_similarity'],
        result['amplitude_similarity'],
        result['phase_coherence']
    ]
    colors = ['steelblue', 'coral', 'seagreen']
    bars = ax8.bar(metrics, values, color=colors)
    ax8.set_title("Similarity Metrics")
    ax8.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    info_text = f"""Token Analysis Summary
    
Word 1: "{word1}" → token: "{result['token1']}" (id: {result['id1']})
Word 2: "{word2}" → token: "{result['token2']}" (id: {result['id2']})

Embedding Similarity: {result['embedding_similarity']:.4f}
Amplitude Similarity: {result['amplitude_similarity']:.4f}
Phase Coherence: {result['phase_coherence']:.4f}

Interpretation:
- Amplitude = "what frequencies are present"
- Phase = "timing/position of each frequency"
- Phase coherence = "consistency of phase relationship"
"""
    ax9.text(0.1, 0.9, info_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f"Wave Analysis: {word1} vs {word2}", fontsize=14, fontweight='bold')
    plt.savefig(output_dir / f"pair_{pair_name}.png", dpi=150, bbox_inches='tight')
    plt.close()


def visualize_frequency_clusters(
    embeddings: torch.Tensor,
    tokenizer,
    output_dir: Path
):
    """
    Visualize how different token categories cluster in frequency space.
    """
    # Define token categories
    categories = {
        'nouns': ['cat', 'dog', 'house', 'tree', 'car', 'book', 'water', 'food'],
        'verbs': ['run', 'walk', 'think', 'speak', 'write', 'read', 'eat', 'sleep'],
        'adjectives': ['happy', 'sad', 'big', 'small', 'fast', 'slow', 'hot', 'cold'],
        'punctuation': ['.', ',', '!', '?', ':', ';', '-', '"'],
        'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'articles': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'some'],
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    all_amplitudes = {}
    
    for idx, (category, words) in enumerate(categories.items()):
        ax = axes[idx]
        amplitudes = []
        valid_words = []
        
        for word in words:
            token_id = get_token_id(tokenizer, word)
            if token_id is not None:
                amp, _ = compute_fft(embeddings[token_id])
                amplitudes.append(amp.numpy())
                valid_words.append(word)
        
        if amplitudes:
            amplitudes = np.array(amplitudes)
            all_amplitudes[category] = amplitudes
            
            # Plot each token's amplitude spectrum
            for i, (amp, word) in enumerate(zip(amplitudes, valid_words)):
                ax.plot(amp[:50], alpha=0.5, label=word)
            
            # Plot mean amplitude
            mean_amp = amplitudes.mean(axis=0)
            ax.plot(mean_amp[:50], 'k-', linewidth=2, label='mean')
            
            ax.set_title(f"{category.capitalize()} (n={len(valid_words)})")
            ax.set_xlabel("Frequency bin")
            ax.set_ylabel("Amplitude")
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle("Amplitude Spectra by Token Category", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "frequency_clusters.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create a summary comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_amplitudes)))
    
    for (category, amplitudes), color in zip(all_amplitudes.items(), colors):
        mean_amp = amplitudes.mean(axis=0)
        std_amp = amplitudes.std(axis=0)
        x = np.arange(len(mean_amp[:50]))
        ax.plot(x, mean_amp[:50], label=category, color=color, linewidth=2)
        ax.fill_between(x, (mean_amp - std_amp)[:50], (mean_amp + std_amp)[:50],
                       alpha=0.2, color=color)
    
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Mean Amplitude")
    ax.set_title("Mean Amplitude Spectrum by Category (with std dev)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "category_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def analyze_dominant_frequencies(
    embeddings: torch.Tensor,
    tokenizer,
    output_dir: Path,
    n_tokens: int = 1000
):
    """
    Analyze which frequencies are most important across all tokens.
    """
    print(f"Analyzing dominant frequencies across {n_tokens} tokens...")
    
    vocab_size = embeddings.size(0)
    n_tokens = min(n_tokens, vocab_size)
    
    # Sample tokens (prefer common ones = lower IDs typically)
    token_ids = list(range(min(n_tokens, vocab_size)))
    
    all_amplitudes = []
    
    for token_id in tqdm(token_ids, desc="Computing FFTs"):
        amp, _ = compute_fft(embeddings[token_id])
        all_amplitudes.append(amp.numpy())
    
    all_amplitudes = np.array(all_amplitudes)  # [n_tokens, n_freqs]
    
    # Compute statistics
    mean_amplitude = all_amplitudes.mean(axis=0)
    std_amplitude = all_amplitudes.std(axis=0)
    max_amplitude = all_amplitudes.max(axis=0)
    
    # Find dominant frequencies
    top_k = 20
    top_freq_indices = np.argsort(mean_amplitude)[-top_k:][::-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean amplitude spectrum
    ax = axes[0, 0]
    ax.plot(mean_amplitude, 'b-', alpha=0.7)
    ax.fill_between(range(len(mean_amplitude)),
                    mean_amplitude - std_amplitude,
                    mean_amplitude + std_amplitude,
                    alpha=0.3)
    ax.set_title("Mean Amplitude Spectrum (all tokens)")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Amplitude")
    
    # Top frequencies bar chart
    ax = axes[0, 1]
    ax.bar(range(top_k), mean_amplitude[top_freq_indices])
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"f={i}" for i in top_freq_indices], rotation=45)
    ax.set_title(f"Top {top_k} Dominant Frequencies")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Mean Amplitude")
    
    # Amplitude heatmap (sample of tokens)
    ax = axes[1, 0]
    sample_size = min(100, len(all_amplitudes))
    heatmap_data = all_amplitudes[:sample_size, :100]  # First 100 freqs
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
    ax.set_title("Amplitude Heatmap (tokens × frequencies)")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Token index")
    plt.colorbar(im, ax=ax, label="Amplitude")
    
    # Frequency importance (Gini coefficient per frequency)
    ax = axes[1, 1]
    # How "specialized" is each frequency? High Gini = few tokens use it strongly
    def gini(arr):
        sorted_arr = np.sort(arr)
        n = len(arr)
        cumsum = np.cumsum(sorted_arr)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    gini_per_freq = [gini(all_amplitudes[:, f]) for f in range(all_amplitudes.shape[1])]
    ax.plot(gini_per_freq[:100], 'g-', alpha=0.7)
    ax.set_title("Gini Coefficient per Frequency")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Gini (higher = more specialized)")
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    plt.suptitle("Dominant Frequency Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "dominant_frequencies.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'top_frequencies': top_freq_indices.tolist(),
        'mean_amplitude': mean_amplitude.tolist(),
        'std_amplitude': std_amplitude.tolist(),
        'gini_per_freq': gini_per_freq,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze wave structure of Qwen3 token embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name to analyze"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--n_tokens",
        type=int,
        default=1000,
        help="Number of tokens to analyze for dominant frequencies"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model
    model, tokenizer = load_qwen3(args.model)
    
    # Get embedding matrix
    embeddings = get_embedding_matrix(model)
    print(f"Embedding matrix shape: {embeddings.shape}")
    
    # Define interesting token pairs
    token_pairs = [
        # Semantic similarity
        ("cat", "dog", "similar_animals"),
        ("happy", "joyful", "synonyms"),
        ("run", "walk", "similar_verbs"),
        
        # Antonyms
        ("happy", "sad", "antonyms_emotion"),
        ("hot", "cold", "antonyms_temp"),
        ("big", "small", "antonyms_size"),
        
        # Grammatical variations
        ("cat", "cats", "singular_plural"),
        ("run", "running", "verb_forms"),
        ("happy", "happily", "adj_adv"),
        
        # Punctuation
        (".", ",", "punct_period_comma"),
        ("!", "?", "punct_exclaim_question"),
        
        # Articles
        ("the", "a", "articles"),
        
        # Numbers
        ("1", "2", "numbers_adjacent"),
        ("0", "9", "numbers_extremes"),
    ]
    
    # Analyze each pair
    print("\n" + "="*60)
    print("Analyzing token pairs...")
    print("="*60)
    
    results = []
    for word1, word2, pair_name in tqdm(token_pairs, desc="Token pairs"):
        result = analyze_token_pair(embeddings, tokenizer, word1, word2)
        if result:
            results.append(result)
            visualize_token_pair(result, output_dir, pair_name)
            print(f"  {word1} vs {word2}: emb_sim={result['embedding_similarity']:.3f}, "
                  f"amp_sim={result['amplitude_similarity']:.3f}, "
                  f"phase_coh={result['phase_coherence']:.3f}")
    
    # Frequency cluster analysis
    print("\n" + "="*60)
    print("Analyzing frequency clusters by category...")
    print("="*60)
    visualize_frequency_clusters(embeddings, tokenizer, output_dir)
    
    # Dominant frequency analysis
    print("\n" + "="*60)
    print("Analyzing dominant frequencies...")
    print("="*60)
    freq_analysis = analyze_dominant_frequencies(embeddings, tokenizer, output_dir, args.n_tokens)
    
    # Save summary
    summary = {
        'model': args.model,
        'embedding_shape': list(embeddings.shape),
        'token_pairs': [
            {
                'word1': r['word1'],
                'word2': r['word2'],
                'embedding_similarity': r['embedding_similarity'],
                'amplitude_similarity': r['amplitude_similarity'],
                'phase_coherence': r['phase_coherence'],
            }
            for r in results
        ],
        'top_frequencies': freq_analysis['top_frequencies'],
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    # Print summary table
    print("\nToken Pair Summary:")
    print("-" * 70)
    print(f"{'Word 1':<12} {'Word 2':<12} {'Emb Sim':<10} {'Amp Sim':<10} {'Phase Coh':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['word1']:<12} {r['word2']:<12} {r['embedding_similarity']:<10.4f} "
              f"{r['amplitude_similarity']:<10.4f} {r['phase_coherence']:<10.4f}")
    print("-" * 70)


if __name__ == "__main__":
    main()
