"""
Training with checkpoints and visualization for wave vs attention comparison.

Saves snapshots at regular intervals for visualization of:
1. Loss/accuracy curves
2. Fourier structure emergence
3. Learned frequency weights (for wave models)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from models import (
    StandardAttentionTransformer, 
    WaveTransformer, 
    train_test_split,
    count_params,
    create_modular_data
)


def compute_fourier_metrics(model, device):
    """Compute Fourier structure metrics."""
    p = model.p
    W_L = model.get_neuron_logit_map().detach()
    
    x = torch.arange(p, device=device, dtype=torch.float32)
    n_freqs = p // 2 + 1
    
    freq_strengths = []
    for k in range(1, n_freqs):
        angle = 2 * np.pi * k * x / p
        cos_basis = torch.cos(angle)
        sin_basis = torch.sin(angle)
        
        cos_proj = (W_L.T @ cos_basis) / p
        sin_proj = (W_L.T @ sin_basis) / p
        strength = (cos_proj**2 + sin_proj**2).sum().item()
        freq_strengths.append(strength)
    
    freq_strengths = np.array(freq_strengths)
    total_var = (W_L**2).sum().item()
    
    if total_var > 0:
        freq_strengths_norm = freq_strengths / total_var
    else:
        freq_strengths_norm = freq_strengths
    
    top_k = min(5, len(freq_strengths))
    key_freq_indices = np.argsort(freq_strengths)[-top_k:][::-1]
    key_frequencies = (key_freq_indices + 1).tolist()
    fourier_strength = freq_strengths_norm[key_freq_indices].sum()
    
    return {
        'fourier_strength': float(fourier_strength),
        'key_frequencies': key_frequencies,
        'freq_spectrum': freq_strengths_norm.tolist(),
    }


def get_neuron_activations_by_sum(model, device):
    """Get MLP neuron activations organized by (a+b) mod p."""
    p = model.p
    a, b, targets = create_modular_data(p, device)
    
    # Forward pass - need to capture MLP activations
    batch_size = a.shape[0]
    eq_tokens = torch.full((batch_size,), model.eq_token, device=device, dtype=torch.long)
    tokens = torch.stack([a, b, eq_tokens], dim=1)
    positions = torch.arange(3, device=device).unsqueeze(0).expand(batch_size, -1)
    x = model.token_embed(tokens) + model.pos_embed(positions)
    
    # Apply mixing (attention or wave)
    if hasattr(model, 'wave_mix'):
        wave_out = model.wave_mix(x)
        x = x + wave_out
    else:
        # Standard attention path
        from einops import einsum
        import math
        q = einsum(x, model.W_Q, 'b s d, h d k -> b h s k')
        k = einsum(x, model.W_K, 'b s d, h d k -> b h s k')
        v = einsum(x, model.W_V, 'b s d, h d k -> b h s k')
        attn_scores = einsum(q, k, 'b h s1 d, b h s2 d -> b h s1 s2') / math.sqrt(model.d_head)
        mask = torch.triu(torch.ones(3, 3, device=device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_pattern = F.softmax(attn_scores, dim=-1)
        attn_out = einsum(attn_pattern, v, 'b h s1 s2, b h s2 d -> b h s1 d')
        attn_out = einsum(attn_out, model.W_O, 'b h s d, h d m -> b s m')
        x = x + attn_out
    
    # MLP
    mlp_hidden = F.relu(model.W_in(x))
    mlp_acts = mlp_hidden[:, 2, :]  # Position 2 (=), shape (p*p, d_mlp)
    
    sums = (a + b) % p
    
    # Mean activation for each sum value
    mean_activations = torch.zeros(p, mlp_acts.shape[1], device=device)
    for s in range(p):
        mask = (sums == s)
        if mask.sum() > 0:
            mean_activations[s] = mlp_acts[mask].mean(dim=0)
    
    return mean_activations.cpu().numpy()


def plot_snapshot(model, history, epoch, output_dir, model_name, device):
    """Generate visualization snapshot at current epoch."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{model_name} - Epoch {epoch}', fontsize=14, fontweight='bold')
    
    # 1. Loss curves
    ax = axes[0, 0]
    epochs = history['epochs']
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', alpha=0.7)
    ax.plot(epochs, history['test_loss'], 'r-', label='Test', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.set_yscale('log')
    ax.axvline(epoch, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train', alpha=0.7)
    ax.plot(epochs, history['test_acc'], 'r-', label='Test', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(epoch, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(1.0, color='green', linestyle=':', alpha=0.3)
    
    # 3. Fourier strength
    ax = axes[0, 2]
    ax.plot(epochs, history['fourier_strength'], 'g-', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Fourier Strength')
    ax.set_title('Fourier Structure Emergence')
    ax.axvline(epoch, color='gray', linestyle='--', alpha=0.5)
    
    # 4. Current frequency spectrum
    ax = axes[1, 0]
    if history['freq_spectrum']:
        spectrum = history['freq_spectrum'][-1]
        freqs = np.arange(1, len(spectrum) + 1)
        ax.bar(freqs, spectrum, alpha=0.7)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Normalized Strength')
        ax.set_title(f'Frequency Spectrum (epoch {epoch})')
    
    # 5. Neuron activations by sum (top 6 neurons)
    ax = axes[1, 1]
    try:
        with torch.no_grad():
            mean_acts = get_neuron_activations_by_sum(model, device)
        
        # Find neurons with highest variance (most structured)
        variances = np.var(mean_acts, axis=0)
        top_neurons = np.argsort(variances)[-6:][::-1]
        
        p = model.p
        x = np.arange(p)
        for i, neuron_idx in enumerate(top_neurons[:3]):
            ax.plot(x, mean_acts[:, neuron_idx], alpha=0.7, label=f'Neuron {neuron_idx}')
        
        ax.set_xlabel('(a + b) mod p')
        ax.set_ylabel('Mean Activation')
        ax.set_title('Top Neuron Activations by Sum')
        ax.legend(fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center', transform=ax.transAxes)
    
    # 6. Learned wave weights (if applicable)
    ax = axes[1, 2]
    if hasattr(model, 'get_freq_weights'):
        freq_weights = model.get_freq_weights()
        if 'freq_gate' in freq_weights:
            gate = freq_weights['freq_gate'].cpu().numpy()
            im = ax.imshow(gate, aspect='auto', cmap='RdBu_r')
            ax.set_xlabel('Hidden dim')
            ax.set_ylabel('Frequency')
            ax.set_title('Learned Frequency Gates')
            plt.colorbar(im, ax=ax)
        elif 'phase_shift' in freq_weights:
            phase = freq_weights['phase_shift'].cpu().numpy()
            im = ax.imshow(phase, aspect='auto', cmap='twilight')
            ax.set_xlabel('Hidden dim')
            ax.set_ylabel('Frequency')
            ax.set_title('Learned Phase Shifts')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No learned freq weights\n(fnet or full mode)', 
                   ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'Standard Attention\n(no frequency weights)', 
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save
    frame_path = output_dir / f"frame_{epoch:06d}.png"
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return frame_path


def train_with_snapshots(
    model,
    train_data,
    test_data,
    n_epochs: int = 30000,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    log_every: int = 100,
    snapshot_every: int = 500,
    device: torch.device = None,
    output_dir: Path = None,
    model_name: str = "model",
):
    """Train with periodic snapshots for visualization."""
    
    a_train, b_train, targets_train = train_data
    a_test, b_test, targets_test = test_data
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    history = {
        'epochs': [],
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'fourier_strength': [],
        'key_frequencies': [],
        'freq_spectrum': [],
    }
    
    # Create output dir
    frames_dir = output_dir / f"frames_{model_name}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm(range(n_epochs), desc=model_name)
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        logits = model(a_train, b_train)
        loss = F.cross_entropy(logits, targets_train)
        loss.backward()
        optimizer.step()
        
        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                train_pred = logits.argmax(dim=-1)
                train_acc = (train_pred == targets_train).float().mean().item()
                
                test_logits = model(a_test, b_test)
                test_loss = F.cross_entropy(test_logits, targets_test).item()
                test_pred = test_logits.argmax(dim=-1)
                test_acc = (test_pred == targets_test).float().mean().item()
                
                fourier = compute_fourier_metrics(model, device)
            
            history['epochs'].append(epoch)
            history['train_loss'].append(loss.item())
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['fourier_strength'].append(fourier['fourier_strength'])
            history['key_frequencies'].append(fourier['key_frequencies'])
            history['freq_spectrum'].append(fourier['freq_spectrum'])
            
            pbar.set_postfix({
                'tr': f"{train_acc:.3f}",
                'te': f"{test_acc:.3f}",
                'four': f"{fourier['fourier_strength']:.3f}",
            })
        
        # Save snapshot
        if epoch % snapshot_every == 0:
            model.eval()
            with torch.no_grad():
                plot_snapshot(model, history, epoch, frames_dir, model_name, device)
    
    # Final snapshot
    model.eval()
    with torch.no_grad():
        plot_snapshot(model, history, n_epochs, frames_dir, model_name, device)
    
    return history


def run_visual_comparison(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    d_mlp: int = 512,
    train_frac: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    n_epochs: int = 30000,
    snapshot_every: int = 500,
    seed: int = 42,
    wave_modes: list = None,
    output_dir: str = "results_visual",
):
    """Run comparison with visualizations."""
    
    if wave_modes is None:
        wave_modes = ["fnet", "learned_gate"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"p={p}, train_frac={train_frac}, epochs={n_epochs}")
    print("=" * 60)
    
    torch.manual_seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data
    train_data, test_data = train_test_split(p, train_frac, device, seed)
    print(f"Train: {len(train_data[0])}, Test: {len(test_data[0])}")
    
    results = {}
    
    # Train standard attention
    print("\n" + "=" * 60)
    print("STANDARD ATTENTION")
    print("=" * 60)
    
    torch.manual_seed(seed)
    standard = StandardAttentionTransformer(
        p=p, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp
    ).to(device)
    print(f"Parameters: {count_params(standard):,}")
    
    history_std = train_with_snapshots(
        standard, train_data, test_data,
        n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
        snapshot_every=snapshot_every,
        device=device, output_dir=output_path, model_name="standard"
    )
    results['standard'] = history_std
    
    # Train wave variants
    for mode in wave_modes:
        print("\n" + "=" * 60)
        print(f"WAVE TRANSFORMER ({mode})")
        print("=" * 60)
        
        torch.manual_seed(seed)
        wave = WaveTransformer(
            p=p, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp,
            wave_mode=mode
        ).to(device)
        print(f"Parameters: {count_params(wave):,}")
        
        history_wave = train_with_snapshots(
            wave, train_data, test_data,
            n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
            snapshot_every=snapshot_every,
            device=device, output_dir=output_path, model_name=f"wave_{mode}"
        )
        results[f'wave_{mode}'] = history_wave
    
    # Save combined comparison plot
    plot_final_comparison(results, output_path)
    
    # Save histories
    with open(output_path / "histories.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def plot_final_comparison(results, output_path):
    """Plot final comparison of all models."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'standard': 'red', 'wave_fnet': 'blue', 'wave_learned_gate': 'green', 
              'wave_phase_shift': 'orange', 'wave_full': 'purple'}
    
    for name, history in results.items():
        color = colors.get(name, 'gray')
        epochs = history['epochs']
        
        # Accuracy
        axes[0].plot(epochs, history['test_acc'], color=color, label=name, alpha=0.8)
        
        # Loss
        axes[1].plot(epochs, history['test_loss'], color=color, label=name, alpha=0.8)
        
        # Fourier
        axes[2].plot(epochs, history['fourier_strength'], color=color, label=name, alpha=0.8)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Grokking Comparison')
    axes[0].legend()
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Test Loss')
    axes[1].legend()
    axes[1].set_yscale('log')
    
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Fourier Strength')
    axes[2].set_title('Fourier Structure')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {output_path / 'comparison.png'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_mlp", type=int, default=512)
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--n_epochs", type=int, default=30000)
    parser.add_argument("--snapshot_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wave_modes", nargs="+", default=["fnet", "learned_gate"])
    parser.add_argument("--output_dir", type=str, default="results_visual")
    
    args = parser.parse_args()
    
    run_visual_comparison(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        train_frac=args.train_frac,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        snapshot_every=args.snapshot_every,
        seed=args.seed,
        wave_modes=args.wave_modes,
        output_dir=args.output_dir,
    )
