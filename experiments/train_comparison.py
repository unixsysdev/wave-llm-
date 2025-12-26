"""
Train and compare Standard Attention vs Wave transformers on modular addition.

Tracks grokking dynamics, Fourier structure emergence, and final performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from models import (
    StandardAttentionTransformer, 
    WaveTransformer, 
    train_test_split,
    count_params
)


def compute_fourier_metrics(model, device):
    """
    Compute Fourier structure metrics.
    Same as phase_transition/train_with_metrics.py
    """
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


def train_model(
    model,
    train_data,
    test_data,
    n_epochs: int = 30000,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    log_every: int = 100,
    device: torch.device = None,
    desc: str = "Training",
):
    """Train a model and return history."""
    
    a_train, b_train, targets_train = train_data
    a_test, b_test, targets_test = test_data
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    history = {
        'epochs': [],
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'fourier_strength': [],
        'key_frequencies': [],
    }
    
    pbar = tqdm(range(n_epochs), desc=desc)
    
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
            
            pbar.set_postfix({
                'tr_acc': f"{train_acc:.3f}",
                'te_acc': f"{test_acc:.3f}",
                'fourier': f"{fourier['fourier_strength']:.3f}",
            })
    
    return history


def run_comparison(
    p: int = 53,
    d_model: int = 128,
    n_heads: int = 4,
    d_mlp: int = 512,
    train_frac: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    n_epochs: int = 30000,
    seed: int = 42,
    wave_modes: list = None,
    output_dir: str = "results",
):
    """Run comparison between standard and wave transformers."""
    
    if wave_modes is None:
        wave_modes = ["fnet", "learned_gate", "phase_shift", "full"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"p={p}, train_frac={train_frac}, epochs={n_epochs}")
    print("=" * 60)
    
    torch.manual_seed(seed)
    
    # Create data
    train_data, test_data = train_test_split(p, train_frac, device, seed)
    print(f"Train: {len(train_data[0])}, Test: {len(test_data[0])}")
    
    results = {}
    
    # Train standard attention
    print("\n" + "=" * 60)
    print("STANDARD ATTENTION")
    print("=" * 60)
    
    standard = StandardAttentionTransformer(
        p=p, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp
    ).to(device)
    print(f"Parameters: {count_params(standard):,}")
    
    history_std = train_model(
        standard, train_data, test_data,
        n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
        device=device, desc="Standard"
    )
    
    results['standard'] = {
        'params': count_params(standard),
        'history': history_std,
        'final_test_acc': history_std['test_acc'][-1],
        'final_fourier': history_std['fourier_strength'][-1],
    }
    
    # Train wave variants
    for mode in wave_modes:
        print("\n" + "=" * 60)
        print(f"WAVE TRANSFORMER ({mode})")
        print("=" * 60)
        
        torch.manual_seed(seed)  # Same init
        
        wave = WaveTransformer(
            p=p, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp,
            wave_mode=mode
        ).to(device)
        print(f"Parameters: {count_params(wave):,}")
        
        history_wave = train_model(
            wave, train_data, test_data,
            n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
            device=device, desc=f"Wave ({mode})"
        )
        
        # Get learned freq weights if applicable
        freq_weights = wave.get_freq_weights()
        freq_weights_serializable = {
            k: v.cpu().numpy().tolist() for k, v in freq_weights.items()
        }
        
        results[f'wave_{mode}'] = {
            'params': count_params(wave),
            'history': history_wave,
            'final_test_acc': history_wave['test_acc'][-1],
            'final_fourier': history_wave['fourier_strength'][-1],
            'freq_weights': freq_weights_serializable,
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Params':>12} {'Test Acc':>10} {'Fourier':>10}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<25} {res['params']:>12,} {res['final_test_acc']:>10.4f} {res['final_fourier']:>10.4f}")
    
    # Find grokking point (when test_acc > 0.95)
    print("\n" + "=" * 60)
    print("GROKKING ANALYSIS")
    print("=" * 60)
    
    for name, res in results.items():
        hist = res['history']
        grok_epoch = None
        for i, acc in enumerate(hist['test_acc']):
            if acc > 0.95:
                grok_epoch = hist['epochs'][i]
                break
        
        if grok_epoch:
            print(f"{name:<25} groks at epoch {grok_epoch}")
        else:
            print(f"{name:<25} did not grok (max acc: {max(hist['test_acc']):.4f})")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"comparison_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=53)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_mlp", type=int, default=512)
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--n_epochs", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wave_modes", nargs="+", default=["fnet", "learned_gate", "phase_shift", "full"])
    parser.add_argument("--output_dir", type=str, default="wave_token_analysis/experiments/results")
    
    args = parser.parse_args()
    
    run_comparison(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        train_frac=args.train_frac,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        seed=args.seed,
        wave_modes=args.wave_modes,
        output_dir=args.output_dir,
    )
