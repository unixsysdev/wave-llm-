#!/usr/bin/env python3
"""
Wave Attention Adapter for Qwen3

This implements a trainable wave attention layer that can be added to
a pretrained model. We compare:
1. Frozen Qwen3 + standard classifier
2. Frozen Qwen3 + wave adapter + classifier

The wave adapter learns to use phase-based attention on top of the
pretrained representations.
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from device_utils import get_device

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
except ImportError as e:
    print(f"Error: {e}")
    print("Run: pip install transformers datasets")
    sys.exit(1)


# ============================================================================
# Wave Attention Layers
# ============================================================================

class WaveAttentionLayer(nn.Module):
    """
    Wave-based attention using phase alignment.
    
    Key insight from our analysis:
    - Amplitude similarity is ~0.80-0.90 for ALL tokens (shared structure)
    - Phase coherence varies and correlates with semantic similarity
    
    So we compute attention based on phase alignment, weighted by amplitude.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Learnable temperature for attention
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, hidden]
            attention_mask: [batch, seq] or [batch, 1, 1, seq]
        Returns:
            output: [batch, seq, hidden]
        """
        batch, seq, _ = x.shape
        residual = x
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # [batch, seq, hidden]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head: [batch, num_heads, seq, head_dim]
        Q = Q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute FFT along head_dim
        Q_fft = torch.fft.rfft(Q, dim=-1)  # [batch, heads, seq, freq]
        K_fft = torch.fft.rfft(K, dim=-1)
        
        # Extract amplitude and phase
        Q_amp = torch.abs(Q_fft)
        K_amp = torch.abs(K_fft)
        Q_phase = torch.angle(Q_fft)
        K_phase = torch.angle(K_fft)
        
        # Amplitude weights (geometric mean)
        amp_weights = torch.sqrt(Q_amp.unsqueeze(3) * K_amp.unsqueeze(2) + 1e-8)
        
        # Phase alignment: cos(phase_q - phase_k)
        phase_diff = Q_phase.unsqueeze(3) - K_phase.unsqueeze(2)  # [batch, heads, seq_q, seq_k, freq]
        phase_alignment = torch.cos(phase_diff)
        
        # Weighted phase alignment
        scores = (phase_alignment * amp_weights).sum(dim=-1)  # [batch, heads, seq_q, seq_k]
        scores = scores * self.temperature
        
        # Apply mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        output = torch.matmul(attn_weights, V)  # [batch, heads, seq, head_dim]
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch, seq, self.hidden_size)
        output = self.out_proj(output)
        
        # Residual + LayerNorm
        output = self.layer_norm(residual + output)
        
        return output


class WaveFFN(nn.Module):
    """
    Feed-forward network that operates in frequency domain.
    
    Instead of just MLP, we:
    1. FFT the input
    2. Apply learnable frequency filter
    3. MLP in frequency domain
    4. IFFT back
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        intermediate_size = intermediate_size or hidden_size * 4
        
        # Learnable frequency filter
        n_freqs = hidden_size // 2 + 1
        self.freq_filter = nn.Parameter(torch.ones(n_freqs))
        
        # Standard FFN
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        
        # Apply learnable filter
        x_fft = x_fft * self.freq_filter.unsqueeze(0).unsqueeze(0)
        
        # IFFT
        x = torch.fft.irfft(x_fft, n=self.hidden_size, dim=-1)
        
        # Standard FFN
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Residual + LayerNorm
        return self.layer_norm(residual + x)


class WaveAdapterBlock(nn.Module):
    """Single wave adapter block: WaveAttention + WaveFFN"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = WaveAttentionLayer(hidden_size, num_heads, dropout)
        self.ffn = WaveFFN(hidden_size, dropout=dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, attention_mask)
        x = self.ffn(x)
        return x


# ============================================================================
# Model Wrappers
# ============================================================================

class QwenWithWaveAdapter(nn.Module):
    """
    Qwen3 with wave adapter layers for classification.
    
    Architecture:
    1. Frozen Qwen3 embeddings + layers
    2. Trainable wave adapter blocks
    3. Classification head
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        num_labels: int = 4,
        num_adapter_layers: int = 2,
        adapter_heads: int = 8,
        dropout: float = 0.1,
        freeze_base: bool = True
    ):
        super().__init__()
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        hidden_size = self.base_model.config.hidden_size
        
        # Wave adapter layers
        self.wave_adapters = nn.ModuleList([
            WaveAdapterBlock(hidden_size, adapter_heads, dropout)
            for _ in range(num_adapter_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        
        self.num_labels = num_labels
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get base model hidden states
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Use last hidden state
        hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        
        # Apply wave adapters
        for adapter in self.wave_adapters:
            hidden = adapter(hidden, attention_mask)
        
        # Pool: use last token (common for causal LM) or mean pool
        if attention_mask is not None:
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = hidden.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        result = {"logits": logits}
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result


class QwenBaseline(nn.Module):
    """Baseline: Frozen Qwen3 + simple classifier (no wave adapter)"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        num_labels: int = 4,
        dropout: float = 0.1,
        freeze_base: bool = True
    ):
        super().__init__()
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        hidden_size = self.base_model.config.hidden_size
        
        # Simple classifier (same architecture as wave model, for fair comparison)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        
        self.num_labels = num_labels
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        hidden = outputs.hidden_states[-1]
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = hidden.mean(dim=1)
        
        logits = self.classifier(pooled)
        
        result = {"logits": logits}
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result


# ============================================================================
# Dataset
# ============================================================================

class AGNewsDataset(Dataset):
    """AG News dataset for text classification (4 classes)"""
    
    def __init__(self, tokenizer, split: str = "train", max_length: int = 128, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        dataset = load_dataset("ag_news", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.texts = dataset["text"]
        self.labels = dataset["label"]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, desc="Training"):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=desc)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device, desc="Evaluating"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            
            total_loss += outputs["loss"].item()
            preds = outputs["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train wave adapter on AG News")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--num_adapter_layers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--baseline_only", action="store_true", help="Only train baseline")
    parser.add_argument("--wave_only", action="store_true", help="Only train wave model")
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("Loading AG News dataset...")
    train_dataset = AGNewsDataset(tokenizer, "train", args.max_length, args.train_samples)
    test_dataset = AGNewsDataset(tokenizer, "test", args.max_length, args.test_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    results = {}
    
    # ========== Train Baseline ==========
    if not args.wave_only:
        print("\n" + "="*60)
        print("Training BASELINE (Frozen Qwen + Classifier)")
        print("="*60)
        
        baseline_model = QwenBaseline(args.model, num_labels=4).to(device)
        print(f"Trainable parameters: {count_parameters(baseline_model):,}")
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, baseline_model.parameters()),
            lr=args.lr
        )
        
        best_baseline_acc = 0
        baseline_history = []
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(
                baseline_model, train_loader, optimizer, device, 
                f"Baseline Epoch {epoch+1}/{args.epochs}"
            )
            test_loss, test_acc = evaluate(baseline_model, test_loader, device, "Baseline Eval")
            
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
            
            baseline_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            })
            
            if test_acc > best_baseline_acc:
                best_baseline_acc = test_acc
        
        results["baseline"] = {
            "best_acc": best_baseline_acc,
            "trainable_params": count_parameters(baseline_model),
            "history": baseline_history
        }
        
        # Clean up
        del baseline_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========== Train Wave Adapter ==========
    if not args.baseline_only:
        print("\n" + "="*60)
        print(f"Training WAVE ADAPTER ({args.num_adapter_layers} layers)")
        print("="*60)
        
        wave_model = QwenWithWaveAdapter(
            args.model, 
            num_labels=4,
            num_adapter_layers=args.num_adapter_layers
        ).to(device)
        print(f"Trainable parameters: {count_parameters(wave_model):,}")
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, wave_model.parameters()),
            lr=args.lr
        )
        
        best_wave_acc = 0
        wave_history = []
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(
                wave_model, train_loader, optimizer, device,
                f"Wave Epoch {epoch+1}/{args.epochs}"
            )
            test_loss, test_acc = evaluate(wave_model, test_loader, device, "Wave Eval")
            
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
            
            wave_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            })
            
            if test_acc > best_wave_acc:
                best_wave_acc = test_acc
        
        results["wave_adapter"] = {
            "best_acc": best_wave_acc,
            "trainable_params": count_parameters(wave_model),
            "num_adapter_layers": args.num_adapter_layers,
            "history": wave_history
        }
    
    # ========== Summary ==========
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if "baseline" in results:
        print(f"Baseline:      {results['baseline']['best_acc']*100:.2f}% "
              f"({results['baseline']['trainable_params']:,} params)")
    
    if "wave_adapter" in results:
        print(f"Wave Adapter:  {results['wave_adapter']['best_acc']*100:.2f}% "
              f"({results['wave_adapter']['trainable_params']:,} params)")
    
    if "baseline" in results and "wave_adapter" in results:
        diff = results['wave_adapter']['best_acc'] - results['baseline']['best_acc']
        print(f"\nDifference: {diff*100:+.2f}%")
        
        if diff > 0:
            print("✅ Wave adapter IMPROVES over baseline!")
        elif diff < 0:
            print("❌ Wave adapter underperforms baseline")
        else:
            print("➖ No difference")
    
    # Save results
    results["config"] = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples
    }
    
    with open(output_dir / "wave_adapter_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'wave_adapter_results.json'}")


if __name__ == "__main__":
    main()
