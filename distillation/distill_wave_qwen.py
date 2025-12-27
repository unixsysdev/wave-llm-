"""
Distill Qwen3 attention into wave layers.

Architecture:
- Teacher: Full Qwen3-0.6B (frozen)
- Student: Qwen3 embeddings + wave layers + Qwen3 MLPs + Qwen3 LM head

We keep the MLPs (the "reasoning" part) and replace attention (the "routing" part).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
import argparse

from wave_layers import get_wave_layer


class WaveQwen(nn.Module):
    """
    Qwen3 with attention replaced by wave layers.
    
    Keeps: embeddings, MLPs, RMSNorm, LM head
    Replaces: attention layers with wave layers
    """
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-0.6B",
        wave_layer_type: str = "learned_gate",
        freeze_mlps: bool = True,
        freeze_embeds: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__()
        
        # Load base model
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
        )
        
        config = base_model.config
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        
        # Keep embeddings
        self.embed_tokens = base_model.model.embed_tokens
        if freeze_embeds:
            for param in self.embed_tokens.parameters():
                param.requires_grad = False
        
        # Build layers: wave attention + original MLP
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            layer_dict = nn.ModuleDict()
            
            # Wave layer (replaces attention)
            layer_dict['wave'] = get_wave_layer(
                wave_layer_type,
                self.hidden_size,
                max_seq_len=max_seq_len,
            )
            
            # Original MLP (keep from base model)
            layer_dict['mlp'] = base_model.model.layers[i].mlp
            if freeze_mlps:
                for param in layer_dict['mlp'].parameters():
                    param.requires_grad = False
            
            # Original norms
            layer_dict['input_layernorm'] = base_model.model.layers[i].input_layernorm
            layer_dict['post_attention_layernorm'] = base_model.model.layers[i].post_attention_layernorm
            
            self.layers.append(layer_dict)
        
        # Final norm and LM head
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head
        
        if freeze_embeds:
            for param in self.norm.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
        
        # Clean up base model
        del base_model
        torch.cuda.empty_cache()
        
        print(f"WaveQwen initialized with {wave_layer_type} layers")
        print(f"Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print(f"Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        # Embed
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through layers
        for layer in self.layers:
            # Wave attention (replaces self-attention)
            # Proper residual: save original, normalize, apply wave, add back original
            residual = hidden_states
            hidden_states = layer['input_layernorm'](hidden_states)
            hidden_states = layer['wave'](hidden_states, attention_mask=attention_mask)
            hidden_states = residual + hidden_states  # Proper residual connection
            
            # MLP with residual
            residual = hidden_states
            hidden_states = layer['post_attention_layernorm'](hidden_states)
            hidden_states = layer['mlp'](hidden_states)
            hidden_states = residual + hidden_states
        
        # Final norm + LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {"logits": logits, "loss": loss}


class TextDataset(Dataset):
    """Simple dataset for distillation."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    
    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings.input_ids[idx],
            "attention_mask": self.encodings.attention_mask[idx],
        }


def load_sample_data(n_samples=1000, min_length=100):
    """
    Load diverse text data for distillation.
    
    Tries datasets in order of preference:
    1. SlimPajama (high quality, diverse)
    2. C4 (large, web crawl)
    3. OpenWebText
    4. WikiText-2 (fallback, small)
    """
    from datasets import load_dataset
    
    datasets_to_try = [
        ("cerebras/SlimPajama-627B", "train", "text"),
        ("allenai/c4", "en", "text"),
        ("openwebtext", "train", "text"),
        ("wikitext", "wikitext-103-raw-v1", "text"),
        ("wikitext", "wikitext-2-raw-v1", "text"),
    ]
    
    for dataset_name, split_or_config, text_field in datasets_to_try:
        try:
            print(f"Trying to load {dataset_name}...")
            
            # Use streaming to avoid downloading entire dataset
            if dataset_name == "allenai/c4":
                dataset = load_dataset(dataset_name, split_or_config, split="train", streaming=True)
            elif dataset_name == "cerebras/SlimPajama-627B":
                dataset = load_dataset(dataset_name, split="train", streaming=True)
            elif dataset_name == "openwebtext":
                dataset = load_dataset(dataset_name, split=split_or_config, streaming=True)
            else:
                # WikiText - not streaming, it's small
                dataset = load_dataset(dataset_name, split_or_config, split="train")
                texts = [t for t in dataset[text_field] if len(t.strip()) > min_length][:n_samples]
                print(f"Loaded {len(texts)} samples from {dataset_name}")
                return texts
            
            # For streaming datasets, collect samples
            texts = []
            for sample in dataset:
                text = sample[text_field]
                if len(text.strip()) > min_length:
                    texts.append(text)
                if len(texts) >= n_samples:
                    break
            
            if len(texts) >= n_samples // 2:  # Accept if we got at least half
                print(f"Loaded {len(texts)} samples from {dataset_name}")
                return texts
            else:
                print(f"Only got {len(texts)} samples from {dataset_name}, trying next...")
                continue
                
        except Exception as e:
            print(f"Couldn't load {dataset_name}: {e}")
            continue
    
    # Final fallback: synthetic (should rarely happen)
    print("WARNING: Using synthetic data - distillation quality will be poor!")
    base_texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Machine learning is transforming how we build software. " * 10,
        "Artificial intelligence systems can now understand natural language. " * 10,
        "The transformer architecture has revolutionized NLP. " * 10,
        "Deep learning models require large amounts of data and compute. " * 10,
        "Neural networks learn representations from examples. " * 10,
        "Natural language processing enables computers to understand text. " * 10,
        "Attention mechanisms allow models to focus on relevant parts. " * 10,
    ]
    return base_texts * (n_samples // len(base_texts) + 1)


def distill(
    teacher_model_name: str = "Qwen/Qwen3-0.6B",
    wave_layer_type: str = "learned_gate",
    n_samples: int = 1000,
    batch_size: int = 4,
    n_epochs: int = 3,
    lr: float = 1e-4,
    max_seq_len: int = 256,
    temperature: float = 2.0,
    alpha: float = 0.5,  # balance between soft and hard targets
    output_dir: str = "results",
    device: str = None,
    save_every_n_epochs: int = 1,  # Save checkpoint every N epochs
    freeze_mlps: bool = True,  # Whether to freeze MLP layers
):
    """
    Distill teacher attention into student wave layers.
    
    Args:
        teacher_model_name: Teacher model (full Qwen3)
        wave_layer_type: Type of wave layer for student
        n_samples: Number of training samples
        batch_size: Batch size
        n_epochs: Number of epochs
        lr: Learning rate
        max_seq_len: Maximum sequence length
        temperature: Distillation temperature
        alpha: Weight for soft targets (1-alpha for hard targets)
        output_dir: Where to save results
        device: Device to use
        save_every_n_epochs: Save checkpoint every N epochs
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"Wave layer type: {wave_layer_type}")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load teacher (frozen)
    print("Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float32,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Create student
    print("Creating student model...")
    student = WaveQwen(
        base_model_name=teacher_model_name,
        wave_layer_type=wave_layer_type,
        freeze_mlps=freeze_mlps,
        freeze_embeds=True,
        max_seq_len=max_seq_len,
    ).to(device)
    
    # Load data
    print("Loading data...")
    texts = load_sample_data(n_samples)
    dataset = TextDataset(texts, tokenizer, max_length=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer (only wave layer params)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    print(f"\nTraining...")
    print(f"Samples: {len(dataset)}, Batches: {len(dataloader)}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Checkpoints will be saved every {save_every_n_epochs} epoch(s)")
    
    history = {
        "epoch": [],
        "loss": [],
        "kl_loss": [],
        "ce_loss": [],
    }
    
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(n_epochs):
        student.train()
        total_loss = 0
        total_kl = 0
        total_ce = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Teacher forward (frozen)
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_outputs.logits
            
            # Student forward
            student_outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = student_outputs["logits"]
            
            # Distillation loss (KL divergence on soft targets)
            soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / temperature, dim=-1)
            kl_loss = F.kl_div(
                soft_student,
                soft_teacher.exp(),
                reduction="batchmean",
            ) * (temperature ** 2)
            
            # Hard target loss (standard cross-entropy on next token prediction)
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, student.vocab_size),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            
            # Combined loss
            loss = alpha * kl_loss + (1 - alpha) * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_kl += kl_loss.item()
            total_ce += ce_loss.item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
                "ce": f"{ce_loss.item():.4f}",
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        avg_ce = total_ce / len(dataloader)
        
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["kl_loss"].append(avg_kl)
        history["ce_loss"].append(avg_ce)
        
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, kl={avg_kl:.4f}, ce={avg_ce:.4f}")
        
        # Track best model
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_epoch = epoch + 1
        
        # Save checkpoint every N epochs
        if (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_file = output_path / f"wave_qwen_{wave_layer_type}_{timestamp}_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "history": history,
                "config": {
                    "teacher": teacher_model_name,
                    "wave_layer_type": wave_layer_type,
                    "n_samples": n_samples,
                    "n_epochs": n_epochs,
                    "lr": lr,
                    "max_seq_len": max_seq_len,
                    "temperature": temperature,
                    "alpha": alpha,
                },
            }, checkpoint_file)
            print(f"  💾 Checkpoint saved: {checkpoint_file}")
            
            # Save best model separately
            if is_best:
                best_file = output_path / f"wave_qwen_{wave_layer_type}_{timestamp}_best.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "history": history,
                    "config": {
                        "teacher": teacher_model_name,
                        "wave_layer_type": wave_layer_type,
                        "n_samples": n_samples,
                        "n_epochs": n_epochs,
                        "lr": lr,
                        "max_seq_len": max_seq_len,
                        "temperature": temperature,
                        "alpha": alpha,
                    },
                }, best_file)
                print(f"  ⭐ New best model! (loss={avg_loss:.4f})")
    
    # Evaluate
    print("\nEvaluating...")
    student.eval()
    teacher.eval()
    
    eval_results = evaluate_models(teacher, student, tokenizer, device, max_seq_len)
    
    # Save final results
    results = {
        "config": {
            "teacher": teacher_model_name,
            "wave_layer_type": wave_layer_type,
            "n_samples": n_samples,
            "n_epochs": n_epochs,
            "lr": lr,
            "temperature": temperature,
            "alpha": alpha,
        },
        "history": history,
        "eval": eval_results,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
    }
    
    results_file = output_path / f"distill_{wave_layer_type}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save final student model
    model_file = output_path / f"wave_qwen_{wave_layer_type}_{timestamp}_final.pt"
    torch.save({
        "epoch": n_epochs,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "history": history,
        "config": {
            "teacher": teacher_model_name,
            "wave_layer_type": wave_layer_type,
            "n_samples": n_samples,
            "n_epochs": n_epochs,
            "lr": lr,
            "max_seq_len": max_seq_len,
            "temperature": temperature,
            "alpha": alpha,
        },
    }, model_file)
    
    print(f"\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best epoch: {best_epoch} (loss={best_loss:.4f})")
    print(f"Results saved to: {results_file}")
    print(f"Final model: {model_file}")
    print(f"Best model: wave_qwen_{wave_layer_type}_{timestamp}_best.pt")
    print("=" * 60)
    
    return results


def evaluate_models(teacher, student, tokenizer, device, max_seq_len):
    """Compare teacher and student on sample prompts."""
    
    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "The quick brown fox",
        "In the year 2024,",
        "Python is a programming language that",
    ]
    
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)
        
        with torch.no_grad():
            # Teacher generation
            teacher_out = teacher.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            teacher_text = tokenizer.decode(teacher_out[0], skip_special_tokens=True)
            
            # Student logits (can't use generate easily, just check logits)
            student_out = student(**inputs)
            student_logits = student_out["logits"]
            
            # Compare next token predictions
            teacher_logits = teacher(**inputs).logits
            
            # Top-5 agreement
            teacher_top5 = teacher_logits[0, -1].topk(5).indices
            student_top5 = student_logits[0, -1].topk(5).indices
            
            agreement = len(set(teacher_top5.tolist()) & set(student_top5.tolist())) / 5
            
            results.append({
                "prompt": prompt,
                "teacher_text": teacher_text,
                "top5_agreement": agreement,
            })
    
    # Average agreement
    avg_agreement = sum(r["top5_agreement"] for r in results) / len(results)
    
    print(f"\nTop-5 Token Agreement: {avg_agreement:.2%}")
    print("\nSample outputs:")
    for r in results[:3]:
        print(f"  Prompt: {r['prompt']}")
        print(f"  Teacher: {r['teacher_text']}")
        print(f"  Agreement: {r['top5_agreement']:.2%}")
        print()
    
    return {
        "avg_top5_agreement": avg_agreement,
        "samples": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--wave_type", type=str, default="learned_gate",
                        choices=["learned_gate", "fnet", "wave_network", "frequency_band"])
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--freeze_mlps", type=bool, default=True, help="Freeze MLP layers (set False to train them)")
    parser.add_argument("--unfreeze_mlps", action="store_true", help="Unfreeze MLP layers for training")
    
    args = parser.parse_args()
    
    # Handle both --freeze_mlps and --unfreeze_mlps flags
    freeze_mlps = args.freeze_mlps and not args.unfreeze_mlps
    
    distill(
        teacher_model_name=args.teacher,
        wave_layer_type=args.wave_type,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        alpha=args.alpha,
        output_dir=args.output_dir,
        save_every_n_epochs=args.save_every,
        freeze_mlps=freeze_mlps,
    )
