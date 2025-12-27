#!/usr/bin/env python3
"""
Generate text from trained WaveQwen model.

Usage:
    python generate_wave.py --model results/wave_qwen_learned_gate_20251227_012550.pt
    python generate_wave.py --model results/wave_qwen_learned_gate_20251227_012550.pt --prompt "Hello world"
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import from distill script
from distill_wave_qwen import WaveQwen


def load_wave_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained WaveQwen model from checkpoint."""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # The checkpoint is just the state_dict directly (not wrapped in a dict)
    # We need to infer config from the checkpoint filename or use defaults
    import os
    filename = os.path.basename(checkpoint_path)
    
    # Default config
    base_model = "Qwen/Qwen3-0.6B"
    wave_type = "learned_gate"
    max_seq_len = 256  # Default used in training
    
    # Try to parse wave type from filename
    if "learned_gate" in filename:
        wave_type = "learned_gate"
    elif "fnet" in filename:
        wave_type = "fnet"
    elif "wave_network" in filename:
        wave_type = "wave_network"
    elif "frequency_band" in filename:
        wave_type = "frequency_band"
    
    # Also check if there's a matching JSON file with config
    json_path = checkpoint_path.replace("wave_qwen_", "distill_").replace(".pt", ".json")
    if os.path.exists(json_path):
        import json
        with open(json_path) as f:
            config = json.load(f).get("config", {})
            base_model = config.get("teacher", base_model)
            wave_type = config.get("wave_layer_type", wave_type)
            max_seq_len = config.get("max_seq_len", max_seq_len)
    
    # Infer max_seq_len from checkpoint if not in config
    # freq_gate_real shape is [n_freqs, hidden_size] where n_freqs = max_seq_len // 2 + 1
    for key, value in checkpoint.items():
        if "freq_gate_real" in key:
            n_freqs = value.shape[0]
            max_seq_len = (n_freqs - 1) * 2
            print(f"Inferred max_seq_len={max_seq_len} from checkpoint (n_freqs={n_freqs})")
            break
    
    print(f"Base model: {base_model}")
    print(f"Wave type: {wave_type}")
    print(f"Max seq len: {max_seq_len}")
    
    # Create model
    model = WaveQwen(
        base_model_name=base_model,
        wave_layer_type=wave_type,
        freeze_mlps=True,
        freeze_embeds=True,
        max_seq_len=max_seq_len,
    )
    
    # Load trained weights - checkpoint is the state_dict directly
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    return model, base_model


def generate_greedy(
    model: WaveQwen,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> str:
    """Generate text using greedy decoding."""
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits
            outputs = model(generated_ids)
            logits = outputs["logits"]
            
            # Get next token (greedy)
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            # Stop if EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def generate_with_sampling(
    model: WaveQwen,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    """Generate text using sampling with temperature."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated_ids)
            logits = outputs["logits"]
            
            # Get next token logits and apply temperature
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def show_top_predictions(
    model: WaveQwen,
    tokenizer,
    prompt: str,
    top_k: int = 10,
    device: str = "cuda",
):
    """Show the top-k predicted next tokens."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        logits = outputs["logits"]
        
        # Get probabilities for next token
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Get top-k
        top_probs, top_indices = probs.topk(top_k)
        
        print(f"\nTop {top_k} predictions for '{prompt}':")
        print("-" * 50)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' (prob: {prob:.4f})")


def compare_with_teacher(
    wave_model: WaveQwen,
    teacher_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    device: str = "cuda",
):
    """Generate from both models and compare side-by-side."""
    
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print('='*60)
    
    # Teacher generation
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        teacher_out = teacher_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        teacher_text = tokenizer.decode(teacher_out[0], skip_special_tokens=True)
    
    # Wave generation (greedy)
    wave_text = generate_greedy(wave_model, tokenizer, prompt, max_new_tokens, device)
    
    print(f"\n📚 Teacher (Qwen3-0.6B):")
    print(f"   {teacher_text}")
    print(f"\n🌊 Wave Model:")
    print(f"   {wave_text}")
    
    # Show top predictions comparison
    print(f"\n📊 Next token predictions comparison:")
    
    with torch.no_grad():
        teacher_logits = teacher_model(**inputs).logits[0, -1, :]
        wave_logits = wave_model(inputs["input_ids"])["logits"][0, -1, :]
        
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        wave_probs = F.softmax(wave_logits, dim=-1)
        
        teacher_top5 = teacher_probs.topk(5)
        wave_top5 = wave_probs.topk(5)
        
        print(f"\n   Teacher top-5:")
        for prob, idx in zip(teacher_top5.values, teacher_top5.indices):
            print(f"      '{tokenizer.decode([idx])}' ({prob:.3f})")
        
        print(f"\n   Wave top-5:")
        for prob, idx in zip(wave_top5.values, wave_top5.indices):
            print(f"      '{tokenizer.decode([idx])}' ({prob:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Generate text from WaveQwen model")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--compare", action="store_true", help="Compare with teacher model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/mps)")
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")
    
    # Load wave model
    wave_model, base_model_name = load_wave_model(args.model, args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load teacher if comparing
    teacher_model = None
    if args.compare:
        print(f"\nLoading teacher model for comparison...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
        ).to(args.device)
        teacher_model.eval()
    
    # Default prompts
    prompts = [
        "The capital of France is",
        "Machine learning is",
        "The quick brown fox",
        "Once upon a time",
        "def fibonacci(n):",
        "To be or not to be",
    ]
    
    if args.prompt:
        prompts = [args.prompt]
    
    print("\n" + "="*60)
    print("WAVE MODEL GENERATION")
    print("="*60)
    
    for prompt in prompts:
        if args.compare and teacher_model is not None:
            compare_with_teacher(
                wave_model, teacher_model, tokenizer, 
                prompt, args.max_tokens, args.device
            )
        else:
            print(f"\n📝 Prompt: {prompt}")
            
            # Show top predictions
            show_top_predictions(wave_model, tokenizer, prompt, top_k=10, device=args.device)
            
            # Greedy generation
            print(f"\n🤖 Greedy generation:")
            greedy_text = generate_greedy(
                wave_model, tokenizer, prompt, 
                args.max_tokens, args.device
            )
            print(f"   {greedy_text}")
            
            # Sampled generation
            print(f"\n🎲 Sampled generation (temp={args.temperature}):")
            sampled_text = generate_with_sampling(
                wave_model, tokenizer, prompt,
                args.max_tokens, args.temperature,
                device=args.device
            )
            print(f"   {sampled_text}")
        
        print()


if __name__ == "__main__":
    main()
