"""
Tiny transformers for grokking comparison: Standard Attention vs FFT/Wave.

Both models have identical structure except for the mixing mechanism:
- Standard: dot-product attention
- Wave: FFT-based mixing

Task: Modular addition (a + b) mod p
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import einsum


class StandardAttentionTransformer(nn.Module):
    """
    One-layer transformer with standard dot-product attention.
    Baseline for comparison.
    """
    
    def __init__(
        self,
        p: int = 53,
        d_model: int = 128,
        n_heads: int = 4,
        d_mlp: int = 512,
    ):
        super().__init__()
        
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp
        
        self.vocab_size = p + 1
        self.eq_token = p
        
        # Embeddings
        self.token_embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed = nn.Embedding(3, d_model)
        
        # Attention
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(n_heads, self.d_head, d_model) / math.sqrt(d_model))
        
        # MLP
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        
        # Unembedding
        self.unembed = nn.Linear(d_model, p, bias=False)
        
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size = a.shape[0]
        device = a.device
        
        # Create input: [a, b, =]
        eq_tokens = torch.full((batch_size,), self.eq_token, device=device, dtype=torch.long)
        tokens = torch.stack([a, b, eq_tokens], dim=1)
        
        # Embeddings
        positions = torch.arange(3, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        
        # Standard attention
        q = einsum(x, self.W_Q, 'b s d, h d k -> b h s k')
        k = einsum(x, self.W_K, 'b s d, h d k -> b h s k')
        v = einsum(x, self.W_V, 'b s d, h d k -> b h s k')
        
        attn_scores = einsum(q, k, 'b h s1 d, b h s2 d -> b h s1 s2') / math.sqrt(self.d_head)
        
        # Causal mask
        mask = torch.triu(torch.ones(3, 3, device=device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_pattern = F.softmax(attn_scores, dim=-1)
        
        attn_out = einsum(attn_pattern, v, 'b h s1 s2, b h s2 d -> b h s1 d')
        attn_out = einsum(attn_out, self.W_O, 'b h s d, h d m -> b s m')
        
        x = x + attn_out
        
        # MLP
        mlp_out = self.W_out(F.relu(self.W_in(x)))
        x = x + mlp_out
        
        # Output from position 2
        logits = self.unembed(x[:, 2, :])
        return logits
    
    def get_neuron_logit_map(self) -> torch.Tensor:
        return self.unembed.weight @ self.W_out.weight


class WaveTransformer(nn.Module):
    """
    One-layer transformer with FFT-based mixing instead of attention.
    
    Key difference: Instead of Q @ K.T -> softmax -> @ V,
    we use FFT to mix information across sequence positions.
    """
    
    def __init__(
        self,
        p: int = 53,
        d_model: int = 128,
        n_heads: int = 4,  # Kept for param parity, used differently
        d_mlp: int = 512,
        wave_mode: str = "fnet",  # "fnet", "learned_gate", "phase_shift"
    ):
        super().__init__()
        
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp
        self.wave_mode = wave_mode
        
        self.vocab_size = p + 1
        self.eq_token = p
        
        # Embeddings (same as standard)
        self.token_embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed = nn.Embedding(3, d_model)
        
        # Wave mixing parameters
        if wave_mode == "fnet":
            # Pure FNet: no learned params in mixing
            pass
        elif wave_mode == "learned_gate":
            # Learn which frequencies to keep
            # For seq_len=3, we have 2 freq components (rfft)
            self.freq_gate = nn.Parameter(torch.ones(2, d_model))
        elif wave_mode == "phase_shift":
            # Learn phase rotations
            self.phase_shift = nn.Parameter(torch.zeros(2, d_model))
        elif wave_mode == "full":
            # Full learnable complex transform (most params, closest to attention)
            # Keep Q/K/V style projections for param parity
            self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
            self.W_K = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
            self.W_V = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
            self.W_O = nn.Parameter(torch.randn(n_heads, self.d_head, d_model) / math.sqrt(d_model))
        
        # MLP (same as standard)
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        
        # Unembedding
        self.unembed = nn.Linear(d_model, p, bias=False)
        
    def wave_mix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mix information across sequence using FFT.
        
        Args:
            x: (batch, seq=3, d_model)
        Returns:
            mixed: (batch, seq=3, d_model)
        """
        if self.wave_mode == "fnet":
            # Pure FNet: FFT along sequence, then hidden dim
            x_fft = torch.fft.fft(torch.fft.fft(x, dim=1), dim=2).real
            return x_fft
            
        elif self.wave_mode == "learned_gate":
            # FFT -> gate frequencies -> IFFT
            x_fft = torch.fft.rfft(x, dim=1)  # (batch, 2, d_model) complex
            x_fft = x_fft * self.freq_gate  # element-wise gate
            return torch.fft.irfft(x_fft, n=3, dim=1)
            
        elif self.wave_mode == "phase_shift":
            # FFT -> rotate phase -> IFFT
            x_fft = torch.fft.rfft(x, dim=1)  # (batch, 2, d_model) complex
            rotation = torch.exp(1j * self.phase_shift)  # complex rotation
            x_fft = x_fft * rotation
            return torch.fft.irfft(x_fft, n=3, dim=1)
            
        elif self.wave_mode == "full":
            # Phase-alignment attention (like in train_wave_adapter.py)
            batch, seq, _ = x.shape
            
            q = einsum(x, self.W_Q, 'b s d, h d k -> b h s k')
            k = einsum(x, self.W_K, 'b s d, h d k -> b h s k')
            v = einsum(x, self.W_V, 'b s d, h d k -> b h s k')
            
            # FFT on head dimension
            q_fft = torch.fft.rfft(q, dim=-1)
            k_fft = torch.fft.rfft(k, dim=-1)
            
            # Phase alignment scores
            q_phase = torch.angle(q_fft)
            k_phase = torch.angle(k_fft)
            q_amp = torch.abs(q_fft)
            k_amp = torch.abs(k_fft)
            
            # cos(phase_q - phase_k) weighted by amplitude
            phase_diff = q_phase.unsqueeze(3) - k_phase.unsqueeze(2)
            amp_weight = torch.sqrt(q_amp.unsqueeze(3) * k_amp.unsqueeze(2) + 1e-8)
            scores = (torch.cos(phase_diff) * amp_weight).sum(dim=-1)
            
            # Causal mask
            mask = torch.triu(torch.ones(3, 3, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            
            out = einsum(attn, v, 'b h s1 s2, b h s2 d -> b h s1 d')
            out = einsum(out, self.W_O, 'b h s d, h d m -> b s m')
            return out
        
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size = a.shape[0]
        device = a.device
        
        # Create input: [a, b, =]
        eq_tokens = torch.full((batch_size,), self.eq_token, device=device, dtype=torch.long)
        tokens = torch.stack([a, b, eq_tokens], dim=1)
        
        # Embeddings
        positions = torch.arange(3, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        
        # Wave mixing (instead of attention)
        wave_out = self.wave_mix(x)
        x = x + wave_out
        
        # MLP
        mlp_out = self.W_out(F.relu(self.W_in(x)))
        x = x + mlp_out
        
        # Output from position 2
        logits = self.unembed(x[:, 2, :])
        return logits
    
    def get_neuron_logit_map(self) -> torch.Tensor:
        return self.unembed.weight @ self.W_out.weight
    
    def get_freq_weights(self):
        """Return learned frequency parameters for analysis."""
        if self.wave_mode == "learned_gate":
            return {"freq_gate": self.freq_gate.detach()}
        elif self.wave_mode == "phase_shift":
            return {"phase_shift": self.phase_shift.detach()}
        elif self.wave_mode == "full":
            return {
                "W_Q": self.W_Q.detach(),
                "W_K": self.W_K.detach(),
            }
        return {}


def create_modular_data(p: int, device: torch.device):
    """Create all (a, b, (a+b) mod p) pairs."""
    a = torch.arange(p, device=device).repeat_interleave(p)
    b = torch.arange(p, device=device).repeat(p)
    targets = (a + b) % p
    return a, b, targets


def train_test_split(p: int, train_frac: float, device: torch.device, seed: int = 42):
    """Split into train/test."""
    torch.manual_seed(seed)
    a, b, targets = create_modular_data(p, device)
    n_total = p * p
    n_train = int(n_total * train_frac)
    
    perm = torch.randperm(n_total, device=device)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    return (a[train_idx], b[train_idx], targets[train_idx]), \
           (a[test_idx], b[test_idx], targets[test_idx])


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = 53
    
    print("Model comparison:")
    print("-" * 40)
    
    standard = StandardAttentionTransformer(p=p).to(device)
    print(f"Standard Attention: {count_params(standard):,} params")
    
    for mode in ["fnet", "learned_gate", "phase_shift", "full"]:
        wave = WaveTransformer(p=p, wave_mode=mode).to(device)
        print(f"Wave ({mode}): {count_params(wave):,} params")
    
    # Test forward
    a = torch.tensor([0, 1, 2], device=device)
    b = torch.tensor([3, 4, 5], device=device)
    
    print("\nForward pass test:")
    print(f"Standard output shape: {standard(a, b).shape}")
    print(f"Wave output shape: {wave(a, b).shape}")
