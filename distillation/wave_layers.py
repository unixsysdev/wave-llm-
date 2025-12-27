"""
Wave layers for distillation experiments.

These replace attention layers in transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnedGateWaveLayer(nn.Module):
    """
    FFT-based mixing with learned frequency gates.
    
    O(n log n) complexity - much faster than attention.
    Learns which frequencies to amplify/suppress.
    
    NOTE: This layer does NOT include residual connection or layernorm.
    The caller (e.g., WaveQwen) is responsible for residual + norm.
    This matches the design in experiments/models.py that works.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Learned frequency gates
        # For rfft, we get seq_len//2 + 1 frequency components
        n_freqs = max_seq_len // 2 + 1
        self.freq_gate_real = nn.Parameter(torch.ones(n_freqs, hidden_size))
        self.freq_gate_imag = nn.Parameter(torch.zeros(n_freqs, hidden_size))
        
        # Dropout only - no layernorm here
        self.dropout = nn.Dropout(dropout)
        
        # Optional projection (like attention output projection)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size) - should be pre-normalized by caller
            attention_mask: (batch, seq_len) - optional, not used for FFT
        Returns:
            output: (batch, seq_len, hidden_size) - the wave mixing output (NO residual added)
        """
        batch, seq_len, hidden = x.shape
        
        # FFT along sequence dimension
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, n_freqs, hidden) complex
        
        # Get the right slice of frequency gates for this seq_len
        n_freqs = x_fft.shape[1]
        gate = self.freq_gate_real[:n_freqs] + 1j * self.freq_gate_imag[:n_freqs]
        
        # Apply learned frequency gates
        x_fft = x_fft * gate
        
        # IFFT back to sequence domain
        x = torch.fft.irfft(x_fft, n=seq_len, dim=1)  # (batch, seq_len, hidden)
        
        # Output projection + dropout
        x = self.out_proj(x)
        x = self.dropout(x)
        
        # NO residual, NO layernorm - caller handles this
        return x


class FNetLayer(nn.Module):
    """
    Pure FNet - no learned parameters in mixing.
    Just FFT along sequence and hidden dims.
    
    NOTE: No internal residual/layernorm - caller handles this.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        # 2D FFT: sequence then hidden
        x = torch.fft.fft2(x).real
        x = self.dropout(x)
        return x  # NO residual, NO layernorm


class WaveNetworkLayer(nn.Module):
    """
    Wave Network style (from arxiv 2411.02674).
    
    Explicit magnitude (global) + phase (local→global relationship).
    O(n) complexity via interference/modulation.
    
    NOTE: No internal residual/layernorm - caller handles this.
    """
    
    def __init__(
        self,
        hidden_size: int,
        mode: str = "interference",  # "interference" or "modulation"
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        
        # Projections to create two variants for wave combination
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        
        # Output
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def token2wave(self, embeddings: torch.Tensor):
        """
        Convert token embeddings to wave representation.
        
        Args:
            embeddings: (batch, seq, hidden)
        Returns:
            Z: complex tensor (batch, seq, hidden)
        """
        # Global magnitude: L2 norm across all tokens per dimension
        G = torch.sqrt((embeddings ** 2).sum(dim=1, keepdim=True) + 1e-8)  # (batch, 1, hidden)
        
        # Phase: each token's relation to global
        ratio = embeddings / (G + 1e-8)  # (batch, seq, hidden)
        ratio = torch.clamp(ratio, -1 + 1e-7, 1 - 1e-7)  # numerical stability
        phase = torch.acos(ratio)  # (batch, seq, hidden)
        
        # Complex representation: G * e^(i * phase)
        Z = G * torch.complex(torch.cos(phase), torch.sin(phase))
        
        return Z, G
    
    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        # Create two variants via projection
        x1 = self.proj1(x)
        x2 = self.proj2(x)
        
        # Convert to wave representation
        Z1, G1 = self.token2wave(x1)
        Z2, G2 = self.token2wave(x2)
        
        # Combine via interference or modulation
        if self.mode == "interference":
            Z_combined = Z1 + Z2  # wave interference
        else:
            Z_combined = Z1 * Z2  # wave modulation
        
        # Take real part or magnitude as output
        x = Z_combined.real
        
        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x  # NO residual, NO layernorm


class FrequencyBandLayer(nn.Module):
    """
    Hybrid: FFT for low frequencies (global), small attention for high frequencies (local).
    
    Balances O(n log n) efficiency with attention expressiveness.
    
    NOTE: No internal residual/layernorm - caller handles this.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_low_freqs: int = 8,  # how many low frequencies to handle with FFT
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_low_freqs = n_low_freqs
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        # FFT gate for low frequencies
        self.low_freq_gate = nn.Parameter(torch.ones(n_low_freqs, hidden_size))
        
        # Small attention for high frequencies
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        
        # FFT
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, n_freqs, hidden)
        n_freqs = x_fft.shape[1]
        
        # Split into low and high frequencies
        n_low = min(self.n_low_freqs, n_freqs)
        
        # Low frequencies: learned gating (cheap, global)
        low_fft = x_fft[:, :n_low, :] * self.low_freq_gate[:n_low]
        
        # High frequencies: convert back and apply small attention
        high_fft = x_fft[:, n_low:, :]
        
        # Reconstruct from low frequencies
        low_out_fft = torch.zeros_like(x_fft)
        low_out_fft[:, :n_low, :] = low_fft
        low_out = torch.fft.irfft(low_out_fft, n=seq_len, dim=1)
        
        # Reconstruct high frequency signal
        high_out_fft = torch.zeros_like(x_fft)
        high_out_fft[:, n_low:, :] = high_fft
        high_signal = torch.fft.irfft(high_out_fft, n=seq_len, dim=1)
        
        # Small attention on high frequency content
        Q = self.q_proj(high_signal).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(high_signal).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(high_signal).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        high_out = torch.matmul(attn, V)
        high_out = high_out.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        
        # Combine
        x = low_out + high_out
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x  # NO residual, NO layernorm


def get_wave_layer(
    layer_type: str,
    hidden_size: int,
    **kwargs
) -> nn.Module:
    """Factory function for wave layers."""
    
    if layer_type == "learned_gate":
        return LearnedGateWaveLayer(hidden_size, **kwargs)
    elif layer_type == "fnet":
        return FNetLayer(hidden_size, **kwargs)
    elif layer_type == "wave_network":
        return WaveNetworkLayer(hidden_size, **kwargs)
    elif layer_type == "frequency_band":
        return FrequencyBandLayer(hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
