#!/usr/bin/env python3
"""
Wave Utilities for Token Representation

Utility functions for converting between vector embeddings and wave representations,
implementing ideas from the Wave Network paper (arXiv:2411.02674).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def embedding_to_wave(embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose an embedding vector into wave parameters via FFT.
    
    This is a lossless decomposition - the original embedding can be
    perfectly reconstructed from the wave parameters.
    
    Args:
        embedding: [hidden_size] tensor
        
    Returns:
        amplitude: [n_freqs] magnitude of each frequency component
        phase: [n_freqs] phase angle of each frequency component
        frequencies: [n_freqs] the frequency values (fixed, not learned)
    """
    fft = torch.fft.rfft(embedding)
    amplitude = torch.abs(fft)
    phase = torch.angle(fft)
    frequencies = torch.fft.rfftfreq(len(embedding))
    return amplitude, phase, frequencies


def wave_to_embedding(amplitude: torch.Tensor, phase: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """
    Reconstruct an embedding from wave parameters.
    
    Args:
        amplitude: [n_freqs] magnitude of each frequency component
        phase: [n_freqs] phase angle of each frequency component
        hidden_size: Original embedding dimension
        
    Returns:
        embedding: [hidden_size] reconstructed vector
    """
    # Reconstruct complex spectrum
    fft = amplitude * torch.exp(1j * phase)
    # Inverse FFT
    embedding = torch.fft.irfft(fft, n=hidden_size)
    return embedding


def compute_wave_similarity(
    emb1: torch.Tensor,
    emb2: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Compute similarity between two embeddings in wave space.
    
    Returns:
        amplitude_sim: Cosine similarity of amplitude spectra
        phase_coherence: How consistent the phase difference is
        embedding_sim: Original cosine similarity
    """
    amp1, phase1, _ = embedding_to_wave(emb1)
    amp2, phase2, _ = embedding_to_wave(emb2)
    
    # Amplitude similarity
    amp_sim = F.cosine_similarity(amp1.unsqueeze(0), amp2.unsqueeze(0)).item()
    
    # Phase coherence
    phase_diff = phase1 - phase2
    phase_coherence = torch.abs(torch.mean(torch.exp(1j * phase_diff))).item()
    
    # Original embedding similarity
    emb_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    return amp_sim, phase_coherence, emb_sim


class WaveRepresentation(nn.Module):
    """
    Convert token embeddings to wave representation.
    
    Following the Wave Network paper, each token is represented as:
    - Magnitude (G): global semantics, shared across sequence
    - Phase (α): each token's relationship to global meaning
    
    Complex representation: z = G * e^(i*α)
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert embeddings to complex wave representation.
        
        Args:
            embeddings: [batch, seq, hidden] token embeddings
            
        Returns:
            complex: [batch, seq, hidden//2] complex tensor
        """
        batch, seq, hidden = embeddings.shape
        
        # Global semantics = mean across sequence
        G = embeddings.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        G = G.expand_as(embeddings)  # [batch, seq, hidden]
        
        # Local phase = angle between token and global
        dot = (embeddings * G).sum(dim=-1, keepdim=True)  # [batch, seq, 1]
        norm_e = embeddings.norm(dim=-1, keepdim=True) + 1e-8
        norm_g = G.norm(dim=-1, keepdim=True) + 1e-8
        cos_angle = (dot / (norm_e * norm_g)).clamp(-1, 1)
        angle = torch.acos(cos_angle)  # [batch, seq, 1]
        
        # Expand angle to match hidden dimension
        # Use different phases for different dimensions
        dim_phases = torch.linspace(0, 2*np.pi, hidden//2, device=embeddings.device)
        angle = angle + dim_phases.view(1, 1, -1)  # [batch, seq, hidden//2]
        
        # Magnitude from global semantics
        magnitude = G[:, :, :hidden//2].norm(dim=-1, keepdim=True)  # [batch, seq, 1]
        
        # Complex representation
        real = magnitude * torch.cos(angle)
        imag = magnitude * torch.sin(angle)
        
        return torch.complex(real, imag)


class WaveInterference(nn.Module):
    """
    Wave interference layer for mixing tokens.
    
    Instead of dot product attention, uses wave interference:
    - When phases align → constructive (amplify)
    - When phases oppose → destructive (cancel)
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_complex = hidden_size // 2
        
        # Projections to Q, K, V
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.to_out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Wave interference attention.
        
        Args:
            x: [batch, seq, hidden] input tensor
            mask: [batch, seq, seq] optional attention mask
            
        Returns:
            output: [batch, seq, hidden] output tensor
        """
        batch, seq, _ = x.shape
        
        # Project to Q, K, V
        Q = self.to_q(x)  # [batch, seq, hidden]
        K = self.to_k(x)
        V = self.to_v(x)
        
        # Convert to complex (split hidden into real/imag)
        Q_c = self._to_complex(Q)  # [batch, seq, hidden//2]
        K_c = self._to_complex(K)
        V_c = self._to_complex(V)
        
        # Wave attention: phase alignment
        # similarity = Re(Q · K*) where K* is complex conjugate
        scores = torch.real(torch.einsum('bqd,bkd->bqk', Q_c, K_c.conj()))
        scores = scores / np.sqrt(self.d_complex)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)  # [batch, seq, seq]
        
        # Combine via interference (complex weighted sum)
        output_c = torch.einsum('bqk,bkd->bqd', weights.to(V_c.dtype), V_c)
        
        # Convert back to real
        output = self._to_real(output_c)
        
        return self.to_out(output)
    
    def _to_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Split hidden dimension into real and imaginary parts."""
        real, imag = x.chunk(2, dim=-1)
        return torch.complex(real, imag)
    
    def _to_real(self, z: torch.Tensor) -> torch.Tensor:
        """Concatenate real and imaginary parts."""
        return torch.cat([z.real, z.imag], dim=-1)


class FFTMixer(nn.Module):
    """
    FFT-based token mixing (similar to FNet).
    
    Replaces attention with O(n log n) FFT operations.
    Much faster than attention for long sequences.
    """
    
    def __init__(self, hidden_size: int, learnable_filter: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        
        if learnable_filter:
            # Learnable frequency filter
            self.freq_filter = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer('freq_filter', torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFT mixing.
        
        Args:
            x: [batch, seq, hidden] input tensor
            
        Returns:
            output: [batch, seq, hidden] mixed tensor
        """
        # 2D FFT over sequence and hidden dimensions
        x_freq = torch.fft.fft2(x.float())
        
        # Apply learnable filter
        x_freq = x_freq * self.freq_filter.view(1, 1, -1)
        
        # Inverse FFT
        output = torch.fft.ifft2(x_freq).real
        
        return output.to(x.dtype)


# Convenience functions

def visualize_wave(embedding: torch.Tensor, n_freqs: int = 20) -> np.ndarray:
    """
    Reconstruct a waveform from an embedding for visualization.
    
    Args:
        embedding: [hidden_size] tensor
        n_freqs: Number of frequencies to use
        
    Returns:
        wave: [n_points] reconstructed waveform
    """
    amp, phase, _ = embedding_to_wave(embedding)
    
    n_points = 200
    t = np.linspace(0, 2*np.pi, n_points)
    wave = np.zeros(n_points)
    
    for k in range(min(n_freqs, len(amp))):
        wave += amp[k].item() * np.cos(k * t + phase[k].item())
    
    return wave


def batch_embedding_to_wave(
    embeddings: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch convert embeddings to wave parameters.
    
    Args:
        embeddings: [batch, seq, hidden] or [n_tokens, hidden]
        
    Returns:
        amplitudes: [..., n_freqs] magnitudes
        phases: [..., n_freqs] phase angles
    """
    fft = torch.fft.rfft(embeddings, dim=-1)
    return torch.abs(fft), torch.angle(fft)
