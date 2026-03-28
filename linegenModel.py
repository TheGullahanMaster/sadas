import os
import sys
import json
import time
import math
import random
import pathlib
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
@torch.jit.script
def heinsen_associative_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor, h0: Optional[torch.Tensor] = None):
    """
    Computes parallel scan in log-space.
    h_t = a_t * h_{t-1} + b_t  -->  log(h_t) = log_a + log(h_{t-1}) (associative)
    """
    # 1. Cumulative sum of log_coeffs (log_A)
    a_star = torch.cumsum(log_coeffs, dim=1)
    
    # 2. Log-Cumulative-Sum-Exp of the values corrected by A
    # log_values is log(b_t)
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    
    # 3. Combine
    log_h = a_star + log_h0_plus_b_star
    
    # 4. Handle initial state if present (h0 is strictly positive in this formulation)
    if h0 is not None:
        # Broadcasting h0 correction: log(h0) + A_t
        log_h0 = torch.log(torch.clamp(h0, min=1e-8)).unsqueeze(1)
        log_h = torch.logaddexp(log_h, a_star + log_h0)
        
    return torch.exp(log_h)

@torch.jit.script
def g_act(x: torch.Tensor):
    # The "g" activation from minGRU paper: linear for positive, sigmoid for negative
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

@torch.jit.script
def log_g_act(x: torch.Tensor):
    # Stable log(g(x))
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))
@torch.jit.script
def pscan_linear_jit(A, X, h0: Optional[torch.Tensor]):
    """
    Stable Linear Recurrence: h_t = A_t * h_{t-1} + X_t
    Compiles to fused CUDA kernel for speed.
    """
    # A, X: (B, T, D)
    # h0: (B, D)
    B, T, D = X.shape
    h_current = h0 if h0 is not None else torch.zeros((B, D), device=X.device, dtype=X.dtype)
    
    output = []
    # Loop over time (JIT fuses this)
    for t in range(T):
        a_t = A[:, t, :]
        x_t = X[:, t, :]
        h_current = a_t * h_current + x_t
        output.append(h_current)
        
    return torch.stack(output, dim=1)
# ========= Optional activations from lamb.py =========
class PostSDPAGate(nn.Module):
    """
    Implementation of the Gated Attention mechanism (G1 position).
    Applies a head-specific (elementwise) sigmoid gate to SDPA output.
    
    Paper: "Gated Attention for Large Language Models" (Qiu et al., 2025)
    Ref: [cite: 858, 1072]
    """
    def __init__(self, d_model):
        super().__init__()
        # The paper recommends elementwise gating (n x q x dk), which 
        # is equivalent to a linear layer projecting to d_model followed by sigmoid.
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Init: standard Xavier, bias 0 (starts roughly near 0.5 gating)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x_input, y_attn):
        """
        Args:
            x_input: The input to the attention block (usually normalized). 
                     Used to compute the gate score[cite: 1086].
            y_attn:  The output of the SDPA (before Wo).
        """
        gate_score = torch.sigmoid(self.gate_proj(x_input))
        return y_attn * gate_score
def robust_log_scan(log_coeffs: torch.Tensor, log_values: torch.Tensor):
    """
    Computes h_t = a_t * h_{t-1} + x_t in log space.
    Stable parallel scan.
    """
    # log_coeffs = log(a)
    # log_values = log(x)
    
    # 1. Accumulate decay: A_t = prod(a_1...a_t) -> log_A = cumsum(log_a)
    log_A = torch.cumsum(log_coeffs, dim=1)
    
    # Pad with 0 (log 1) at start for alignment
    log_A_pad = F.pad(log_A, (0,0,1,0))

    # 2. Accumulate input: sum(x_k / A_k) -> logsumexp(log_x - log_A)
    # We want to compute the "inner sum" efficiently in parallel
    
    # This subtraction aligns the decay schedule
    shifted_values = log_values - log_A_pad[:, :-1] # Align time steps
    
    # The "Parallel" magic: logcumsumexp is heavily optimized in PyTorch
    acc = torch.logcumsumexp(shifted_values, dim=1)
    
    # 3. Combine: h_t = A_t * acc_t
    log_h = log_A + acc
    
    # Safety Clamp to prevent float32 infinity (NaNs)
    return torch.exp(log_h.clamp(max=50.0))

def parallel_scan_split(A, X, h0: Optional[torch.Tensor]=None):
    """
    Handles x_t = a_t * h_{t-1} + x_t for SIGNED x_t using parallel log-scan.
    Splits X into Positive and Negative streams to allow log-space math.
    """
    # A: (B,T,D) in [0,1] (gates)
    # X: (B,T,D) real values (can be negative)
    
    # 1. Prepare Coefficients (Log Space)
    # Clamp A for stability (prevent log(0))
    log_a = torch.log(A.clamp(min=1e-6))
    
    # 2. Split Input into Pos/Neg streams
    x_pos = X.clamp(min=0)
    x_neg = -X.clamp(max=0)
    
    # Avoid log(0) by masking
    # (We use a tiny epsilon in log, but masked values won't contribute due to exp later)
    log_x_pos = torch.log(x_pos + 1e-12)
    log_x_neg = torch.log(x_neg + 1e-12)
    
    # 3. Handle Initial State h0
    # We fold h0 into the first timestep of the scan effectively
    if h0 is not None:
        h0_pos = h0.clamp(min=0)
        h0_neg = -h0.clamp(max=0)
        # We can't easily prepend to parallel scan without re-padding.
        # Simpler strategy: Add decaying h0 term explicitly at end.
        # h_t_total = h_scan_t + (A_1...A_t)*h0
        pass # handled below
        
    # 4. Run Parallel Scans
    h_pos = robust_log_scan(log_a, log_x_pos)
    h_neg = robust_log_scan(log_a, log_x_neg)
    
    h_out = h_pos - h_neg
    
    # 5. Add Initial State Decay
    if h0 is not None:
        # Decay chain: A_cum = cumprod(A)
        # term = h0 * A_cum
        # But we have log_a, so use exp(cumsum(log_a))
        A_cum = torch.exp(torch.cumsum(log_a, dim=1))
        h_out = h_out + h0.unsqueeze(1) * A_cum
        
    return h_out
# (TTanh / ATanU already used; we also import ASigU, atan_u, asig_u)
try:
    from lamb import TTanh, ATanU, ASigU, atan_u, asig_u
except Exception:
    class TTanh(nn.Module):
        def forward(self, x): return torch.tanh(1.25 * x)
    class ATanU(nn.Module):
        def forward(self, x): return (2 / math.pi) * torch.atan(x)
    # Function fallbacks (used by ATanULSTM fallback)
    def atan_u(x): return (2 / math.pi) * torch.atan(x)
    def asig_u(x, k=2.0): return torch.sigmoid(k * x)
    class ASigU(nn.Module):
        def __init__(self, k=2.0): super().__init__(); self.k = k
        def forward(self, x): return asig_u(x, k=self.k)

# ========= Try to import custom recurrent cores from lstm.py =========
ExtIndRNN = None
ExtATanULSTM = None
try:
    from lstm import IndRNN as ExtIndRNN  # your file
    from lstm import ATanULSTM as ExtATanULSTM
except Exception:
    pass  # We'll provide robust fallbacks below so the script still runs.

# ========= Activations for MLPs =========
# ========= Activation registry (modular) =========
_ACT_REGISTRY = {}

def register_activation(name: str, factory):
    """factory: () -> nn.Module OR callable(x)->Tensor for functional acts"""
    _ACT_REGISTRY[name.lower()] = factory

def get_activation(name: str) -> nn.Module:
    n = (name or "linear").lower()
    if n not in _ACT_REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. Available: {sorted(_ACT_REGISTRY.keys())}")
    act = _ACT_REGISTRY[n]
    return act() if isinstance(act, type) or callable(getattr(act, "__call__", None)) and isinstance(act, type(nn.Module)) else act()
from lamb import *
# Built-ins + your customs
register_activation("linear", lambda: nn.Identity())
register_activation("sigmoid", lambda: nn.Sigmoid())
register_activation("tanh", lambda: nn.Tanh())
register_activation("relu",  lambda: nn.ReLU())
register_activation("lrelu", lambda: nn.LeakyReLU(0.2))
register_activation("gelu", lambda: nn.Mish())
register_activation("ttanh", lambda: TTanh())
register_activation("atanu", lambda: ATanU())
register_activation("sns", lambda: SNS())
register_activation("capsech", lambda: CapSech())
register_activation("salu", lambda: SALU())

# Menu is now dynamic:
def activation_menu_text() -> str:
    names = sorted(_ACT_REGISTRY.keys())
    lines = ["Choose activation (for MLPs/TCN/Transformer):"]
    for i, n in enumerate(names):
        lines.append(f"{i} - {n}")
    return "\n".join(lines)

def activation_names() -> list[str]:
    return sorted(_ACT_REGISTRY.keys())


# ========= MLPs =========
class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers, act_name):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        layers = []
        act = get_activation(act_name)
        for i in range(n_layers):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(act if i < n_layers-1 else nn.Identity())
        self.net = nn.Sequential(*layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    def forward(self, idx):
        x = self.embed(idx)              # (B,T,C)
        B,T,C = x.shape
        x = x.reshape(B*T, C)
        x = self.net(x)
        return self.lm_head(x).reshape(B,T,-1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linegenModel.py - Extended with gMLP and aMLP implementations

This file implements gMLP and aMLP architectures based on the paper:
"Pay Attention to MLPs" by Liu et al. (2021)

Key adaptations for autoregressive modeling:
- Causal masking in spatial projections
- Support for variable sequence lengths
- Compatible with existing training infrastructure
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========= Activation Functions =========
def get_activation(name):
    """Return activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.Mish(),
        "silu": nn.Mish(),
        "swish": nn.Mish(),  # Swish is same as SiLU
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(),
    }
    return activations.get(name.lower(), nn.ReLU())


# ========= Positional Encodings =========
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as used in the original Transformer."""
    def __init__(self, d_model: int, max_len: int = 65536):
        super().__init__()
        self.d_model = d_model
        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
    
    def forward(self, T: int, device=None):
        """Return positional encodings for sequence length T."""
        if device is None:
            return self.pe[:T].unsqueeze(0)  # (1, T, d_model)
        return self.pe[:T].to(device).unsqueeze(0)


# ========= Baseline MLP Models (for reference) =========
class ResidualBlock(nn.Module):
    """Basic residual MLP block."""
    def __init__(self, dim, act_name):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.act = get_activation(act_name)
        self.lin2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        h = self.lin1(x)
        h = self.act(h)
        h = self.lin2(h)
        return self.norm(x + h)


class ResidualMLPClassifier(nn.Module):
    """Baseline residual MLP with sinusoidal positional encoding."""
    def __init__(self, vocab_size, embed_dim, n_layers, act_name):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = SinusoidalPositionalEncoding(embed_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(embed_dim, act_name) for _ in range(n_layers)])
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, idx):
        x = self.embed(idx)  # (B, T, C)
        B, T, C = x.shape
        x = x + self.pos(T, device=idx.device)
        x = x.reshape(B * T, C)
        x = self.blocks(x)
        return self.lm_head(x).reshape(B, T, -1)


# ========= gMLP Implementation =========
class SpatialGatingUnit(nn.Module):
    """
    Spatial Gating Unit with causal masking for autoregressive modeling.
    
    Based on the paper "Pay Attention to MLPs" (Liu et al., 2021).
    The SGU performs spatial (cross-token) interactions using a learned linear projection
    combined with multiplicative gating.
    """
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        self.seq_len = seq_len
        
        # Spatial projection: projects along sequence dimension
        # For autoregressive tasks, we apply causal masking to the weight matrix
        self.spatial_proj = nn.Linear(seq_len, seq_len, bias=True)
        
        # Initialize bias to 1 as recommended in the paper
        # This ensures the block acts like a standard FFN at initialization
        nn.init.ones_(self.spatial_proj.bias)
        
        # Initialize weights to near-zero for training stability
        nn.init.normal_(self.spatial_proj.weight, mean=0.0, std=1e-6)
        
        # Register causal mask as buffer (won't be trained)
        # This masks the weight matrix to enforce causality
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, d_ffn)
        
        Returns:
            Tensor of shape (B, T, d_ffn//2)
        """
        B, T, C = x.shape
        
        # Check sequence length compatibility
        if T > self.seq_len:
            raise ValueError(
                f"Input sequence length ({T}) exceeds model's maximum sequence length ({self.seq_len}). "
                f"Please initialize the model with seq_len >= {T}."
            )
        
        # Split along channel dimension for gating
        u, v = x.chunk(2, dim=-1)  # each: (B, T, d_ffn/2)
        
        # Normalize v
        v = self.norm(v)
        
        # Spatial projection with causal masking
        # Transpose to (B, d_ffn/2, T) for projection
        v = v.transpose(1, 2)  # (B, d_ffn/2, T)
        
        # Apply causal mask to weight matrix during forward pass
        # This ensures position i can only see positions <= i
        W = self.spatial_proj.weight[:T, :T]  # Slice to current seq length
        W_masked = W * self.causal_mask[:T, :T]  # Apply causal mask
        b = self.spatial_proj.bias[:T] if T < self.seq_len else self.spatial_proj.bias
        
        # Manual linear projection with masked weights
        v = F.linear(v, W_masked, b)  # (B, d_ffn/2, T)
        
        v = v.transpose(1, 2)  # (B, T, d_ffn/2)
        
        # Multiplicative gating
        return u * v


class gMLPBlock(nn.Module):
    """
    A single gMLP block as described in 'Pay Attention to MLPs'.
    
    Structure:
    1. LayerNorm
    2. Channel expansion (linear projection)
    3. GELU activation
    4. Spatial Gating Unit (SGU)
    5. Channel projection back to original dimension
    6. Residual connection
    """
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.Mish()
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.channel_proj2 = nn.Linear(d_ffn // 2, d_model)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, d_model)
        
        Returns:
            Tensor of shape (B, T, d_model)
        """
        shortcut = x
        x = self.norm(x)
        x = self.channel_proj1(x)  # (B, T, d_ffn)
        x = self.activation(x)
        x = self.sgu(x)  # (B, T, d_ffn/2)
        x = self.channel_proj2(x)  # (B, T, d_model)
        return x + shortcut


class gMLPLanguageModel(nn.Module):
    """
    gMLP for autoregressive language modeling.
    
    Key features:
    - No explicit positional encodings (position info captured in spatial weights)
    - Causal masking for autoregressive generation
    - Multiplicative gating for spatial interactions
    
    **IMPORTANT**: The seq_len parameter MUST match the maximum sequence length 
    used during training! The spatial projection weights are fixed-size based on seq_len.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension (d_model)
        n_layers: Number of gMLP blocks
        d_ffn: Hidden dimension in feed-forward layers (typically 4 * embed_dim)
               MUST be even (will be split in half for gating)
        seq_len: Maximum sequence length - THIS MUST MATCH YOUR TRAINING SEQ_LEN!
    """
    def __init__(self, vocab_size, embed_dim, n_layers, d_ffn, seq_len):
        super().__init__()
        
        # Validate d_ffn is even
        if d_ffn % 2 != 0:
            raise ValueError(f"d_ffn must be even (got {d_ffn}). It will be split in half for gating.")
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.seq_len = seq_len
        
        # Stack of gMLP blocks
        self.blocks = nn.ModuleList([
            gMLPBlock(embed_dim, d_ffn, seq_len) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, idx):
        """
        Args:
            idx: Token indices of shape (B, T) where T <= seq_len
        
        Returns:
            Logits of shape (B, T, vocab_size)
        """
        x = self.embed(idx)  # (B, T, embed_dim)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits


# ========= aMLP Implementation (gMLP + Tiny Attention) =========
class TinyAttention(nn.Module):
    """
    Tiny single-head causal self-attention for aMLP.
    Updated with Post-SDPA Gating[cite: 858].
    """
    def __init__(self, d_model, d_attn=64):
        super().__init__()
        self.d_attn = d_attn
        self.qkv_proj = nn.Linear(d_model, 3 * d_attn)
        self.out_proj = nn.Linear(d_attn, d_model)
        self.scale = d_attn ** -0.5
        
        # === NEW: Post-SDPA Gate ===
        # Note: The gate acts on the attention dimension (d_attn) before projection,
        # or we can project the gate to d_attn. 
        # The paper applies G1 to the SDPA output (dim = heads * d_k). 
        # Here we gate the `d_attn` dimension.
        self.sdpa_gate = nn.Linear(d_model, d_attn) 
        nn.init.zeros_(self.sdpa_gate.bias)
        # ===========================

    def forward(self, x):
        # x is (B, T, d_model) - assumed to be normalized input [cite: 1086]
        B, T, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1) # (B, T, d_attn)
        
        # --- SDPA ---
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
            ).squeeze(1)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        
        # === NEW: Apply Gating ===
        # Gate depends on input x [cite: 1195]
        # Y' = Y * sigmoid(XW)
        gate = torch.sigmoid(self.sdpa_gate(x)) # (B, T, d_attn)
        out = out * gate
        # =========================

        # Project back to d_model
        out = self.out_proj(out)
        return out


class SpatialGatingUnitWithAttention(nn.Module):
    """
    SGU enhanced with tiny attention (for aMLP).
    
    This combines the spatial gating mechanism of gMLP with a tiny attention module.
    The attention is used to capture cross-sentence alignment patterns that pure
    spatial projection might miss.
    """
    def __init__(self, d_model, d_ffn, seq_len, d_attn=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        self.seq_len = seq_len
        
        # Spatial projection (same as gMLP)
        self.spatial_proj = nn.Linear(seq_len, seq_len, bias=True)
        nn.init.ones_(self.spatial_proj.bias)
        nn.init.normal_(self.spatial_proj.weight, mean=0.0, std=1e-6)
        
        # Tiny attention module
        self.tiny_attn = TinyAttention(d_model, d_attn)
        # Projection to convert attention output to SGU dimension
        self.attn_proj = nn.Linear(d_model, d_ffn // 2)
        
        # Register causal mask for spatial projection
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
    
    def forward(self, x, x_norm):
        """
        Args:
            x: Tensor of shape (B, T, d_ffn) - output from channel expansion
            x_norm: Tensor of shape (B, T, d_model) - normalized input (for attention)
        
        Returns:
            Tensor of shape (B, T, d_ffn//2)
        """
        B, T, C = x.shape
        
        # Check sequence length compatibility
        if T > self.seq_len:
            raise ValueError(
                f"Input sequence length ({T}) exceeds model's maximum sequence length ({self.seq_len}). "
                f"Please initialize the model with seq_len >= {T}."
            )
        
        # Standard SGU path
        u, v = x.chunk(2, dim=-1)  # each: (B, T, d_ffn/2)
        v = self.norm(v)
        
        # Spatial projection with causal masking
        v = v.transpose(1, 2)  # (B, d_ffn/2, T)
        
        # Apply causal mask to weight matrix
        W = self.spatial_proj.weight[:T, :T]
        W_masked = W * self.causal_mask[:T, :T]
        b = self.spatial_proj.bias[:T] if T < self.seq_len else self.spatial_proj.bias
        
        v = F.linear(v, W_masked, b)  # (B, d_ffn/2, T)
        v = v.transpose(1, 2)  # (B, T, d_ffn/2)
        
        # Add tiny attention contribution
        attn_out = self.tiny_attn(x_norm)  # (B, T, d_model)
        attn_contrib = self.attn_proj(attn_out)  # (B, T, d_ffn/2)
        v = v + attn_contrib
        
        return u * v


class aMLPBlock(nn.Module):
    """
    aMLP block: gMLP + tiny attention.
    
    This hybrid architecture combines the efficiency of spatial gating with
    the flexibility of self-attention, using only a tiny attention module.
    """
    def __init__(self, d_model, d_ffn, seq_len, d_attn=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.Mish()
        self.sgu_with_attn = SpatialGatingUnitWithAttention(d_model, d_ffn, seq_len, d_attn)
        self.channel_proj2 = nn.Linear(d_ffn // 2, d_model)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, d_model)
        
        Returns:
            Tensor of shape (B, T, d_model)
        """
        shortcut = x
        x_norm = self.norm(x)
        x = self.channel_proj1(x_norm)
        x = self.activation(x)
        x = self.sgu_with_attn(x, x_norm)
        x = self.channel_proj2(x)
        return x + shortcut


class aMLPLanguageModel(nn.Module):
    """
    aMLP for autoregressive language modeling (gMLP + tiny attention).
    
    This model enhances gMLP with small attention modules that help with
    cross-sentence alignment tasks. According to the paper, a single-head
    attention with dimension 64-128 is sufficient to close the gap with
    full Transformers on many NLP tasks.
    
    **IMPORTANT**: The seq_len parameter MUST match the maximum sequence length 
    used during training! The spatial projection weights are fixed-size based on seq_len.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension (d_model)
        n_layers: Number of aMLP blocks
        d_ffn: Hidden dimension in feed-forward layers
               MUST be even (will be split in half for gating)
        seq_len: Maximum sequence length - THIS MUST MATCH YOUR TRAINING SEQ_LEN!
        d_attn: Attention dimension (typically 64 or 128)
    """
    def __init__(self, vocab_size, embed_dim, n_layers, d_ffn, seq_len, d_attn=64):
        super().__init__()
        
        # Validate d_ffn is even
        if d_ffn % 2 != 0:
            raise ValueError(f"d_ffn must be even (got {d_ffn}). It will be split in half for gating.")
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.seq_len = seq_len
        
        # Stack of aMLP blocks
        self.blocks = nn.ModuleList([
            aMLPBlock(embed_dim, d_ffn, seq_len, d_attn) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, idx):
        """
        Args:
            idx: Token indices of shape (B, T) where T <= seq_len
        
        Returns:
            Logits of shape (B, T, vocab_size)
        """
        x = self.embed(idx)  # (B, T, embed_dim)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

class OneHotWindowMLPClassifier(nn.Module):
    """
    MLP that consumes a rolling window of one-hot tokens (no nn.Embedding).
    For a sequence length K=seq_len (a.k.a. block_size), we roll the input K times.
    Each roll inserts a special token at the start (id = vocab_size) and shifts the rest.
    We one-hot encode each rolled sequence and concatenate along features:
        input dim to first Linear = K * (vocab_size + 1)
    Output is per-position logits: (B, T, vocab_size)
    """
    def __init__(self, vocab_size: int, seq_len: int, embed_dim: int, n_layers: int, act_name: str):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len      # block_size
        self.input_dim = (vocab_size + 1) * seq_len  # (+1) for the special BOS/blank id
        layers = []

        # First layer: big one-hot window -> hidden
        layers.append(nn.Linear(self.input_dim, embed_dim))
        layers.append(get_activation(act_name))

        # Hidden layers
        for _ in range(max(0, n_layers - 1)):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(get_activation(act_name))

        # Head to vocab
        layers.append(nn.Linear(embed_dim, vocab_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, idx: torch.Tensor):
        """
        idx: (B, T) long
        Returns logits: (B, T, vocab_size)
        """
        B, T_orig = idx.shape
    
        # If the current T is shorter than model's configured seq_len, left-pad with the special token
        if T_orig < self.seq_len:
            pad = idx.new_full((B, self.seq_len - T_orig), self.vocab_size)  # special id = vocab_size
            idx_work = torch.cat([pad, idx], dim=1)  # (B, T_pad)
        else:
            idx_work = idx
    
        # Build rolling one-hot window over exactly self.seq_len rolls
        cur = idx_work
        onehots = []
        for _ in range(self.seq_len):
            # (B, T_pad, V+1)
            oh = F.one_hot(cur.clamp_max(self.vocab_size), num_classes=self.vocab_size + 1).float()
            onehots.append(oh)
            cur = torch.roll(cur, shifts=1, dims=1)
            cur[:, 0] = self.vocab_size  # insert special token at the new front
    
        # Concatenate window features → (B, T_pad, seq_len*(V+1))
        x = torch.cat(onehots, dim=-1)
    
        # MLP: preserves time dimension
        logits = self.mlp(x)  # (B, T_pad, vocab)
    
        # If we padded, trim back to the original sequence length (right-aligned)
        if logits.size(1) != T_orig:
            logits = logits[:, -T_orig:, :]
    
        return logits


# ========= Builtin RNNs =========
import math
import torch
import torch.nn as nn
import torch.nn.init as init

class OGBuiltinRNNWrapper(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, mode, tie_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.mode = mode
        self.tie_weights = tie_weights

        self.embed = nn.Embedding(vocab_size, hidden)

        if mode == 'rnn_tanh':
            self.core = nn.RNN(hidden, hidden, num_layers=n_layers,
                               nonlinearity='tanh', batch_first=True)
        elif mode == 'rnn_relu':
            self.core = nn.RNN(hidden, hidden, num_layers=n_layers,
                               nonlinearity='relu', batch_first=True)
        elif mode == 'gru':
            self.core = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True)
        elif mode == 'lstm':
            self.core = nn.LSTM(hidden, hidden, num_layers=n_layers, batch_first=True)
        else:
            raise ValueError("Unknown mode")

        self.lm_head = nn.Linear(hidden, vocab_size, bias=True)

        # init + optional tie
        self._init_parameters()
        if tie_weights:
            # requires embed_dim == hidden_dim used by lm_head
            if self.embed.weight.shape[1] != self.lm_head.in_features:
                raise ValueError("Cannot tie weights: embed dim != hidden dim")
            self.lm_head.weight = self.embed.weight  # weight tying

    def forward(self, idx, state=None):
        x = self.embed(idx)
        out, state = self.core(x, state)
        return self.lm_head(out), state

    # ---------- init helpers ----------
    @torch.no_grad()
    def _init_parameters(self):
        # Embedding: common choice is normal(0, 1/sqrt(hidden)) or uniform
        init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden))

        # Output head bias: zero
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

        # Initialize recurrent module per mode
        if isinstance(self.core, nn.RNN):
            if self.core.nonlinearity == 'tanh':
                self._init_rnn_tanh()
            else:
                self._init_rnn_relu()
        elif isinstance(self.core, nn.GRU):
            self._init_gru()
        elif isinstance(self.core, nn.LSTM):
            self._init_lstm()

    @torch.no_grad()
    def _init_rnn_tanh(self):
        gain = nn.init.calculate_gain('tanh')  # ~5/3
        for l in range(self.n_layers):
            w_ih = getattr(self.core, f'weight_ih_l{l}')
            w_hh = getattr(self.core, f'weight_hh_l{l}')
            b_ih = getattr(self.core, f'bias_ih_l{l}', None)
            b_hh = getattr(self.core, f'bias_hh_l{l}', None)

            init.xavier_uniform_(w_ih, gain=gain)
            init.orthogonal_(w_hh, gain=gain)
            if b_ih is not None: nn.init.zeros_(b_ih)
            if b_hh is not None: nn.init.zeros_(b_hh)

    @torch.no_grad()
    def _init_rnn_relu(self, rho: float = 0.97):
        """Scaled-identity recurrent init for long sequences."""
        relu_gain = nn.init.calculate_gain('relu')
    
        for l in range(self.n_layers):
            w_ih = getattr(self.core, f'weight_ih_l{l}')
            w_hh = getattr(self.core, f'weight_hh_l{l}')
            b_ih = getattr(self.core, f'bias_ih_l{l}', None)
            b_hh = getattr(self.core, f'bias_hh_l{l}', None)
    
            # Input: standard He init for ReLU
            nn.init.kaiming_uniform_(w_ih, a=0.0, nonlinearity='relu')
    
            # Recurrent: scaled identity
            hidden_size = w_hh.shape[0]
            w_hh.zero_()
            w_hh.view(hidden_size, hidden_size).copy_(torch.eye(hidden_size) * rho)
    
            # Biases: small positive to avoid dead ReLUs
            if b_ih is not None: nn.init.zeros_(b_ih)
            if b_hh is not None: b_hh.fill_(0.01)


    @torch.no_grad()
    def _init_gru(self):
        # PyTorch gate order: [reset, update, new] => chunks along dim 0
        for l in range(self.n_layers):
            w_ih = getattr(self.core, f'weight_ih_l{l}')
            w_hh = getattr(self.core, f'weight_hh_l{l}')
            b_ih = getattr(self.core, f'bias_ih_l{l}', None)
            b_hh = getattr(self.core, f'bias_hh_l{l}', None)

            # Input weights: Xavier per gate (sigmoid gates gain=1, tanh gate gain=tanh)
            r_ih, z_ih, n_ih = w_ih.chunk(3, dim=0)
            r_hh, z_hh, n_hh = w_hh.chunk(3, dim=0)

            init.xavier_uniform_(r_ih, gain=1.0)                    # reset (sigmoid)
            init.xavier_uniform_(z_ih, gain=1.0)                    # update (sigmoid)
            init.xavier_uniform_(n_ih, gain=nn.init.calculate_gain('tanh'))  # new (tanh)

            # Recurrent weights: orthogonal per gate
            init.orthogonal_(r_hh, gain=1.0)
            init.orthogonal_(z_hh, gain=1.0)
            init.orthogonal_(n_hh, gain=nn.init.calculate_gain('tanh'))

            if b_ih is not None: nn.init.zeros_(b_ih)
            if b_hh is not None: nn.init.zeros_(b_hh)

    @torch.no_grad()
    def _init_lstm(self):
        # PyTorch gate order: [ingate, forgetgate, cellgate, outgate]
        tanh_gain = nn.init.calculate_gain('tanh')
        for l in range(self.n_layers):
            w_ih = getattr(self.core, f'weight_ih_l{l}')
            w_hh = getattr(self.core, f'weight_hh_l{l}')
            b_ih = getattr(self.core, f'bias_ih_l{l}', None)
            b_hh = getattr(self.core, f'bias_hh_l{l}', None)

            i_ih, f_ih, g_ih, o_ih = w_ih.chunk(4, dim=0)
            i_hh, f_hh, g_hh, o_hh = w_hh.chunk(4, dim=0)

            # Input weights: Xavier (sigmoid gates gain=1, cell/tanh gate uses tanh gain)
            init.xavier_uniform_(i_ih, gain=1.0)
            init.xavier_uniform_(f_ih, gain=1.0)
            init.xavier_uniform_(o_ih, gain=1.0)
            init.xavier_uniform_(g_ih, gain=tanh_gain)

            # Recurrent weights: orthogonal gate-wise
            init.orthogonal_(i_hh, gain=1.0)
            init.orthogonal_(f_hh, gain=1.0)
            init.orthogonal_(o_hh, gain=1.0)
            init.orthogonal_(g_hh, gain=tanh_gain)

            # Biases: zero, then forget-gate bias trick
            if b_ih is not None:
                nn.init.zeros_(b_ih)
                # add +1.0 to forget gate (bias_ih slice)
                hidden = self.hidden
                b_ih[hidden:2*hidden].add_(1.0)
            if b_hh is not None:
                nn.init.zeros_(b_hh)


import math
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """RMSNorm over last dim: x * g / rms(x)"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).sqrt_()
        return x * (self.g / rms)

class BuiltinRNNWrapper(nn.Module):
    """
    Stack of num_layers separate 1-layer RNN/GRU/LSTM cores.
    Keeps cuDNN fast path; allows Norm/Dropout/Residuals between layers.
    """
    def __init__(self, vocab_size, hidden, num_layers, mode,
                 tie_weights=True,
                 use_norm=2,          # 0=None, 1=BatchNorm, 2=LayerNorm, 3=RMSNorm
                 res_every=2,         # 0 disables; otherwise every n layers
                 res_type=0,          # 0=add, 1=concat(+proj), 2=ReZero(scalar), 3=ReZero(elementwise)
                 dropout=0.0,         # inter-layer dropout prob
                 use_multiplier=0,    # 0=None, 1=scalar per-layer, 2=vector per-layer
                 # --- new: long-seq init knobs ---
                 tanh_spectral_radius=0.99,
                 relu_identity_scale=1.0,
                 # --- new: capture/visualizer ---
                 enable_capture=False):
        super().__init__()
        assert mode in ('rnn_tanh', 'rnn_relu', 'gru', 'lstm')
        assert use_norm in (0,1,2,3,4,5,6)
        assert res_type in (0,1,2,3)
        assert use_multiplier in (0,1,2)

        self.vocab_size = vocab_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.mode = mode
        self.tie_weights = tie_weights
        self.use_norm = int(use_norm)
        self.res_every = int(res_every)
        self.res_type = int(res_type)
        self.dropout_p = float(dropout)
        self.use_multiplier = int(use_multiplier)

        # long-seq init knobs
        self.tanh_spectral_radius = float(tanh_spectral_radius)
        self.relu_identity_scale = float(relu_identity_scale)

        # capture buffers
        self._capture_enabled = bool(enable_capture)
        self._captured = None  

        self.embed = nn.Embedding(vocab_size, hidden)

        # Build 1-layer cores (cuDNN fast path)
        cores = []
        for _ in range(num_layers):
            if mode == 'rnn_tanh':
                core = nn.RNN(hidden, hidden, num_layers=1, nonlinearity='tanh', batch_first=True, dropout=0.0)
            elif mode == 'rnn_relu':
                core = nn.RNN(hidden, hidden, num_layers=1, nonlinearity='relu', batch_first=True, dropout=0.0)
            elif mode == 'gru':
                core = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, dropout=0.0)
            else:
                core = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True, dropout=0.0)
            cores.append(core)
        self.cores = nn.ModuleList(cores)

        # Inter-layer normalization modules
        if self.use_norm == 0:
            self.norms = None
        elif self.use_norm == 1:
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers - 1)])
        elif self.use_norm == 2:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers - 1)])
        elif self.use_norm == 3:  # RMSNorm
            self.norms = nn.ModuleList([RMSNorm(hidden) for _ in range(num_layers - 1)])
        elif self.use_norm == 4:  # TTanh
            self.norms = nn.ModuleList([TTanh() for _ in range(num_layers - 1)])
        elif self.use_norm == 5:  # ETTanh
            self.norms = nn.ModuleList([ETTanh(hidden) for _ in range(num_layers - 1)])
        elif self.use_norm == 6:  # DyT
            self.norms = nn.ModuleList([DyT(hidden) for _ in range(num_layers - 1)])

        # Inter-layer dropout
        self.drops = nn.ModuleList([nn.Dropout(self.dropout_p) for _ in range(num_layers - 1)]) if self.dropout_p > 0 else None

        # 4. Residual Handlers
        # === FIX IS HERE ===
        self.do_res = (self.res_every > 0)
        
        num_res_hops = 0
        if self.do_res:  # Guard against res_every=0
            for i in range(num_layers - 1):
                if ((i + 1) % self.res_every) == 0:
                    num_res_hops += 1

        self.res_mixers = None
        self.alphas = None
        self.betas = None

        if self.do_res and num_res_hops > 0:
            if self.res_type == 0:
                mixers = []
                for i in range(num_layers - 1):
                    if ((i + 1) % self.res_every) == 0:
                        mixers.append(nn.Linear(hidden, hidden))
                    else:
                        mixers.append(nn.Identity()) 
                self.res_mixers = nn.ModuleList(mixers)

            elif self.res_type == 1:
                mixers = []
                for i in range(num_layers - 1):
                    if ((i + 1) % self.res_every) == 0:
                        mixers.append(nn.Linear(hidden * 2, hidden))
                    else:
                        mixers.append(nn.Identity())
                self.res_mixers = nn.ModuleList(mixers)

            elif self.res_type in (2, 3):
                param_shape = 1 if self.res_type == 2 else hidden
                self.alphas = nn.ParameterList()
                self.betas = nn.ParameterList() 
                for i in range(num_layers - 1):
                    if ((i + 1) % self.res_every) == 0:
                        self.alphas.append(nn.Parameter(torch.zeros(param_shape)))
                        self.betas.append(nn.Parameter(torch.ones(param_shape)))

        # 5. Multipliers
        if self.use_multiplier == 0:
            self.multipliers = None
        elif self.use_multiplier == 1:
            self.multipliers = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_layers)])
        else:
            self.multipliers = nn.ParameterList([nn.Parameter(torch.ones(hidden)) for _ in range(num_layers)])

        self.lm_head = nn.Linear(hidden, vocab_size, bias=True)

        self._init_parameters()
        if tie_weights:
            if self.embed.weight.shape[1] != self.lm_head.in_features:
                raise ValueError("Cannot tie weights: embed dim != hidden dim")
            self.lm_head.weight = self.embed.weight

    # ... (Rest of the class methods: start_capture, stop_capture, get_captured, _maybe_capture, _zero_state, _apply_norm, _apply_residual, _apply_multiplier, forward, _init_parameters, etc. remain unchanged) ...
    # Be sure to include the rest of the methods below if copy-pasting!
    
    def start_capture(self):
        self._capture_enabled = True
        self._captured = [[] for _ in range(self.num_layers)]

    def stop_capture(self):
        self._capture_enabled = False

    @torch.no_grad()
    def get_captured(self):
        if self._captured is None: return None
        out = []
        for layer_list in self._captured:
            if len(layer_list) == 0:
                out.append(None)
            else:
                xs = [x.unsqueeze(1) if x.dim() == 2 else x for x in layer_list]
                stacked = torch.cat(xs, dim=1) 
                out.append(stacked)
        return out

    def _maybe_capture(self, li, y):
        if not self._capture_enabled: return
        if self._captured is None: self._captured = [[] for _ in range(self.num_layers)]
        y_last = y[:, -1, :].detach().to('cpu')
        self._captured[li].append(y_last)

    def _zero_state(self, B, device, dtype):
        if self.mode in ('gru', 'rnn_tanh', 'rnn_relu'):
            return [torch.zeros(1, B, self.hidden, device=device, dtype=dtype) for _ in range(self.num_layers)]
        else:  
            return [(torch.zeros(1, B, self.hidden, device=device, dtype=dtype),
                     torch.zeros(1, B, self.hidden, device=device, dtype=dtype)) for _ in range(self.num_layers)]

    def _apply_norm(self, li, y):
        if self.norms is None: return y
        if self.use_norm == 1:
            B, T, H = y.shape
            y2 = y.contiguous().view(B*T, H)
            y2 = self.norms[li](y2)
            return y2.view(B, T, H)
        else:
            return self.norms[li](y)

    def _apply_residual(self, li, y_in, y_out, res_idx):
        if not self.do_res: return y_out
        if ((li + 1) % self.res_every) != 0: return y_out

        if self.res_type == 0:
            mixed = self.res_mixers[li](y_out)
            return y_in + mixed
        elif self.res_type == 1:
            cat = torch.cat([y_out, y_in], dim=-1)
            return self.res_mixers[li](cat)
        elif self.res_type in (2, 3):
            alpha = self.alphas[res_idx]
            beta = self.betas[res_idx]
            return (y_out * alpha) + (y_in * beta)
        return y_out

    def _apply_multiplier(self, li, y):
        if self.multipliers is None: return y
        m = self.multipliers[li]
        if self.use_multiplier == 1:
            return y * m
        else:
            return y * m.view(1, 1, -1)

    def forward(self, idx, state=None):
        B, T = idx.size(0), idx.size(1)
        x0 = self.embed(idx)

        if state is None:
            state = self._zero_state(B, x0.device, x0.dtype)

        new_state = []
        y = x0
        res_hop_count = 0

        for li, core in enumerate(self.cores):
            s_in = state[li]
            y = self._apply_multiplier(li, y)
            y_in = y
            
            if li > 0:
                y = self._apply_norm(li - 1, y)
            
            y, s_out = core(y, s_in) 

            if li < self.num_layers - 1:
                y = self._apply_residual(li, y_in, y, res_hop_count)
                if self.do_res and ((li + 1) % self.res_every) == 0:
                    res_hop_count += 1
                if self.drops is not None:
                    y = self.drops[li](y)

            self._maybe_capture(li, y)
            new_state.append(s_out)

        logits = self.lm_head(y)
        return logits, new_state

    # ... (Keep _init_parameters and the specific init methods as they were in your file) ...
    @torch.no_grad()
    def _init_parameters(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden))
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

        for core in self.cores:
            if isinstance(core, nn.RNN):
                if core.nonlinearity == 'tanh':
                    self._init_rnn_tanh_longseq(core, spectral_radius=self.tanh_spectral_radius)
                else:
                    self._init_rnn_relu_longseq(core, identity_scale=self.relu_identity_scale)
            elif isinstance(core, nn.GRU):
                r_ih, z_ih, n_ih = core.weight_ih_l0.chunk(3, 0)
                r_hh, z_hh, n_hh = core.weight_hh_l0.chunk(3, 0)
                nn.init.xavier_uniform_(r_ih, gain=1.0)
                nn.init.xavier_uniform_(z_ih, gain=1.0)
                nn.init.xavier_uniform_(n_ih, gain=nn.init.calculate_gain('tanh'))
                nn.init.orthogonal_(r_hh, gain=1.0)
                nn.init.orthogonal_(z_hh, gain=1.0)
                nn.init.orthogonal_(n_hh, gain=nn.init.calculate_gain('tanh'))
                if core.bias:
                    nn.init.ones_(core.bias_ih_l0); nn.init.zeros_(core.bias_hh_l0)
            else:  # LSTM
                i_ih, f_ih, g_ih, o_ih = core.weight_ih_l0.chunk(4, 0)
                i_hh, f_hh, g_hh, o_hh = core.weight_hh_l0.chunk(4, 0)
                tg = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(i_ih, gain=1.0)
                nn.init.xavier_uniform_(f_ih, gain=1.0)
                nn.init.xavier_uniform_(o_ih, gain=1.0)
                nn.init.xavier_uniform_(g_ih, gain=tg)
                nn.init.orthogonal_(i_hh, gain=1.0)
                nn.init.orthogonal_(f_hh, gain=1.0)
                nn.init.orthogonal_(o_hh, gain=1.0)
                nn.init.orthogonal_(g_hh, gain=tg)
                if core.bias:
                    nn.init.zeros_(core.bias_ih_l0); nn.init.zeros_(core.bias_hh_l0)
                    H = core.hidden_size
                    core.bias_ih_l0[H:2*H].add_(1.0)  # forget bias +1

    @torch.no_grad()
    def _init_rnn_tanh_longseq(self, core: nn.RNN, spectral_radius: float = 0.99):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(core.weight_ih_l0, gain=gain)
        nn.init.orthogonal_(core.weight_hh_l0, gain=1.0)
        with torch.no_grad():
            core.weight_hh_l0.mul_(spectral_radius)
        if core.bias:
            nn.init.zeros_(core.bias_ih_l0); nn.init.zeros_(core.bias_hh_l0)

    @torch.no_grad()
    def _init_rnn_relu_longseq(self, core: nn.RNN, identity_scale: float = 1.0):
        H = core.hidden_size
        with torch.no_grad():
            core.weight_hh_l0.zero_()
            eye = torch.eye(H, device=core.weight_hh_l0.device, dtype=core.weight_hh_l0.dtype)
            core.weight_hh_l0[:H, :H].copy_(identity_scale * eye)
        nn.init.kaiming_uniform_(core.weight_ih_l0, a=0.0, nonlinearity='relu')
        core.weight_ih_l0.mul_(1e-3)  
        if core.bias:
            nn.init.zeros_(core.bias_ih_l0); nn.init.zeros_(core.bias_hh_l0)



# ========= Custom RNN-like wrappers =========
# ==============================================================================
# COMPILE-SAFE WRAPPER & RNN CORES (Jiri's Fix)
# ==============================================================================

class CustomRNNWrapper(nn.Module):
    """
    Compiler-Safe Wrapper.
    1. Initializes Embedding explicitly (Fixes AttributeError: embed).
    2. Maps string names to classes manually.
    3. Handles the input projection flow correctly.
    """
    def __init__(self, cell_type, vocab_size, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        # 1. Define Embedding (Crucial!)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # 2. Define Head
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # 3. Map & Instantiate Core
        # Ensure we pass hidden_size as input_size because embedding dim == hidden dim
        cell_map = {
            "indrnn": IndRNN,
            "indygru": IndyGRU,
            "janet": JANET,
            "liquid": LiquidRNN,
            "atanulstm": ExtATanULSTM if 'ExtATanULSTM' in globals() else None
        }
        
        c_type = cell_type.lower()
        if c_type not in cell_map or cell_map[c_type] is None:
            raise ValueError(f"Unknown or unavailable cell type: {cell_type}")
            
        self.rnn = cell_map[c_type](
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            **kwargs
        )

    def forward(self, idx, state=None):
        # 1. Embed indices -> Vectors (Fixes 'reduction dim' error)
        x = self.embed(idx) 
        
        # 2. Run Core
        out, state = self.rnn(x, state)
        
        # 3. Project to Vocab
        return self.lm_head(out), state


class IndRNN(nn.Module):
    """
    Compiler-Safe IndRNN.
    - Uses module attributes instead of dicts (Fixes 'getitem' error).
    - Uses .size() instead of unpacking (Fixes AssertionError).
    - Pre-allocates output tensors (Fixes Graph Breaks).
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Create a simple Module to hold params (Fixes 'FloatTensor is not Module')
            layer = nn.Module()
            layer.W_ih = nn.Linear(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            
            # Direct attribute assignment (safest for Dynamo)
            layer.u_hh = nn.Parameter(torch.empty(hidden_size).uniform_(-0.5, 0.5))
            layer.b_hh = nn.Parameter(torch.zeros(hidden_size)) if bias else None
            
            self.layers.append(layer)
            
        self.act = nn.Tanh()

    def forward(self, x, state=None):
        # 1. Safe Unpacking (Fixes Dynamo AssertionError)
        B = x.size(0)
        T = x.size(1)
        
        # 2. Handle None state inside compiled graph
        if state is None:
            state = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        
        new_states = []
        layer_input = x
        
        for i, layer in enumerate(self.layers):
            h = state[i]
            # 3. Pre-allocate output (Fixes Append Graph Break)
            y = torch.empty((B, T, self.hidden_size), device=x.device, dtype=x.dtype)
            
            # 4. Attribute Access (Fixes 'getitem' error)
            # Compute input projection for whole sequence
            preact = layer.W_ih(layer_input)
            u = layer.u_hh
            b = layer.b_hh if layer.b_hh is not None else 0.0
            
            # 5. Fused Loop
            for t in range(T):
                z = preact[:, t, :] + h * u + b
                h = self.act(z)
                y[:, t, :] = h
                
            new_states.append(h)
            layer_input = y
            
        return layer_input, torch.stack(new_states)


class IndyGRU(nn.Module):
    """Compiler-Safe IndyGRU"""
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = float(dropout)
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Module()
            in_sz = input_size if i == 0 else hidden_size
            
            # Input projections
            layer.W_gate = nn.Linear(in_sz, 2 * hidden_size, bias=bias)
            layer.W_cand = nn.Linear(in_sz, hidden_size, bias=bias)
            
            # Diagonal Recurrent weights (Direct Attributes)
            layer.u_gate = nn.Parameter(torch.ones(2 * hidden_size) * 0.5)
            layer.u_cand = nn.Parameter(torch.ones(hidden_size) * 0.5)
            
            self.layers.append(layer)
            
        self._drop = nn.Dropout(self.dropout)

    def forward(self, x, state=None):
        B = x.size(0)
        T = x.size(1)
        
        if state is None:
            state = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]

        layer_input = x
        new_states = []
        
        for i, layer in enumerate(self.layers):
            h = state[i]
            y = torch.empty((B, T, self.hidden_size), device=x.device, dtype=x.dtype)
            
            # Attribute Access
            gate_in = layer.W_gate(layer_input)
            cand_in = layer.W_cand(layer_input)
            u_gate = layer.u_gate
            u_cand = layer.u_cand
            
            for t in range(T):
                # Fused Gate Logic
                gates = torch.sigmoid(gate_in[:, t] + h.repeat(1, 2) * u_gate)
                r, z = gates.chunk(2, dim=1)
                
                h_tilde = torch.tanh(cand_in[:, t] + (r * h) * u_cand)
                h = (1 - z) * h + z * h_tilde
                y[:, t] = h
                
            new_states.append(h)
            if i != self.num_layers - 1 and self.dropout > 0:
                y = self._drop(y)
            layer_input = y
            
        return layer_input, torch.stack(new_states)


class JANET(nn.Module):
    """Compiler-Safe JANET"""
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, beta=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Module()
            in_sz = input_size if i == 0 else hidden_size
            
            layer.W_f = nn.Linear(in_sz, hidden_size, bias=bias)
            layer.W_c = nn.Linear(in_sz, hidden_size, bias=bias)
            layer.U_f = nn.Linear(hidden_size, hidden_size, bias=bias)
            layer.U_c = nn.Linear(hidden_size, hidden_size, bias=bias)
            
            # Chrono Init Logic (Simplified)
            nn.init.constant_(layer.W_f.bias, 1.0)
            
            self.layers.append(layer)
        self._drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        B = x.size(0)
        T = x.size(1)
        
        if state is None:
            state = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
            
        layer_in = x
        new_states = []
        
        for i, layer in enumerate(self.layers):
            c = state[i]
            y = torch.empty((B, T, self.hidden_size), device=x.device, dtype=x.dtype)
            
            for t in range(T):
                xt = layer_in[:, t]
                # Attribute Access
                f = torch.sigmoid(layer.W_f(xt) + layer.U_f(c))
                cand = torch.tanh(layer.W_c(xt) + layer.U_c(c))
                
                # Beta shift logic
                f_beta = 1.0 - torch.sigmoid(layer.W_f(xt) + layer.U_f(c) - self.beta)
                
                c = f * c + f_beta * cand
                y[:, t] = c
                
            new_states.append(c)
            if i != self.num_layers - 1 and self.dropout > 0:
                y = self._drop(y)
            layer_in = y
            
        return layer_in, torch.stack(new_states)

# ========= Fallback implementations if lstm.py not present =========
# (We only define these if imports failed; matches your provided code.)
if ExtIndRNN is None:
    class ExtIndRNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity="tanh",
                     bias=True, batch_first=True, dropout=0.0, bidirectional=False,
                     activation_modules: Optional[List[nn.Module]] = None):
            super().__init__()
            if nonlinearity not in ("tanh","relu"): raise ValueError("nonlinearity must be 'tanh' or 'relu'")
            self.input_size = input_size; self.hidden_size = hidden_size
            self.num_layers = num_layers; self.bias = bias
            self.batch_first = batch_first; self.dropout = float(dropout)
            self.bidirectional = bidirectional; self.num_directions = 2 if bidirectional else 1
            for layer in range(num_layers):
                suffix = f"_l{layer}"
                in_features = input_size if layer==0 else hidden_size*self.num_directions
                self.register_parameter("weight_ih"+suffix, nn.Parameter(torch.empty(hidden_size, in_features)))
                self.register_parameter("weight_hh"+suffix, nn.Parameter(torch.empty(hidden_size)))
                if bias:
                    self.register_parameter("bias_ih"+suffix, nn.Parameter(torch.empty(hidden_size)))
                    self.register_parameter("bias_hh"+suffix, nn.Parameter(torch.empty(hidden_size)))
            if activation_modules is not None:
                assert len(activation_modules) == num_layers * self.num_directions
                self._activations = nn.ModuleList(activation_modules)
            else:
                acts = [CapSech() if nonlinearity=="tanh" else nn.ReLU() for _ in range(num_layers*self.num_directions)]
                self._activations = nn.ModuleList(acts)
            self.reset_parameters()
        def _p(self, name): return getattr(self, name)
        def _get_params(self, layer, direction):
            suffix = f"_l{layer}"
            W_ih = self._p("weight_ih"+suffix); u_hh = self._p("weight_hh"+suffix)
            b_ih = self._p("bias_ih"+suffix) if self.bias else None
            b_hh = self._p("bias_hh"+suffix) if self.bias else None
            act = self._activations[layer*self.num_directions+direction]
            return W_ih, u_hh, b_ih, b_hh, act
        def reset_parameters(self):
            for layer in range(self.num_layers):
                W_ih, u_hh, b_ih, b_hh, _ = self._get_params(layer, 0)
                nn.init.xavier_uniform_(W_ih)
                nn.init.uniform_(u_hh, -0.5, 0.5)
                if self.bias:
                    fan_in = W_ih.size(1); bound_b = 1/math.sqrt(fan_in) if fan_in>0 else 0
                    nn.init.uniform_(b_ih, -bound_b, bound_b); nn.init.uniform_(b_hh, -bound_b, bound_b)
        def flatten_parameters(self): return
        def forward(self, x, hx=None):
            if self.batch_first: x = x.transpose(0,1)  # (T,B,C)
            T,B,_ = x.shape
            if hx is None: hx = x.new_zeros(self.num_layers, B, self.hidden_size)
            out = x
            finals = []
            for layer in range(self.num_layers):
                W_ih, u_hh, b_ih, b_hh, act = self._get_params(layer, 0)
                pre = torch.matmul(out, W_ih.t())
                if b_ih is not None: pre = pre + b_ih
                h = hx[layer]
                ys = []
                for t in range(T):
                    z = pre[t] + h * u_hh
                    if b_hh is not None: z = z + b_hh
                    h = act(z); ys.append(h)
                y = torch.stack(ys, dim=0)
                finals.append(h.unsqueeze(0))
                if layer != self.num_layers-1 and self.training: y = F.dropout(y, p=0.0)
                out = y
            if self.batch_first: out = out.transpose(0,1)
            return out, torch.cat(finals, dim=0)

if ExtATanULSTM is None:
    class ExtATanULSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                     batch_first=True, dropout=0.0, forget_bias=1.0):
            super().__init__()
            self.input_size = input_size; self.hidden_size = hidden_size
            self.num_layers = num_layers; self.bias = bias
            self.batch_first = batch_first; self.dropout = float(dropout)
            self.forget_bias = float(forget_bias)
            self.layers = nn.ModuleList()
            in_sizes = [input_size] + [hidden_size]*(num_layers-1)
            for in_sz in in_sizes:
                mod = nn.Module()
                mod.W_ih = nn.Parameter(torch.empty(4*hidden_size, in_sz))
                mod.W_hh = nn.Parameter(torch.empty(4*hidden_size, hidden_size))
                if bias:
                    mod.b_ih = nn.Parameter(torch.zeros(4*hidden_size))
                    mod.b_hh = nn.Parameter(torch.zeros(4*hidden_size))
                else:
                    mod.register_parameter('b_ih', None); mod.register_parameter('b_hh', None)
                self.layers.append(mod)
            self._drop = nn.Dropout(self.dropout); self.reset_parameters()
        def reset_parameters(self):
            H = self.hidden_size
            for mod in self.layers:
                nn.init.xavier_uniform_(mod.W_ih); nn.init.orthogonal_(mod.W_hh)
                if self.bias:
                    nn.init.zeros_(mod.b_ih); nn.init.zeros_(mod.b_hh)
                    mod.b_ih.data[H:2*H].add_(self.forget_bias)  # forget gate bias
        def _layer_forward(self, xseq, h0, c0, mod):
            T,B,_ = xseq.shape; H = self.hidden_size
            h = h0; c = c0; outs=[]
            for t in range(T):
                gates = F.linear(xseq[t], mod.W_ih, mod.b_ih) + F.linear(h, mod.W_hh, mod.b_hh)
                i_lin, f_lin, g_lin, o_lin = gates.chunk(4, dim=1)
                i = asig_u(i_lin, k=2.0); f = asig_u(f_lin, k=2.0)
                g = atan_u(g_lin);       o = asig_u(o_lin, k=2.0)
                c = f * c + i * g
                h = o * atan_u(c)
                outs.append(h)
            return torch.stack(outs, dim=0), h, c
        def forward(self, x, hx=None):
            if self.batch_first: x = x.transpose(0,1)
            T,B,_ = x.shape
            if hx is None:
                h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
                c0 = x.new_zeros(self.num_layers, B, self.hidden_size)
            else:
                h0,c0 = hx
            layer_in = x; hn=[]; cn=[]
            for li,mod in enumerate(self.layers):
                y,hT,cT = self._layer_forward(layer_in, h0[li], c0[li], mod)
                if li != self.num_layers-1 and self.dropout>0 and self.training:
                    y = self._drop(y)
                layer_in = y; hn.append(hT); cn.append(cT)
            y = layer_in
            hn = torch.stack(hn, dim=0); cn = torch.stack(cn, dim=0)
            if self.batch_first: y = y.transpose(0,1)
            return y,(hn,cn)

# ========= TCN =========
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.pad = pad

    def forward(self, x):
        y = self.conv(x)
        return y[:, :, :-self.pad] if self.pad > 0 else y


class TCNBlock(nn.Module):
    def __init__(self, channels, act_name="relu", k=3, dilation=1):
        super().__init__()
        self.c1 = CausalConv1d(channels, channels, k, dilation)
        self.c2 = CausalConv1d(channels, channels, k, dilation)
        self.n1 = nn.LayerNorm(channels)
        self.n2 = nn.LayerNorm(channels)
        self.act = get_activation(act_name)

    def forward(self, x):
        h = self.c1(x)
        # Conv1d outputs [B, C, L], but LayerNorm expects [B, L, C]
        h = h.transpose(1, 2)
        h = self.n1(self.act(h))
        h = h.transpose(1, 2)

        h = self.c2(h)
        h = h.transpose(1, 2)
        h = self.n2(h)
        h = h.transpose(1, 2)

        return self.act(x + h)


class TemporalConvNet(nn.Module):
    def __init__(self, vocab_size, channels, n_layers, act_name="relu", k=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, channels)
        self.blocks = nn.Sequential(
            *[TCNBlock(channels, act_name=act_name, k=k, dilation=2**i) for i in range(n_layers)]
        )
        self.head = nn.Linear(channels, vocab_size)

    def forward(self, idx):
        x = self.embed(idx).transpose(1, 2)  # [B, C, L]
        x = self.blocks(x)
        x = x.transpose(1, 2)  # back to [B, L, C]
        return self.head(x)



# ========= GPT (simple) =========

# ================= GPT-2 (drop-in) =================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPT2Config:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_seq_len: int
    ff_mult: int = 4
    dropout: float = 0.0
    attn_dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True
    use_flash: bool = False   # uses F.scaled_dot_product_attention if available

def _resolve_act(act_name: str):
    # Use your project's get_activation if present; else GELU
    ga = globals().get("get_activation", None)
    if ga is not None:
        try:
            act = ga(act_name)
            if isinstance(act, nn.Module):
                return act
            # if it returned a function, wrap it
            class _Fn(nn.Module):
                def forward(self, x): return act(x)
            return _Fn()
        except Exception:
            pass
    class _GELU(nn.Module):
        def forward(self, x): return F.mish(x)
    return _GELU()

class MultiHeadCausalAttn(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, resid_dropout=0.0, bias=True, use_flash=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")
        self.qkv = nn.Linear(d_model, 3*d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)
        
        # === NEW: Post-SDPA Gate [cite: 858] ===
        self.gate = PostSDPAGate(d_model)
        # =======================================

    def forward(self, x, past_kv=None):
        # x is the normalized input (from Block: self.attn(self.ln1(x)))
        B,T,C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.hd).transpose(1,2)
        q, k, v = qkv[:,0].transpose(1,2), qkv[:,1].transpose(1,2), qkv[:,2].transpose(1,2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v)

        # SDPA Calculation
        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2,-1)) / math.sqrt(self.hd)
            Tq, Tk = att.size(-2), att.size(-1)
            causal = torch.triu(torch.ones(Tq, Tk, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(causal, float("-inf"))
            w = F.softmax(att, dim=-1)
            w = self.attn_drop(w)
            y = w @ v

        # Reshape SDPA output
        y = y.transpose(1,2).contiguous().view(B, T, C)
        
        # === NEW: Apply Gating ===
        # Applied after SDPA, before output projection (Wo) [cite: 1068, 1079]
        # x is used as the gating input dependency
        y = self.gate(x, y) 
        # =========================

        y = self.resid_drop(self.proj(y))
        return y, present

class MLP(nn.Module):
    def __init__(self, d_model, ff_mult=4, dropout=0.0, bias=True, act: nn.Module = None):
        super().__init__()
        self.fc = nn.Linear(d_model, ff_mult*d_model, bias=bias)
        self.act = act if act is not None else _resolve_act("gelu")
        self.proj = nn.Linear(ff_mult*d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.proj(x)
        return self.drop(x)

class Block(nn.Module):
    def __init__(self, cfg: GPT2Config, act: nn.Module):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadCausalAttn(cfg.d_model, cfg.n_heads, cfg.attn_dropout, cfg.dropout, cfg.bias, cfg.use_flash)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.ff_mult, cfg.dropout, cfg.bias, act)

    def forward(self, x, past_kv=None):
        a, present = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present

class GPT2Core(nn.Module):
    """Core GPT-2; returns (logits, presents)."""
    def __init__(self, cfg: GPT2Config, act_name: str = "gelu"):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        act = _resolve_act(act_name)
        self.blocks = nn.ModuleList([Block(cfg, act) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.head.weight = self.tok.weight

        self.apply(self._init_weights)
        # GPT-2 residual proj scaling
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*cfg.n_layers))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, idx, past_kv=None):
        B,T = idx.shape
        if T > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len:]; T = idx.size(1)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        x = self.drop(x)

        presents = []
        for i, block in enumerate(self.blocks):
            pkv = None if past_kv is None else past_kv[i]
            x, present = block(x, past_kv=pkv)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)
        return logits, presents

class GPT2ForLM(nn.Module):
    """Thin wrapper: return logits only to match your training code."""
    def __init__(self, cfg: GPT2Config, act_name="gelu"):
        super().__init__()
        self.core = GPT2Core(cfg, act_name)

    def forward(self, idx):
        logits, _ = self.core(idx, past_kv=None)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, repetition_penalty=1.0):
        self.eval()
        past = [None]*len(self.core.blocks)
        
        # Keep track of the full sequence for output, but feed core only the new token
        full_idx = idx
        
        # Initial step: feed full context to prime 'past'
        logits, past = self.core(idx, past_kv=past)
        next_tok = self._sample_token(logits, temperature, top_k, repetition_penalty, full_idx)
        full_idx = torch.cat([full_idx, next_tok], dim=1)
        
        # Generation loop
        for _ in range(max_new_tokens - 1):
            # FIX: Feed ONLY the last token (next_tok) when we have 'past'
            logits, past = self.core(next_tok, past_kv=past)
            
            next_tok = self._sample_token(logits, temperature, top_k, repetition_penalty, full_idx)
            full_idx = torch.cat([full_idx, next_tok], dim=1)
            
        return full_idx

    def _sample_token(self, logits, temperature, top_k, repetition_penalty, full_idx):
        logits = logits[:, -1, :]
        if repetition_penalty != 1.0:
            for b in range(logits.size(0)):
                logits[b, full_idx[b].unique()] /= repetition_penalty
        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None and top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            thresh = v[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
class ScanBlock_GateLoop(nn.Module):
    def __init__(self, dim: int, mag_act: str = "sigmoid",
                 clamp_mag: float = 0.995, floor_mag: float = 1e-3, ln_eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim, eps=ln_eps)

        self.Wq = nn.Linear(dim, dim, bias=True)
        self.Wk = nn.Linear(dim, dim, bias=True)
        self.Wv = nn.Linear(dim, dim, bias=True)
        self.Wmag = nn.Linear(dim, dim, bias=True)
        self.Wtheta = nn.Linear(dim, dim, bias=True)

        self.post = GatedMLP(dim, mult=4, act_name="gelu")

        self._mag_fn = torch.sigmoid if mag_act == "sigmoid" else torch.sigmoid
        self._clamp_mag_hi = float(clamp_mag)
        self._clamp_mag_lo = float(floor_mag)

        # Gentle init
        for W in (self.Wq, self.Wk, self.Wv, self.Wmag, self.Wtheta):
            nn.init.xavier_uniform_(W.weight); nn.init.zeros_(W.bias)

    @torch.cuda.amp.autocast(enabled=False)  # complex math in full precision
    def _complex_a(self, xhat: torch.Tensor) -> torch.Tensor:
        # xhat is real; compute in float32 → complex64
        xr = xhat.float()
        mag = self._mag_fn(self.Wmag(xr)).clamp(self._clamp_mag_lo, self._clamp_mag_hi)  # [lo, hi]
        theta = self.Wtheta(xr)
        # torch.polar expects (real, real) => complex
        a = torch.polar(mag, theta).to(torch.cfloat)  # complex64
        return a

    def _gate_triplets(self, xhat: torch.Tensor):
        xr = xhat.float()
        q = torch.sigmoid(self.Wq(xr))      # [0,1]
        k = self.Wk(xr)
        v = self.Wv(xr)
        return q, k, v

    @torch.cuda.amp.autocast(enabled=False)
    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        # 1. Compute Parameters
        # Magnitude and Phase for A
        mag = torch.sigmoid(self.Wmag(x_norm))
        theta = self.Wtheta(x_norm)
        
        # Create Complex A
        # (B,T,D) complex64
        a_complex = torch.polar(mag, theta)
        
        # Q, K, V
        q = torch.sigmoid(self.Wq(x_norm))
        k = self.Wk(x_norm)
        v = self.Wv(x_norm)
        
        # Input to scan: K * V (Complex)
        # Note: In standard GateLoop, input is just (K*V) complexified? 
        # Actually usually it's Real K, Real V -> Complex KV via some transform, 
        # or just treated as real input to complex state. 
        # Let's treat K*V as the complex input b_t (pure real, imag=0)
        kv_complex = (k * v).to(a_complex.dtype)
        
        # 2. Parallel Scan (Linear, Complex)
        # h_t = a_t * h_{t-1} + kv_t
        s0 = h0.to(a_complex.dtype) if h0 is not None else None
        h_complex = parallel_scan_linear(a_complex, kv_complex, s0)
        
        # 3. Output
        # y = Re(q * h)
        y = (q * h_complex.real)
        
        out = x + self.post(y)
        return out, h_complex[:, -1, :]

    @torch.cuda.amp.autocast(enabled=False)
    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_tn = self.ln(x_t)
        
        mag = torch.sigmoid(self.Wmag(x_tn))
        theta = self.Wtheta(x_tn)
        a_t = torch.polar(mag, theta)
        
        q = torch.sigmoid(self.Wq(x_tn))
        k = self.Wk(x_tn)
        v = self.Wv(x_tn)
        kv_t = (k * v).to(a_t.dtype)
        
        h_prev_c = h_prev.to(a_t.dtype) if h_prev is not None else torch.zeros_like(a_t)
        
        h_t = a_t * h_prev_c + kv_t
        y = q * h_t.real
        
        out = x_t + self.post(y)
        return out, h_t


class BlockDiagLinear(nn.Module):
    """
    Block-diagonal linear: split last dim into Nh heads of size d_h, apply per-head Linear(d_h->d_h).
    Used for recurrent ('R*') matrices to realize head-wise memory mixing (no cross-head mixing).
    """
    def __init__(self, dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.dh = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, self.dh, self.dh))
        self.bias = nn.Parameter(torch.empty(num_heads, self.dh)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.num_heads):
            nn.init.xavier_uniform_(self.weight[h])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) or (B, D)
        is_seq = (x.dim() == 3)
        if not is_seq: x = x.unsqueeze(1)
        B,T,D = x.shape
        Nh, dh = self.num_heads, self.dh
        xh = x.view(B, T, Nh, dh)
        # y[b,t,h,:] = xh @ W[h].T + b[h]
        y = torch.einsum('bt hd, hkd->bt hk', xh, self.weight.transpose(1,2))
        if self.bias is not None:
            y = y + self.bias.view(1,1,Nh,dh)
        y = y.reshape(B, T, D)
        if not is_seq: y = y.squeeze(1)
        return y


class sLSTMCore(nn.Module):
    """
    sLSTM with exponential input/forget, normalizer n, stabilizer m, and head-wise memory mixing.
    Equations (8)-(17). Heads implemented via block-diagonal recurrent matrices. 
    """
    def __init__(self, dim: int, num_heads: int = 1, phi: str = "tanh", forget_activation: str = "exp"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.phi = getattr(torch, "tanh") if phi == "tanh" else torch.nn.functional.silu
        assert forget_activation in ("exp", "sigmoid")
        self.forget_activation = forget_activation

        # Input projections (W*)
        self.Wz = nn.Linear(dim, dim, bias=True)
        self.Wi = nn.Linear(dim, dim, bias=True)
        self.Wf = nn.Linear(dim, dim, bias=True)
        self.Wo = nn.Linear(dim, dim, bias=True)

        # Recurrent (R*): block-diagonal mixing within heads only
        self.Rz = BlockDiagLinear(dim, num_heads, bias=True)
        self.Ri = BlockDiagLinear(dim, num_heads, bias=True)
        self.Rf = BlockDiagLinear(dim, num_heads, bias=True)
        self.Ro = BlockDiagLinear(dim, num_heads, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.Wz, self.Wi, self.Wf, self.Wo]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _gate_forget(self, x):
        return torch.exp(x) if self.forget_activation == "exp" else torch.sigmoid(x)

    def forward(self, x: torch.Tensor, state=None):
        """
        x: (B,T,D); state is a dict with keys ['h','c','n','m'] each (B,D)
        Returns y:(B,T,D), new_state
        """
        B,T,D = x.shape
        if state is None:
            h = x.new_zeros(B, D); c = x.new_zeros(B, D); n = x.new_zeros(B, D)
            m = x.new_zeros(B, D)  # stabilizer state (log-domain max tracker)
        else:
            h = state["h"]; c = state["c"]; n = state["n"]; m = state["m"]

        ys = []
        for t in range(T):
            xt = x[:, t, :]

            z_tilde = self.Wz(xt) + self.Rz(h)
            i_tilde = self.Wi(xt) + self.Ri(h)
            f_tilde = self.Wf(xt) + self.Rf(h)
            o_tilde = self.Wo(xt) + self.Ro(h)

            z = self.phi(z_tilde)
            i = torch.exp(i_tilde)                    # (12)
            f = self._gate_forget(f_tilde)            # (13)
            o = torch.sigmoid(o_tilde)                # (14)

            # Stabilizer m_t = max( log f + m_{t-1}, log i )  (15)
            logf = torch.log(torch.clamp(f, min=1e-20))
            logi = i_tilde                             # log(exp(i_tilde))
            m_new = torch.maximum(logf + m, logi)

            # Stabilized gates i', f'   (16,17)
            i_hat = torch.exp(i_tilde - m_new)
            f_hat = torch.exp(logf + m - m_new)

            # State updates (8,9)
            c = f_hat * c + i_hat * z
            n = f_hat * n + i_hat

            # Hidden (10)
            h_tilde = c / torch.clamp(n, min=1e-12)
            h = o * h_tilde

            m = m_new
            ys.append(h.unsqueeze(1))

        y = torch.cat(ys, dim=1)
        new_state = {"h": h, "c": c, "n": n, "m": m}
        return y, new_state


class mLSTMCore(nn.Module):
    """
    mLSTM with matrix memory C (per head), covariance update, normalizer n, stabilized gates.
    Equations (19)-(27). Multiple heads => separate (C, n, m) per head with head dimension d_h.
    """
    def __init__(self, dim: int, num_heads: int = 1, forget_activation: str = "exp"):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.dh = dim // num_heads
        assert forget_activation in ("exp", "sigmoid")
        self.forget_activation = forget_activation

        # Projections (head-shared for simplicity; split per head by reshape)
        self.Wq = nn.Linear(dim, dim, bias=True)
        self.Wk = nn.Linear(dim, dim, bias=True)
        self.Wv = nn.Linear(dim, dim, bias=True)
        self.Wo = nn.Linear(dim, dim, bias=True)

        # Gates i, f are input-dependent (best per ablations)
        self.Wi = nn.Linear(dim, dim, bias=True)
        self.Wf = nn.Linear(dim, dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.Wq, self.Wk, self.Wv, self.Wo, self.Wi, self.Wf]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _gate_forget(self, x):
        return torch.exp(x) if self.forget_activation == "exp" else torch.sigmoid(x)

    def forward(self, x: torch.Tensor, state=None):
        """
        x: (B,T,D)
        state: dict with 'C' (B,Nh,dh,dh), 'n' (B,Nh,dh), 'm' (B,Nh,dh)
        Returns y:(B,T,D), new_state
        """
        B,T,D = x.shape
        Nh, dh = self.num_heads, self.dh
        if state is None:
            C = x.new_zeros(B, Nh, dh, dh)
            n = x.new_zeros(B, Nh, dh)
            m = x.new_zeros(B, Nh, dh)
        else:
            C = state["C"]; n = state["n"]; m = state["m"]

        ys = []
        for t in range(T):
            xt = x[:, t, :]

            q = self.Wq(xt).view(B, Nh, dh)             # (22)
            k = self.Wk(xt).view(B, Nh, dh) / math.sqrt(dh)  # (23)
            v = self.Wv(xt).view(B, Nh, dh)             # (24)
            o = torch.sigmoid(self.Wo(xt)).view(B, Nh, dh)   # (27)

            i_tilde = self.Wi(xt).view(B, Nh, dh)       # (25)
            f_tilde = self.Wf(xt).view(B, Nh, dh)       # (26)
            i = torch.exp(i_tilde)
            f = self._gate_forget(f_tilde)

            # Stabilizer per head/channel (like sLSTM): (15)-(17)
            logf = torch.log(torch.clamp(f, min=1e-20))
            m_new = torch.maximum(logf + m, i_tilde)
            i_hat = torch.exp(i_tilde - m_new)
            f_hat = torch.exp(logf + m - m_new)

            # Updates (19), (20): matrix memory and normalizer (per head)
            # C <- f_hat*C + i_hat * v ⊗ k
            C = f_hat.unsqueeze(-1) * C + torch.einsum('bhd,bhe->bhde', i_hat * v, k)
            n = f_hat * n + i_hat * k

            # Retrieval (21): h̃ = (C @ q) / max(|n^T q|, 1)
            num = torch.einsum('bhde,bhe->bhd', C, q)
            denom = torch.clamp(torch.abs(torch.einsum('bhd,bhd->bh', n, q)) , min=1.0).unsqueeze(-1)
            h_tilde = num / denom
            h = o * h_tilde

            ys.append(h.reshape(B, D).unsqueeze(1))
            m = m_new

        y = torch.cat(ys, dim=1)  # (B,T,D)
        new_state = {"C": C, "n": n, "m": m}
        return y, new_state


# ======= GatedMLP (scaled residual) =======
class GatedMLP(nn.Module):
    """Simple gated MLP with a learnable residual scale (safer in deep stacks)."""
    def __init__(self, dim: int, mult: float = 4/3, act_name: str = "gelu"):
        super().__init__()
        hid = int(dim * mult)
        self.fc1  = nn.Linear(dim, hid)
        self.fc2  = nn.Linear(hid, dim)
        self.gate = nn.Linear(dim, dim)
        self.act  = get_activation(act_name)
        # Residual scale starts modest to prevent early amplification
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))          # gate in [0,1]
        y = self.fc2(self.act(self.fc1(x)))      # payload
        return self.res_scale * (y * g)          # scaled residual payload


class XBlock_sLSTM(nn.Module):
    """Pre-LN residual block with sLSTM core and post up-projection (gated MLP)."""
    def __init__(self, dim: int, num_heads: int, act_name: str):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.core = sLSTMCore(dim, num_heads=num_heads, phi="tanh", forget_activation="exp")
        self.post = GatedMLP(dim, mult=4, act_name=act_name)

    def forward(self, x, state):
        x_norm = self.ln(x)
        y, new_state = self.core(x_norm, state)
        y = self.post(y)
        return x + y, new_state


class XBlock_mLSTM(nn.Module):
    """Pre-LN residual block with pre up-projection, mLSTM in high-dim, and down-projection."""
    def __init__(self, dim: int, num_heads: int, up_mult: float, act_name: str):
        super().__init__()
        up = int(dim * up_mult)
        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, up)
        self.core = mLSTMCore(up, num_heads=num_heads, forget_activation="exp")
        self.down = nn.Linear(up, dim)
        self.out_gate = nn.Linear(dim, dim)   # externalized component-wise output gate
        self.skip = nn.Parameter(torch.tensor(1.0))  # learnable skip (Fig. 11)
        self.act = get_activation(act_name)

    def forward(self, x, state):
        x_norm = self.ln(x)
        u = self.up(x_norm)
        u = self.act(u)
        y, new_state = self.core(u, state)
        y = self.down(y)
        # externalized output gate like Fig. 11
        y = torch.sigmoid(self.out_gate(x_norm)) * y
        return x * self.skip + y, new_state


class XlstmLM(nn.Module):
    """
    Full xLSTM language model: embedding -> stacked blocks -> head.
    Exposes 3 configs via build_model:
      - xLSTM_s: only sLSTM blocks
      - xLSTM_m: only mLSTM blocks
      - xLSTM_mix: mixed with ratio cfg['xlstm_m_to_s'] (e.g., 7:1)
    """
    def __init__(self, vocab_size: int, dim: int, n_blocks: int, num_heads: int,
                 act_name: str, kind: str = "mix", m_to_s: Tuple[int,int] = (7,1), up_mult_m: float = 2.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.kind = kind
        self.blocks = nn.ModuleList()
        if kind == "s":
            for _ in range(n_blocks):
                self.blocks.append(XBlock_sLSTM(dim, num_heads, act_name))
        elif kind == "m":
            for _ in range(n_blocks):
                self.blocks.append(XBlock_mLSTM(dim, num_heads, up_mult_m, act_name))
        else:
            a,b = m_to_s
            pattern = ["m"]*a + ["s"]*b
            for i in range(n_blocks):
                t = pattern[i % len(pattern)]
                if t == "m":
                    self.blocks.append(XBlock_mLSTM(dim, num_heads, up_mult_m, act_name))
                else:
                    self.blocks.append(XBlock_sLSTM(dim, num_heads, act_name))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, idx: torch.Tensor, state=None):
        """
        idx: (B,T) -> logits: (B,T,V), state is list of per-block states
        """
        B,T = idx.shape
        x = self.embed(idx)
        if state is None:
            state = [None]*len(self.blocks)
        new_states = []
        for blk, st in zip(self.blocks, state):
            x, st_new = blk(x, st)
            new_states.append(st_new)
        x = self.ln(x)
        logits = self.head(x)
        return logits, new_states
# ========= Parallel-scan utilities (linear recurrences) =========
# (parallel_scan_linear defined below, after log-space scan)
# ========= Log-space parallel scan (stable) =========
def parallel_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor):
    """
    Compute all h_t for recurrence h_t = a_t ⊙ h_{t-1} + b_t using logs.
      log_coeffs: (B,T,D)  = log(a_1..T)
      log_values: (B,T+1,D)= [log(h0), log(b_1..T)]
    Returns: h (B,T,D)
    """
    # prefix log-products of a_t over time
    logA = torch.cumsum(log_coeffs, dim=1)          # (B,T,D)
    logA_pad = F.pad(logA, (0,0,1,0))               # (B,T+1,D) with logA_0 = 0

    # h_t = exp( logA_t + logsumexp_{k=0..t}( log_values[k] - logA_k ) )
    acc = torch.logcumsumexp(log_values - logA_pad, dim=1)  # (B,T+1,D)
    log_h = logA_pad[:, 1:, :] + acc[:, 1:, :]
    
    # === SAFETY FIX ===
    # Clamp to prevent float32 overflow (e^88 is approx max float32)
    # We clamp to 50.0 to be safe (e^50 is ~5e21, plenty for a hidden state)
    log_h = torch.clamp(log_h, max=50.0)
    
    return torch.exp(log_h)

# ========= Paper's positive surrogate g and its log =========
def pos_surrogate_g(x: torch.Tensor):
    # g(x) = { x+0.5 if x>=0;  sigmoid(x) otherwise }
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

def log_g(x: torch.Tensor):
    # log g(x) = { log(x+0.5) if x>=0;  -softplus(-x) otherwise }
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
class ScanBlock_minGRU(nn.Module):
    def __init__(self, dim: int, log_space: bool = True):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.Wz = nn.Linear(dim, dim) # Gate
        self.Wh = nn.Linear(dim, dim) # Candidate
        self.post = GatedMLP(dim, mult=4, act_name="gelu")
        
        # Init
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Wh.weight)
        nn.init.constant_(self.Wz.bias, -4.0)
        nn.init.zeros_(self.Wh.bias)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        # x: (B, T, D)
        x_norm = self.ln(x)
        
        # 1. Projections
        z_raw = self.Wz(x_norm)
        h_tilde_raw = self.Wh(x_norm)
        
        # 2. Calculate Log-Coeffs and Log-Values for Heinsen Scan
        # We need (1 - z) in log space.
        # log(1 - sigmoid(z)) = log(sigmoid(-z)) = -softplus(z)
        log_coeffs = -F.softplus(z_raw) 
        
        # We need (z * h_tilde) in log space.
        # log(sigmoid(z)) = -softplus(-z_raw)
        # h_tilde must be positive for log-scan -> use g(h)
        log_z = -F.softplus(-z_raw)
        log_h_tilde = log_g_act(h_tilde_raw)
        log_values = log_z + log_h_tilde
        
        # 3. Parallel Scan
        h_seq = heinsen_associative_scan_log(log_coeffs, log_values, h0)
        
        # 4. Output
        out = x + self.post(h_seq)
        return out, h_seq[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_tn = self.ln(x_t)
        
        # 1. Projections
        z_raw = self.Wz(x_tn)
        h_tilde_raw = self.Wh(x_tn)
        
        # 2. Gate and Candidate
        z = torch.sigmoid(z_raw)
        h_tilde = g_act(h_tilde_raw) # Crucial: Match g_act from training
        
        # 3. GRU Update
        # h_t = (1-z) * h_{t-1} + z * h_tilde
        h = (1.0 - z) * h_prev + z * h_tilde
        
        out = x_t + self.post(h)
        return out, h


class ScanBlock_minLSTM(nn.Module):
    def __init__(self, dim: int, log_space: bool = True):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.Wf = nn.Linear(dim, dim)
        self.Wi = nn.Linear(dim, dim)
        self.Wh = nn.Linear(dim, dim)
        self.post = GatedMLP(dim, mult=4, act_name="gelu")
        
        # Init
        nn.init.xavier_uniform_(self.Wf.weight); nn.init.zeros_(self.Wf.bias)
        nn.init.xavier_uniform_(self.Wi.weight); nn.init.zeros_(self.Wi.bias)
        nn.init.xavier_uniform_(self.Wh.weight); nn.init.zeros_(self.Wh.bias)
        # Bias f to be open initially
        with torch.no_grad():
            self.Wf.bias.fill_(4.0)
            self.Wi.bias.fill_( -4.0)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        f_raw = self.Wf(x_norm)
        i_raw = self.Wi(x_norm)
        h_tilde_raw = self.Wh(x_norm)

        # Log-space Gate Normalization
        # log(f) = -softplus(-f_raw)
        # log(i) = -softplus(-i_raw)
        log_f = -F.softplus(-f_raw)
        log_i = -F.softplus(-i_raw)
        
        # Normalization denominator: log(exp(log_f) + exp(log_i))
        log_denom = torch.logaddexp(log_f, log_i)
        
        # Normalized logs: f' = f / (f+i) -> log(f') = log(f) - log_denom
        log_f_prime = log_f - log_denom
        log_i_prime = log_i - log_denom
        
        # Values: i' * h_tilde
        log_values = log_i_prime + log_g_act(h_tilde_raw)
        
        # Scan
        h_seq = heinsen_associative_scan_log(log_f_prime, log_values, h0)
        
        out = x + self.post(h_seq)
        return out, h_seq[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_tn = self.ln(x_t)
        
        f = torch.sigmoid(self.Wf(x_tn))
        i = torch.sigmoid(self.Wi(x_tn))
        h_tilde = g_act(self.Wh(x_tn))
        
        # Normalize
        denom = f + i + 1e-8
        f_prime = f / denom
        i_prime = i / denom
        
        h = f_prime * h_prev + i_prime * h_tilde
        out = x_t + self.post(h)
        return out, h

# Add this helper
def parallel_scan_linear(a: torch.Tensor, b: torch.Tensor, h0: Optional[torch.Tensor] = None):
    """
    Vectorized linear scan: h_t = a_t * h_{t-1} + b_t
    Uses cumprod for speed. Falls back to sequential loop for long sequences
    where cumprod may overflow/underflow.
    
    Args:
        a: (B, T, D) decay coefficients (should be in [0, 1] for stability)
        b: (B, T, D) input signals
        h0: (B, D) optional initial state
    Returns:
        h: (B, T, D) all hidden states
    """
    B, T, D = a.shape
    
    # For very long sequences, cumprod can overflow/underflow.
    # Fall back to a chunked approach.
    if T > 512:
        CHUNK = 256
        h_chunks = []
        h_prev = h0
        for t0 in range(0, T, CHUNK):
            t1 = min(t0 + CHUNK, T)
            a_c = a[:, t0:t1, :]
            b_c = b[:, t0:t1, :]
            A_cum = torch.cumprod(a_c, dim=1)
            A_cum_safe = A_cum.clamp(min=1e-12)
            A_cum_inv = 1.0 / A_cum_safe
            S = torch.cumsum(b_c * A_cum_inv, dim=1)
            h_c = A_cum * S
            if h_prev is not None:
                h_c = h_c + A_cum * h_prev.unsqueeze(1)
            h_chunks.append(h_c)
            h_prev = h_c[:, -1, :]
        return torch.cat(h_chunks, dim=1)
    
    # Fast path: single cumprod for short sequences
    A_cum = torch.cumprod(a, dim=1)
    A_cum_inv = 1.0 / (A_cum + 1e-12)
    S = torch.cumsum(b * A_cum_inv, dim=1)
    h = A_cum * S
    if h0 is not None:
        h = h + A_cum * h0.unsqueeze(1)
    return h
# ======= ScanBlock_Mamba (stabilized) =======
class ScanBlock_Mamba(nn.Module):
    """
    Corrected Mamba-style selective scan block.
    Fixes:
      - x_proj now projects to d_inner (was 2*d_inner with half wasted)
      - Causal conv uses explicit left-padding via F.pad for clarity
      - Conv buffer saves raw pre-conv x_ssm inputs for correct TBPTT/step continuation
      - forward_seq and step use identical SSM logic
    """
    def __init__(self, dim: int, kernel_size: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_inner = dim * expand
        self.k = int(kernel_size)

        # 1. Input Projection (split into SSM path + gate)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        # 2. Depthwise Convolution (no built-in padding; we handle causal padding explicitly)
        self.dw = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=self.k,
                            groups=self.d_inner, padding=0, bias=True)
        self.act = nn.Mish()

        # 3. Inner Normalization (crucial for stability)
        self.inner_norm = nn.LayerNorm(self.d_inner)

        # 4. SSM Projections
        #    B_signal projection: d_inner -> d_inner (was 2*d_inner with half unused)
        self.x_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # 5. Decay (Log-A)
        A = torch.arange(1, self.d_inner + 1, dtype=torch.float32)
        self.log_A = nn.Parameter(torch.log(A))

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Init weights
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.dw.bias)
        dt_init = math.log(math.exp(0.001) - 1)
        nn.init.constant_(self.dt_proj.bias, dt_init)
        nn.init.xavier_uniform_(self.dt_proj.weight, gain=0.1)

    def _safe_softplus(self, x):
        return torch.where(x > 20, x, torch.log1p(torch.exp(x)))

    def _ssm_step(self, x_conv, log_A_clamped):
        """Shared SSM parameter computation for both forward_seq and step."""
        dt = self._safe_softplus(self.dt_proj(x_conv)).clamp(max=4.0)
        A_bar = torch.exp(-torch.exp(log_A_clamped) * dt)
        B_signal = self.x_proj(x_conv)
        X_bar = B_signal * dt
        return A_bar, X_bar

    def forward_seq(self, x: torch.Tensor, state: Optional[dict] = None):
        # x: (B, T, D)
        B, T, _ = x.shape

        # 1. Expand into SSM path and gate
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # 2. Causal convolution with explicit left-padding
        prev_buf = state["buf"] if (state and "buf" in state) else None
        if prev_buf is not None and prev_buf.shape[1] > 0:
            # Prepend buffer from previous TBPTT window
            x_conv_in = torch.cat([prev_buf, x_ssm], dim=1)  # (B, k-1+T, d_inner)
        else:
            # No previous state: zero-pad on the left for causality
            x_conv_in = F.pad(x_ssm.transpose(1, 2), (self.k - 1, 0)).transpose(1, 2)

        x_conv = self.dw(x_conv_in.transpose(1, 2)).transpose(1, 2)  # (B, T, d_inner)
        x_conv = self.act(x_conv)
        x_conv = self.inner_norm(x_conv)

        # 3. SSM Parameters
        log_A_clamped = self.log_A.clamp(min=-20, max=20)
        A_bar, X_bar = self._ssm_step(x_conv, log_A_clamped)

        # 4. Parallel scan
        s0 = state["s"] if (state and "s" in state) else None
        s = parallel_scan_linear(A_bar, X_bar, s0)

        # 5. Output gating
        y = s * self.act(z)
        out = self.out_proj(y)

        # 6. Save state: SSM hidden + conv buffer (raw pre-conv inputs)
        final_s = s[:, -1, :]
        if self.k > 1:
            next_buf = x_ssm[:, -(self.k - 1):, :].detach().clone()
        else:
            next_buf = x_ssm.new_zeros(B, 0, self.d_inner)

        return x + out, {'s': final_s, 'buf': next_buf}

    def step(self, x_t: torch.Tensor, state: dict):
        """Single-step inference. x_t: (B, D)"""
        # 1. Expand
        xz = self.in_proj(x_t)
        x_ssm_t, z_t = xz.chunk(2, dim=-1)  # each (B, d_inner)

        # 2. Conv (buffered)
        buf = state.get('buf', x_t.new_zeros(x_t.size(0), 0, self.d_inner))
        # Append current input to buffer
        x_cat = torch.cat([buf, x_ssm_t.unsqueeze(1)], dim=1)  # (B, buf_len+1, d_inner)

        # Ensure we have exactly k timesteps for the conv (left-pad with zeros if needed)
        if x_cat.size(1) < self.k:
            pad_len = self.k - x_cat.size(1)
            x_input = F.pad(x_cat.transpose(1, 2), (pad_len, 0)).transpose(1, 2)
        else:
            x_input = x_cat[:, -self.k:, :]

        x_conv_t = self.dw(x_input.transpose(1, 2)).squeeze(-1)  # (B, d_inner)
        x_conv_t = self.act(x_conv_t)
        x_conv_t = self.inner_norm(x_conv_t)

        # 3. SSM (same logic as forward_seq)
        log_A_clamped = self.log_A.clamp(min=-20, max=20)
        A_bar, X_bar = self._ssm_step(x_conv_t, log_A_clamped)

        s_prev = state.get('s', torch.zeros_like(x_conv_t))
        s_t = A_bar * s_prev + X_bar

        # 4. Output gating
        y_t = s_t * self.act(z_t)
        out = self.out_proj(y_t)

        # 5. Update buffer: keep last k-1 raw inputs
        if self.k > 1:
            next_buf = x_cat[:, -(self.k - 1):, :]
        else:
            next_buf = buf

        return x_t + out, {'s': s_t, 'buf': next_buf}


class RWKVBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ln_time = nn.LayerNorm(dim)
        self.ln_chan = nn.LayerNorm(dim)

        # Time-mix parameters
        self.time_mix_r = nn.Parameter(torch.zeros(dim))
        self.time_mix_k = nn.Parameter(torch.zeros(dim))
        self.time_mix_v = nn.Parameter(torch.zeros(dim))
        self.time_mix_u = nn.Parameter(torch.zeros(dim))

        # Projections
        self.Wr = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wu = nn.Linear(dim, dim, bias=False) # Boost
        self.Wo = nn.Linear(dim, dim, bias=False)

        # Decay w (parameterized as -exp(w))
        self.w = nn.Parameter(torch.zeros(dim)) 
        
        # Channel mix
        self.ff = GatedMLP(dim, mult=4, act_name="gelu")
        
        # Init (RWKV standard)
        nn.init.zeros_(self.time_mix_r); nn.init.zeros_(self.time_mix_k)
        nn.init.zeros_(self.time_mix_v); nn.init.zeros_(self.time_mix_u)
        # Initialize w to generate decent decay curves
        nn.init.uniform_(self.w, -6, -5)
    def init_state(self, x: torch.Tensor):
        """
        Initialize zero state for RWKV.
        x: Input tensor to derive batch size, device, and dtype from.
        """
        B = x.size(0)
        D = self.dim
        device = x.device
        dtype = x.dtype
        
        return {
            "s": torch.zeros(B, D, device=device, dtype=dtype),
            "n": torch.zeros(B, D, device=device, dtype=dtype),
            "xprev": torch.zeros(B, D, device=device, dtype=dtype)
        }
    def _decay(self):
        # a = exp(-exp(w)) in (0,1)
        return torch.exp(-torch.exp(self.w))

    def _mix_current_prev(self, x_cur, x_prev):
        # Token shift
        mr, mk, mv, mu = map(torch.sigmoid, [self.time_mix_r, self.time_mix_k, self.time_mix_v, self.time_mix_u])
        xr = x_prev * (1 - mr) + x_cur * mr
        xk = x_prev * (1 - mk) + x_cur * mk
        xv = x_prev * (1 - mv) + x_cur * mv
        xu = x_prev * (1 - mu) + x_cur * mu
        return xr, xk, xv, xu

    def forward_seq(self, x: torch.Tensor, state: Optional[dict] = None):
        # x: (B,T,D)
        B, T, D = x.shape
        x_n = self.ln_time(x)
        
        # 1. Token Shift (Vectorized)
        if state is None:
            x_prev_token = x_n.new_zeros(B, D)
            s_prev, n_prev = None, None
        else:
            x_prev_token = state.get("xprev", x_n.new_zeros(B, D))
            s_prev = state.get("s", None)
            n_prev = state.get("n", None)

        # Shift x to right by 1 to get x_{t-1}
        x_shifted = torch.cat([x_prev_token.unsqueeze(1), x_n[:, :-1, :]], dim=1) # (B,T,D)
        
        # Mix
        xr, xk, xv, xu = self._mix_current_prev(x_n, x_shifted)
        
        # 2. Projections
        r = self.Wr(xr)
        k = self.Wk(xk)
        v = self.Wv(xv)
        u = self.Wu(xu) # This is the "u" boost key
        
        # 3. Prepare for Scan
        # RWKV: S_t = a * S_{t-1} + exp(k) * v
        #       N_t = a * N_{t-1} + exp(k)
        
        a = self._decay().view(1, 1, D).expand(B, T, D) # Decay
        ek = torch.exp(torch.clamp(k, max=30.0))        # Exponentiated k
        
        # Inputs to scan
        b_num = ek * v
        b_den = ek
        
        # 4. Run Parallel Scan
        # These return inclusive states S_t, N_t
        S_inclusive = parallel_scan_linear(a, b_num, h0=s_prev)
        N_inclusive = parallel_scan_linear(a, b_den, h0=n_prev)
        
        # 5. Calculate Output (WKV)
        # WKV_t = ( e^(u+k) * v + a * S_{t-1} ) / ( e^(u+k) + a * N_{t-1} )
        # Note: S_{t-1} is simply S_inclusive shifted right (or S_inclusive - b_t / a_t? No, shift is safer)
        
        S_prev_step = torch.cat([
            (s_prev.unsqueeze(1) if s_prev is not None else torch.zeros(B, 1, D, device=x.device)), 
            S_inclusive[:, :-1, :]
        ], dim=1)
        
        N_prev_step = torch.cat([
            (n_prev.unsqueeze(1) if n_prev is not None else torch.zeros(B, 1, D, device=x.device)), 
            N_inclusive[:, :-1, :]
        ], dim=1)
        
        euk = torch.exp(torch.clamp(u + k, max=30.0))
        
        num = euk * v + a * S_prev_step
        den = euk     + a * N_prev_step
        
        wkv = num / (den + 1e-12)
        
        # 6. Output Gate
        y_time = self.Wo(torch.sigmoid(r) * wkv)
        
        # 7. Channel Mix & Residual
        y = x + y_time
        y = y + self.ff(self.ln_chan(y))
        
        # Save state for next chunk
        new_state = {
            "s": S_inclusive[:, -1, :],
            "n": N_inclusive[:, -1, :],
            "xprev": x_n[:, -1, :]
        }
        
        return y, new_state

    def step(self, x_t: torch.Tensor, state: dict):
        # Sequential inference step (matches forward_seq logic)
        x_tn = self.ln_time(x_t)
        xprev = state.get("xprev", torch.zeros_like(x_tn))
        
        xr, xk, xv, xu = self._mix_current_prev(x_tn, xprev)
        
        r = self.Wr(xr)
        k = self.Wk(xk)
        v = self.Wv(xv)
        u = self.Wu(xu)
        
        a = self._decay()
        ek = torch.exp(torch.clamp(k, max=30.0))
        euk = torch.exp(torch.clamp(u + k, max=30.0))
        
        s_prev = state.get("s", torch.zeros_like(x_tn))
        n_prev = state.get("n", torch.zeros_like(x_tn))
        
        # WKV calculation
        num = euk * v + a * s_prev
        den = euk     + a * n_prev
        wkv = num / (den + 1e-12)
        
        y_time = self.Wo(torch.sigmoid(r) * wkv)
        
        # Update State
        s_new = a * s_prev + ek * v
        n_new = a * n_prev + ek
        
        y = x_t + y_time
        y = y + self.ff(self.ln_chan(y))
        
        return y, {"s": s_new, "n": n_new, "xprev": x_tn}

class ScanLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, kind: str,
                 n_blocks: int = 2, mamba_kernel: int = 4, log_space: bool = True,
                 minrnn_act: int = 0): # Added minrnn_act arg
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.emb_ln = nn.LayerNorm(dim) 
        self.kind = kind
        self.blocks = nn.ModuleList()
        
        for _ in range(n_blocks):
            if kind == "mingru":
                self.blocks.append(ScanBlock_minGRU(dim, log_space=log_space))
            elif kind == "minrnn":
                # Use the new Generalized Block
                self.blocks.append(ScanBlock_MinRNN_Gen(dim, act_type=minrnn_act))
            elif kind == "minlstm":
                self.blocks.append(ScanBlock_minLSTM(dim, log_space=log_space))
            elif kind == "mamba":
                self.blocks.append(ScanBlock_Mamba(dim, kernel_size=mamba_kernel))
            elif kind == "rwkv":
                self.blocks.append(RWKVBlock(dim))
            elif kind == "gateloop":
                self.blocks.append(ScanBlock_GateLoop(dim))
            elif kind == "minindrnn":
                self.blocks.append(ScanBlock_MinIndRNN(dim, act_type=minrnn_act))
            elif kind == "minjanet":
                self.blocks.append(ScanBlock_MinJANET(dim))
            elif kind == "minindygru":  # <--- NEW
                self.blocks.append(ScanBlock_MinIndyGRU(dim, log_space=log_space))
            elif kind == "minindylstm": # <--- NEW
                self.blocks.append(ScanBlock_MinIndyLSTM(dim, log_space=log_space))
            else:
                raise ValueError(f"Unknown scan kind: {kind}")
                
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx: torch.Tensor, state=None):
        # tiny pre-norm in embeddings
        x = self.emb_ln(self.embed(idx))  # (B,T,D)

        # === TRAINING ===
        # If state is provided, we are doing TBPTT and must thread initial states through.
        if self.training and (state is not None):
            new_states = []
            for b, st in zip(self.blocks, (state if isinstance(state, list) else [None]*len(self.blocks))):
                if isinstance(b, (ScanBlock_Mamba, RWKVBlock)):
                    x, st_new = b.forward_seq(x, state=st)
                else:
                    # minGRU / minLSTM: pass initial h0 = Tensor (B,D) or None
                    h0 = st if (torch.is_tensor(st) or st is None) else None
                    x, h_last = b.forward_seq(x, h0=h0)
                    st_new = h_last  # carry last hidden as state
                new_states.append(st_new)
            x = self.ln(x)
            return self.head(x), new_states

        # === TRAINING (no TBPTT) ===
        if self.training:
            st = None
            for b in self.blocks:
                if isinstance(b, (ScanBlock_Mamba, RWKVBlock)):
                    x, st = b.forward_seq(x, state=None)   # stateless
                else:
                    x, _ = b.forward_seq(x, h0=None)
            x = self.ln(x)
            return self.head(x), None

        # === EVAL / SEQUENTIAL ===
        if state is None:
            state = [None] * len(self.blocks)

        outs = []
        for t in range(x.size(1)):
            x_t = x[:, t, :]  # (B,D)
            new_states = []
            for b, st in zip(self.blocks, state):
                if isinstance(b, ScanBlock_Mamba):
                    x_t, st_new = b.step(x_t, st or {})
                elif isinstance(b, RWKVBlock):
                    x_t, st_new = b.step(x_t, st or b.init_state(x_t))
                else:
                    h_prev = st if st is not None else x_t.new_zeros(x_t.size(0), x_t.size(-1))
                    x_t, st_new = b.step(x_t, h_prev)
                new_states.append(st_new)
            state = new_states
            outs.append(x_t.unsqueeze(1))
        y = torch.cat(outs, dim=1)
        y = self.ln(y)
        return self.head(y), state


class JanetRNN(nn.Module):
    """
    Multi-layer JANET (forget-gate-only LSTM), batch_first=True.

    Equations (JANET):
        f_t = σ(U_f h_{t-1} + W_f x_t + b_f)
        c~_t = tanh(U_c h_{t-1} + W_c x_t + b_c)
        c_t = f_t ⊙ c_{t-1} + (1 - σ(U_f h_{t-1} + W_f x_t + b_f - β)) ⊙ c~_t
        h_t = c_t

    State we carry per layer = c_t (shape (B, H)).
    Returned state = stacked (num_layers, B, H) tensor (like IndRNN).
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=True, dropout=0.0, beta: float = 1.0,
                 chrono_Tmax: Optional[int] = None):
        super().__init__()
        assert batch_first, "This JanetRNN expects batch_first=True"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = float(dropout)
        self.beta = float(beta)
        self.chrono_Tmax = chrono_Tmax  # used for chrono init of forget biases

        self.layers = nn.ModuleList()
        in_sizes = [input_size] + [hidden_size]*(num_layers-1)
        for in_sz in in_sizes:
            mod = nn.Module()
            # f gate projections
            mod.W_f = nn.Linear(in_sz, hidden_size, bias=bias)
            mod.U_f = nn.Linear(hidden_size, hidden_size, bias=bias)
            # candidate c~ projections
            mod.W_c = nn.Linear(in_sz, hidden_size, bias=bias)
            mod.U_c = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.layers.append(mod)

        self._drop = nn.Dropout(self.dropout)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # Xavier-uniform for inputs; orthogonal for hidden like LSTM practice
        for li, mod in enumerate(self.layers):
            nn.init.xavier_uniform_(mod.W_f.weight); nn.init.xavier_uniform_(mod.W_c.weight)
            nn.init.orthogonal_(mod.U_f.weight);     nn.init.orthogonal_(mod.U_c.weight)
            if self.bias:
                # Start b_c at zero
                nn.init.zeros_(mod.W_c.bias); nn.init.zeros_(mod.U_c.bias)
                # Chrono init for forget gate bias b_f
                Tmax = self.chrono_Tmax if (self.chrono_Tmax is not None and self.chrono_Tmax >= 2) else 2
                # b_f ~ log(U[1, Tmax-1]); shape (H,)
                low = torch.ones(self.hidden_size)
                high = torch.full((self.hidden_size,), float(max(1, Tmax - 1)))
                bf = torch.log(torch.rand_like(low) * (high - low) + low)
                # We have two biases contributing to f preact: W_f.bias and U_f.bias; split bf across them.
                if self.bias:
                    nn.init.zeros_(mod.W_f.bias); nn.init.zeros_(mod.U_f.bias)
                    mod.W_f.bias.add_(0.5 * bf)
                    mod.U_f.bias.add_(0.5 * bf)

    def forward(self, x, state=None):
        # x: (B,T,in)
        B, T, _ = x.shape
        # state: (num_layers,B,H) with carried c_t per layer
        if state is None:
            c_list = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        else:
            assert torch.is_tensor(state) and state.shape == (self.num_layers, B, self.hidden_size)
            c_list = [state[li] for li in range(self.num_layers)]

        layer_in = x
        new_c = []
        for li, mod in enumerate(self.layers):
            c = c_list[li]
            outs = []
            # precompute affine terms
            # Note: using explicit loop over T to keep it simple & consistent with other custom cores
            for t in range(T):
                xt = layer_in[:, t, :]
                f = torch.sigmoid(mod.W_f(xt) + mod.U_f(c))                      # f_t
                cand = torch.tanh(mod.W_c(xt) + mod.U_c(c))                      # c~_t
                # apply beta shift on the (1 - f) branch as in the paper (β=1 recommended)
                one_minus_f_beta = 1.0 - torch.sigmoid(mod.W_f(xt) + mod.U_f(c) - self.beta)
                c = f * c + one_minus_f_beta * cand
                outs.append(c.unsqueeze(1))
            y = torch.cat(outs, dim=1)  # (B,T,H)
            if li != self.num_layers-1 and self.training and self.dropout > 0.0:
                y = self._drop(y)
            layer_in = y
            new_c.append(c)
        # h_t = c_t
        out = layer_in
        new_state = torch.stack(new_c, dim=0)  # (L,B,H)
        return out, new_state
# ========= HyperMixer =========
class MultiHeadHyperMixing(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads=4, tie_in_out=True, dropout=0.0, causal=True):
        super().__init__()
        assert d_hidden % n_heads == 0, "d_hidden must be divisible by n_heads"
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.n_heads = int(n_heads)
        self.d_head  = self.d_hidden // self.n_heads
        self.tie = bool(tie_in_out)
        self.causal = bool(causal)

        # Hypernets output per-token, per-head parameters
        out_dim = self.n_heads * self.d_head
        self.hyper_in  = nn.Sequential(nn.Linear(d_model, d_model), nn.Mish(), nn.Linear(d_model, out_dim))
        self.hyper_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Mish(), nn.Linear(d_model, out_dim))

        self.out_proj = nn.Linear(self.n_heads * d_model, d_model)  # fuse heads
        self.ln_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape

        # Hyper weights per token → reshape to heads
        W1 = self.hyper_in(x).view(B, T, self.n_heads, self.d_head)            # (B, T, H, Dh)
        W2 = W1 if self.tie else self.hyper_out(x).view(B, T, self.n_heads, self.d_head)  # (B, T, H, Dh)

        # Shared values across heads (like V without per-head projection)
        # values: (B, D, T)
        values = x.transpose(1, 2)

        # Build per-head token kernels: K[b, h, τ, t]
        #   K_h[τ,t] = <W2[τ,h,:], W1[t,h,:]>
        # einsum over Dh
        # W1: (B, T, H, Dh); W2: (B, T, H, Dh) → K: (B, H, T(τ), T(t))
        K = torch.einsum('bthk,bshk->bhst', W1, W2)

        if self.causal:
            # causal mask over (τ, t): only t <= τ
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            # Use -inf for masked positions to zero them after softmax-like use; here it's linear mixing,
            # so we can just zero the masked positions directly:
            K = K * mask  # broadcast over B,H

        # Mix values with each head's kernel:
        # y_h[b, h, d, τ] = sum_t values[b, d, t] * K[b, h, τ, t]
        # → (B, H, D, T)
        y_heads = torch.einsum('bdt,bhst->bhds', values, K)

        # Reorder to (B, T, H*D) then fuse
        y_heads = y_heads.permute(0, 3, 1, 2).contiguous()   # (B, T, H, D)
        y_cat   = y_heads.view(B, T, self.n_heads * D)       # (B, T, H*D)
        y       = self.out_proj(y_cat)                       # (B, T, D)

        y = self.drop(y)
        return self.ln_out(y)



class _FeatureMLP(nn.Module):
    """Feature mixing MLP (token-wise) with your activation registry + gated style."""
    def __init__(self, d_model: int, d_ff: int, act_name: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = get_activation(act_name)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        z = self.fc2(self.act(self.fc1(self.ln(x))))
        return self.drop(z)

class HyperMixerBlock(nn.Module):
    """
    Pre-LN residual block:
      x = x + MultiHeadHyperMixing(x)
      x = x + FeatureMLP(x)
    """
    def __init__(self, d_model: int, d_hidden: int, d_ff: int, act_name: str = "gelu",
                 tie_hyper: bool = True, drop_token: float = 0.0, drop_ff: float = 0.0,
                 n_heads: int = 4, causal: bool = True):
        super().__init__()
        self.tmix = MultiHeadHyperMixing(
            d_model, d_hidden, n_heads=n_heads,
            tie_in_out=tie_hyper, dropout=drop_token, causal=causal
        )
        self.fmix = _FeatureMLP(d_model, d_ff, act_name=act_name, dropout=drop_ff)

    def forward(self, x):
        x = x + self.tmix(x)
        x = x + self.fmix(x)
        return x


class HyperMixerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 d_hidden: int = None, d_ff: int = None, act_name: str = "gelu",
                 max_seq_len: int = 65536, tie_hyper: bool = True, dropout: float = 0.0,
                 n_heads: int = 4, causal: bool = True):
        super().__init__()
        d_hidden = int(d_hidden or max(64, d_model))
        d_ff     = int(d_ff or (4 * d_model))
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.blocks = nn.ModuleList([
            HyperMixerBlock(d_model, d_hidden, d_ff, act_name=act_name,
                            tie_hyper=tie_hyper, drop_token=dropout, drop_ff=dropout,
                            n_heads=n_heads, causal=causal)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        if T > self.max_seq_len:
            idx = idx[:, -self.max_seq_len:]
            T = idx.size(1)
        x = self.tok(idx) + self.pos(T, device=idx.device)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.head(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.w3(F.mish(self.w1(x)) * self.w2(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x, seq_len=None):
        # x: [B, T, n_heads, head_dim]
        if seq_len > self.max_seq_len: seq_len = self.max_seq_len
        
        if self.cached_cos is None or self.cached_cos.size(0) < seq_len:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos()[None, :, None, :]
            self.cached_sin = emb.sin()[None, :, None, :]

        return self.cached_cos[:, :seq_len], self.cached_sin[:, :seq_len]

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, T, H, D]
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ==============================================================================
# 1. MinRNN (ScanBlock Variant)
# ==============================================================================
# ==============================================================================
# 1. MinRNN (ScanBlock Variant) - Tanh Edition
# ==============================================================================
class ScanBlock_MinRNN(nn.Module):
    """
    Minimal RNN compatible with parallel scan.
    Modified to use Tanh for the recurrence gate to improve gradient flow
    in deep networks ("punch through").
    
    Formulation:
    h_t = a_t * h_{t-1} + b_t
       where a_t = tanh(W_z x_t + bias)
             b_t = W_x x_t
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ln = RMSNorm(dim)
        self.Wx = nn.Linear(dim, dim, bias=False)
        
        # Use bias=True for Wz to allow initializing near boundary (high memory)
        self.Wz = nn.Linear(dim, dim, bias=False) 
        self.out = nn.Linear(dim, dim, bias=False)
        
        # Init Wz bias to 2.0 so tanh(bias) ~= 0.96 (Long memory init)
        #nn.init.constant_(self.Wz.bias, 2.0)
        nn.init.xavier_uniform_(self.Wz.weight, gain=0.1) # Small random jitter

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        # Candidate / Input (b_t)
        b = self.Wx(x_norm)
        
        # Decay (a_t) using Tanh
        # Tanh allows a range of (-1, 1). 
        # Gradients are steeper (max 1.0) compared to sigmoid (max 0.25).
        a = torch.tanh(self.Wz(x_norm))
        
        # Linear Scan
        h = parallel_scan_linear(a, b, h0)
        
        return self.out(h + x), h[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_norm = self.ln(x_t)
        
        # Candidate
        b = self.Wx(x_norm)
        
        # Decay
        a = torch.tanh(self.Wz(x_norm))
        
        # Update
        h = a * h_prev + b
        
        return self.out(h + x_t), h


# ==============================================================================
# 2. Causal MLPMixer
# ==============================================================================
class CausalMixingBlock(nn.Module):
    """
    Autoregressive Mixer:
    1. Token Mixing (Time): Causal Masked Linear
    2. Channel Mixing (Feature): Standard MLP
    """
    def __init__(self, dim, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        # Time Mixing: (B, D, T) -> (B, D, T) with causal mask
        self.time_mix = nn.Linear(seq_len, seq_len)
        self.register_buffer("causal_mask", torch.tril(torch.ones(seq_len, seq_len)))
        
        # Channel Mixing
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.Mish(),
            nn.Linear(4*dim, dim)
        )

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        shortcut = x
        x = self.ln1(x)
        
        # Time mix: Transpose to (B, D, T)
        x = x.transpose(1, 2)
        
        # Apply masked linear manually for causality
        # y = x @ W.T + b
        # We need W to be masked. 
        # Ideally, we crop the weight matrix to T x T and mask it.
        W = self.time_mix.weight[:T, :T] * self.causal_mask[:T, :T]
        b = self.time_mix.bias[:T]
        
        x = F.linear(x, W, b)
        
        x = x.transpose(1, 2) # Back to (B, T, D)
        x = x + shortcut
        
        # Channel mix
        shortcut = x
        x = self.ln2(x)
        x = self.channel_mix(x)
        x = x + shortcut
        return x

class CausalMLPMixer(nn.Module):
    def __init__(self, vocab_size, dim, depth, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([CausalMixingBlock(dim, seq_len) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, idx):
        B, T = idx.shape
        if T > self.seq_len: idx = idx[:, -self.seq_len:]
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

# ==============================================================================
# 3. Modern Transformer (Llama Style)
# ==============================================================================
class ModernAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, max_seq_len=10000):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # === NEW: Post-SDPA Gate [cite: 858] ===
        self.gate = PostSDPAGate(dim)
        # =======================================

    def forward(self, x):
        # x is normalized input
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        cos, sin = self.rope(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # SDPA
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # === NEW: Apply Gating ===
        # Applied after SDPA, before output projection 
        out = self.gate(x, out)
        # =========================

        return self.wo(out)

class ModernTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.attn = ModernAttention(dim, n_heads, n_kv_heads)
        self.ffn = SwiGLU(dim, int(dim * 2.68)) # approximation of 8/3 * d
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class ModernTransformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(dim, n_heads, n_heads) for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

# ==============================================================================
# 4. Griffin (RG-LRU + Local Attn)
# ==============================================================================
class RGLRU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.input_gate = nn.Linear(dim, dim)
        self.recur_gate = nn.Linear(dim, dim)
        self.out_gate = nn.Linear(dim, dim)

    def forward(self, x, state=None):
        # x: (B,T,D)
        i = torch.sigmoid(self.input_gate(x))
        log_a = F.logsigmoid(self.recur_gate(x))
        u = i * x 
        a_lin = torch.exp(log_a)
        # Use JIT scan for stability
        h = pscan_linear_jit(a_lin, u, state)
        return h * torch.sigmoid(self.out_gate(x)), h[:, -1, :]

class GriffinBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        self.rglru = RGLRU(dim)
        self.mlp = SwiGLU(dim, dim*4)

    def forward(self, x, state=None):
        # state is now a tuple: (rnn_hidden, conv_buffer)
        rnn_state = None
        conv_buf = None
        if state is not None:
            if isinstance(state, tuple):
                rnn_state, conv_buf = state
            else:
                rnn_state = state # legacy fallback

        shortcut = x
        x = self.norm(x)
        B, T, D = x.shape

        # === FIXED CONV CACHE LOGIC ===
        if T == 1 and conv_buf is not None:
            # Inference step: append to buffer
            # buffer holds previous 3 tokens. 
            # We need input of length 4 to convolution to produce 1 valid output at end
            x_cat = torch.cat([conv_buf, x], dim=1) # (B, 4, D)
            # Run conv
            x_conv = self.conv(x_cat.transpose(1, 2)).transpose(1, 2) # (B, 4+pad, D)
            # We want the last valid token.
            # Conv padding=3. Output length for input L=4 is 4+3-3 = 4?
            # padding=3 on kernel=4 -> output size L + 3 - 4 + 1 = L.
            # We just want the last timestep.
            x_conv_out = x_conv[:, -1:, :] # (B, 1, D)
            
            # Update buffer: keep last 3
            new_conv_buf = x_cat[:, 1:, :]
            x_processed = F.mish(x_conv_out)
            
        else:
            # Training mode or first step
            # Standard conv with padding handled by layer
            # Conv input (B, D, T). Output (B, D, T)
            x_in = x.transpose(1, 2)
            if conv_buf is not None and T > 1:
                # If we have a buffer but receiving a chunk, concat?
                # Simplify: assume training resets buffer or handles long seq.
                pass
            
            x_conv = self.conv(x_in)[:, :, :-3].transpose(1, 2) # remove lookahead padding
            x_processed = F.mish(x_conv)
            
            # Create buffer for next time (last 3 tokens)
            if T >= 3:
                new_conv_buf = x[:, -3:, :]
            else:
                new_conv_buf = F.pad(x, (0, 0, 3-T, 0)) # Pad time dimension
        
        # RG-LRU
        x_rnn, new_rnn_state = self.rglru(x_processed, rnn_state)
        
        x = x_rnn + shortcut
        x = x + self.mlp(self.norm(x))
        
        return x, (new_rnn_state, new_conv_buf)

class GriffinLM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([GriffinBlock(dim) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(self.norm(x)), new_states

# ==============================================================================
# 5. DeltaNet
# ==============================================================================
# ==============================================================================
# 5. DeltaNet - FIXED (Normalized Keys)
# ==============================================================================
class DeltaNetBlock(nn.Module):
    def __init__(self, dim, head_dim=64):
        super().__init__()
        self.dim = dim
        if dim < head_dim:
            self.head_dim = dim
            self.n_heads = 1
        else:
            self.n_heads = max(1, dim // head_dim)
            self.head_dim = dim // self.n_heads
            
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim*4)

    def forward(self, x, state=None):
        B, T, C = x.shape
        shortcut = x
        x_norm = self.norm(x)
        
        q = self.q(x_norm).view(B, T, self.n_heads, self.head_dim)
        k = self.k(x_norm).view(B, T, self.n_heads, self.head_dim)
        v = self.v(x_norm).view(B, T, self.n_heads, self.head_dim)
        beta = torch.sigmoid(self.beta(x_norm)).view(B, T, self.n_heads, self.head_dim)
        
        # === KEY NORMALIZATION FIX ===
        # L2 Normalize K to prevent explosion
        k = F.normalize(k, p=2, dim=-1)
        
        if state is None:
            state = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device)
            
        outs = []
        H = state
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)
        
        for t in range(T):
            qt, kt, vt, bt = q[:, :, t, :], k[:, :, t, :], v[:, :, t, :], beta[:, :, t, :]
            
            # Standard Delta Rule
            Rk = torch.matmul(H, kt.unsqueeze(-1)).squeeze(-1)
            diff = vt - Rk
            update = torch.matmul((bt * diff).unsqueeze(-1), kt.unsqueeze(-2))
            H = H + update
            
            ot = torch.matmul(H, qt.unsqueeze(-1)).squeeze(-1)
            outs.append(ot)
            
        y = torch.stack(outs, dim=2).transpose(1, 2).reshape(B, T, C)
        y = self.o(y)
        x = shortcut + y
        x = x + self.mlp(self.norm(x))
        return x, H

class DeltaNetLM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([DeltaNetBlock(dim) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(self.norm(x)), new_states

# ==============================================================================
# 6. RetNet (Simple Linear Retention)
# ==============================================================================
class RetNetBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.gn = nn.GroupNorm(n_heads, dim)
        self.swiglu = SwiGLU(dim, dim*2)
        self.ln = RMSNorm(dim)
        gammas = 1.0 - 2.0 ** (-5.0 - torch.arange(n_heads).float())
        self.register_buffer("gammas", gammas)

    def forward(self, x, state=None):
        shortcut = x
        x_norm = self.ln(x)
        B, T, C = x.shape
        q = self.wq(x_norm).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x_norm).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x_norm).view(B, T, self.n_heads, self.head_dim)
        
        kv = torch.einsum('bthd,bthe->bthde', k, v).flatten(3) 
        gamma = self.gammas.view(1, 1, self.n_heads, 1).expand(B, T, -1, self.head_dim*self.head_dim)
        
        if state is None: state = None
        else: state = state.flatten(2)
        
        # Use JIT scan (handles vanishing gamma correctly without explosion)
        h = pscan_linear_jit(gamma, kv, state)
        
        h_mat = h.view(B, T, self.n_heads, self.head_dim, self.head_dim)
        out = torch.einsum('bthd,bthde->bthe', q, h_mat).flatten(2)
        out = self.gn(out.transpose(1, 2)).transpose(1, 2)
        x = shortcut + self.wo(out)
        x = x + self.swiglu(self.ln(x))
        return x, h[:, -1, :].view(B, self.n_heads, self.head_dim, self.head_dim)

class RetNetLM(nn.Module):
    def __init__(self, vocab_size, dim, depth, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RetNetBlock(dim, n_heads) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(x), new_states

# ==============================================================================
# 7. HGRN - FIXED STABILITY
# ==============================================================================
class HGRNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.i_gate = nn.Linear(dim, dim)
        self.f_gate = nn.Linear(dim, dim)
        self.g_gate = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, state=None):
        shortcut = x
        x = self.norm(x)
        i = torch.sigmoid(self.i_gate(x))
        g = torch.tanh(self.g_gate(x))
        f = torch.sigmoid(self.f_gate(x))
        u = i * g
        # Use JIT scan
        h = pscan_linear_jit(f, u, state)
        out = self.out(h)
        return shortcut + out, h[:, -1, :]

class HGRN_LM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([HGRNBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(x), new_states

# ==============================================================================
# 8. Liquid Neural Network (LTC Cell Wrapper)
# ==============================================================================
class LiquidCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Cell.
    dh/dt = - (1/tau + W x) * h + A
    Approximated by explicit Euler or semi-implicit step.
    This is sequential.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Parameters
        self._w_tau = nn.Linear(hidden_size, 1) # Time constant network
        self._w_x = nn.Linear(input_size, hidden_size)
        self._w_h = nn.Linear(hidden_size, hidden_size)
        self._w_in = nn.Linear(input_size, hidden_size)
        
    def forward(self, x, h):
        # x: B D, h: B D
        
        # Compute time constant tau (bounded)
        tau = torch.sigmoid(self._w_tau(h)) + 1e-2 
        
        # Compute input-dependent non-linearity
        # A = tanh(Wx + Wh)
        numerator = torch.tanh(self._w_in(x) + self._w_h(h))
        
        # Update rule:
        # h_new = h + (dt/tau) * (-h + numerator)
        # Using sigmoid gating form effectively
        
        # Simplified LTC update:
        # h_t = (1 - tau) * h_{t-1} + tau * numerator
        h_new = (1 - tau) * h + tau * numerator
        
        return h_new + x

class LiquidRNN(nn.Module):
    """Wraps LiquidCell into a layer for the CustomRNNWrapper"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.cells = nn.ModuleList([
            LiquidCell(input_size if i==0 else hidden_size, hidden_size) 
            for i in range(num_layers)
        ])
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, state=None):
        if self.batch_first: x = x.transpose(0, 1) # T B D
        T, B, _ = x.shape
        
        if state is None:
            state = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        
        outs = []
        for t in range(T):
            xt = x[t]
            new_state_t = []
            for i, cell in enumerate(self.cells):
                ht = state[i]
                ht_new = cell(xt, ht)
                new_state_t.append(ht_new)
                xt = ht_new
            state = new_state_t
            outs.append(xt)
        
        out = torch.stack(outs, dim=0)
        if self.batch_first: out = out.transpose(0, 1)
        
        # Stack state for return: (L, B, D)
        final_state = torch.stack(state, dim=0)
        return out, final_state

# ==============================================================================
# 9. MEGABYTE - FIXED MODULE LIST ERROR
# ==============================================================================
class MegaByteLM(nn.Module):
    def __init__(self, vocab_size, dim, depth, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.global_dim = dim
        self.local_dim = dim // 2
        self.embed = nn.Embedding(vocab_size, self.local_dim)
        self.global_proj = nn.Linear(self.local_dim * patch_size, self.global_dim)
        
        # We assume ModernTransformer has a .blocks ModuleList
        self.global_transformer = ModernTransformer(0, self.global_dim, depth // 2, n_heads=4)
        self.global_transformer.embed = nn.Identity()
        self.global_out = nn.Linear(self.global_dim, self.local_dim * patch_size)
        
        self.local_transformer = ModernTransformer(0, self.local_dim, depth // 2, n_heads=4)
        self.local_transformer.embed = nn.Identity()
        self.head = nn.Linear(self.local_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        P = self.patch_size
        pad = (P - (T % P)) % P
        if pad > 0: idx = F.pad(idx, (0, pad), value=0)
        T_pad = idx.size(1)

        x = self.embed(idx)
        x_patches = x.view(B, -1, P, self.local_dim)
        N_patches = x_patches.shape[1]
        
        x_global_in = self.global_proj(x_patches.view(B, N_patches, -1))
        
        # === FIX: Iterate explicitly ===
        x_g = x_global_in
        for blk in self.global_transformer.blocks:
            x_g = blk(x_g)
        x_global_ctx = self.global_transformer.norm(x_g)
        
        global_repr = self.global_out(x_global_ctx).view(B, N_patches, P, self.local_dim)
        x_local_in = (x_patches + global_repr).view(B, T_pad, self.local_dim)
        
        # === FIX: Iterate explicitly ===
        x_l = x_local_in
        for blk in self.local_transformer.blocks:
            x_l = blk(x_l)
        x_local_out = self.local_transformer.norm(x_l)
        
        logits = self.head(x_local_out)
        return logits[:, :T, :]
# ==============================================================================
# MinRNN Generalized (Multi-Activation)
# ==============================================================================
class ScanBlock_MinRNN_Gen(nn.Module):
    """
    MinRNN with configurable activation for the recurrence gate.
    h_t = a_t * h_{t-1} + b_t
    b_t = W_x x_t
    a_t = Activation(W_z x_t + bias)
    """
    def __init__(self, dim: int, act_type: int = 0):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim) # Standard LayerNorm for stability
        
        self.Wx = nn.Linear(dim, dim, bias=False)
        
        # FIXED: bias=True is required because you access self.Wz.bias in init logic below
        self.Wz = nn.Linear(dim, dim, bias=True) 
        
        self.out = nn.Linear(dim, dim, bias=False)
        self.act_type = act_type
        
        # Init Wz to produce values close to identity or stable decay initially
        if act_type == 4: # Sigmoid
            # Initialize bias to start with high retention (sigmoid(2.0) ~= 0.88)
            nn.init.constant_(self.Wz.bias, 2.0) 
            # Keep weights small so initially the gate is mostly controlled by bias
            nn.init.xavier_uniform_(self.Wz.weight, gain=0.01) 
            
        elif act_type == 0: # Tanh
             # MinRNN paper approach: Tanh generally decays (-1 to 1).
             # We want to avoid 0 (forgetting) or -1 (oscillation) initially.
             # Small weights ensure we stay in the linear region or near 0, 
             # but strictly speaking Tanh isn't great for "holding" memory compared to Sigmoid.
             nn.init.xavier_uniform_(self.Wz.weight, gain=0.1)
             nn.init.zeros_(self.Wz.bias)
        else:
            # ReLU, SiLU, GELU are unbounded positive. 
            # We want small initial values to avoid explosion (a_t > 1.0).
            nn.init.xavier_uniform_(self.Wz.weight, gain=0.01)
            nn.init.zeros_(self.Wz.bias)

    def _get_a(self, z):
        if self.act_type == 0: return torch.tanh(z)
        if self.act_type == 1: return F.relu(z)
        if self.act_type == 2: return F.mish(z)
        if self.act_type == 3: return F.mish(z)
        if self.act_type == 4: return torch.sigmoid(z)
        return torch.sigmoid(z)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        b = self.Wx(x_norm)
        
        # Calculate decay gate
        z = self.Wz(x_norm)
        #a = self._get_a(z)
        
        # Parallel Scan: h_t = a_t * h_{t-1} + b_t
        if self.act_type == 5:
            # Log-Space Scan (minGRU style)
            # a_t = g(z), b_t = g(b_raw)
            # log_a = log_g(z), log_b = log_g(b_raw)
            log_a = log_g_act(z)
            log_b = log_g_act(b) # Candidate must be positive for log scan
            h = heinsen_associative_scan_log(log_a, log_b, h0)
        else:
            # Linear Scan
            a = self._get_a(z)
            h = parallel_scan_linear(a, b, h0)
        
        # FIXED: Transformer-style Residual
        # x + OutputProjection(Branch)
        # Allows 'x' to flow cleanly and 'self.out' to center the RNN signal.
        return x + self.out(h), h[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_norm = self.ln(x_t)
        b = self.Wx(x_norm)
        z = self.Wz(x_norm)
        
        if self.act_type == 5:
            # Log-space path: use g_act for both gate and candidate (matches forward_seq)
            a = g_act(z)
            b = g_act(b)
        else:
            a = self._get_a(z)
        
        h = a * h_prev + b
        
        # FIXED: Match forward_seq
        return x_t + self.out(h), h


# ==============================================================================
# MinIndRNN (Parallel Scan Compatible IndRNN) with Extended Activations
# ==============================================================================
class ScanBlock_MinIndRNN(nn.Module):
    """
    MinIndRNN: Independent RNN adapted for Parallel Scan with extended activation support.
    
    Structure: x + Linear(RNN(Norm(x)))
    This creates a stable residual block where the Linear layer centers the RNN output.
    """
    def __init__(self, dim: int, act_type: int = 0):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim) # Standard LayerNorm
        
        self.Wx = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=False)
        self.act_type = act_type
        
        # --- Handle Stateful Activations ---
        self.act_layer = None
        
        # PReLU (init 0.0)
        if self.act_type == 3:
            self.act_layer = nn.PReLU(num_parameters=dim, init=0.0)
        # PReLU (default init 0.25)
        elif self.act_type == 4:
            self.act_layer = nn.PReLU(num_parameters=dim, init=0.25)
        # Snake Activation parameter (alpha)
        elif self.act_type == 11:
            self.snake_alpha = nn.Parameter(torch.ones(dim))
        elif self.act_type == 17:
            pass # g_act handled functionally
        
        # --- Recurrent Weights ---
        # Static recurrent weight 'u'. 
        # Parameterized as sigmoid(p) to ensure stability in (0, 1) range
        self.u_param = nn.Parameter(torch.Tensor(dim))
        nn.init.uniform_(self.u_param, 2.0, 4.0) # Init high for long memory

    def _act(self, x):
        # 0: Tanh
        if self.act_type == 0: 
            return torch.tanh(x)
        # 1: ReLU
        if self.act_type == 1: 
            return F.relu(x)
        # 2: SiLU
        if self.act_type == 2: 
            return F.mish(x)
        
        # 3 & 4: PReLU (Stateful)
        if self.act_type in [3, 4]: 
            # If input is (B, T, D), transpose to (B, D, T) for PReLU
            if x.dim() == 3:
                return self.act_layer(x.transpose(1, 2)).transpose(1, 2)
            # If input is (B, D) (during step/inference), it works as is
            return self.act_layer(x)
        
        # 5: LeakyReLU (0.2)
        if self.act_type == 5: 
            return F.leaky_relu(x, negative_slope=0.2)
        # 6: LeakyReLU (0.01)
        if self.act_type == 6: 
            return F.leaky_relu(x, negative_slope=0.01)
        
        # 7: GELU
        if self.act_type == 7: 
            return F.mish(x)
        
        # 8: BentIdentity
        if self.act_type == 8:
            return ((torch.sqrt(x.pow(2) + 1) - 1) / 2) + x
            
        # 9: Sine
        if self.act_type == 9: 
            return torch.sin(x)
        
        # 10: Cosine
        if self.act_type == 10: 
            return torch.cos(x)
            
        # 11: Snake
        if self.act_type == 11:
            return x + (1.0 / (self.snake_alpha + 1e-9)) * torch.pow(torch.sin(self.snake_alpha * x), 2)
            
        # 12: x + sin(x)
        if self.act_type == 12: 
            return x + torch.sin(x)
        
        # 13: x + cos(x)
        if self.act_type == 13: 
            return x + torch.cos(x)
            
        # 14: Mish
        if self.act_type == 14:
            return x * torch.tanh(F.softplus(x))
            
        # 15: Cone (Triangle)
        if self.act_type == 15:
            return 1.0 - torch.abs(x - 1)
            
        # 16: SquareReLU
        if self.act_type == 16: 
            return torch.square(F.relu(x))

        # 17: g_act (minGRU style)
        if self.act_type == 17:
            return g_act(x)

        return torch.tanh(x)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        """
        Parallel forward pass (requires parallel_scan_linear).
        """
        B, T, D = x.shape
        x_norm = self.ln(x)
        
        if self.act_type == 17:
            # Log-Space Scan
            # b_t = g_act(Wx) -> log_b = log_g_act(Wx)
            log_b = log_g_act(self.Wx(x_norm))
            
            # a_t = sigmoid(u) -> log_a = logsigmoid(u)
            log_a = F.logsigmoid(self.u_param).view(1, 1, D).expand(B, T, D)
            
            h = heinsen_associative_scan_log(log_a, log_b, h0)
        else:
            # Linear Scan
            b = self._act(self.Wx(x_norm))
            # Recurrent Decay: a_t = Sigmoid(u) [Static across time]
            u = torch.sigmoid(self.u_param).view(1, 1, D).expand(B, T, D)
            h = parallel_scan_linear(u, b, h0)
        
        # FIXED: Transformer-style Residual
        # x + OutputProjection(Branch)
        # This keeps 'x' gradient flow clean (Identity Mapping)
        # and lets 'self.out' center the signal.
        return x + self.out(h), h[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        """
        Sequential inference step.
        """
        x_norm = self.ln(x_t)
        b = self._act(self.Wx(x_norm))
        u = torch.sigmoid(self.u_param)
        
        # h_t = u * h_{t-1} + b_t
        h = u * h_prev + b
        
        # FIXED: Match forward_seq
        return x_t + self.out(h), h

# ==============================================================================
# z (Parallel Scan Compatible JANET)
# ==============================================================================
class ScanBlock_MinJANET(nn.Module):
    """
    MinJANET adapted for Log-Space Parallel Scan (minGRU style).
    
    Changes from Standard JANET:
    1. Uses 'g_act' (linear/sigmoid hybrid) instead of 'tanh' for the candidate.
       This is necessary because heinsen_associative_scan_log assumes positive values.
    2. Computes the recurrence in log-space for stability over long sequences.
    
    Recurrence: h_t = f_t * h_{t-1} + (1 - f_t) * c_t
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        
        self.Wf = nn.Linear(dim, dim, bias=True) # Forget Gate
        self.Wc = nn.Linear(dim, dim, bias=True) # Candidate
        self.out = nn.Linear(dim, dim, bias=False)
        self.post = GatedMLP(dim, mult=4, act_name="gelu") # Optional: Match minGRU post-processing
        
        # Chrono Init: bias f to be open (1.0) initially
        # sigmoid(2.0) ~= 0.88, sigmoid(4.0) ~= 0.98
        nn.init.constant_(self.Wf.bias, 2.0) 
        nn.init.xavier_uniform_(self.Wc.weight)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        # 1. Projections
        z_f = self.Wf(x_norm)
        z_c = self.Wc(x_norm)
        
        # 2. Log-Space Math
        # We need log(f) and log((1-f)*c)
        
        # log(f) where f = sigmoid(z_f)
        # log(sigmoid(x)) = -softplus(-x)
        log_f = -F.softplus(-z_f)
        
        # log(1-f) where f = sigmoid(z_f)
        # Identity: 1 - sigmoid(x) = sigmoid(-x)
        # log(sigmoid(-x)) = -softplus(x)
        log_1_minus_f = -F.softplus(z_f)
        
        # log(c) using stable log_g_act (from your file)
        log_c = log_g_act(z_c)
        
        # Combine input term: b_t = (1-f) * c
        # log(b_t) = log(1-f) + log(c)
        log_values = log_1_minus_f + log_c
        
        # 3. Parallel Scan (Heinsen Log Scan)
        h = heinsen_associative_scan_log(log_f, log_values, h0)
        
        # 4. Output (Residual + Post-MLP like minGRU)
        # Using 0.5 blending for residual stability
        y = x + self.out(h)
        # Optional: Add the GatedMLP post-layer if you want parity with minGRU block
        y = y + self.post(y)
        
        return y, h[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_norm = self.ln(x_t)
        
        # 1. Projections
        z_f = self.Wf(x_norm)
        z_c = self.Wc(x_norm)
        
        # 2. Activations (Linear Space)
        f = torch.sigmoid(z_f)
        c = g_act(z_c) # Use g_act to match log_g_act from forward_seq
        
        # 3. Recurrence
        # h_t = f * h_{t-1} + (1-f) * c
        h = f * h_prev + (1.0 - f) * c
        
        y = x_t + self.out(h)
        # if using post MLP:
        y = y + self.post(y)
        
        return y, h
# ==============================================================================
# 10. KAN-Transformer (Chebyshev Implementation)
# ==============================================================================
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        
        # Chebyshev coefficients
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1.0 / (input_dim * (degree + 1)))
        
        # Base linear activation (residual)
        self.base_linear = nn.Linear(input_dim, output_dim)
        self.act = nn.Mish()

    def forward(self, x):
        # x: (..., input_dim)
        # Normalize x to [-1, 1] for Chebyshev stability (using tanh)
        x_norm = torch.tanh(x)
        
        # Compute Chebyshev polynomials recursively
        # T_0(x) = 1, T_1(x) = x, T_n(x) = 2xT_{n-1} - T_{n-2}
        polys = [torch.ones_like(x_norm), x_norm]
        for i in range(2, self.degree + 1):
            polys.append(2 * x_norm * polys[-1] - polys[-2])
        
        # Stack: (..., input_dim, degree+1)
        poly_stack = torch.stack(polys, dim=-1)
        
        # y = Sum( c_ij * T_j(x_i) )
        # Contract: (...In, Deg) * (In, Out, Deg) -> (...Out)
        y = torch.einsum("...id,iod->...o", poly_stack, self.cheby_coeffs)
        
        # Add base linear transformation
        base = self.base_linear(self.act(x))
        return y + base

class KANBlock(nn.Module):
    def __init__(self, dim, degree=3):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        # Replaces Standard Attention with a KAN-Mixer for this variant
        # or replaces FFN. Here we replace FFN with KAN and keep standard Attn.
        self.attn = ModernAttention(dim, n_heads=8)
        self.kan_ffn = nn.Sequential(
            ChebyKANLayer(dim, dim * 2, degree=degree),
            nn.LayerNorm(dim * 2), # Norm inside KAN helps stability
            ChebyKANLayer(dim * 2, dim, degree=degree)
        )

    def forward(self, x):
        x = x + self.attn(self.ln(x))
        x = x + self.kan_ffn(self.ln(x))
        return x

class KAN_LM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([KANBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

# ==============================================================================
# 11. Linear Transformer (Recurrent Form)
# ==============================================================================
class LinearAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, dim*4)

    def feature_map(self, x):
        # Katharopoulos feature map: elu(x) + 1
        return F.elu(x) + 1.0

    def forward(self, x, state=None):
        B, T, C = x.shape
        shortcut = x
        x_norm = self.norm(x)
        
        q = self.wq(x_norm).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x_norm).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x_norm).view(B, T, self.n_heads, self.head_dim)
        
        Q = self.feature_map(q)
        K = self.feature_map(k)
        
        # KV calculation for recurrence: outer product K^T * V
        # shape: (B, T, H, D, 1) * (B, T, H, 1, D) -> (B, T, H, D, D)
        KV = torch.einsum("bthd,bthe->bthde", K, v)
        
        # Recurrence: S_t = S_{t-1} + K_t V_t^T
        if state is None:
            # Full sequence cumsum
            S = torch.cumsum(KV, dim=1)
        else:
            # Stepwise or continued
            # state is (B, H, D, D)
            S_inc = torch.cumsum(KV, dim=1)
            S = S_inc + state.unsqueeze(1)
            
        # Normalization factor Z_t = Z_{t-1} + K_t
        K_sum = torch.cumsum(K, dim=1) 

        # Y_t = (Q_t * S_t) / (Q_t * Z_t)
        num = torch.einsum("bthd,bthde->bthe", Q, S)
        den = torch.einsum("bthd,bthd->bth", Q, K_sum).clamp(min=1e-4)
        
        y = num / den.unsqueeze(-1)
        y = y.reshape(B, T, C)
        y = self.wo(y)
        
        x = shortcut + y
        x = x + self.mlp(self.norm(x))
        
        # Return last state for generation
        last_state = S[:, -1, :, :, :]
        return x, last_state

    def forward_seq(self, x, h0=None, state=None):
        # Alias for linegen compatibility
        st = state if state is not None else h0
        return self.forward(x, state=st)

class LinearTransformerLM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([LinearAttentionBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(x), new_states

# ==============================================================================
# 12. H3 (Hungry Hungry Hippos) - Simplified
# ==============================================================================
class H3Block(nn.Module):
    def __init__(self, dim, head_dim=64):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Shift SSM (Causal convolution — padding on the left only)
        self.shift_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=0, groups=dim)
        
        # Diagonal SSM parameters
        self.dt_proj = nn.Linear(dim, dim)
        self.A_log = nn.Parameter(torch.log(torch.rand(dim) + 0.5)) # Decay
        
        self.mlp = SwiGLU(dim, dim*4)

    def forward(self, x, state=None):
        # State: tuple(ssm_state, conv_buffer)
        rnn_state, buf = (None, None)
        if state is not None: 
            if isinstance(state, tuple):
                rnn_state, buf = state
            else:
                rnn_state = state

        shortcut = x
        x = self.norm(x)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 1. Shift-SSM on K (Causal Local Context — left-padded conv)
        if buf is None:
            # Training / Fresh — pad 2 zeros on the left for causal kernel_size=3
            k_T = k.transpose(1, 2)
            k_T_padded = F.pad(k_T, (2, 0))  # (left=2, right=0)
            k_shift = self.shift_conv(k_T_padded).transpose(1, 2)
            # Create buffer for next step (last 2 tokens)
            new_buf = k[:, -2:, :]
        else:
            # Step — prepend buffer for causal context
            k_cat = torch.cat([buf, k], dim=1)
            k_shift = self.shift_conv(k_cat.transpose(1, 2)).transpose(1, 2)[:, -x.shape[1]:, :]
            new_buf = k_cat[:, -2:, :]

        # 2. Diagonal SSM on V * K_shifted
        x_ssm = v * k_shift
        
        # Parameters
        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log.clamp(max=5.0))
        D_decay = torch.exp((A * dt).clamp(min=-30.0, max=0.0))  # bounded decay in (0, 1]
        
        # Scan: h_t = D_decay * h_{t-1} + x_ssm
        h = parallel_scan_linear(D_decay, x_ssm, rnn_state)
        
        # 3. Output Gating
        y = q * h
        y = self.out_proj(y)
        
        x = shortcut + y
        x = x + self.mlp(self.norm(x))
        
        return x, (h[:, -1, :], new_buf)

    def forward_seq(self, x, h0=None, state=None):
        # Alias for linegen compatibility
        st = state if state is not None else h0
        return self.forward(x, state=st)

class H3LM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([H3Block(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(x), new_states

# ==============================================================================
# 13. DCT-Former (Discrete Cosine Transform Mixing)
# ==============================================================================
import torch.fft

class DCTMixingBlock(nn.Module):
    """
    Causal frequency-aware token mixing block.
    
    Original used FFT which is fundamentally non-causal (every output depends on
    all inputs). This replacement uses a causal linear projection with DCT-inspired
    initialization, preserving the frequency-domain character while being strictly causal.
    
    The weight matrix W is masked with a lower-triangular (causal) mask, and 
    initialized from a truncated DCT basis so the model starts with frequency-like
    mixing patterns that respect causality.
    """
    def __init__(self, dim, seq_len):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.ln = nn.LayerNorm(dim)
        
        # Causal mixing weight: (seq_len, seq_len) masked lower-triangular
        # Initialize from DCT basis (truncated to causal)
        W = torch.zeros(seq_len, seq_len)
        for k in range(seq_len):
            for n in range(k + 1):  # only causal entries (n <= k)
                W[k, n] = math.cos(math.pi * (2*n + 1) * k / (2 * seq_len))
        # Normalize rows
        row_norms = W.norm(dim=1, keepdim=True).clamp(min=1e-6)
        W = W / row_norms * 0.02  # scale down for stable init
        self.mix_weight = nn.Parameter(W)
        
        # Learnable per-channel frequency scaling
        self.channel_scale = nn.Parameter(torch.ones(dim) * 0.1)
        
        # Register causal mask as buffer (not a parameter)
        self.register_buffer('causal_mask', torch.tril(torch.ones(seq_len, seq_len)))
        
        self.mlp = SwiGLU(dim, dim * 4)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        shortcut = x
        x = self.ln(x)
        
        # Apply causal mask to weight matrix, then mix tokens
        T_eff = min(T, self.seq_len)
        W = self.mix_weight[:T_eff, :T_eff] * self.causal_mask[:T_eff, :T_eff]
        
        # Token mixing: (B, T, D) -> einsum with (T, T) -> (B, T, D)
        # Scale per channel
        x_mix = torch.einsum('ij,bjd->bid', W, x[:, :T_eff, :]) * self.channel_scale.unsqueeze(0).unsqueeze(0)
        
        if T > self.seq_len:
            # For sequences longer than seq_len, pass through unmixed (rare edge case)
            x_out = torch.cat([x_mix, x[:, T_eff:, :] * 0.0], dim=1)
        else:
            x_out = x_mix
        
        x = shortcut + x_out
        x = x + self.mlp(self.ln(x))
        return x

class DCTFormerLM(nn.Module):
    def __init__(self, vocab_size, dim, depth, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([DCTMixingBlock(dim, seq_len) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)



class ScanBlock_MinIndyGRU(nn.Module):
    """
    MinIndyGRU: minGRU + Independent Recurrent Scaling (IndRNN).
    Recurrence: h_t = (u * (1-z_t)) * h_{t-1} + z_t * h_tilde
    
    The 'u' parameter allows the model to scale the memory state independently 
    of the gate, enabling better gradient flow and long-term memory (u ~ 1.0) 
    or rapid flushing (u < 1.0) per channel.
    """
    def __init__(self, dim: int, log_space: bool = True):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.Wz = nn.Linear(dim, dim) # Gate
        self.Wh = nn.Linear(dim, dim) # Candidate
        
        # IndRNN scaling parameter 'u' (parameterized in log-space)
        # Init to 0.0 (linear 1.0) to start with standard minGRU behavior
        self.u_log = nn.Parameter(torch.zeros(dim))
        
        self.post = GatedMLP(dim, mult=4, act_name="gelu")
        
        # Init
        nn.init.xavier_uniform_(self.Wz.weight); nn.init.zeros_(self.Wz.bias)
        nn.init.xavier_uniform_(self.Wh.weight); nn.init.zeros_(self.Wh.bias)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        z_raw = self.Wz(x_norm)
        h_tilde_raw = self.Wh(x_norm)
        
        # Log-Coeffs: log(u * (1-z)) = log(u) + log(1-z)
        # log(1 - sigmoid(z)) = -softplus(z)
        log_coeffs = self.u_log - F.softplus(z_raw)
        
        # Log-Values: log(z * h_tilde)
        log_z = -F.softplus(-z_raw)
        log_h_tilde = log_g_act(h_tilde_raw)
        log_values = log_z + log_h_tilde
        
        h_seq = heinsen_associative_scan_log(log_coeffs, log_values, h0)
        
        out = x + self.post(h_seq)
        return out, h_seq[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_tn = self.ln(x_t)
        
        z = torch.sigmoid(self.Wz(x_tn))
        h_tilde = g_act(self.Wh(x_tn))
        
        # Linear space u
        u = torch.exp(self.u_log)
        
        # h_t = u * (1-z) * h_{t-1} + z * h_tilde
        h = (u * (1.0 - z)) * h_prev + z * h_tilde
        
        out = x_t + self.post(h)
        return out, h


class ScanBlock_MinIndyLSTM(nn.Module):
    """
    MinIndyLSTM: minLSTM + Independent Recurrent Scaling.
    Recurrence: h_t = (u * f'_t) * h_{t-1} + i'_t * h_tilde
    
    Allows the 'forgetting' mechanic to be scaled by a learnable static vector 'u'.
    """
    def __init__(self, dim: int, log_space: bool = True):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.Wf = nn.Linear(dim, dim)
        self.Wi = nn.Linear(dim, dim)
        self.Wh = nn.Linear(dim, dim)
        
        # IndRNN parameter
        self.u_log = nn.Parameter(torch.zeros(dim))
        
        self.post = GatedMLP(dim, mult=4, act_name="gelu")
        
        nn.init.xavier_uniform_(self.Wf.weight); nn.init.zeros_(self.Wf.bias)
        nn.init.xavier_uniform_(self.Wi.weight); nn.init.zeros_(self.Wi.bias)
        nn.init.xavier_uniform_(self.Wh.weight); nn.init.zeros_(self.Wh.bias)
        with torch.no_grad(): self.Wf.bias.fill_(1.0)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        f_raw = self.Wf(x_norm)
        i_raw = self.Wi(x_norm)
        h_tilde_raw = self.Wh(x_norm)

        # 1. Normalize Gates (minLSTM logic)
        log_f = -F.softplus(-f_raw)
        log_i = -F.softplus(-i_raw)
        log_denom = torch.logaddexp(log_f, log_i)
        
        log_f_prime = log_f - log_denom
        log_i_prime = log_i - log_denom
        
        # 2. Inject Independent Scaling 'u' into decay
        # log(a_t) = log(u) + log(f')
        log_coeffs = self.u_log + log_f_prime
        
        # log(b_t) = log(i') + log(h_tilde)
        log_values = log_i_prime + log_g_act(h_tilde_raw)
        
        h_seq = heinsen_associative_scan_log(log_coeffs, log_values, h0)
        
        out = x + self.post(h_seq)
        return out, h_seq[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_tn = self.ln(x_t)
        
        f = torch.sigmoid(self.Wf(x_tn))
        i = torch.sigmoid(self.Wi(x_tn))
        h_tilde = g_act(self.Wh(x_tn))
        
        denom = f + i + 1e-8
        f_prime = f / denom
        i_prime = i / denom
        
        u = torch.exp(self.u_log)
        
        # h = u * f' * h_prev + i' * h_tilde
        h = (u * f_prime) * h_prev + i_prime * h_tilde
        
        out = x_t + self.post(h)
        return out, h
