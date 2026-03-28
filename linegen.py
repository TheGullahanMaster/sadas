#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import random
import shutil
import pathlib
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Set
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lamb import *
import logging
import torch._dynamo
import torch._inductor.config
import struct
import re
import binascii
from tqdm import tqdm

# Enable TF32 for Matrix Multiplications (Linear Layers)
torch.backends.cuda.matmul.allow_tf32 = True

# Enable TF32 for Convolutions (if you use TCNs/CNNs)
torch.backends.cudnn.allow_tf32 = True

print(f"🚀 TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
# Debug flags - set to True only when debugging torch.compile issues
torch._dynamo.config.verbose = False
torch._inductor.config.debug = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "model.pt"
CONFIG_PATH = "textgen.json"
BOS_TOKEN = "<BOS>"
# at top near BOS_TOKEN
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

SEED = 1337
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# === ADD: tiktoken (optional) ===
try:
    import tiktoken
except Exception:
    tiktoken = None

def _ensure_tiktoken():
    if tiktoken is None:
        raise RuntimeError("tiktoken not installed. Run: pip install tiktoken")

def save_activation_capture_bin(
    path: str,
    tokens_text: List[str],
    captured: List[Optional[torch.Tensor]]
):
    """
    Write a compact binary file with per-timestep activations + decoded token.

    File format (little-endian):
      magic:   4 bytes  = b'ACTV'
      version: uint16   = 1
      L:       uint16   = num_layers
      S:       uint32   = steps (timesteps captured)
      B:       uint32   = batch size (always 1 for current UI)
      H_l...:  For l in [0..L-1]: uint32 hidden_size_l

      For each step s in [0..S-1]:
        Tlen:  uint16 = byte-length of decoded token text at step s (UTF-8)
        T:     bytes  = token text
        For each layer l:
          activations: H_l * float32  (row = layer l, step s, batch 0)

    Notes:
      - If a layer wasn't captured (None), we write H_l=0 and skip its data.
      - Batch is fixed to 1 in the current sampling UI.
    """
    # Infer steps and hidden sizes from captured tensors
    # captured[l]: Tensor [B, S, H_l] on CPU (as returned by .get_captured())
    L = len(captured)
    B = 1
    # Determine S as the maximum S found (missing layers -> treated as H=0)
    S = 0
    Hs = []
    for t in captured:
        if t is None:
            Hs.append(0)
            continue
        assert t.dim() == 3 and t.size(0) == 1, "Expect captured as [1, S, H]"
        Hs.append(int(t.size(2)))
        S = max(S, int(t.size(1)))

    # sanity: tokens_text should match S (if not, we clamp)
    S = min(S, len(tokens_text))

    with open(path, "wb") as f:
        # header
        f.write(b"ACTV")
        f.write(struct.pack("<H", 1))            # version
        f.write(struct.pack("<H", L))            # num layers
        f.write(struct.pack("<I", S))            # steps
        f.write(struct.pack("<I", B))            # batch size
        for h in Hs:
            f.write(struct.pack("<I", h))        # hidden size per layer

        # body
        for s in range(S):
            token_bytes = tokens_text[s].encode("utf-8", "ignore")
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)
            for l in range(L):
                h = Hs[l]
                if h == 0:
                    continue
                # slice [1, S, H] -> [H] at step s
                step_vec = captured[l][0, s, :].contiguous().view(-1)
                f.write(struct.pack("<%sf" % h, *step_vec.tolist()))


def split_indices(n, frac=0.9):
    idx = torch.randperm(n)
    k = int(frac * n)
    return idx[:k], idx[k:]


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def readable_num(n):
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n >= 1_000: return f"{n/1_000:.2f}k"
    return str(n)

# ========= Tokenization =========
# ==== NEW HELPERS (put near other small utils) ====

BOLD_ON  = "\033[1m"
BOLD_OFF = "\033[0m"
def bold(s: str) -> str:
    return f"{BOLD_ON}{s}{BOLD_OFF}"

# ─── ANSI colour / CLI formatting ─────────────────────────────────────────────
_R  = "\033[0m"       # reset
_B  = "\033[1m"       # bold
_DIM= "\033[2m"       # dim
_CY = "\033[36m"      # cyan
_YL = "\033[33m"      # yellow
_GR = "\033[32m"      # green
_MG = "\033[35m"      # magenta
_RD = "\033[31m"      # red
_WH = "\033[39m"      # default foreground (adapts to light/dark terminals)
_BL = "\033[34m"      # blue

def _c(*parts) -> str:
    """Concatenate ANSI codes + text + reset."""
    return "".join(str(p) for p in parts) + _R

def cli_banner(title: str, subtitle: str = "", width: int = 64) -> None:
    bar = "═" * (width - 2)
    inner = width - 2
    t_pad = (inner - len(title)) // 2
    print(f"\n{_c(_CY, '╔' + bar + '╗')}")
    print(f"{_c(_CY, '║')}{' ' * t_pad}{_c(_B, _WH, title)}{' ' * (inner - t_pad - len(title))}{_c(_CY, '║')}")
    if subtitle:
        s_pad = (inner - len(subtitle)) // 2
        print(f"{_c(_CY, '║')}{' ' * s_pad}{_c(_DIM, subtitle)}{' ' * (inner - s_pad - len(subtitle))}{_c(_CY, '║')}")
    print(f"{_c(_CY, '╚' + bar + '╝')}\n")

def cli_section(title: str, width: int = 64) -> None:
    dash = "─" * (width - len(title) - 5)
    print(f"\n  {_c(_CY, _B, '┌─')} {_c(_WH, _B, title)} {_c(_CY, '─' * max(2, width - len(title) - 7) + '┐')}")

def cli_section_end(width: int = 64) -> None:
    print(f"  {_c(_CY, '└' + '─' * (width - 4) + '┘')}")

def cli_rule(width: int = 64) -> None:
    print(f"  {_c(_DIM, '─' * width)}")

def cli_opt(key, label: str, desc: str = "", kw: int = 4, lw: int = 24) -> None:
    """Print a numbered/keyed option row with an optional description."""
    k_str = _c(_YL, _B, f"{str(key):>{kw}}")
    l_str = _c(_WH, f"  {label:<{lw}}")
    d_str = f"  {_c(_DIM, desc)}" if desc else ""
    print(f"  │ {k_str}{l_str}{d_str}")

def cli_blank_row() -> None:
    print(f"  │")

def cli_group(label: str, width: int = 60) -> None:
    """Print a group header row inside a box."""
    dash = "─" * max(2, width - len(label) - 4)
    print(f"  │  {_c(_DIM, '── ' + label + ' ' + dash)}")

def pinfo(msg: str) -> None:
    """Informational line (cyan bullet)."""
    print(f"  {_c(_CY, '·')} {msg}")

def pwarn(msg: str) -> None:
    """Warning line."""
    print(f"  {_c(_YL, '!')} {msg}")

def pok(msg: str) -> None:
    """Success/result line."""
    print(f"  {_c(_GR, '✓')} {msg}")

def prompt_label(msg: str, default=None) -> str:
    """Format a prompt label with optional default hint."""
    hint = f" {_c(_DIM, f'[{default}]')}" if default is not None else ""
    return f"  {_c(_YL, _B, '▸')} {_c(_WH, msg)}{hint} "

def print_model_menu() -> None:
    """Pretty-print the model selection menu."""
    W = 66
    bar = "─" * (W - 4)
    print(f"\n  {_c(_CY, _B, '┌─')} {_c(_WH, _B, 'Model Selection')} {_c(_CY, '─' * (W - 22) + '┐')}")

    groups = [
        ("MLPs", [
            (0,  "MLP (basic)",          "One-hot window → feedforward, no embedding lookup"),
            (1,  "MLP (residual)",        "Transformer-style FF blocks, no attention or recurrence"),
        ]),
        ("Classic RNNs  (step-by-step, stateful)", [
            (2,  "RNN – Tanh",            "Vanilla Elman RNN with tanh activation"),
            (3,  "RNN – ReLU",            "Vanilla Elman RNN with ReLU activation"),
            (4,  "GRU",                   "Gated Recurrent Unit"),
            (5,  "LSTM",                  "Long Short-Term Memory"),
            (6,  "IndRNN",                "Independently Recurrent NN (diagonal hidden-to-hidden)"),
            (7,  "IndyGRU",               "IndRNN-style GRU"),
            (8,  "ATanU-LSTM",            "LSTM with ArcTan unit activation"),
            (18, "JANET",                 "Forget-gate-only LSTM (simplified)"),
            (23, "Liquid / LTC",          "Liquid Time-Constant Neural Network"),
        ]),
        ("Non-recurrent  (attention / conv / mixer)", [
            (9,  "Temporal ConvNet",      "Causal dilated 1-D TCN"),
            (10, "GPT-2 Transformer",     "Decoder-only transformer, SDPA / Flash-Attn"),
            (19, "HyperMixer",            "MLP-Mixer variant with hypernetwork token mixing"),
            (21, "gMLP",                  "Gated MLP with spatial gating unit (causal)"),
            (22, "aMLP",                  "gMLP + tiny self-attention gate"),
            (24, "MLP-Mixer (causal)",    "Patch-style MLP mixer adapted for sequences"),
            (25, "Modern Transformer",    "Llama-3 style: RMSNorm, SwiGLU, RoPE, GQA"),
            (31, "MEGABYTE",              "Patch-based hierarchical byte-level LM"),
            (34, "KAN-Transformer",       "Transformer with Chebyshev KAN feed-forward"),
            (37, "DCT-Former",            "DCT-based spectral attention transformer"),
        ]),
        ("xLSTM", [
            (11, "xLSTM – sLSTM only",   "Exponential gating scalar LSTM blocks"),
            (12, "xLSTM – mLSTM only",   "Matrix-memory mLSTM blocks"),
            (13, "xLSTM – mixed m:s",    "Interleaved mLSTM + sLSTM (7:1 default)"),
        ]),
        ("State-Space / Linear Recurrence  (scan / parallel)", [
            (14, "Mamba",                 "Selective state-space model (S6 scan)"),
            (15, "minGRU",               "Parallelized minimal GRU (log-space scan)"),
            (16, "minLSTM",              "Parallelized minimal LSTM (log-space scan)"),
            (17, "RWKV",                  "Receptance Weighted Key Value (scan)"),
            (20, "GateLoop",              "Data-controlled linear recurrence"),
            (26, "MinRNN ★",             "Parallelized vanilla RNN — multiple activation options"),
            (27, "Griffin / RG-LRU",      "Real-gated linear recurrent unit"),
            (28, "DeltaNet",              "Delta-rule linear recurrence"),
            (29, "RetNet",                "Retentive network (multi-scale retention)"),
            (30, "HGRN",                  "Hierarchical Gated Recurrent Network"),
            (32, "MinIndRNN ★",          "Parallelized IndRNN — many activation choices"),
            (33, "MinJANET",             "Parallelized JANET forget-gate model"),
            (35, "Linear Transformer",    "Linear attention recurrent form"),
            (36, "H3",                    "Hungry Hungry Hippos SSM"),
            (38, "MinIndyGRU",            "Parallelized IndyGRU (scan)"),
            (39, "MinIndyLSTM",           "Parallelized IndyLSTM (scan)"),
        ]),
    ]

    for gname, models in groups:
        cli_blank_row()
        cli_group(gname, W - 4)
        for mid, mname, mdesc in models:
            cli_opt(mid, mname, mdesc, kw=3, lw=26)

    cli_blank_row()
    print(f"  {_c(_CY, '└' + '─' * (W - 4) + '┘')}")

def ensure_filegen_clean():
    """Clear and recreate FileGen/."""
    out_dir = pathlib.Path("FileGen")
    if out_dir.exists():
        for p in out_dir.iterdir():
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception:
                pass
        try: out_dir.rmdir()
        except Exception: pass
    out_dir.mkdir(parents=True, exist_ok=True)

# ========= Tokenization =========
class BaseVocab:
    line_mode: bool = False
    bos_id: Optional[int] = None
    def encode(self, s): raise NotImplementedError
    def decode(self, ids): raise NotImplementedError
    @property
    def size(self): raise NotImplementedError

class CharVocab(BaseVocab):
    def __init__(self, texts: List[str], line_mode: bool):
        charset = set()
        for t in texts:
            charset.update(t)
        self.line_mode = line_mode

        self.tokens = sorted(list(charset))
        if line_mode:
            for sp in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                if sp in self.tokens:
                    self.tokens.remove(sp)
            self.tokens = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN] + self.tokens

        self.stoi = {ch: i for i, ch in enumerate(self.tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.bos_id = self.stoi[BOS_TOKEN] if line_mode else None
        self.eos_id = self.stoi[EOS_TOKEN] if line_mode else None
        self.pad_id = self.stoi[PAD_TOKEN] if line_mode else None

    def encode(self, s: str) -> List[int]:
        if not self.line_mode:
            return [self.stoi[c] for c in s]
        # In line mode we allow literal `<BOS>` in input (optional), but we always append `<EOS>`
        out = []
        i = 0
        L = len(s)
        if s.startswith(BOS_TOKEN):
            out.append(self.bos_id)
            i += len(BOS_TOKEN)
        else:
            out.append(self.bos_id)
        while i < L:
            if s.startswith(BOS_TOKEN, i):
                out.append(self.bos_id); i += len(BOS_TOKEN)
            elif s.startswith(EOS_TOKEN, i):
                out.append(self.eos_id); i += len(EOS_TOKEN)
            else:
                out.append(self.stoi[s[i]]); i += 1
        out.append(self.eos_id)
        return out

    def decode(self, ids: List[int]) -> str:
        if not self.line_mode:
            return "".join(self.itos[i] for i in ids)
        parts = []
        for i in ids:
            if i == self.bos_id: parts.append(BOS_TOKEN)
            elif i == self.eos_id: parts.append(EOS_TOKEN)
            elif i == self.pad_id: parts.append("")  # hide PAD in decode
            else: parts.append(self.itos[i])
        return "".join(parts)

    @property
    def size(self): return len(self.tokens)
# === ADD: TiktokenVocab ===
class TiktokenVocab(BaseVocab):
    """
    Wrapper for tiktoken encodings. In line_mode we reserve three IDs above base vocab
    for BOS/EOS/PAD so sampling/stop logic works like other vocabs.
    
    NEW: scan_file() builds a filtered set of actually-used tokens from the dataset,
    so random prompt generation only picks tokens that actually exist in training data.
    """
    def __init__(self, encoding_name: str, line_mode: bool):
        _ensure_tiktoken()
        self.enc = tiktoken.get_encoding(encoding_name)
        self.line_mode = line_mode
        self.n_base = int(self.enc.n_vocab)
        self._active_tokens: Optional[List[int]] = None

        if line_mode:
            self.bos_id = self.n_base
            self.eos_id = self.n_base + 1
            self.pad_id = self.n_base + 2
            self._size = self.n_base + 3
        else:
            self.bos_id = None
            self.eos_id = None
            self.pad_id = None
            self._size = self.n_base

    def scan_file(self, txt_path: str, max_bytes: int = 8 * 1024 * 1024):
        """Scan the dataset and build a set of actually-used tiktoken IDs."""
        print(f"[TikToken] Scanning {txt_path} for active tokens (up to {max_bytes//1024//1024}MB)...")
        active = set()
        bytes_read = 0
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            while bytes_read < max_bytes:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                bytes_read += len(chunk.encode("utf-8", "ignore"))
                ids = self.enc.encode(chunk)
                active.update(ids)
        self._active_tokens = sorted(active)
        print(f"[TikToken] Found {len(self._active_tokens)} unique tokens (of {self.n_base} total)")

    def get_random_token(self) -> int:
        """Return a random token from the active set if available."""
        if self._active_tokens and len(self._active_tokens) > 0:
            return random.choice(self._active_tokens)
        return random.randrange(self.n_base)

    def encode(self, s: str) -> List[int]:
        ids = self.enc.encode(s if isinstance(s, str) else str(s))
        if self.line_mode:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        if self.line_mode:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        base_ids = [i for i in ids if 0 <= int(i) < self.n_base]
        return self.enc.decode(base_ids)

    @property
    def size(self): return self._size

    @property
    def tokens(self):
        if self._active_tokens is not None:
            return self._active_tokens
        return list(range(self._size))
    
    @property
    def active_token_count(self) -> int:
        if self._active_tokens is not None:
            return len(self._active_tokens)
        return self._size
# === Custom Efficient BPE ===
class CustomBPEVocab(BaseVocab):
    """
    Byte-level BPE tokenizer (like GPT-2) implemented in pure Python.
    Designed to train on a subset of data to avoid RAM issues on large files.
    """
    def __init__(self, vocab_path: str, line_mode: bool, expected_size: int = 4096):
        self.line_mode = line_mode
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.base_size = 256
        
        # Load if exists
        self.vocab_path = vocab_path
        if os.path.exists(vocab_path):
            self._load(vocab_path)
        else:
            print(f"[BPE] Vocab file {vocab_path} not found. Will train on data.")
            # We initialize empty, train() must be called externally
            
        # Determine IDs for specials
        if line_mode:
            self.bos_id = expected_size
            self.eos_id = expected_size + 1
            self.pad_id = expected_size + 2
            self._size = expected_size + 3
        else:
            self.bos_id = None
            self.eos_id = None
            self.pad_id = None
            self._size = expected_size

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_ids(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, txt_path: str, vocab_size: int):
        """
        Efficient BPE training with incremental pair count updates.
        Instead of rescanning the entire sequence each merge (O(n) per step = O(n*m) total),
        we update pair counts incrementally around each merge site (amortized O(1) per site).
        """
        print(f"[BPE] Training Custom BPE (target size: {vocab_size})...")
        
        MAX_BYTES = 8 * 1024 * 1024 
        with open(txt_path, "rb") as f:
            raw_bytes = f.read(MAX_BYTES)
        
        ids = list(raw_bytes)
        print(f"[BPE] Loaded {len(ids)/1024/1024:.2f} MB of sample data ({len(ids)} tokens).")
        
        num_merges = vocab_size - 256
        if num_merges <= 0:
            print("[BPE] Vocab size <= 256, no BPE training needed.")
            return

        # Build initial pair counts
        import collections as _collections
        pair_counts = _collections.Counter()
        for i in range(len(ids) - 1):
            pair_counts[(ids[i], ids[i+1])] += 1

        pbar = tqdm(total=num_merges, desc="BPE Merge")
        for merge_idx in range(num_merges):
            if not pair_counts:
                print(f"[BPE] No more pairs at vocab size {256+merge_idx}. Stopping.")
                break
            
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            
            if best_count < 2:
                print(f"[BPE] Best pair count={best_count} at step {merge_idx}. Stopping early.")
                break
            
            new_idx = 256 + merge_idx
            p0, p1 = best_pair
            
            self.merges[best_pair] = new_idx
            self.vocab[new_idx] = self.vocab[p0] + self.vocab[p1]
            
            # Apply merge with incremental pair count updates
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
                    # Subtract old neighbouring pairs
                    if new_ids:
                        prev = new_ids[-1]
                        old_left = (prev, p0)
                        if old_left in pair_counts:
                            pair_counts[old_left] -= 1
                            if pair_counts[old_left] <= 0:
                                del pair_counts[old_left]
                    
                    if best_pair in pair_counts:
                        pair_counts[best_pair] -= 1
                        if pair_counts[best_pair] <= 0:
                            del pair_counts[best_pair]
                    
                    if i + 2 < len(ids):
                        nxt = ids[i+2]
                        old_right = (p1, nxt)
                        if old_right in pair_counts:
                            pair_counts[old_right] -= 1
                            if pair_counts[old_right] <= 0:
                                del pair_counts[old_right]
                    
                    # Add new neighbouring pairs
                    if new_ids:
                        new_left = (new_ids[-1], new_idx)
                        pair_counts[new_left] = pair_counts.get(new_left, 0) + 1
                    
                    new_ids.append(new_idx)
                    
                    if i + 2 < len(ids):
                        nxt = ids[i+2]
                        new_right = (new_idx, nxt)
                        pair_counts[new_right] = pair_counts.get(new_right, 0) + 1
                    
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            
            ids = new_ids
            pbar.update(1)
            if merge_idx % 500 == 0:
                pbar.set_postfix(vocab=256+merge_idx, seq_len=len(ids), top_freq=best_count)
        
        pbar.close()
        print(f"[BPE] Training complete. Final vocab size: {256 + len(self.merges)}, seq compressed to {len(ids)}")
        self._save(self.vocab_path)

    def _save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Custom BPE Merges v1\n")
            sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
            for (p0, p1), idx in sorted_merges:
                f.write(f"{p0} {p1} {idx}\n")

    def _load(self, path):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) != 3: continue
                p0, p1, idx = int(parts[0]), int(parts[1]), int(parts[2])
                self.merges[(p0, p1)] = idx
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
    
    def encode(self, s) -> List[int]:
        if isinstance(s, str):
            s = s.encode("utf-8")
        
        if isinstance(s, (bytes, bytearray)):
            ids = list(s)
        elif isinstance(s, list):
            ids = s
        else:
            raise ValueError("BPE encode expects str, bytes, or list[int]")

        while len(ids) >= 2:
            candidates = {}
            for pair in zip(ids, ids[1:]):
                if pair in self.merges:
                    candidates[pair] = self.merges[pair]
            
            if not candidates:
                break
            
            best_pair = min(candidates, key=candidates.get)
            ids = self._merge_ids(ids, best_pair, self.merges[best_pair])
            
        if self.line_mode:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        if self.line_mode:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        
        res = b""
        for i in ids:
            res += self.vocab.get(i, b"")
        return res.decode("utf-8", "ignore")

    @property
    def size(self):
        return self._size

    @property
    def tokens(self):
        return list(range(self._size))
class WordVocab(BaseVocab):
    """Whitespace-split word-level tokenizer (+ optional BOS)."""
    def __init__(self, lines: List[str], line_mode: bool):
        self.line_mode = line_mode
        words = set()
        for s in lines:
            words.update(s.split())
        self.tokens = sorted(list(words))
        if line_mode:
            if BOS_TOKEN in self.tokens:
                self.tokens.remove(BOS_TOKEN)
            self.tokens = [BOS_TOKEN] + self.tokens
        self.stoi = {w:i for i,w in enumerate(self.tokens)}
        self.itos = {i:w for w,i in self.stoi.items()}
        self.bos_id = self.stoi[BOS_TOKEN] if line_mode else None

    def encode(self, s: str) -> List[int]:
        if self.line_mode and s.startswith(BOS_TOKEN):
            # keep BOS as the very first token, then split the rest
            rest = s[len(BOS_TOKEN):].lstrip()
            return [self.bos_id] + [self.stoi[w] for w in rest.split()]
        else:
            return [self.stoi[w] for w in s.split()]

    def decode(self, ids: List[int]) -> str:
        if not self.line_mode:
            return " ".join(self.itos[i] for i in ids)
        out = []
        first = True
        for i in ids:
            if i == self.bos_id:
                # keep BOS literal; caller can hide it
                out.append(BOS_TOKEN)
            else:
                out.append(self.itos[i])
            first = False
        return " ".join(out)

    @property
    def size(self): return len(self.tokens)

class BinaryVocab(BaseVocab):
    """Binary tokens {0,1} (+ BOS if line mode)."""
    def __init__(self, line_mode: bool):
        self.line_mode = line_mode
        self.tokens = [BOS_TOKEN, "0", "1"] if line_mode else ["0", "1"]
        self.stoi = {t:i for i,t in enumerate(self.tokens)}
        self.itos = {i:t for t,i in self.stoi.items()}
        self.bos_id = self.stoi[BOS_TOKEN] if line_mode else None

    def encode(self, s: str) -> List[int]:
        # Keep only 0/1; ignore other characters
        ids = []
        if self.line_mode and s.startswith(BOS_TOKEN):
            ids.append(self.bos_id)
            s = s[len(BOS_TOKEN):]
        for ch in s:
            if ch in ("0","1"):
                ids.append(self.stoi[ch])
        return ids

    def decode(self, ids: List[int]) -> str:
        if not self.line_mode:
            return "".join(self.itos[i] for i in ids)
        return "".join(BOS_TOKEN if i==self.bos_id else self.itos[i] for i in ids)

    @property
    def size(self): return len(self.tokens)

HEX_RE = re.compile(r'^(?:0x)?[0-9a-fA-F]+(?:\s+(?:0x)?[0-9a-fA-F]+)*$')

def _hex_to_bytes(s: str) -> bytes:
    # normalize: remove whitespace, allow optional 0x prefixes
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'0x', '', s, flags=re.IGNORECASE)
    if len(s) % 2 == 1:
        s = '0' + s  # odd nibble → pad on the left
    return bytes.fromhex(s)

class ByteVocab(BaseVocab):
    """0..255 bytes (+ BOS if line mode)."""
    def __init__(self, line_mode: bool):
        self.line_mode = line_mode
        self.bos_id = 256 if line_mode else None
        self._size = 257 if line_mode else 256

    def encode(self, s) -> List[int]:
        """
        Accepts:
          - bytes/bytearray/memoryview → direct mapping
          - str → latin1 by default; BUT if it looks like hex (or starts with 'hex:'), parse as hex bytes
                   If line_mode and string starts with BOS_TOKEN, it's honored first.
          - int in [0,255] → single byte token
          - list/tuple of ints in [0,255] → sequence of byte tokens
        """
        # ints
        if isinstance(s, int):
            if not (0 <= s <= 255):
                raise ValueError("ByteVocab.encode int must be in [0,255]")
            ids = [s]
            return ([self.bos_id] + ids) if self.line_mode else ids

        # list/tuple of ints
        if isinstance(s, (list, tuple)) and all(isinstance(x, int) for x in s):
            ids = [x for x in s if 0 <= x <= 255]
            return ([self.bos_id] + ids) if self.line_mode else ids

        # bytes-like
        if isinstance(s, (bytes, bytearray, memoryview)):
            b = bytes(s)
            ids = list(b)
            return ([self.bos_id] + ids) if self.line_mode else ids

        # strings (latin1 vs hex)
        if isinstance(s, str):
            # Handle BOS_TOKEN literally at the front in line_mode
            add_bos = False
            if self.line_mode and s.startswith(BOS_TOKEN):
                s = s[len(BOS_TOKEN):]
                add_bos = True

            # explicit "hex:" prefix OR looks like hex (with optional 0x and spaces)
            looks_hex = s.lower().startswith("hex:") or bool(HEX_RE.match(s))
            if looks_hex:
                if s.lower().startswith("hex:"):
                    s = s[4:].lstrip()
                try:
                    b = _hex_to_bytes(s)
                except ValueError:
                    raise ValueError("Invalid hex string for ByteVocab.encode")
                ids = list(b)
                if self.line_mode and add_bos:
                    return [self.bos_id] + ids
                return ([self.bos_id] + ids) if (self.line_mode and not add_bos) else ids

            # fallback: latin1
            ids = list(s.encode("latin1", "ignore"))
            return ([self.bos_id] + ids) if self.line_mode else ids

        raise TypeError("ByteVocab.encode expects bytes, str, int, or list[int]")

    def decode(self, ids: List[int]) -> str:
        if self.line_mode:
            ids = [i for i in ids if i != self.bos_id]
        return bytes([i for i in ids if 0 <= i <= 255]).decode("latin1", "ignore")

    def to_bytes(self, ids: List[int]) -> bytes:
        if self.line_mode:
            ids = [i for i in ids if i != self.bos_id]
        return bytes([i for i in ids if 0 <= i <= 255])

    @property
    def size(self): 
        return self._size

    @property
    def tokens(self):
        # for random sampling in callers that expect a tokens list
        return list(range(257)) if self.line_mode else list(range(256))






# ========= Datasets =========
class ClassicCorpus:
    def __init__(self, text: str, vocab: CharVocab, seq_len: int):
        self.vocab = vocab
        self.ids = torch.tensor(vocab.encode(text), dtype=torch.long)
        self.seq_len = seq_len
    def get_batch(self, batch_size: int):
        L = len(self.ids) - (self.seq_len + 1)
        idx = torch.randint(0, max(1, L), (batch_size,))
        x = torch.stack([self.ids[i:i+self.seq_len] for i in idx])
        y = torch.stack([self.ids[i+1:i+self.seq_len+1] for i in idx])
        return x.to(DEVICE), y.to(DEVICE)

class LineDataset:
    def __init__(self, lines: List[str], vocab: BaseVocab):
        assert vocab.line_mode
        self.vocab = vocab
        # each line gets BOS ... EOS via vocab.encode
        enc = [vocab.encode(ln) for ln in lines]  # variable lengths, already BOS...EOS
        self.lines_enc = enc

        self.max_len = max(len(e) for e in enc)
        data = []
        for e in enc:
            pad_len = self.max_len - len(e)
            data.append(e + [vocab.pad_id]*pad_len)
        self.data = torch.tensor(data, dtype=torch.long)

    def get_batch(self, batch_size: int):
        idx = torch.randint(0, self.data.size(0), (batch_size,))
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        # Important: we will ignore PAD in loss via ignore_index
        return x.to(DEVICE), y.to(DEVICE)


class LineDatasetSubset(LineDataset):
    def __init__(self, parent: LineDataset, rows: torch.Tensor):
        self.vocab = parent.vocab
        self.lines_enc = [parent.lines_enc[i] for i in rows.tolist()]
        self.max_len = max(len(e) for e in self.lines_enc) if self.lines_enc else 1
        # rebuild padded tensor for the classic random batching path
        data = []
        for e in self.lines_enc:
            pad_len = self.max_len - len(e)
            pad_id = self.vocab.pad_id
            data.append(e + [pad_id] * pad_len)
        self.data = torch.tensor(data, dtype=torch.long) if data else torch.empty(0, 1, dtype=torch.long)
from linegenModel import *

class LineTBPTTStream:
    """
    Maintains B independent streams over a set of encoded lines (each already has BOS at index 0).
    Windowed TBPTT: returns (x, y, reset_mask) each step with shape (B, W).

    - All streams start at BOS on the first step.
    - When a stream would exceed the line (need W+1 tokens for y), we re-sample a new line and start at BOS,
      and mark reset_mask[b]=True for that step.
    """
    def __init__(self, lines_enc: List[List[int]], window: int, batch_size: int, bos_id: int):
        self.bos_id = bos_id
        self.W = max(1, int(window))
        self.B = int(batch_size)

        # Keep only lines with at least two tokens (BOS + 1 char)
        self.lines: List[torch.Tensor] = [torch.tensor(l, dtype=torch.long) for l in lines_enc if len(l) >= 2]
        if len(self.lines) == 0:
            # degenerate fallback: one fake line of just BOS BOS to avoid crashes
            self.lines = [torch.tensor([bos_id, bos_id], dtype=torch.long)]

        # Precompute eligible lines (len >= W+1 so we can form x and y of length W)
        self._refresh_eligible()

        # Per-stream (line_idx, pos)
        self.line_idx = torch.zeros(self.B, dtype=torch.long)
        self.pos = torch.zeros(self.B, dtype=torch.long)
        self._init_streams()

    def _refresh_eligible(self):
        need = self.W + 1
        self.eligible = [i for i, t in enumerate(self.lines) if t.numel() >= need]
        if not self.eligible:
            # If none satisfy current W, fall back to the longest line(s)
            maxL = max(t.numel() for t in self.lines)
            self.eligible = [i for i, t in enumerate(self.lines) if t.numel() == maxL]
            # and clamp W to maxL-1 to ensure windows exist
            self.W = max(1, maxL - 1)

    def _pick_line(self) -> int:
        return random.choice(self.eligible)

    def _init_streams(self):
        # All streams begin at BOS of a random eligible line
        for b in range(self.B):
            self.line_idx[b] = self._pick_line()
            self.pos[b] = 0

    def get_next(self, device):
        B = self.B; W = self.W
        x = torch.empty(B, W, dtype=torch.long)
        y = torch.empty(B, W, dtype=torch.long)
        reset = torch.zeros(B, dtype=torch.bool)

        for b in range(B):
            li = int(self.line_idx[b].item())
            line = self.lines[li]
            L = line.numel()
            p = int(self.pos[b].item())

            # If we can't take a full W+1 from current position, reset to BOS of a new line
            if p + W >= L:
                li = self._pick_line()
                line = self.lines[li]; L = line.numel()
                p = 0
                self.line_idx[b] = li
                self.pos[b] = 0
                reset[b] = True

            # Slice window
            x[b] = line[p : p + W]
            y[b] = line[p + 1 : p + W + 1]

            # Advance
            self.pos[b] = p + W

        return x.to(device), y.to(device), reset.to(device)


# ==================================================
class MemmapClassicDataset:
    def __init__(self, txt_path: str, vocab: BaseVocab, seq_len: int, split_range=(0.0, 1.0)):
        self.seq_len = seq_len
        self.vocab = vocab
        self.bin_path = str(pathlib.Path(txt_path).with_suffix(".bin"))
        
        if not os.path.exists(self.bin_path):
            print(f"[Dataset] Pre-tokenizing {txt_path} -> {self.bin_path} ...")
            self._tokenize_and_save(txt_path)
        else:
            print(f"[Dataset] Found existing {self.bin_path}, loading...")

        self.dtype = np.uint16 if vocab.size < 65535 else np.int32
        self.full_data = np.memmap(self.bin_path, dtype=self.dtype, mode='r')
        
        # === NEW: Slice the memmap logically ===
        total_len = len(self.full_data)
        start_pct, end_pct = split_range
        self.start_idx = int(start_pct * total_len)
        self.end_idx = int(end_pct * total_len)
        
        # Safety: ensure we have at least one sequence
        if self.end_idx - self.start_idx <= seq_len + 1:
            print(f"[Dataset] Warning: Split {split_range} is too small! Using full.")
            self.start_idx = 0
            self.end_idx = total_len
            
        print(f"[Dataset] Loaded segment {split_range} ({readable_num(self.end_idx - self.start_idx)} tokens).")

    # ... (keep _tokenize_and_save exactly as it was) ...
    def _tokenize_and_save(self, txt_path):
        # [Use your existing code here, no changes needed]
        # Just ensure you copy the method from your file into this class
        dtype = np.uint16 if self.vocab.size < 65535 else np.int32
        file_size = os.path.getsize(txt_path)
        temp_path = self.bin_path + ".tmp"
        CHUNK_SIZE = 1024 * 1024 
        
        with open(temp_path, "wb") as f_out:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f_in:
                with tqdm(total=file_size, unit="B", unit_scale=True, desc="Tokenizing") as pbar:
                    buffer = ""
                    while True:
                        chunk = f_in.read(CHUNK_SIZE)
                        if not chunk:
                            if buffer: 
                                ids = self.vocab.encode(buffer)
                                f_out.write(np.array(ids, dtype=dtype).tobytes())
                            break
                        buffer += chunk
                        pbar.update(len(chunk.encode('utf-8')))
                        
                        if hasattr(self.vocab, "enc") or isinstance(self.vocab, TiktokenVocab): 
                            last_nl = buffer.rfind('\n')
                            if last_nl != -1:
                                to_process = buffer[:last_nl+1]
                                buffer = buffer[last_nl+1:]
                                ids = self.vocab.encode(to_process)
                                f_out.write(np.array(ids, dtype=dtype).tobytes())
                            elif len(buffer) > 10 * CHUNK_SIZE:
                                ids = self.vocab.encode(buffer)
                                f_out.write(np.array(ids, dtype=dtype).tobytes())
                                buffer = ""
                        else:
                            ids = self.vocab.encode(buffer)
                            f_out.write(np.array(ids, dtype=dtype).tobytes())
                            buffer = ""

        if os.path.exists(self.bin_path): os.remove(self.bin_path)
        os.rename(temp_path, self.bin_path)

    def get_batch(self, batch_size: int):
        # Effective length of OUR slice
        slice_len = self.end_idx - self.start_idx
        high = slice_len - self.seq_len - 1
        
        if high <= 0: 
            return torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=DEVICE), \
                   torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=DEVICE)
        
        # Pick random offsets relative to start_idx
        ix = torch.randint(0, high, (batch_size,)) + self.start_idx
        
        x_list = []
        y_list = []
        for i in ix:
            i_int = int(i)
            # Read from the memmap
            chunk = np.array(self.full_data[i_int : i_int + self.seq_len + 1], dtype=np.int64)
            x_list.append(torch.from_numpy(chunk[:-1]))
            y_list.append(torch.from_numpy(chunk[1:]))
            
        x = torch.stack(x_list).to(DEVICE)
        y = torch.stack(y_list).to(DEVICE)
        return x, y
    
    @property
    def ids(self):
        # Return only the slice if requested via property (for TBPTT)
        # Fix: Cast to int64 because torch.from_numpy doesn't support uint16
        chunk = self.full_data[self.start_idx : self.end_idx]
        return torch.from_numpy(chunk.astype(np.int64))

class IndexedLineDataset:
    """
    Efficient Line-based dataset that indexes file offsets instead of loading lines.
    Allows random access to 10GB+ line-based files with minimal RAM.
    """
    def __init__(self, txt_path: str, vocab: BaseVocab):
        self.path = txt_path
        self.vocab = vocab
        # FIX: Explicitly add .npy so numpy doesn't silently append it later
        self.index_path = str(pathlib.Path(txt_path).with_suffix(".idx.npy"))
        
        if not os.path.exists(self.index_path):
            print(f"[Dataset] Indexing lines in {txt_path} ...")
            self._build_index()
        else:
            print(f"[Dataset] Loading line index {self.index_path} ...")
        
        self.offsets = np.load(self.index_path) 
        print(f"[Dataset] Indexed {readable_num(len(self.offsets))} lines.")
        self.max_len = int(np.max(self.offsets[:, 1])) if len(self.offsets) > 0 else 0
        self.f = open(self.path, "rb")

    def _build_index(self):
        offsets = []
        cur_offset = 0
        seen_hashes = set()
        
        with open(self.path, "rb") as f:
            # We use tqdm for progress
            for line in tqdm(f, desc="Indexing"):
                length = len(line)
                
                # Strip for content check (handles \r\n, \n, whitespace)
                content = line.strip()
                
                # 1. Skip empty lines
                if len(content) == 0:
                    cur_offset += length
                    continue
                
                # 2. Skip duplicates using hash
                h = hash(content)
                if h in seen_hashes:
                    cur_offset += length
                    continue
                
                seen_hashes.add(h)
                offsets.append((cur_offset, length))
                cur_offset += length

        np.save(self.index_path, np.array(offsets, dtype=np.int64))

    def get_batch(self, batch_size: int):
        idx = np.random.randint(0, len(self.offsets), size=batch_size)
        batch_tokens = []
        max_batch_len = 0
        
        for i in idx:
            off, length = self.offsets[i]
            self.f.seek(off)
            line_bytes = self.f.read(length)
            
            if isinstance(self.vocab, ByteVocab):
                line_content = line_bytes.rstrip(b'\n\r')
                enc = self.vocab.encode(line_content)
            else:
                line_content = line_bytes.decode("utf-8", "ignore").rstrip('\n\r')
                enc = self.vocab.encode(line_content)
                
            batch_tokens.append(enc)
            max_batch_len = max(max_batch_len, len(enc))
            
        pad_id = self.vocab.pad_id if self.vocab.pad_id is not None else 0
        x_tensor = torch.full((batch_size, max_batch_len - 1), pad_id, dtype=torch.long)
        y_tensor = torch.full((batch_size, max_batch_len - 1), pad_id, dtype=torch.long)
        
        for i, seq in enumerate(batch_tokens):
            if len(seq) < 2: continue 
            seq_t = torch.tensor(seq, dtype=torch.long)
            l = len(seq) - 1
            x_tensor[i, :l] = seq_t[:-1]
            y_tensor[i, :l] = seq_t[1:]
            
        return x_tensor.to(DEVICE), y_tensor.to(DEVICE)
        
    def close(self):
        self.f.close()

class IndexedLineDatasetSubset(IndexedLineDataset):
    def __init__(self, parent: IndexedLineDataset, indices: np.ndarray):
        self.path = parent.path
        self.vocab = parent.vocab
        self.f = open(self.path, "rb") 
        self.offsets = parent.offsets[indices]
        self.max_len = parent.max_len



# ========= Model selection menu (with separators) =========
MODEL_MENU = """
# ==== MLPs ====
0 - MLP (basic)
1 - MLP (residual)

# ==== RNNs ====
2 - RNN (tanh)
3 - RNNReLU
4 - GRU
5 - LSTM
6 - IndRNN
7 - IndyGRU
8 - ATanULSTM
18 - JANET (forget-gate LSTM)
23 - Liquid Neural Network (LTC)

# ==== Non-recurrents ====
9  - Temporal ConvNet
10 - GPT transformer
19 - HyperMixer
21 - gMLP
22 - aMLP
24 - MLPMixer (Causal)
25 - Modern Transformer (Llama3-style)
31 - MEGABYTE
34 - KAN-Transformer (Chebyshev)

# ==== xLSTM ====
11 - xLSTM (sLSTM only)
12 - xLSTM (mLSTM only)
13 - xLSTM (mixed m:s)

# ==== Space State Machines / Linear Recurrence ====
14 - Mamba (selective scan)
15 - minGRU (scan)
16 - minLSTM (scan)
17 - RWKV (scan)
20 - GateLoop (scan)
26 - MinRNN (scan - Multi-Act)
27 - Griffin (RG-LRU)
28 - DeltaNet
29 - RetNet
30 - HGRN
32 - MinIndRNN (scan)
33 - MinJANET (scan)
35 - Linear Transformer (Recurrent)
36 - H3 (Hungry Hungry Hippos)
37 - DCT-Former
38 - MinIndyGRU (scan)
39 - MinIndyLSTM (scan)
"""

# Update set of Scan models
ACT_MENU = activation_menu_text()
ACT_NAMES = activation_names()

RNN_MODEL_IDS = {2,3,4,5,6,7,8,11,12,13,14,15,16,17,18,20,23,26,27,28,29,30,32,33,35,36,38,39}
# Add 35 and 36 to this set
SCAN_MODEL_IDS = {14, 15, 16, 17, 20, 26, 27, 28, 29, 30, 32, 33, 35, 36, 38, 39}
# Models that actually use head_count (attention heads or xLSTM heads)
ATTN_MODEL_IDS = {10, 11, 12, 13, 19, 25, 29}
class TBPTTClassicStream:
    """
    Streaming TBPTT over a single long 1D tensor `ids`.
    Guarantees every (x,y) window has length exactly W, by choosing starts that have >= W+1 tokens left.
    One stream (b=0) always starts at position 0.
    """
    def __init__(self, ids: torch.Tensor, window: int, batch_size: int, total_len: int):
        assert ids.dim() == 1, "ids must be 1D"
        self.ids = ids
        self.N = ids.numel()
        self.W = max(1, int(window))
        self.B = int(batch_size)
        self.total_len = max(0, int(total_len))

        # If file is too short, clamp W so we can form (x,y)
        if self.N < self.W + 1:
            self.W = max(1, self.N - 1)

        # If a user provided a truncated segment smaller than W+1, clamp W
        if 0 < self.total_len < (self.W + 1):
            self.W = max(1, self.total_len - 1)

        # Per-stream position and segment end (exclusive)
        self.pos = torch.zeros(self.B, dtype=torch.long)
        self.seg_end = torch.zeros(self.B, dtype=torch.long)
        self._init_streams()

    def _new_start(self):
        """
        Choose a (start,end) such that end-start >= W+1.
        If total_len>0, we honor it (and we already clamped W accordingly).
        Otherwise choose start uniformly in [0, N-(W+1)] and end=N.
        """
        need = self.W + 1
        if self.total_len > 0:
            # segment length is fixed to total_len (>= need due to clamping)
            start_max = max(0, self.N - self.total_len)
            start = random.randrange(0, start_max + 1) if start_max > 0 else 0
            end = min(start + self.total_len, self.N)
        else:
            if self.N <= need:
                # degenerate but safe: whole file acts as a single window, W already clamped
                return 0, self.N
            start_max = self.N - need
            start = random.randrange(0, start_max + 1)
            end = self.N
        return start, end

    def _init_streams(self):
        # stream 0 starts at 0; ensure its segment has at least W+1 tokens
        self.pos[0] = 0
        if self.total_len > 0:
            self.seg_end[0] = min(self.total_len, self.N)
        else:
            self.seg_end[0] = self.N
        # if even stream 0's segment is too short, widen it safely
        if int(self.seg_end[0].item()) - int(self.pos[0].item()) < (self.W + 1):
            self.seg_end[0] = min(self.pos[0] + self.W + 1, self.N)

        # others random valid segments
        for b in range(1, self.B):
            s, e = self._new_start()
            self.pos[b] = s
            self.seg_end[b] = e

    def get_next(self, device):
        B, W = self.B, self.W
        x = torch.empty(B, W, dtype=torch.long)
        y = torch.empty(B, W, dtype=torch.long)
        reset = torch.zeros(B, dtype=torch.bool)

        need = W + 1
        for b in range(B):
            p = int(self.pos[b].item())
            e = int(self.seg_end[b].item())

            # If not enough room for a full window, resample a fresh valid segment and mark reset
            if (e - p) < need or (self.N - p) < need:
                s, ee = self._new_start()
                self.pos[b] = s
                self.seg_end[b] = ee
                p, e = s, ee
                reset[b] = True

            # Now guaranteed: e - p >= need
            x[b] = self.ids[p : p + W]
            y[b] = self.ids[p + 1 : p + W + 1]
            self.pos[b] = p + W

            # If we exactly hit the boundary, next step will have to reset
            if (e - int(self.pos[b].item())) < need:
                reset[b] = True

        return x.to(device), y.to(device), reset.to(device)

def reset_rnn_state(state, reset_mask, model, msel):
    """Zero the hidden states for batch indices where reset_mask==True."""
    if state is None or reset_mask is None or reset_mask.numel() == 0:
        return state

    # --- BuiltinRNNWrapper: stacked 1-layer cores ---
    # GRU/RNN: state is List[Tensor] where each Tensor is (1, B, H)
    # LSTM:    state is List[Tuple[Tensor, Tensor]] where each is (1, B, H)
    if isinstance(model, BuiltinRNNWrapper):
        if model.mode == 'lstm':
            new_state = []
            for (h, c) in state:
                # shapes: (1, B, H); batch dimension is 1
                h[:, reset_mask, :] = 0
                c[:, reset_mask, :] = 0
                new_state.append((h, c))
            return new_state
        else:
            new_state = []
            for h in state:
                # shape: (1, B, H); batch dimension is 1
                h[:, reset_mask, :] = 0
                new_state.append(h)
            return new_state

    # --- CustomRNNWrapper with IndRNN/IndyGRU/JANET/LiquidRNN/ExtATanULSTM ---
    if isinstance(model, CustomRNNWrapper):
        core = model.rnn
        # ExtATanULSTM returns (hn, cn) tuple of (num_layers, B, H)
        if isinstance(core, ExtATanULSTM):
            hn, cn = state
            hn[:, reset_mask, :] = 0
            cn[:, reset_mask, :] = 0
            return (hn, cn)
        # IndRNN, IndyGRU, JANET, LiquidRNN all return stacked (num_layers, B, H)
        if state is not None:
            state[:, reset_mask, :] = 0
        return state

    # --- xLSTM: list of dict states (per block) ---
    if isinstance(model, XlstmLM):
        if state is None: return None
        new_state = []
        for st in state:
            if st is None:
                new_state.append(None); continue
            st2 = {}
            for k,v in st.items():
                if v is None:
                    st2[k] = None
                elif torch.is_tensor(v) and v.dim() >= 2:
                    vv = v.clone()
                    # batch is dim 0 for these states
                    vv[reset_mask] = 0
                    st2[k] = vv
                else:
                    st2[k] = v
            new_state.append(st2)
        return new_state

    # --- ScanLM: dict/Tensor per block (batch is dim 0) ---
    if isinstance(model, ScanLM):
        if state is None:
            return None
        new_state = []
        for st in state:
            if st is None:
                new_state.append(None); continue
            if isinstance(st, dict):
                st2 = {}
                for k, v in st.items():
                    if torch.is_tensor(v) and v.dim() >= 2:
                        vv = v.clone()
                        vv[reset_mask] = 0
                        st2[k] = vv
                    else:
                        st2[k] = v
                new_state.append(st2)
            elif torch.is_tensor(st):
                st2 = st.clone()
                st2[reset_mask] = 0
                new_state.append(st2)
            else:
                new_state.append(st)
        return new_state

    # Unknown model: best-effort recursive zero on tensors assuming batch is first dim
    def _zero_any(x):
        if torch.is_tensor(x):
            if x.dim() >= 2:
                try:
                    x[reset_mask] = 0
                except Exception:
                    # fallback if batch is not leading dim
                    if x.dim() >= 3 and x.size(0) == 1:
                        x[:, reset_mask, :] = 0
            return x
        if isinstance(x, (list, tuple)):
            xs = [_zero_any(t) for t in x]
            return tuple(xs) if isinstance(x, tuple) else xs
        if isinstance(x, dict):
            return {k: _zero_any(v) for k, v in x.items()}
        return x

    return _zero_any(state)

def detach_state(state):
    """Detach hidden state from autograd graph (handles Tensor, tuple, list, dict, nested)."""
    if state is None:
        return None
    if torch.is_tensor(state):
        return state.detach()
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        items = [detach_state(s) for s in state]
        return tuple(items) if isinstance(state, tuple) else items
    return state

def build_model(cfg, vocab_size):
    msel = cfg["model_selection"]
    embed = cfg["embed_dim"]
    layers = cfg["layer_count"]
    seq_len = cfg["seq_len"]
    heads = cfg.get("head_count", 4)
    RNN_MAP = {
        6: "indrnn",
        7: "indygru",
        18: "janet",
        8: "atanulstm"
    }
    if msel in RNN_MAP:
        cell_type_str = RNN_MAP[msel]
        # Ensure vocab_size and embed are passed correctly
        return CustomRNNWrapper(cell_type_str, vocab_size, embed, cfg["layer_count"]).to(DEVICE)

    if msel == 0:
        # Basic MLP now uses rolling one-hot window input (MLPOG-style), needs seq_len
        return OneHotWindowMLPClassifier(
            vocab_size=vocab_size,
            seq_len=cfg["seq_len"],
            embed_dim=embed,
            n_layers=layers,
            act_name=cfg["activation_name"]
        ).to(DEVICE)

    if msel == 1:
        return ResidualMLPClassifier(vocab_size, embed, layers, cfg["activation_name"]).to(DEVICE)

    if msel == 2:
        return BuiltinRNNWrapper(
            vocab_size, embed, layers, 'rnn_tanh',
            tie_weights=bool(cfg.get("tie_weights", True)),
            use_norm=int(cfg.get("use_norm", 0)),
            res_every=int(cfg.get("res_every", 0)),
            res_type=int(cfg.get("res_type", 0)),
            dropout=float(cfg.get("dropout", 0.0)),
            use_multiplier=int(cfg.get("use_multiplier", 0))
        ).to(DEVICE)

    if msel == 3:
        return BuiltinRNNWrapper(
            vocab_size, embed, layers, 'rnn_relu',
            tie_weights=bool(cfg.get("tie_weights", True)),
            use_norm=int(cfg.get("use_norm", 0)),
            res_every=int(cfg.get("res_every", 0)),
            res_type=int(cfg.get("res_type", 0)),
            dropout=float(cfg.get("dropout", 0.0)),
            use_multiplier=int(cfg.get("use_multiplier", 0))
        ).to(DEVICE)

    if msel == 4:
        return BuiltinRNNWrapper(
            vocab_size, embed, layers, 'gru',
            tie_weights=bool(cfg.get("tie_weights", True)),
            use_norm=int(cfg.get("use_norm", 0)),
            res_every=int(cfg.get("res_every", 0)),
            res_type=int(cfg.get("res_type", 0)),
            dropout=float(cfg.get("dropout", 0.0)),
            use_multiplier=int(cfg.get("use_multiplier", 0))
        ).to(DEVICE)

    if msel == 5:
        return BuiltinRNNWrapper(
            vocab_size, embed, layers, 'lstm',
            tie_weights=bool(cfg.get("tie_weights", True)),
            use_norm=int(cfg.get("use_norm", 0)),
            res_every=int(cfg.get("res_every", 0)),
            res_type=int(cfg.get("res_type", 0)),
            dropout=float(cfg.get("dropout", 0.0)),
            use_multiplier=int(cfg.get("use_multiplier", 0))
        ).to(DEVICE)

    if msel == 9:
        return TemporalConvNet(vocab_size, embed, layers, act_name=cfg["activation_name"], k=3).to(DEVICE)
    if msel == 10:
        # GPT-2 style decoder-only LM
        cfg_gpt2 = GPT2Config(
            vocab_size=vocab_size,
            d_model=embed,
            n_layers=layers,
            n_heads=heads,
            max_seq_len=seq_len,
            ff_mult=int(cfg.get("ff_mult", 4)),
            dropout=float(cfg.get("dropout", 0.0)),
            attn_dropout=float(cfg.get("attn_dropout", cfg.get("dropout", 0.0))),
            bias=bool(cfg.get("bias", True)),
            tie_weights=bool(cfg.get("tie_weights", True)),
            use_flash=bool(cfg.get("use_flash", True)),
        )
        act_name = cfg.get("activation_name", "gelu")
        return GPT2ForLM(cfg_gpt2, act_name=act_name).to(DEVICE)
    if msel in (11,12,13):
        # Heads = cfg['head_count']; act = cfg['activation_name']
        # Mixed ratio (a:b) taken from cfg or defaults to 7:1
        a = int(cfg.get("xlstm_m_blocks", 7))
        b = int(cfg.get("xlstm_s_blocks", 1))
        kind = "s" if msel==11 else ("m" if msel==12 else "mix")
        return XlstmLM(
            vocab_size=vocab_size,
            dim=embed,
            n_blocks=layers,
            num_heads=heads,
            act_name=cfg["activation_name"],
            kind=kind,
            m_to_s=(a,b),
            up_mult_m=cfg.get("xlstm_m_up_mult", 2.0),
        ).to(DEVICE)
    if msel == 14:  # Mamba (selective scan)
        return ScanLM(
            vocab_size=vocab_size,
            dim=embed,
            kind="mamba",
            n_blocks=cfg["layer_count"],
        ).to(DEVICE)

    if msel == 15:  # minGRU (scan)
        return ScanLM(
            vocab_size=vocab_size,
            dim=embed,
            kind="mingru",
            n_blocks=cfg["layer_count"],
        ).to(DEVICE)

    if msel == 16:  # minLSTM (scan)
        return ScanLM(
            vocab_size=vocab_size,
            dim=embed,
            kind="minlstm",
            n_blocks=cfg["layer_count"],
        ).to(DEVICE)
    if msel == 17:  # RWKV (scan)
        return ScanLM(
            vocab_size=vocab_size,
            dim=embed,
            kind="rwkv",
            n_blocks=cfg["layer_count"],
        ).to(DEVICE)
    if msel == 19:
        return HyperMixerLM(
            vocab_size=vocab_size,
            d_model=embed,
            n_layers=layers,
            d_hidden=int(cfg.get("hm_hidden", embed)),
            d_ff=int(cfg.get("hm_ff", 4*embed)),
            act_name=cfg.get("activation_name", "gelu"),
            max_seq_len=int(cfg.get("seq_len", 65536)),
            tie_hyper=bool(cfg.get("hm_tie", True)),
            dropout=float(cfg.get("dropout", 0.0)),
            n_heads=int(cfg.get("head_count", 4)),
            causal=bool(cfg.get("causal", True)),
        ).to(DEVICE)
    if msel == 20:  # GateLoop (scan)
        return ScanLM(
            vocab_size=vocab_size,
            dim=embed,
            kind="gateloop",
            n_blocks=cfg["layer_count"],
        ).to(DEVICE)
    if msel == 21:  # gMLP
        return gMLPLanguageModel(vocab_size, embed, layers, embed*4, seq_len).to(DEVICE)
    if msel == 22:  # aMLP
        return aMLPLanguageModel(vocab_size, embed, layers, embed*4, seq_len, d_attn=64).to(DEVICE)
    if msel == 23: # Liquid
        return CustomRNNWrapper("liquid", vocab_size, embed, layers).to(DEVICE)

    if msel == 24: # Causal MLPMixer
        return CausalMLPMixer(vocab_size, embed, layers, seq_len).to(DEVICE)
    
    if msel == 25: # Modern Transformer
        return ModernTransformer(vocab_size, embed, layers, heads).to(DEVICE)
    if msel == 27: # Griffin
        return GriffinLM(vocab_size, embed, layers).to(DEVICE)

    if msel == 28: # DeltaNet
        return DeltaNetLM(vocab_size, embed, layers).to(DEVICE)

    if msel == 29: # RetNet
        return RetNetLM(vocab_size, embed, layers, heads).to(DEVICE)

    if msel == 30: # HGRN
        return HGRN_LM(vocab_size, embed, layers).to(DEVICE)
        
    if msel == 31: # MEGABYTE
        return MegaByteLM(vocab_size, embed, layers, patch_size=4).to(DEVICE)
    if msel == 26: # MinRNN (Generalized)
        return ScanLM(vocab_size, embed, kind="minrnn", n_blocks=layers, minrnn_act=cfg.get("minrnn_act", 0)).to(DEVICE)

    if msel == 32: # MinIndRNN
        return ScanLM(vocab_size, embed, kind="minindrnn", n_blocks=layers, minrnn_act=cfg.get("minrnn_act", 0)).to(DEVICE)

    if msel == 33: # MinJANET
        return ScanLM(vocab_size, embed, kind="minjanet", n_blocks=layers).to(DEVICE)

    if msel == 34: # KAN-Transformer
        return KAN_LM(vocab_size, embed, layers).to(DEVICE)

    if msel == 35: # Linear Transformer
        return LinearTransformerLM(vocab_size, embed, layers).to(DEVICE)

    if msel == 36: # H3
        return H3LM(vocab_size, embed, layers).to(DEVICE)

    if msel == 37: # DCT-Former
        return DCTFormerLM(vocab_size, embed, layers, seq_len).to(DEVICE)
    
    if msel == 38: # MinIndyGRU
        return ScanLM(vocab_size, embed, kind="minindygru", n_blocks=layers).to(DEVICE)

    if msel == 39: # MinIndyLSTM
        return ScanLM(vocab_size, embed, kind="minindylstm", n_blocks=layers).to(DEVICE)


    raise ValueError("Bad model selection")

@torch.no_grad()
def sample_step(logits, temperature=1.0, top_k=0, top_p=0.0,
                repetition_penalty=1.0, last_tokens=None):
    """
    Advanced sampling with top-k, top-p (nucleus), and repetition penalty.
    """
    if temperature <= 0: 
        return torch.argmax(logits, dim=-1)
    
    # Apply repetition penalty
    if repetition_penalty != 1.0 and last_tokens is not None and len(last_tokens) > 0:
        penalty_ids = torch.tensor(last_tokens, dtype=torch.long, device=logits.device)
        if logits.dim() == 1:
            for pid in penalty_ids:
                if logits[pid] > 0:
                    logits[pid] /= repetition_penalty
                else:
                    logits[pid] *= repetition_penalty
        else:
            for pid in penalty_ids:
                mask_pos = logits[:, pid] > 0
                logits[:, pid] = torch.where(mask_pos, logits[:, pid] / repetition_penalty,
                                              logits[:, pid] * repetition_penalty)
    
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        v = logits.size(-1)
        top_k = min(top_k, v)
        topk_vals, _ = torch.topk(logits, top_k, dim=-1)
        threshold = topk_vals[..., -1:]
        logits = torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)
    
    # Top-p (nucleus) filtering
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float('-inf')
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
    
    probs = F.softmax(logits, dim=-1)
    
    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
        vocab_size = logits.size(-1)
        if logits.dim() > 1:
            return torch.randint(0, vocab_size, (logits.size(0),), device=logits.device)
        else:
            return torch.randint(0, vocab_size, (1,), device=logits.device).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)
@torch.no_grad()
def _init_scan_state_from_prompt(scan_model: ScanLM, idx_prompt: torch.Tensor):
    """
    Given a full prompt (B=1, T>=1), run the blocks' parallel path to
    initialize per-block states, then return that state list.
    """
    # FIX: Apply the embedding LayerNorm!
    # Without this, the hidden states are initialized with un-normalized embedding magnitudes,
    # causing immediate distribution shift and garbage output.
    x = scan_model.emb_ln(scan_model.embed(idx_prompt))
    
    states = []
    for b in scan_model.blocks:
        if isinstance(b, (ScanBlock_Mamba, RWKVBlock)):
            x, st = b.forward_seq(x, state=None)
        else:
            x, st_last = b.forward_seq(x, h0=None)  # st_last: (B,D)
            st = st_last
        states.append(st)
    return states

@torch.no_grad()
def generate_classic(model, cfg, vocab: CharVocab, prompt_ids: List[int], max_len: int, stream=True):
    msel = cfg["model_selection"]
    seq_len = cfg["seq_len"]
    model.eval()
    out_ids = list(prompt_ids)

    # === RNN / SCAN PATH ===
    if msel in SCAN_MODEL_IDS or msel in RNN_MODEL_IDS:
        # [Logic preserved from your provided file, ensuring robustness]
        state = None
        
        # 1. Warmup / Init State
        # If we have a prompt, we scan all tokens *except* the last one to build state.
        if len(prompt_ids) > 1:
            # Context is everything up to the last token
            ctx_ids = prompt_ids[:-1]
            x_ctx = torch.tensor([ctx_ids], dtype=torch.long, device=DEVICE)
            
            if msel in SCAN_MODEL_IDS:
                state = _init_scan_state_from_prompt(model, x_ctx)
            else:
                _, state = model(x_ctx, None)
            
            cur = prompt_ids[-1]
        elif len(prompt_ids) == 1:
            cur = prompt_ids[0]
        else:
            cur = random.randrange(vocab.size)
            # If prompt was empty, we generated a token, so we should add it to out_ids
            if len(out_ids) == 0: out_ids.append(cur)

        # 2. Generation Loop
        for _ in range(max_len):
            x = torch.tensor([[cur]], dtype=torch.long, device=DEVICE)
            logits, state = model(x, state)
            nxt = sample_step(logits[:, -1, :], cfg.get("temperature", 1.0), top_k=cfg.get("_top_k", 0), top_p=cfg.get("_top_p", 0.0), repetition_penalty=cfg.get("_rep_penalty", 1.0)).item()
            out_ids.append(nxt)
            if stream:
                sys.stdout.write(vocab.decode([nxt])); sys.stdout.flush()
            cur = nxt

    # === SLIDING WINDOW PATH (MLP / Transformers / Mixers) ===
    else:
        # 1. Initialize Context
        # Ensure we have a valid starting context window
        if len(prompt_ids) > 0:
            cur_ctx = prompt_ids[-seq_len:]
        else:
            # If empty prompt, seed with random token
            start_token = random.randrange(vocab.size)
            cur_ctx = [start_token]
            out_ids.append(start_token)
            if stream:
                sys.stdout.write(vocab.decode([start_token])); sys.stdout.flush()

        # 2. Generation Loop
        for _ in range(max_len):
            # Ensure context doesn't exceed seq_len (safety clip)
            cur_ctx = cur_ctx[-seq_len:]
            
            x = torch.tensor([cur_ctx], dtype=torch.long, device=DEVICE)
            logits = model(x)
            
            # Sample from the last position
            nxt = sample_step(logits[:, -1, :], cfg.get("temperature", 1.0), top_k=cfg.get("_top_k", 0), top_p=cfg.get("_top_p", 0.0), repetition_penalty=cfg.get("_rep_penalty", 1.0)).item()
            out_ids.append(nxt)
            
            if stream:
                sys.stdout.write(vocab.decode([nxt])); sys.stdout.flush()
            
            # Slide window
            cur_ctx.append(nxt)

    if stream: print()
    return out_ids


@torch.no_grad()
def generate_line_mode(model, cfg, vocab: BaseVocab, prompt_ids: List[int], limit_len: int):
    msel = cfg["model_selection"]
    seq_len = cfg["seq_len"]
    bos = vocab.bos_id
    eos = getattr(vocab, "eos_id", None)

    model.eval()
    out_ids = list(prompt_ids)
    stop = False

    # 1. Prepare Priming Sequence
    # In line mode, we generally want to start from BOS if input is empty.
    # We also strip any trailing EOS from the input so the model can continue generating.
    priming = list(prompt_ids)
    if eos is not None and priming and priming[-1] == eos:
        priming = priming[:-1]
    
    # === RNN / SCAN PATH ===
    if msel in SCAN_MODEL_IDS or msel in RNN_MODEL_IDS:
        state = None
        cur = priming[-1] if priming else bos
        
        # Determine history for state initialization
        # If we have [BOS, 'H', 'e'], we run state on [BOS, 'H'] and feed 'e' as cur.
        history = priming[:-1] if len(priming) > 0 else []
        
        # If prompt was totally empty, cur=BOS. History empty. 
        # If prompt was just BOS, cur=BOS. History empty. (Assuming BOS isn't duplicated)
        
        if len(history) > 0:
            x_ctx = torch.tensor([history], dtype=torch.long, device=DEVICE)
            if msel in SCAN_MODEL_IDS:
                state = _init_scan_state_from_prompt(model, x_ctx)
            else:
                _, state = model(x_ctx, None)

        for _ in range(limit_len):
            x = torch.tensor([[cur]], dtype=torch.long, device=DEVICE)
            logits, state = model(x, state)
            nxt = sample_step(logits[:, -1, :], cfg.get("temperature", 1.0), top_k=cfg.get("_top_k", 0), top_p=cfg.get("_top_p", 0.0), repetition_penalty=cfg.get("_rep_penalty", 1.0)).item()
            out_ids.append(nxt)
            cur = nxt
            if eos is not None and nxt == eos:
                stop = True; break

    # === SLIDING WINDOW PATH (MLP / Transformers / Mixers) ===
    else:
        # 1. Initialize Context
        # If priming exists, take the last seq_len tokens.
        # If priming is empty (empty prompt), start with [BOS].
        if len(priming) > 0:
            cur_ctx = priming[-seq_len:]
        else:
            cur_ctx = [bos]
            # Note: We don't append BOS to out_ids here because usually 
            # prompt_ids already contained it, or the caller handles it.
            # If prompt_ids was truly empty, we might want to ensure BOS is in output,
            # but usually line mode prompts start with BOS.

        for _ in range(limit_len):
            # Ensure context doesn't exceed seq_len
            cur_ctx = cur_ctx[-seq_len:]
            
            x = torch.tensor([cur_ctx], dtype=torch.long, device=DEVICE)
            logits = model(x)
            
            # Sample
            nxt = sample_step(logits[:, -1, :], cfg.get("temperature", 1.0), top_k=cfg.get("_top_k", 0), top_p=cfg.get("_top_p", 0.0), repetition_penalty=cfg.get("_rep_penalty", 1.0)).item()
            out_ids.append(nxt)
            
            # Slide window
            cur_ctx.append(nxt)
            
            if eos is not None and nxt == eos:
                stop = True; break

    # Remove the trailing EOS from the result if present (optional, standardizes output)
    if stop and len(out_ids) > 0 and eos is not None and out_ids[-1] == eos:
        out_ids = out_ids[:-1]
        
    return out_ids

# ========= Training =========
def train_loop(cfg, model, optimizer, dataset, valid_ds, vocab, line_mode):
    iters = 0; total_tokens = 0; last_log = time.time(); losses = []
    pad_id = None
    if hasattr(vocab, "pad_id"): pad_id = vocab.pad_id
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
    is_scan = cfg["model_selection"] in SCAN_MODEL_IDS

    # ---- Advanced training features ----
    use_amp = bool(cfg.get("use_amp", False)) and DEVICE == "cuda"
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    log_interval = int(cfg.get("log_interval", 50))
    sample_interval = int(cfg.get("sample_interval", 500))
    val_interval = int(cfg.get("val_interval", 500))
    save_interval = int(cfg.get("save_interval", 10000))
    
    # Early stopping
    use_early_stop = bool(cfg.get("early_stopping", False))
    patience = int(cfg.get("patience", 10))
    best_valid_loss = float('inf')
    patience_counter = 0
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp: print("\u26a1 Mixed Precision (AMP) enabled")
    if grad_accum_steps > 1: print(f"\U0001f4e6 Gradient accumulation: {grad_accum_steps} steps (effective batch = {cfg['batch_size'] * grad_accum_steps})")

    use_tbptt = bool(cfg.get("use_tbptt", False))

    bptt_window = int(cfg.get("bptt_window", 0)) or max(1, cfg["seq_len"])

    tbptt_stream = None
    if use_tbptt and cfg["dataset_type"] == 0:
        # dataset.ids handles the conversion from memmap to torch tensor if needed
        ids_ref = dataset.ids if hasattr(dataset, "ids") else dataset.data
        tbptt_stream = TBPTTClassicStream(
            ids_ref, window=bptt_window,
            batch_size=cfg["batch_size"], total_len=cfg.get("tbptt_total_len", 0))
    elif use_tbptt and cfg["dataset_type"] == 1:
        line_stream = LineTBPTTStream(
            lines_enc=dataset.lines_enc, window=bptt_window,
            batch_size=cfg["batch_size"], bos_id=vocab.bos_id)

    print(f"Training on {DEVICE} ... (Ctrl+C to save & exit)")
    try:
        rnn_state = None
        for epoch in range(cfg["epoch_count"]):
            model.train()
            
            # === FIX: Robust size calculation for Numpy (memmap) vs Torch ===
            if hasattr(dataset, "data"):
                # len() works on both Numpy arrays and Torch tensors
                ds_len = len(dataset.data)
            elif hasattr(dataset, "offsets"): 
                # IndexedLineDataset uses offsets
                ds_len = len(dataset.offsets)
            elif hasattr(dataset, "lines_enc"):
                ds_len = len(dataset.lines_enc)
            else:
                ds_len = 1000 * cfg["batch_size"] # Fallback

            steps_per_epoch = max(1, math.ceil(ds_len / cfg["batch_size"]))
            # ==============================================================

            for _ in range(steps_per_epoch):
                # ===== Batching =====
                if use_tbptt:
                    if cfg["dataset_type"] == 0:
                        x, y, reset_mask = tbptt_stream.get_next(DEVICE)
                    else:
                        x, y, reset_mask = line_stream.get_next(DEVICE)
                else:
                    x, y = dataset.get_batch(cfg["batch_size"])
                    reset_mask = None

                # ===== Forward / Backward =====
                if use_tbptt and (is_scan or cfg["model_selection"] in RNN_MODEL_IDS):
                    # Detach and selective reset across TBPTT windows
                    rnn_state = detach_state(rnn_state)
                    if rnn_state is not None and reset_mask is not None:
                        rnn_state = reset_rnn_state(rnn_state, reset_mask, model, cfg["model_selection"])
                    logits, rnn_state = model(x, rnn_state)
                else:
                    if is_scan:
                        # Parallel training path (stateless)
                        logits, _ = model(x)
                    elif cfg["model_selection"] in RNN_MODEL_IDS:
                        logits, rnn_state = model(x, None)
                    else:
                        logits = model(x)

                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

                # NaN/Inf guard — skip the step entirely to avoid poisoning optimizer state
                if torch.isnan(loss) or torch.isinf(loss):
                    if iters == 0:
                        print(f"[WARNING] NaN/Inf loss at first step — model may be numerically unstable")
                    else:
                        print(f"[WARNING] NaN/Inf loss at iter {iters+1} — skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    iters += 1
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                iters += 1; losses.append(loss.item())
                total_tokens += x.numel()
                if iters % log_interval == 0:
                    avg_loss = sum(losses[-100:]) / min(100, len(losses))
                    tok_s = total_tokens / max(1e-6, (time.time() - last_log))
                    lr_now = optimizer.param_groups[0]['lr']
                    ppl_str = f"ppl {math.exp(min(avg_loss, 20)):.1f}" if avg_loss < 20 else "ppl inf"
                    print(f"[e{epoch+1} iter {readable_num(iters)}] loss {avg_loss:.4f} | {ppl_str} | lr {lr_now:.2e} | tok/s ~{int(tok_s)}")
                    last_log = time.time(); total_tokens = 0

                # ----- Validation (keeps your existing heuristics) -----
                want_valid = (valid_ds is not None)
                if want_valid and iters % val_interval == 0:
                    vloss = eval_valid_loss(
                        model, cfg, valid_ds, vocab,
                        line_mode=(cfg["dataset_type"]==1),
                        max_samples=1000
                    )
                    if vloss is not None:
                        print(f"[valid @ {readable_num(iters)}] loss {vloss:.4f}")

                if iters % sample_interval == 0:
                    with torch.no_grad():
                        do_training_sample(cfg, model, vocab, line_mode)
                if iters % save_interval == 0:
                    cfg["iterations_done"] = iters
                    torch.save(model.state_dict(), CHECKPOINT_PATH)
                    save_json(CONFIG_PATH, cfg)
                    print(f"\n[checkpoint] saved {CHECKPOINT_PATH} + {CONFIG_PATH}")

        cfg["iterations_done"] = iters
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        save_json(CONFIG_PATH, cfg)
        print(f"\n[checkpoint] saved {CHECKPOINT_PATH} + {CONFIG_PATH}")
    except KeyboardInterrupt:
        print("\n[interrupt] saving...")
        cfg["iterations_done"] = iters
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        save_json(CONFIG_PATH, cfg)


def do_training_sample(cfg, model, vocab, line_mode):
    was_training = model.training
    try:
        model.eval()
        tmode = int(cfg.get("tokenizer_mode", 1))
        byte_text = bool(cfg.get("byte_output_text", False))
        out_dir = pathlib.Path("FileGen")
        if tmode == 0:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Retrieve new settings
        sample_len = int(cfg.get("train_sample_len", 20000))
        sample_count = int(cfg.get("train_sample_count", 1))
        custom_prompt_str = cfg.get("train_sample_prompt", "")

        temps = [1.0, 0.5, 0.25]

        # --- Helper to determine prompt IDs ---
        def get_train_prompt_ids():
            # 1. Line mode (Always starts with BOS)
            if line_mode:
                return [vocab.bos_id]
            
            # 2. Corpus mode with Custom Prompt
            if custom_prompt_str:
                if tmode == 0: # Byte mode -> Parse HEX
                    try:
                        # Remove spaces/0x and convert to bytes
                        clean_hex = custom_prompt_str.replace(" ", "").replace("0x", "")
                        raw_bytes = binascii.unhexlify(clean_hex)
                        return vocab.encode(raw_bytes)
                    except Exception as e:
                        print(f"[Warn] Invalid Hex prompt '{custom_prompt_str}': {e}. Using random.")
                        # Fallthrough to random
                else: # Text mode
                    return vocab.encode(custom_prompt_str)

            # 3. Corpus mode Random (use active tokens for tiktoken)
            if isinstance(vocab, TiktokenVocab):
                return [vocab.get_random_token()]
            start_id = random.randrange(vocab.size)
            return [start_id]

        for temp in temps:
            cfg["temperature"] = temp
            
            for i in range(sample_count):
                prompt_ids = get_train_prompt_ids()
                
                # Visual Logging
                if not line_mode:
                    if tmode == 0:
                        p_vis = bytes(vocab.to_bytes(prompt_ids)).hex()
                    else:
                        p_vis = vocab.decode(prompt_ids)
                    if len(p_vis) > 40: p_vis = p_vis[:40] + "..."
                    print(f"[sample t={temp} #{i+1}] Prompt: {p_vis}")
                else:
                    print(f"[sample t={temp} #{i+1}]")

                # Generate
                if line_mode:
                    out = generate_line_mode(
                        model, cfg, vocab, prompt_ids, limit_len=cfg["seq_len"]
                    )
                else:
                    out = generate_classic(
                        model, cfg, vocab, prompt_ids, max_len=sample_len, stream=False
                    )

                # Output / Save
                if tmode == 0: # Byte mode
                    data = vocab.to_bytes(out) if hasattr(vocab, "to_bytes") else bytes()
                    
                    if byte_text:
                        text_rep = vocab.decode(out[1:] if line_mode else out)
                        print(f"{text_rep}\n")
                    else:
                        iter_num = cfg.get('iterations_done', 0)
                        fname = f"train_iter{iter_num}_t{temp}_{i+1}.bin"
                        (out_dir / fname).write_bytes(data)
                        print(f"   -> Saved {fname} ({len(data)} bytes)")
                else: # Text mode
                    disp_ids = out[1:] if (line_mode and len(out) > 0) else out
                    text = vocab.decode(disp_ids)
                    print(f"{text}\n")

    finally:
        if was_training:
            model.train()




# ========= Config / UI =========
@dataclass
class RunConfig:
    dataset_path: str
    dataset_type: int
    model_selection: int
    activation_name: str
    embed_dim: int
    head_count: int
    layer_count: int
    seq_len: int
    epoch_count: int
    batch_size: int
    learning_rate: float
    temperature: float = 1.0
    iterations_done: int = 0
    vocab_tokens: Optional[List[str]] = None
    line_max_len: Optional[int] = None
    tokenizer_mode: int = 1  # -1=binary, 0=byte, 1=char, 2=word
    use_norm: int = 0     # 0=None, 1=BatchNorm, 2=LayerNorm, 3=RMSNorm
    res_every: int = 0    # 0 disables; otherwise every n layers
    res_type: int = 0     # 0=add, 1=concat(+proj), 2=ReZero scalar, 3=ReZero elementwise
    dropout: float = 0.0  # inter-layer dropout prob
    use_multiplier: int = 0 #
    train_sample_len: int = 200     # Length of generated samples during training (corpus mode)
    train_sample_count: int = 1     # Number of samples per temperature
    train_sample_prompt: str = ""   # Custom prompt string (Hex if byte mode, Text otherwise)
    def to_dict(self): return asdict(self)


def read_dataset(path: str, dataset_type: int, tokenizer_mode: int = 1):
    """
    Returns list of items:
      - for char/word/binary: list[str]
      - for byte tokenizer (0): list[bytes]  (raw)
    """
    if tokenizer_mode == 0:
        # byte mode → read raw bytes
        if dataset_type == 0:
            with open(path, "rb") as f:
                return [f.read()]
        else:
            # line mode: split by b'\n' but keep raw bytes per line
            with open(path, "rb") as f:
                return [ln.rstrip(b"\n") for ln in f.readlines()]
    else:
        # text modes
        if dataset_type == 0:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return [f.read()]
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return [ln.rstrip("\n") for ln in f.readlines()]


def prompt_int(msg, valid=None, default=None):
    label = prompt_label(msg, default)
    while True:
        s = input(label).strip()
        if s == "" and default is not None: return default
        try:
            v = int(s)
            if (valid is None) or (v in valid): return v
        except: pass
        print(f"  {_c(_RD, '✗')} Please enter a valid integer{(' in ' + str(valid)) if valid else ''}.")

def prompt_float(msg, default=None):
    label = prompt_label(msg, default)
    while True:
        s = input(label).strip()
        if s == "" and default is not None: return default
        try: return float(s)
        except: print(f"  {_c(_RD, '✗')} Please enter a valid number.")

def prompt_str(msg, default=None):
    label = prompt_label(msg, default)
    s = input(label).strip()
    if s == "" and default is not None: return default
    return s

def build_config_new():
    # ── Dataset ────────────────────────────────────────────────────────────────
    cli_section("Dataset", 64)
    print(f"  │")
    dataset_path = prompt_str("Dataset file path")

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Dataset type:')}")
    cli_opt(0, "Standard (corpus)", "Continuous text stream — random sliding windows")
    cli_opt(1, "Line mode",         "One example per line — BOS/EOS padded sequences")
    print(f"  │")
    dataset_type = prompt_int("Dataset type", valid={0,1})

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Tokenizer:')}")
    cli_opt(-1, "Binary",   "Raw bytes interpreted as binary (no text decoding)")
    cli_opt( 0, "Byte",     "Byte-level 0–255, universal, no vocab building")
    cli_opt( 1, "Char",     "Character-level — vocab built from dataset characters")
    cli_opt( 2, "Word",     "Whitespace-split word tokens — vocab from dataset")
    cli_opt( 3, "Tiktoken", "GPT-4 cl100k_base BPE tokenizer (50k+ vocab)")
    cli_opt( 4, "BPE",      "Custom byte-pair encoding trained on your data")
    print(f"  │")
    tokenizer_mode = prompt_int("Tokenizer", valid={-1,0,1,2,3,4})

    vocab_size_bpe = 0
    if tokenizer_mode == 3:
        tke = prompt_str("Tiktoken encoding  (gpt2 / r50k_base / cl100k_base)", default="cl100k_base")
    elif tokenizer_mode == 4:
        vocab_size_bpe = prompt_int("BPE vocabulary size", default=4096)
    elif tokenizer_mode == 0:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Byte output mode:')}")
        cli_opt(0, "Binary → FileGen/", "Write raw binary files (images, audio, etc.)")
        cli_opt(1, "Print as text",     "Decode output as latin-1 and print to terminal")
        print(f"  │")
        byte_out = prompt_int("Output mode", valid={0,1}, default=0)
    cli_section_end(64)

    # ── Model ──────────────────────────────────────────────────────────────────
    print_model_menu()

    msel = prompt_int("Model #", valid=set(range(99)))

    # MinRNN / MinIndRNN activation sub-menus
    minrnn_act = 0
    if msel == 26:
        print()
        cli_section("MinRNN Activation", 64)
        cli_opt(0, "Tanh",       "Original MinRNN formulation")
        cli_opt(1, "ReLU",       "Unbounded, sparse activations")
        cli_opt(2, "SiLU",       "Smooth gated linear unit")
        cli_opt(3, "GELU",       "Gaussian error linear unit")
        cli_opt(4, "Sigmoid",    "Bounded 0–1 gate-like activation")
        cli_opt(5, "g_act",      "Log-space scan (minGRU-style, numerically stable)")
        cli_section_end(64)
        minrnn_act = prompt_int("Activation", valid={0,1,2,3,4,5})

    if msel == 32:
        print()
        cli_section("MinIndRNN Activation", 64)
        print(f"  │  {_c(_DIM, 'Applied to the input projection inside the parallel IndRNN scan.')}")
        print(f"  │")
        _indrnn_acts = [
            (0,  "Tanh",            "Saturating, symmetric — original MinRNN"),
            (1,  "ReLU",            "Unbounded, sparse — fast but can explode"),
            (2,  "SiLU / Swish",    "Smooth gating, non-monotone — often best"),
            (3,  "PReLU (α=0.0)",   "Leaky ReLU with learned slope initialised at 0"),
            (4,  "PReLU (default)", "Leaky ReLU with learned slope initialised at 0.25"),
            (5,  "LReLU 0.2",       "Fixed leaky slope 0.2 — robust negative-side gradient"),
            (6,  "LReLU 0.01",      "Fixed leaky slope 0.01 — near-ReLU"),
            (7,  "GELU",            "Gaussian-weighted linear unit — Transformer-style"),
            (8,  "BentIdentity",    "(√(x²+1)−1)/2 + x — smooth near-linear"),
            (9,  "Sine",            "sin(x) — periodic, good for positional signals"),
            (10, "Cosine",          "cos(x) — periodic variant"),
            (11, "Snake",           "x + sin²(x) — periodic + monotone blend"),
            (12, "Stepping Sine",   "Quantised sine — discrete periodic steps"),
            (13, "Stepping Cosine", "Quantised cosine — discrete periodic steps"),
            (14, "Mish",            "x·tanh(softplus(x)) — very smooth, self-regularising"),
            (15, "Cone",            "Triangular bump function — local receptive field"),
            (16, "ReLU²",          "max(0,x)² — sparse and strictly positive"),
            (17, "g_act",           "Log-space scan gate (minGRU-style) — numerically stable"),
        ]
        for idx, name, desc in _indrnn_acts:
            cli_opt(idx, name, desc, kw=3, lw=22)
        cli_section_end(64)
        minrnn_act = prompt_int("Activation", valid=set(range(18)))

    # Activation for MLPs / TCN
    activation_name = "linear"
    if msel in (0, 1, 9, 10, 19):
        print()
        cli_section("Activation Function", 64)
        names = activation_names()
        for i, n in enumerate(names):
            cli_opt(i, n, kw=3, lw=20)
        cli_section_end(64)
        a_idx = prompt_int("Activation #", valid=set(range(len(names))))
        activation_name = names[a_idx]

    # ── Architecture ───────────────────────────────────────────────────────────
    cli_section("Architecture", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Embedding / hidden dim — size of every vector in the model.')}")
    print(f"  │  {_c(_DIM, 'Larger = more capacity, more VRAM, slower training.')}")
    embed_dim  = prompt_int("Embedding / hidden dim")

    head_count = 4
    if msel in ATTN_MODEL_IDS:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Head count — splits embed_dim into parallel attention heads.')}")
        print(f"  │  {_c(_DIM, 'Must divide embed_dim evenly. More heads = finer-grained attention.')}")
        head_count = prompt_int("Attention head count", default=4)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Layer count — number of stacked blocks / cells.')}")
    print(f"  │  {_c(_DIM, 'Deeper models learn longer-range patterns at higher compute cost.')}")
    layer_count = prompt_int("Layer count")

    # Classic RNN extras
    use_norm = 0; res_every = 0; res_type = 0; rnn_dropout = 0.0; use_multiplier = 0
    if msel in (2,3,4,5,6,7,8,18,23):
        print(f"  │")
        print(f"  │  {_c(_DIM, 'RNN structure options — extra stabilisation for step-by-step RNNs.')}")
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Normalisation — applied after each recurrent cell output:')}")
        cli_opt(0, "None",      "No normalisation — fastest, can be unstable in deep nets")
        cli_opt(1, "BatchNorm", "Normalise over the batch dim — sensitive to small batch sizes")
        cli_opt(2, "LayerNorm", "Normalise over the feature dim — robust, recommended default")
        cli_opt(3, "RMSNorm",   "LayerNorm without mean-centering — cheaper, similar quality")
        cli_opt(4, "TTanh",     "Tanh-based trainable normaliser — experimental")
        cli_opt(5, "ETTanh",    "Extended TTanh with elementwise scale — experimental")
        cli_opt(6, "DyT",       "Dynamic Tanh — replaces LayerNorm, no mean/var statistics")
        print(f"  │")
        use_norm = prompt_int("Norm type", valid={0,1,2,3,4,5,6})
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Residual connections — skip connections between layers that help')}")
        print(f"  │  {_c(_DIM, 'gradients flow and allow training much deeper RNN stacks.')}")
        res_every = prompt_int("Residual every N layers  (0 = off)", default=0)
        if res_every > 0:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Residual type:')}")
            cli_opt(0, "Add",            "x = x + f(x)  — standard, zero overhead")
            cli_opt(1, "Concat + proj",  "x = proj([x, f(x)])  — richer but adds parameters")
            cli_opt(2, "ReZero scalar",  "x = x + α·f(x), α=0 init — very stable training start")
            cli_opt(3, "ReZero vector",  "x = x + α⊙f(x), per-dim α — most expressive ReZero")
            print(f"  │")
            res_type = prompt_int("Residual type", valid={0,1,2,3})
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Dropout — randomly zeros inter-layer activations during training,')}")
        print(f"  │  {_c(_DIM, 'acting as regularisation. 0.1–0.3 for most tasks; 0 to disable.')}")
        rnn_dropout = prompt_float("Inter-layer dropout  (0.0 = off)", default=0.0)
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Output multiplier — a learnable gate on the final hidden→logit')}")
        print(f"  │  {_c(_DIM, 'projection. Can help calibrate output scale, especially early.')}")
        cli_opt(0, "Off",           "No multiplier — standard behaviour")
        cli_opt(1, "Scalar",        "One learnable scalar multiplies all outputs")
        cli_opt(2, "Per-dim vector","One learnable value per hidden dimension")
        print(f"  │")
        use_multiplier = prompt_int("Multiplier", default=0)
    cli_section_end(64)

    # ── Sequence / Data ────────────────────────────────────────────────────────
    cli_section("Sequence & Validation", 64)
    print(f"  │")
    if dataset_type == 0:
        print(f"  │  {_c(_DIM, 'Sequence length — tokens per training window.')}")
        print(f"  │  {_c(_DIM, 'Longer = more context, more memory. Typical: 128–2048.')}")
        seq_len = prompt_int("Sequence length  (tokens per window)")
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Validation file — a separate held-out text file for measuring')}")
        print(f"  │  {_c(_DIM, 'generalisation. Leave blank to auto-split the training file.')}")
        classic_val_path = prompt_str("Validation file  (leave blank to split)", default="")
        if not classic_val_path:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Validation split — fraction of the file reserved for validation.')}")
            print(f"  │  {_c(_DIM, 'e.g. 0.1 = last 10% of the file. Set 0 to disable validation.')}")
            val_split = prompt_float("Validation split fraction  (0 = disable)", default=0.1)
        else:
            val_split = 0.0
    else:
        seq_len = 0; classic_val_path = ""
        print(f"  │  {_c(_DIM, 'Sequence length is set automatically from the longest line.')}")
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Validation split — fraction of lines held out for evaluation.')}")
        print(f"  │  {_c(_DIM, 'Lines are shuffled before splitting. Set 0 to use all for training.')}")
        val_split = prompt_float("Validation split fraction  (0 = disable)", default=0.1)
    cli_section_end(64)

    # ── Training ───────────────────────────────────────────────────────────────
    cli_section("Training", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Epoch count — full passes through the dataset.')}")
    print(f"  │  {_c(_DIM, 'One epoch = ceil(dataset_size / batch_size) gradient steps.')}")
    epoch_count   = prompt_int("Epoch count")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Batch size — sequences processed per gradient step.')}")
    print(f"  │  {_c(_DIM, 'Larger batches = more stable gradients but more VRAM.')}")
    batch_size    = prompt_int("Batch size")
    cli_section_end(64)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optim_cfg = prompt_optimizer_config()
    learning_rate = optim_cfg["optim_params"].get("lr", 1.0)

    # ── Sampling ──────────────────────────────────────────────────────────────
    cli_section("Sampling", 64)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'During-training sampling — periodically generates text so you can')}")
    print(f"  │  {_c(_DIM, 'watch the model improve without waiting for training to finish.')}")
    train_sample_count = prompt_int("Samples per training-sample call", default=2)
    train_sample_len   = 0
    train_sample_prompt = ""
    if dataset_type == 0:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Sample length — how many tokens to generate each time.')}")
        train_sample_len = prompt_int("Sample length  (tokens)", default=200)
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Training prompt — text (or hex for byte mode) to seed each sample.')}")
        print(f"  │  {_c(_DIM, 'Leave blank to pick a random token from the vocab each time.')}")
        if tokenizer_mode == 0:
            train_sample_prompt = prompt_str("Training prompt  (hex, blank = random)", default="")
        else:
            train_sample_prompt = prompt_str("Training prompt  (text, blank = random)", default="")
    cli_section_end(64)

    # ── Advanced ───────────────────────────────────────────────────────────────
    cli_section("Advanced Training", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Gradient accumulation — accumulates gradients over N mini-batches')}")
    print(f"  │  {_c(_DIM, 'before a weight update. Simulates a larger effective batch size')}")
    print(f"  │  {_c(_DIM, 'without extra VRAM. e.g. accum=4, batch=32 → effective batch 128.')}")
    grad_accum    = prompt_int("Gradient accumulation steps  (1 = off)", default=1)
    use_amp_val   = 0
    if DEVICE == "cuda":
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Automatic Mixed Precision (AMP) — runs forward pass in float16,')}")
        print(f"  │  {_c(_DIM, 'keeping master weights in float32. Halves VRAM and speeds up')}")
        print(f"  │  {_c(_DIM, 'matmuls on Ampere+ GPUs (RTX 30xx / A100 and newer).')}")
        use_amp_val = prompt_int("Mixed precision / AMP  (0=off 1=on)", valid={0,1}, default=0)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'LR scheduler — adjusts learning rate over the course of training.')}")
    print(f"  │  {_c(_DIM, 'Has less impact with Prodigy (which self-tunes) but can still help.')}")
    cli_opt(0, "None",           "Fixed learning rate — Prodigy handles it already")
    cli_opt(1, "Cosine + warmup","Linear ramp for N steps, then cosine decay to 0")
    cli_opt(2, "Cosine",         "Cosine decay from step 0 — no warmup")
    cli_opt(3, "One-cycle",      "Ramps up then aggressively down — fast convergence")
    print(f"  │")
    sched_choice  = prompt_int("Scheduler", valid={0,1,2,3}, default=0)
    sched_map     = {0:"none", 1:"cosine_warmup", 2:"cosine", 3:"one_cycle"}
    lr_scheduler  = sched_map[sched_choice]
    warmup_steps  = 0
    if lr_scheduler == "cosine_warmup":
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Warmup steps — LR grows linearly from 0 to base LR over this many')}")
        print(f"  │  {_c(_DIM, 'steps. Helps stabilise early training. Typical: 100–500.')}")
        warmup_steps = prompt_int("Warmup steps", default=100)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Early stopping — halts training if validation loss stops improving,')}")
    print(f"  │  {_c(_DIM, 'preventing overfitting. Requires a validation split or file.')}")
    early_stop    = prompt_int("Early stopping  (0=off 1=on)", valid={0,1}, default=0)
    patience_val  = 10
    if early_stop:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Patience — number of validation checks without improvement before')}")
        print(f"  │  {_c(_DIM, 'training stops. Higher = more tolerance for temporary plateaus.')}")
        patience_val = prompt_int("Patience  (checks without improvement)", default=10)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Logging / sampling / validation intervals — how often (in steps)')}")
    print(f"  │  {_c(_DIM, 'each event fires. Lower = more feedback, slightly more overhead.')}")
    log_interval        = prompt_int("Log loss every N steps", default=50)
    sample_interval_val = prompt_int("Sample text every N steps", default=500)
    val_interval_val    = prompt_int("Run validation every N steps", default=500)

    # TBPTT (recurrent / scan only)
    use_tbptt = False; bptt_window = 0; tbptt_total_len = 0
    if msel in RNN_MODEL_IDS or msel in SCAN_MODEL_IDS:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Truncated BPTT (TBPTT) — instead of fitting one sequence per step,')}")
        print(f"  │  {_c(_DIM, 'the model processes a long stream and back-props through a short')}")
        print(f"  │  {_c(_DIM, 'window while carrying the hidden state forward. This lets RNNs')}")
        print(f"  │  {_c(_DIM, 'learn very long-range dependencies without exploding memory.')}")
        yn = prompt_str("Enable TBPTT  (y/n)", default="n").lower()
        use_tbptt = yn in ("y","yes","1")
        if use_tbptt:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'BPTT window — tokens processed per gradient step.')}")
            print(f"  │  {_c(_DIM, 'Shorter = faster steps, shallower gradient signal.')}")
            if dataset_type == 1:
                bptt_window = prompt_int("BPTT window  (tokens per step, ≤ max line length)", default=64)
            else:
                bptt_window    = prompt_int("BPTT window  (tokens per step)", default=64)
                print(f"  │")
                print(f"  │  {_c(_DIM, 'Total TBPTT length — total tokens streamed before resetting state.')}")
                print(f"  │  {_c(_DIM, 'Set 0 to stream until EOF then wrap around.')}")
                tbptt_total_len = prompt_int("Total TBPTT length  (0 = until EOF)", default=0)
    cli_section_end(64)

    # ── Assemble config ────────────────────────────────────────────────────────
    cfg = RunConfig(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        model_selection=msel,
        activation_name=activation_name,
        embed_dim=embed_dim,
        head_count=head_count,
        layer_count=layer_count,
        seq_len=seq_len,
        epoch_count=epoch_count,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tokenizer_mode=tokenizer_mode,
        use_norm=use_norm,
        res_every=res_every,
        res_type=res_type,
        dropout=rnn_dropout,
        use_multiplier=use_multiplier,
        train_sample_len=train_sample_len,
        train_sample_count=train_sample_count,
        train_sample_prompt=train_sample_prompt,
    ).to_dict()

    if classic_val_path: cfg["classic_val_path"] = classic_val_path
    cfg["val_split"]         = val_split
    cfg["use_tbptt"]         = use_tbptt
    cfg["bptt_window"]       = bptt_window
    cfg["tbptt_total_len"]   = tbptt_total_len
    cfg["minrnn_act"]        = minrnn_act
    cfg["grad_accum_steps"]  = grad_accum
    cfg["use_amp"]           = bool(use_amp_val)
    cfg["lr_scheduler"]      = lr_scheduler
    cfg["warmup_steps"]      = warmup_steps
    cfg["early_stopping"]    = bool(early_stop)
    cfg["patience"]          = patience_val
    cfg["log_interval"]      = log_interval
    cfg["sample_interval"]   = sample_interval_val
    cfg["val_interval"]      = val_interval_val

    cfg["optimizer"]         = optim_cfg["optimizer"]
    cfg["optim_params"]      = optim_cfg["optim_params"]

    if tokenizer_mode == 3: cfg["tiktoken_encoding"] = tke
    if tokenizer_mode == 0: cfg["byte_output_text"]  = bool(byte_out == 1)
    if tokenizer_mode == 4: cfg["custom_bpe_size"]   = vocab_size_bpe
    return cfg


@torch.no_grad()
def eval_valid_loss(model, cfg, ds, vocab, line_mode, max_samples=1000):
    is_scan = cfg["model_selection"] in SCAN_MODEL_IDS
    
    # === FIX: Ignore PAD tokens in validation loss ===
    pad_id = getattr(vocab, "pad_id", -100)
    if pad_id is None: pad_id = -100
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    total_loss = 0.0
    total_tokens = 0

    # Save original state
    orig_training = model.training
    
    try:
        # Scan models: keep in train() mode to use the fast parallel scan path.
        # torch.no_grad() (from decorator) already disables autograd.
        # Switching to eval() forces some scan models into slow step-by-step mode.
        if is_scan:
            model.train()  # Keep parallel scan path active
        else:
            model.eval() 

        # 1. New IndexedLineDataset (Disk-based)
        if line_mode and hasattr(ds, 'offsets'): 
            batch_size = min(32, cfg["batch_size"])  # Smaller for speed
            steps = max(1, min(max_samples // batch_size, 30))  # Cap at 30 batches
            
            for _ in range(steps):
                x, y = ds.get_batch(batch_size)
                
                if is_scan or cfg["model_selection"] in RNN_MODEL_IDS:
                    out = model(x) # State is usually reset or None for random batches
                    logits = out[0] if isinstance(out, tuple) else out
                else:
                    logits = model(x)

                # Flatten and compute
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                
                # Weighted average by number of non-pad tokens (more accurate)
                # or just numel() if we want simple batch averaging. 
                # Using numel() is standard but strictly speaking we should mask pads.
                # For consistency with train_loop, numel() is fine as long as loss is masked.
                total_loss += loss.item() * x.numel()
                total_tokens += x.numel()

        # 2. Old LineDataset (Memory-based)
        elif line_mode and hasattr(ds, 'data'):
            n = ds.data.size(0)
            if n == 0: return None
            sample_count = min(n, max_samples)
            idx = torch.randperm(n)[:sample_count]
            BATCH = min(64, sample_count)
            for i in range(0, sample_count, BATCH):
                x = ds.data[idx][i:i+BATCH,:-1].to(DEVICE)
                y = ds.data[idx][i:i+BATCH,1:].to(DEVICE)
                
                if is_scan or cfg["model_selection"] in RNN_MODEL_IDS:
                    out = model(x)
                    logits = out[0] if isinstance(out, tuple) else out
                else:
                    logits = model(x)
                    
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item() * x.numel()
                total_tokens += x.numel()
        
        # 3. Classic mode (Memmap or Standard)
        else:
            steps = 20 # 20 batches is usually enough for a quick check
            bs = min(64, cfg["batch_size"])
            for _ in range(steps):
                x, y = ds.get_batch(bs)
                if is_scan or cfg["model_selection"] in RNN_MODEL_IDS:
                    out = model(x)
                    logits = out[0] if isinstance(out, tuple) else out
                else:
                    logits = model(x)
                
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item() * x.numel()
                total_tokens += x.numel()

    finally:
        # Restore original state (Train/Eval)
        if orig_training:
            model.train()

    return (total_loss / max(1, total_tokens)) if total_tokens > 0 else None


def load_or_make_vocab(cfg, txt_path: str) -> BaseVocab:
    line_mode = (cfg["dataset_type"] == 1)
    tmode = int(cfg.get("tokenizer_mode", 1))

    # 1. Non-text tokenizers (Binary, Byte, Tiktoken) don't need saving/loading
    if tmode == -1: return BinaryVocab(line_mode)
    if tmode == 0: return ByteVocab(line_mode)
    if tmode == 3:
        v = TiktokenVocab(cfg.get("tiktoken_encoding", "cl100k_base"), line_mode)
        # Scan file for active tokens (for filtered random prompts)
        if os.path.exists(txt_path):
            v.scan_file(txt_path)
        return v
    if tmode == 4:
        # Save .bpe.vocab next to the text file
        vocab_file = str(pathlib.Path(txt_path).with_suffix(".bpe.vocab"))
        target_size = int(cfg.get("custom_bpe_size", 4096))
        
        bpe = CustomBPEVocab(vocab_file, line_mode, expected_size=target_size)
        
        # Train if empty (meaning file didn't exist)
        if len(bpe.merges) == 0:
            bpe.train(txt_path, target_size)
            
        return bpe

    # 2. Check if vocab is already in the config (this prevents recreation)
    if cfg.get("vocab_tokens") and len(cfg["vocab_tokens"]) > 0:
        print(f"[Vocab] Loading {len(cfg['vocab_tokens'])} tokens from config...")
        # === FIX: Reconstruct CharVocab directly from token list ===
        v = CharVocab.__new__(CharVocab)
        v.line_mode = line_mode
        v.tokens = cfg["vocab_tokens"]
        v.stoi = {ch: i for i, ch in enumerate(v.tokens)}
        v.itos = {i: ch for ch, i in v.stoi.items()}
        v.bos_id = v.stoi.get(BOS_TOKEN) if line_mode else None
        v.eos_id = v.stoi.get(EOS_TOKEN) if line_mode else None
        v.pad_id = v.stoi.get(PAD_TOKEN) if line_mode else None
        return v

    # 3. If not found, scan the file to build it
    print(f"[Vocab] Scanning {txt_path} to build vocabulary...")
    unique_tokens = set()
    
    # Use the streaming read to avoid memory issues
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Building Vocab"):
            if tmode == 1: unique_tokens.update(line)
            elif tmode == 2: unique_tokens.update(line.split())
    
    vocab_list = sorted(list(unique_tokens))
    print(f"[Vocab] Found {len(vocab_list)} unique tokens.")
    
    v = CharVocab(vocab_list, line_mode)
    
    # 4. Save immediately to config
    cfg["vocab_tokens"] = v.tokens
    save_json(CONFIG_PATH, cfg)
    print(f"[Vocab] Saved tokens to {CONFIG_PATH}")
    
    return v


def build_datasets(cfg, vocab):
    path = cfg["dataset_path"]
    dtype = cfg["dataset_type"]
    val_split = float(cfg.get("val_split", 0.0))
    
    if dtype == 0:
        # === Corpus Mode ===
        # If external validation file exists, use it and ignore split
        val_path = cfg.get("classic_val_path")
        if val_path and os.path.exists(val_path):
            print(f"[Dataset] Using explicit validation file: {val_path}")
            train_ds = MemmapClassicDataset(path, vocab, cfg["seq_len"], split_range=(0.0, 1.0))
            valid_ds = MemmapClassicDataset(val_path, vocab, cfg["seq_len"], split_range=(0.0, 1.0))
        elif val_split > 0.0:
            print(f"[Dataset] Splitting single file: {1.0-val_split:.0%} Train / {val_split:.0%} Valid")
            # Train gets 0.0 -> (1.0 - split)
            # Valid gets (1.0 - split) -> 1.0
            split_pt = 1.0 - val_split
            train_ds = MemmapClassicDataset(path, vocab, cfg["seq_len"], split_range=(0.0, split_pt))
            valid_ds = MemmapClassicDataset(path, vocab, cfg["seq_len"], split_range=(split_pt, 1.0))
        else:
            print("[Dataset] No validation split.")
            train_ds = MemmapClassicDataset(path, vocab, cfg["seq_len"], split_range=(0.0, 1.0))
            valid_ds = None
            
        return train_ds, valid_ds
    
    else:
        # === Line Mode ===
        full_ds = IndexedLineDataset(path, vocab)
        
        # Use existing indices/logic but parameterize the split
        n = len(full_ds.offsets)
        if val_split > 0.0:
            perm = np.random.permutation(n)
            cut = int((1.0 - val_split) * n)
            train_idx = perm[:cut]
            valid_idx = perm[cut:]
            
            print(f"[Dataset] Lines Split: {len(train_idx)} Train / {len(valid_idx)} Valid")
            train_ds = IndexedLineDatasetSubset(full_ds, train_idx)
            valid_ds = IndexedLineDatasetSubset(full_ds, valid_idx) if len(valid_idx) > 0 else None
            cfg["valid_examples"] = len(valid_idx)
        else:
            print(f"[Dataset] Using all {n} lines for training (no validation).")
            # Create a subset containing all indices to keep types consistent
            all_idx = np.arange(n)
            train_ds = IndexedLineDatasetSubset(full_ds, all_idx)
            valid_ds = None
            cfg["valid_examples"] = 0
        
        cfg["line_max_len"] = full_ds.max_len
        # Ensure seq_len doesn't exceed the longest line in the file
        cfg["seq_len"] = min(full_ds.max_len, 2048) 
        
        return train_ds, valid_ds



def resume_adjustments(cfg, _model):
    cli_section("Resume Adjustments", 64)
    print(f"  │  {_c(_DIM, 'Update training settings before continuing from the checkpoint.')}")
    print(f"  │  {_c(_DIM, 'Architecture and vocabulary cannot be changed on resume.')}")

    if cfg["model_selection"] in RNN_MODEL_IDS:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Sequence length — RNNs allow changing this on resume because')}")
        print(f"  │  {_c(_DIM, 'the weights do not depend on a fixed context window size.')}")
        cfg["seq_len"] = prompt_int("New sequence length", default=cfg["seq_len"])

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Epoch count — total epochs for this continued run.')}")
    cfg["epoch_count"] = prompt_int("New epoch count")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Batch size — can be changed freely on resume.')}")
    cfg["batch_size"] = prompt_int("New batch size")

    tmode = int(cfg.get("tokenizer_mode", 1))
    if tmode in (-1, 0, 3):  # binary / byte / tiktoken — vocab-size independent
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Dataset switch — only available for tokenizers whose vocab size')}")
        print(f"  │  {_c(_DIM, 'is fixed (binary, byte, tiktoken). Char/word/BPE cannot switch')}")
        print(f"  │  {_c(_DIM, 'datasets because the embedding matrix is tied to vocab size.')}")
        change_ds = prompt_str("Switch to a different dataset?  (y/n)", default="n").lower() in ("y", "yes", "1")
        if change_ds:
            new_path = prompt_str("New dataset path  (blank = keep current)", default=cfg["dataset_path"])
            cfg["_changed_dataset"] = False
            if new_path and new_path != cfg["dataset_path"]:
                cfg["dataset_path"] = new_path
                cfg["_changed_dataset"] = True
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Dataset type — 0 = sliding-window corpus, 1 = per-line examples.')}")
            dt = prompt_str("New dataset type  (0=standard  1=line  blank=keep)", default="")
            if dt.strip() in ("0", "1"):
                cfg["dataset_type"] = int(dt.strip())
                cfg["_changed_dataset"] = True
    cli_section_end(64)
    return cfg


# ── Optimizer Registry ────────────────────────────────────────────────────────
# Each entry: { "name": str, "class": callable_or_str, "defaults": {param: value},
#               "params": [ { "key": str, "prompt": str, "type": "float"|"int", "default": value } ] }
# To add a new optimizer: append an entry here and it will appear in the menu automatically.

OPTIMIZER_REGISTRY = [
    {
        "name": "Prodigy",
        "class": "prodigy",
        "defaults": {"lr": 1.0},
        "params": [
            {"key": "lr",      "prompt": "Learning rate  (Prodigy auto-scales; 1.0 is usually fine)", "type": "float", "default": 1.0},
            {"key": "slice_p", "prompt": "Slice P  (gradient slicing factor)", "type": "int", "default": 8},
        ],
    },
    {
        "name": "Adam",
        "class": "adam",
        "defaults": {"lr": 3e-4},
        "params": [
            {"key": "lr",      "prompt": "Learning rate", "type": "float", "default": 3e-4},
            {"key": "betas",   "prompt": "Betas  (comma-separated, e.g. 0.9,0.999)", "type": "betas", "default": (0.9, 0.999)},
            {"key": "eps",     "prompt": "Epsilon", "type": "float", "default": 1e-8},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "SGD",
        "class": "sgd",
        "defaults": {"lr": 1e-2},
        "params": [
            {"key": "lr",       "prompt": "Learning rate", "type": "float", "default": 1e-2},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.9},
            {"key": "dampening","prompt": "Dampening", "type": "float", "default": 0.0},
            {"key": "nesterov", "prompt": "Nesterov  (0=off 1=on)", "type": "bool", "default": False},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "RMSprop",
        "class": "rmsprop",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr",       "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "alpha",    "prompt": "Alpha  (smoothing constant)", "type": "float", "default": 0.99},
            {"key": "eps",      "prompt": "Epsilon", "type": "float", "default": 1e-8},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.0},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "Rprop",
        "class": "rprop",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr",   "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "etas", "prompt": "Etas  (comma-separated, e.g. 0.5,1.2)", "type": "betas", "default": (0.5, 1.2)},
        ],
    },
    {
        "name": "Adagrad",
        "class": "adagrad",
        "defaults": {"lr": 1e-2},
        "params": [
            {"key": "lr",                "prompt": "Learning rate", "type": "float", "default": 1e-2},
            {"key": "lr_decay",          "prompt": "LR decay", "type": "float", "default": 0.0},
            {"key": "eps",               "prompt": "Epsilon", "type": "float", "default": 1e-10},
            {"key": "weight_decay",      "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "Adadelta",
        "class": "adadelta",
        "defaults": {"lr": 1.0},
        "params": [
            {"key": "lr",    "prompt": "Learning rate", "type": "float", "default": 1.0},
            {"key": "rho",   "prompt": "Rho  (decay rate)", "type": "float", "default": 0.9},
            {"key": "eps",   "prompt": "Epsilon", "type": "float", "default": 1e-6},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
]

def _get_optimizer_names():
    return [o["name"] for o in OPTIMIZER_REGISTRY]

def _parse_optim_param(raw_str, ptype, default):
    """Parse a single optimizer parameter from user input string."""
    if raw_str.strip() == "":
        return default
    if ptype == "float":
        return float(raw_str)
    if ptype == "int":
        return int(raw_str)
    if ptype == "bool":
        v = raw_str.strip().lower()
        return v in ("1", "true", "yes", "y")
    if ptype == "betas":
        parts = [float(x.strip()) for x in raw_str.split(",")]
        return tuple(parts)
    return raw_str

def prompt_optimizer_config():
    """Interactive optimizer selection + per-optimizer param prompts.
    Returns dict: {"optimizer": str, "optim_params": {key: value, ...}}
    """
    cli_section("Optimizer", 64)
    print(f"  │")
    for i, entry in enumerate(OPTIMIZER_REGISTRY):
        desc = ", ".join(f"{p['key']}={p['default']}" for p in entry["params"][:3])
        cli_opt(i, entry["name"], desc)
    print(f"  │")
    choice = prompt_int("Optimizer", valid=set(range(len(OPTIMIZER_REGISTRY))), default=0)
    entry = OPTIMIZER_REGISTRY[choice]

    print(f"  │")
    opt_name = entry["name"]
    print(f"  │  {_c(_DIM, f'Configure {opt_name} parameters  (press Enter for default):')}")
    optim_params = {}
    for p in entry["params"]:
        label = prompt_label(p["prompt"], p["default"])
        raw = input(label).strip()
        optim_params[p["key"]] = _parse_optim_param(raw, p["type"], p["default"])

    cli_section_end(64)
    return {"optimizer": entry["class"], "optim_params": optim_params}


def build_optimizer(model, cfg):
    wd = cfg.get("optim_params", {}).get("weight_decay", 0.0)
    decay_params = []
    no_decay_params = []

    # Substrings to identify parameters that should NOT decay
    blacklist = [
        'embed', 'pos', 'tok',       # Embeddings (including GPT2 'tok')
        'norm', 'ln', 'gn',          # Normalization (LayerNorm, RMSNorm, GroupNorm)
        'alphas', 'betas',           # ReZero scalars
        'res_scale',                 # GatedMLP residual scale
        'gamma',                     # RetNet gammas
        'time_mix', 'w_tau',         # RWKV / Liquid parameters
        'log_A'                      # Mamba/Scan decay parameters
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in blacklist):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    print(f"[Optimizer] Decay: {len(decay_params)} tensors | No Decay: {len(no_decay_params)} tensors (Embeds/Norms/Gates)")

    optim_groups = [
        {'params': decay_params, 'weight_decay': wd},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    optimizer_key = cfg.get("optimizer", "prodigy")
    op = cfg.get("optim_params", {})

    if optimizer_key == "prodigy":
        return Prodigy(optim_groups, lr=op.get("lr", 1.0), slice_p=op.get("slice_p", 8))
    elif optimizer_key == "adam":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return torch.optim.Adam(optim_groups, lr=op.get("lr", 3e-4),
                                betas=betas, eps=op.get("eps", 1e-8))
    elif optimizer_key == "sgd":
        return torch.optim.SGD(optim_groups, lr=op.get("lr", 1e-2),
                               momentum=op.get("momentum", 0.9),
                               dampening=op.get("dampening", 0.0),
                               nesterov=op.get("nesterov", False))
    elif optimizer_key == "rmsprop":
        return torch.optim.RMSprop(optim_groups, lr=op.get("lr", 1e-3),
                                   alpha=op.get("alpha", 0.99),
                                   eps=op.get("eps", 1e-8),
                                   momentum=op.get("momentum", 0.0))
    elif optimizer_key == "rprop":
        etas = op.get("etas", (0.5, 1.2))
        if isinstance(etas, list): etas = tuple(etas)
        return torch.optim.Rprop(optim_groups, lr=op.get("lr", 1e-3), etas=etas)
    elif optimizer_key == "adagrad":
        return torch.optim.Adagrad(optim_groups, lr=op.get("lr", 1e-2),
                                   lr_decay=op.get("lr_decay", 0.0),
                                   eps=op.get("eps", 1e-10))
    elif optimizer_key == "adadelta":
        return torch.optim.Adadelta(optim_groups, lr=op.get("lr", 1.0),
                                    rho=op.get("rho", 0.9),
                                    eps=op.get("eps", 1e-6))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_key}")
def wrap_model_with_compile(model, cfg):
    msel = cfg["model_selection"]
    
    # 1. Skip Scan Models (Vectorized cumsum/prod are already efficient)
    # Compiling these often leads to overhead with no gain on 11.7
    if msel in SCAN_MODEL_IDS:
        print("ℹ️ Scan Model detected: Skipping torch.compile (using Eager mode)")
        return model

    # 2. Modern Transformers (Attention-heavy)
    # Mode: max-autotune enables CUDA Graphs and optimized SDPA (Flash Attention)
    if msel in {10, 25}: # GPT and Modern Transformer
        print("🚀 Transformer detected: Compiling with mode='max-autotune'")
        return torch.compile(model, mode="max-autotune")

    # 3. Custom RNNs (Loop-heavy)
    # Mode: max-autotune is best for these custom cells as it generates 
    # optimized Triton kernels for your specific math (ATanU, etc.)
    if msel in RNN_MODEL_IDS:
        print("⚡ Custom RNN detected: Compiling with mode='reduce-overhead'")
        return torch.compile(model, mode="default")

    # 4. Fallback for MLPs / TCN
    print("✨ Generic model detected: Compiling with mode='reduce-overhead'")
    return torch.compile(model, mode="reduce-overhead")
# ========= NEW MODES =========

def run_interactive_chat(cfg, model, vocab):
    """Interactive multi-turn text generation."""
    model.eval()
    line_mode = (cfg["dataset_type"] == 1)

    cli_banner("Interactive Chat", "Multi-turn text generation session", width=64)
    W = 62
    print(f"  {_c(_CY, _B, '┌─')} {_c(_WH, _B, 'Runtime Commands')} {_c(_CY, '─' * (W - 24) + '┐')}")
    cli_opt("/temp N",  "Set temperature",      "e.g. /temp 0.8  — 0 = greedy, 1 = unmodified")
    cli_opt("/topk N",  "Set top-k",            "e.g. /topk 50   — keep N most likely tokens")
    cli_opt("/topp N",  "Set top-p",            "e.g. /topp 0.9  — nucleus sampling threshold")
    cli_opt("/rep N",   "Set rep. penalty",     "e.g. /rep 1.2   — penalises repeated tokens")
    cli_opt("/len N",   "Set generation length","e.g. /len 300   — max tokens per response")
    cli_opt("/help",    "Show this list",       "")
    cli_opt("/quit",    "Exit chat",            "Also Ctrl-C")
    cli_blank_row()
    print(f"  {_c(_CY, '└' + '─' * (W - 2) + '┘')}\n")

    temp = cfg.get("temperature", 1.0)
    top_k = 0; top_p = 0.0; rep_penalty = 1.0
    gen_len = 200 if not line_mode else cfg.get("seq_len", 512)
    
    while True:
        try:
            user_input = input("You> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat."); break
        if not user_input.strip(): continue
        
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            if cmd in ("/quit", "/exit"): break
            elif cmd == "/temp" and len(parts) > 1: temp = float(parts[1]); print(f"  Temperature={temp}"); continue
            elif cmd == "/topk" and len(parts) > 1: top_k = int(parts[1]); print(f"  Top-k={top_k}"); continue
            elif cmd == "/topp" and len(parts) > 1: top_p = float(parts[1]); print(f"  Top-p={top_p}"); continue
            elif cmd == "/rep" and len(parts) > 1: rep_penalty = float(parts[1]); print(f"  Rep penalty={rep_penalty}"); continue
            elif cmd == "/len" and len(parts) > 1: gen_len = int(parts[1]); print(f"  Gen length={gen_len}"); continue
            elif cmd == "/help": print("  /temp /topk /topp /rep /len /quit"); continue
            else: print(f"  Unknown: {cmd}"); continue
        
        cfg["temperature"] = temp
        cfg["_top_k"] = top_k; cfg["_top_p"] = top_p; cfg["_rep_penalty"] = rep_penalty
        
        if line_mode:
            p_ids = vocab.encode(BOS_TOKEN + user_input)
            out_ids = generate_line_mode(model, cfg, vocab, p_ids, limit_len=gen_len)
            text = vocab.decode(out_ids[1:]) if len(out_ids) > 1 else ""
            print(f"Model> {text}\n")
        else:
            p_ids = vocab.encode(user_input)
            sys.stdout.write("Model> ")
            generate_classic(model, cfg, vocab, p_ids, max_len=gen_len, stream=True)
            print()


@torch.no_grad()
def run_perplexity_eval():
    """Comprehensive perplexity evaluation."""
    cli_banner("Perplexity Evaluation", "Measure how well the model predicts held-out text", width=64)
    if not os.path.exists(CONFIG_PATH):
        pwarn("No config found — train a model first."); return

    cfg = load_json(CONFIG_PATH)
    vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
    model = build_model(cfg, vocab.size)
    model.to(DEVICE)
    pinfo(f"Loading checkpoint from {CHECKPOINT_PATH} …")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    is_scan = cfg["model_selection"] in SCAN_MODEL_IDS
    if is_scan: model.train()
    else: model.eval()

    cli_section("Evaluation Settings", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Evaluation file — the text file to measure perplexity on.')}")
    print(f"  │  {_c(_DIM, 'Should be held-out data the model has never seen during training.')}")
    print(f"  │  {_c(_DIM, 'Defaults to the training file if left blank.')}")
    eval_path = prompt_str("Evaluation file path", default=cfg["dataset_path"])

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Batch count — how many random batches to sample from the file.')}")
    print(f"  │  {_c(_DIM, 'More batches = more accurate estimate, more time. 100 is typical.')}")
    max_batches = prompt_int("Number of batches", default=100)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Batch size — sequences per batch. Larger is faster but uses more')}")
    print(f"  │  {_c(_DIM, 'VRAM. Must fit in memory alongside the model.')}")
    batch_size = prompt_int("Batch size", default=cfg["batch_size"])
    cli_section_end(64)

    line_mode = (cfg["dataset_type"] == 1)
    pad_id = getattr(vocab, "pad_id", -100)
    if pad_id is None: pad_id = -100
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')

    if line_mode:
        eval_ds = IndexedLineDataset(eval_path, vocab)
    else:
        eval_ds = MemmapClassicDataset(eval_path, vocab, cfg["seq_len"])

    total_loss = 0.0; total_tokens = 0
    n_batches = max_batches if max_batches > 0 else 100

    for _ in tqdm(range(n_batches), desc="Perplexity"):
        x, y = eval_ds.get_batch(batch_size)
        if is_scan or cfg["model_selection"] in RNN_MODEL_IDS:
            out = model(x); logits = out[0] if isinstance(out, tuple) else out
        else:
            logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        non_pad = (y != pad_id).sum().item() if pad_id >= 0 else y.numel()
        total_loss += loss.item(); total_tokens += non_pad

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 100))
    bpc = avg_loss / math.log(2)

    W = 62
    print()
    cli_section("Results", W)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Tokens evaluated :')}  {_c(_WH, readable_num(total_tokens))}")
    print(f"  │  {_c(_DIM, 'Cross-entropy loss:')}  {_c(_WH, f'{avg_loss:.4f}')}  {_c(_DIM, '(nats/token)')}")
    print(f"  │  {_c(_DIM, 'Perplexity        :')}  {_c(_WH, _B, f'{ppl:.2f}')}  {_c(_DIM, '(lower = better; random baseline ≈ vocab size)')}")
    print(f"  │  {_c(_DIM, 'Bits per char     :')}  {_c(_WH, f'{bpc:.4f}')}  {_c(_DIM, '(loss base-2; well-trained char models reach ~1.2)')}")
    cli_section_end(W)


def run_model_stats():
    """Display model statistics."""
    cli_banner("Model Statistics", "Parameter counts and layer breakdown", width=64)
    if not os.path.exists(CONFIG_PATH):
        pwarn("No config found — train a model first."); return
    cfg = load_json(CONFIG_PATH)
    vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
    model = build_model(cfg, vocab.size)

    msel = cfg["model_selection"]
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen       = total_params - trainable

    W = 62
    cli_section("Model Overview", W)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Architecture   :')}  {_c(_WH, _B, MODEL_NAMES.get(msel, '?'))}")
    print(f"  │  {_c(_DIM, 'Embed dim      :')}  {_c(_WH, cfg['embed_dim'])}")
    print(f"  │  {_c(_DIM, 'Layers         :')}  {_c(_WH, cfg['layer_count'])}")
    print(f"  │  {_c(_DIM, 'Sequence len   :')}  {_c(_WH, cfg['seq_len'])}")
    print(f"  │  {_c(_DIM, 'Vocab size     :')}  {_c(_WH, vocab.size)}")
    print(f"  │  {_c(_DIM, 'Iterations done:')}  {_c(_WH, cfg.get('iterations_done', 0))}")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Total params   :')}  {_c(_WH, _B, readable_num(total_params))}  {_c(_DIM, f'({total_params:,})')}")
    print(f"  │  {_c(_DIM, 'Trainable      :')}  {_c(_WH, readable_num(trainable))}  {_c(_DIM, f'({100*trainable/max(1,total_params):.1f}%)')}")
    if frozen:
        print(f"  │  {_c(_DIM, 'Frozen         :')}  {_c(_WH, readable_num(frozen))}")
    print(f"  │  {_c(_DIM, 'Size (fp32)    :')}  {_c(_WH, f'{total_params * 4 / 1024 / 1024:.1f} MB')}  {_c(_DIM, '(fp16 would be half that)')}")
    cli_section_end(W)

    layer_params = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        layer_params[top] = layer_params.get(top, 0) + param.numel()

    cli_section("Layer Breakdown", W)
    print(f"  │  {_c(_DIM, 'Grouped by top-level module name, sorted by parameter count.')}")
    print(f"  │")
    for name, count in sorted(layer_params.items(), key=lambda x: -x[1]):
        pct   = 100 * count / max(1, total_params)
        bar_w = int(pct / 2)            # 1 char per 2%
        bar   = _c(_CY, "█" * bar_w) + _c(_DIM, "░" * (50 - bar_w))
        print(f"  │  {_c(_WH, f'{name:<24}')} {_c(_DIM, readable_num(count)):>12}  {_c(_YL, f'{pct:5.1f}%')}  {bar}")
    cli_section_end(W)


def run_token_analysis():
    """Analyze token distribution in a dataset."""
    cli_banner("Token Analysis", "Vocabulary coverage, frequency, and entropy", width=64)

    cli_section("Settings", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Dataset file to analyse. The first 10 MB will be read.')}")
    dataset_path = prompt_str("Dataset file path")
    if not os.path.exists(dataset_path):
        pwarn("File not found."); return

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Tokenizer — must match the one used during training for')}")
    print(f"  │  {_c(_DIM, 'the statistics to be meaningful.')}")
    cli_opt(-1, "Binary",   "Raw bytes as binary — 256 possible values")
    cli_opt( 0, "Byte",     "Byte-level 0–255 — universal, no text decoding needed")
    cli_opt( 1, "Char",     "Character-level — vocab from dataset characters")
    cli_opt( 2, "Word",     "Whitespace-split words")
    cli_opt( 3, "Tiktoken", "GPT-4 cl100k_base BPE (50k vocab)")
    cli_opt( 4, "BPE",      "Custom BPE trained on your data")
    print(f"  │")
    tokenizer_mode = prompt_int("Tokenizer", valid={-1,0,1,2,3,4})
    cfg_dummy = {"dataset_type": 0, "tokenizer_mode": tokenizer_mode, "custom_bpe_size": 4096}
    if tokenizer_mode == 3:
        cfg_dummy["tiktoken_encoding"] = prompt_str("Tiktoken encoding", default="cl100k_base")
    cli_section_end(64)

    vocab = load_or_make_vocab(cfg_dummy, dataset_path)

    max_read = 10 * 1024 * 1024
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(max_read)
    ids = vocab.encode(text)

    counter = collections.Counter(ids)
    unique = len(counter); total = len(ids)
    coverage = 100 * unique / max(1, vocab.size)
    compression = len(text.encode("utf-8")) / max(1, total * 2)

    W = 62
    cli_section("Statistics", W)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Total tokens  :')}  {_c(_WH, _B, readable_num(total))}")
    print(f"  │  {_c(_DIM, 'Unique tokens :')}  {_c(_WH, unique)}  {_c(_DIM, f'of {vocab.size} in vocab  ({coverage:.1f}% used)')}")
    print(f"  │  {_c(_DIM, 'Compression   :')}  {_c(_WH, f'{compression:.2f}x')}  {_c(_DIM, '(UTF-8 bytes / (tokens × 2)  — >1 = vocab saves space)')}")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Top-N — show the most frequent token types.')}")
    print(f"  │  {_c(_DIM, 'High concentration in a few tokens = low entropy dataset.')}")
    top_n = prompt_int("Show top N tokens", default=20)
    cli_section_end(W)

    print()
    cli_section("Top Tokens", W)
    print(f"  │  {'Rank':<5}  {'Token':<32}  {'Count':>8}  {'Freq':>6}")
    cli_rule(W - 2)
    for rank, (tid, cnt) in enumerate(counter.most_common(top_n), 1):
        pct = 100 * cnt / total
        try: decoded = repr(vocab.decode([tid]))
        except: decoded = f"<id={tid}>"
        if len(decoded) > 30: decoded = decoded[:27] + "..."
        print(f"  │  {rank:<5}  {_c(_WH, f'{decoded:<32}')}  {cnt:>8,}  {_c(_YL, f'{pct:5.2f}%')}")
    cli_section_end(W)

    probs = np.array([c / total for c in counter.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = math.log2(unique)
    efficiency = 100 * entropy / max(1, max_entropy)

    print()
    cli_section("Entropy", W)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Shannon entropy  :')}  {_c(_WH, _B, f'{entropy:.4f}')} bits/token")
    print(f"  │  {_c(_DIM, 'Max possible     :')}  {_c(_WH, f'{max_entropy:.4f}')} bits/token  {_c(_DIM, '(uniform distribution)')}")
    print(f"  │  {_c(_DIM, 'Efficiency       :')}  {_c(_WH, f'{efficiency:.1f}%')}  {_c(_DIM, '(100% = perfectly uniform vocab usage)')}")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Low entropy = vocabulary is dominated by a few tokens.')}")
    print(f"  │  {_c(_DIM, 'High entropy = tokens are spread more evenly.')}")
    cli_section_end(W)


def run_export():
    """Export model."""
    cli_banner("Export", "Save model in various portable formats", width=64)
    if not os.path.exists(CONFIG_PATH):
        pwarn("No config found — train a model first."); return
    cfg = load_json(CONFIG_PATH)
    vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
    model = build_model(cfg, vocab.size)
    model.to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    cli_section("Export Format", 64)
    print(f"  │")
    cli_opt(0, "TorchScript (.pt)",    "Serialised traced graph — runs without Python source")
    cli_opt(1, "ONNX (.onnx)",         "Open standard — compatible with TensorRT, ONNX Runtime, etc.")
    cli_opt(2, "State dict (.pth)",    "Raw weight dictionary + JSON config — easiest to reload")
    cli_opt(3, "Quantised int8 (.pth)","Weights quantised to 8-bit — ~4× smaller file, slight quality loss")
    print(f"  │")
    cli_section_end(64)
    choice = prompt_int("Export format", valid={0,1,2,3})
    out_name = prompt_str("Output filename  (no extension)", default="exported_model")
    
    if choice == 2:
        torch.save(model.state_dict(), f"{out_name}.pth")
        save_json(f"{out_name}_config.json", cfg)
        print(f"Saved {out_name}.pth + config")
    elif choice == 3:
        sd = model.state_dict()
        q = {}
        for k, v in sd.items():
            if v.dtype in (torch.float32, torch.float16) and v.numel() > 100:
                scale = v.abs().max() / 127.0
                q[k] = {"data": (v / scale).to(torch.int8), "scale": scale}
            else: q[k] = v
        torch.save(q, f"{out_name}_int8.pth")
        orig = os.path.getsize(CHECKPOINT_PATH) / 1024 / 1024
        new = os.path.getsize(f"{out_name}_int8.pth") / 1024 / 1024
        print(f"Saved {out_name}_int8.pth ({orig:.1f}MB -> {new:.1f}MB)")
    elif choice == 0:
        try:
            dummy = torch.randint(0, vocab.size, (1, cfg["seq_len"]), device=DEVICE)
            if cfg["model_selection"] in RNN_MODEL_IDS:
                scripted = torch.jit.trace(model, (dummy, None))
            else:
                scripted = torch.jit.trace(model, dummy)
            scripted.save(f"{out_name}.pt")
            print(f"Saved TorchScript to {out_name}.pt")
        except Exception as e:
            print(f"TorchScript failed: {e}")
    elif choice == 1:
        try:
            dummy = torch.randint(0, vocab.size, (1, cfg["seq_len"]), device=DEVICE)
            torch.onnx.export(model, dummy, f"{out_name}.onnx",
                input_names=["input_ids"], output_names=["logits"])
            print(f"Saved ONNX to {out_name}.onnx")
        except Exception as e:
            print(f"ONNX failed: {e}")


def run_hyperparam_sweep():
    """Simple hyperparameter search."""
    cli_banner("Hyperparameter Sweep", "Grid search across embed / layers / lr / batch", width=64)

    cli_section("Dataset & Tokenizer", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'All configs in the sweep share the same dataset and tokenizer.')}")
    dataset_path   = prompt_str("Dataset file path")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Dataset type — 0 = sliding-window corpus, 1 = one example per line.')}")
    dataset_type   = prompt_int("Dataset type  (0=standard  1=line)", valid={0,1})
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Tokenizer — must be consistent across the whole sweep.')}")
    cli_opt(-1,"Binary"); cli_opt(0,"Byte"); cli_opt(1,"Char")
    cli_opt(2,"Word");    cli_opt(3,"Tiktoken"); cli_opt(4,"BPE")
    print(f"  │")
    tokenizer_mode = prompt_int("Tokenizer", valid={-1,0,1,2,3,4})
    cli_section_end(64)

    cfg_dummy = {"dataset_type": dataset_type, "tokenizer_mode": tokenizer_mode, "custom_bpe_size": 4096}
    if tokenizer_mode == 3: cfg_dummy["tiktoken_encoding"] = "cl100k_base"
    vocab = load_or_make_vocab(cfg_dummy, dataset_path)

    print_model_menu()
    msel = prompt_int("Model #", valid=set(range(99)))

    cli_section("Sweep Grid", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Enter comma-separated values for each axis.')}")
    print(f"  │  {_c(_DIM, 'Every combination is tried — 2 dims × 2 lrs × 2 batches = 8 runs.')}")
    print(f"  │")

    if dataset_type == 0:
        print(f"  │  {_c(_DIM, 'Sequence length — fixed for all configs in this sweep.')}")
        seq_len = prompt_int("Sequence length", default=128)
    else:
        seq_len = 0

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Iterations per run — gradient steps each config trains for.')}")
    print(f"  │  {_c(_DIM, 'Keep low (200–1000) so the sweep completes in reasonable time.')}")
    iters_per = prompt_int("Iterations per run", default=500)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Embedding dimensions to test  (comma-separated):')}")
    embed_dims   = [int(x) for x in prompt_str("Embed dims", default="256").split(",")]
    print(f"  │  {_c(_DIM, 'Layer counts to test  (comma-separated):')}")
    layer_counts = [int(x) for x in prompt_str("Layer counts", default="4").split(",")]
    print(f"  │  {_c(_DIM, 'Learning rates to test  (comma-separated, e.g. 1e-3,5e-4):')}")
    lrs          = [float(x) for x in prompt_str("Learning rates", default="1e-3").split(",")]
    print(f"  │  {_c(_DIM, 'Batch sizes to test  (comma-separated):')}")
    batch_sizes  = [int(x) for x in prompt_str("Batch sizes", default="32").split(",")]
    cli_section_end(64)

    cfg_data = {"dataset_path": dataset_path, "dataset_type": dataset_type, "seq_len": seq_len,
                "vocab_tokens": getattr(vocab, "tokens", None), "val_split": 0.1, "valid_examples": 0}
    train_ds, valid_ds = build_datasets(cfg_data, vocab)
    if dataset_type == 1:
        cfg_data["seq_len"] = min(train_ds.max_len, 2048)
        seq_len = cfg_data["seq_len"]

    configs = [{"embed_dim": e, "layer_count": l, "learning_rate": lr, "batch_size": b}
               for e in embed_dims for l in layer_counts for lr in lrs for b in batch_sizes]

    results = []
    print(f"\n  {_c(_GR, _B, '▸')} Running {_c(_WH, len(configs))} configurations…\n")
    for idx, hp in enumerate(configs, 1):
        cfg = cfg_data.copy()
        cfg.update({"model_selection": msel, "head_count": 4, "activation_name": "gelu",
                    "dropout": 0.0, "tokenizer_mode": tokenizer_mode, "seq_len": seq_len, **hp})
        desc = f"e={hp['embed_dim']} L={hp['layer_count']} lr={hp['learning_rate']:.0e} bs={hp['batch_size']}"
        try:
            model = build_model(cfg, vocab.size); model.to(DEVICE)
            opt   = build_optimizer(model, cfg)
            score, _, status = bench_train_loop(cfg, model, opt, train_ds, valid_ds, vocab,
                                                (dataset_type==1), iters_per, 0, True)
            results.append({"hp": hp, "score": score, "status": status})
            pok(f"[{idx}/{len(configs)}]  {_c(_WH, desc)}  →  {_c(_GR, f'{score:.5f}')}  {_c(_DIM, status)}")
            del model, opt; torch.cuda.empty_cache()
        except Exception as e:
            results.append({"hp": hp, "score": float("inf"), "status": str(e)})
            pwarn(f"[{idx}/{len(configs)}]  {_c(_WH, desc)}  →  {_c(_RD, 'CRASH:')} {e}")

    results.sort(key=lambda x: x["score"])
    W = 72
    print()
    cli_section("Sweep Results", W)
    hdr = f"  {'#':<4}  {'embed':>5}  {'layers':>6}  {'lr':>8}  {'batch':>5}  {'score':>10}  status"
    print(f"  │{_c(_DIM, hdr)}")
    cli_rule(W - 2)
    for i, r in enumerate(results, 1):
        hp      = r["hp"]
        score_s = f"{r['score']:.5f}" if r["score"] != float("inf") else "∞"
        score_c = _c(_GR, _B, score_s) if i == 1 else (_c(_RD, score_s) if r["score"] == float("inf") else score_s)
        rank_c  = _c(_YL, _B, f"{i:<4}") if i == 1 else f"{i:<4}"
        print(f"  │  {rank_c}  {hp['embed_dim']:>5}  {hp['layer_count']:>6}  {hp['learning_rate']:>8.1e}  {hp['batch_size']:>5}  {score_c:>10}  {_c(_DIM, r['status'])}")
    cli_section_end(W)
    if results and results[0]["score"] != float("inf"):
        b = results[0]["hp"]
        blr = f"{b['learning_rate']:.2e}"
        print(f"\n  {_c(_GR, '★')} Best config: embed={_c(_WH,b['embed_dim'])} layers={_c(_WH,b['layer_count'])} lr={_c(_WH,blr)} batch={_c(_WH,b['batch_size'])}\n")


def run_speed_benchmark():
    """Measure throughput (tokens/second) for models."""
    cli_banner("Speed Benchmark", "Forward + backward pass throughput in tokens / second", width=64)
    if not os.path.exists(CONFIG_PATH):
        pwarn("No config found — train a model first."); return
    cfg = load_json(CONFIG_PATH)
    vocab = load_or_make_vocab(cfg, cfg["dataset_path"])

    cli_section("Settings", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Sequence length and batch size determine the tokens-per-step.')}")
    print(f"  │  {_c(_DIM, 'Use the same values you train with for a realistic comparison.')}")
    seq_len    = prompt_int("Sequence length", default=cfg["seq_len"])
    batch_size = prompt_int("Batch size",      default=cfg["batch_size"])
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Warmup steps — run these steps first to let CUDA / JIT settle.')}")
    print(f"  │  {_c(_DIM, 'Their time is discarded. 10 is usually enough.')}")
    warmup     = prompt_int("Warmup steps",    default=10)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Measure steps — the steps actually timed and averaged.')}")
    print(f"  │  {_c(_DIM, 'More steps = more stable result. 50 is a good default.')}")
    measure    = prompt_int("Measure steps",   default=50)
    cli_section_end(64)

    print_model_menu()
    speed_hint = "Enter comma-separated IDs, or 'all' to test every architecture."
    print(f"  {_c(_DIM, speed_hint)}\n")
    raw = prompt_str("Model IDs  (comma-separated or 'all')", default="all")
    if raw.lower() == "all":
        model_ids = sorted(MODEL_NAMES.keys())
    else:
        model_ids = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]

    results = []
    for msel in model_ids:
        cfg_t = cfg.copy()
        cfg_t["model_selection"] = msel
        cfg_t["seq_len"]         = seq_len
        cfg_t["batch_size"]      = batch_size
        name = MODEL_NAMES.get(msel, f"Model {msel}")
        try:
            model   = build_model(cfg_t, vocab.size); model.to(DEVICE); model.train()
            total_p = sum(p.numel() for p in model.parameters())
            opt     = build_optimizer(model, cfg_t)
            dx = torch.randint(0, vocab.size, (batch_size, seq_len), device=DEVICE)
            dy = torch.randint(0, vocab.size, (batch_size, seq_len), device=DEVICE)
            crit = nn.CrossEntropyLoss()
            for _ in range(warmup):
                if msel in RNN_MODEL_IDS: lg = model(dx, None)[0]
                elif msel in SCAN_MODEL_IDS: out = model(dx); lg = out[0] if isinstance(out, tuple) else out
                else: lg = model(dx)
                crit(lg.reshape(-1, lg.size(-1)), dy.reshape(-1)).backward(); opt.step(); opt.zero_grad(set_to_none=True)
            if DEVICE == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(measure):
                if msel in RNN_MODEL_IDS: lg = model(dx, None)[0]
                elif msel in SCAN_MODEL_IDS: out = model(dx); lg = out[0] if isinstance(out, tuple) else out
                else: lg = model(dx)
                crit(lg.reshape(-1, lg.size(-1)), dy.reshape(-1)).backward(); opt.step(); opt.zero_grad(set_to_none=True)
            if DEVICE == "cuda": torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tps = measure * batch_size * seq_len / elapsed
            results.append({"name": name, "tok_s": tps, "params": total_p})
            pok(f"{_c(_WH, f'{name:<42}')} {_c(_GR, _B, f'{tps:>10,.0f}')} tok/s  {_c(_DIM, readable_num(total_p) + ' params')}")
            del model, opt; torch.cuda.empty_cache()
        except Exception as e:
            pwarn(f"{_c(_WH, f'{name:<42}')} {_c(_RD, 'FAILED:')} {e}")
            results.append({"name": name, "tok_s": 0, "params": 0})

    results.sort(key=lambda x: -x["tok_s"])
    W = 70
    print()
    cli_section("Speed Ranking  (fastest first)", W)
    hdr = f"  {'#':<4}  {'Model':<42}  {'tok/s':>12}  Params"
    print(f"  │{_c(_DIM, hdr)}")
    cli_rule(W - 2)
    for i, r in enumerate(results, 1):
        tps_s = f"{r['tok_s']:>12,.0f}" if r["tok_s"] else f"{'FAILED':>12}"
        tps_c = _c(_GR, _B, tps_s) if i == 1 else (_c(_RD, tps_s) if not r["tok_s"] else tps_s)
        rank_c = _c(_YL, _B, f"{i:<4}") if i == 1 else f"{i:<4}"
        print(f"  │  {rank_c}  {r['name']:<42}  {tps_c}  {_c(_DIM, readable_num(r['params']))}")
    cli_section_end(W)


def interactive_train():
    cli_banner("LineGen", "Neural Text Generation Framework", width=64)

    W = 62
    print(f"  {_c(_CY, _B, '┌─')} {_c(_WH, _B, 'Modes')} {_c(_CY, '─' * (W - 12) + '┐')}")
    cli_opt("0 / t", "Train",        "Train a new model or resume from checkpoint")
    cli_opt("1 / s", "Sample",       "Generate text from a saved checkpoint")
    cli_opt("2 / b", "Benchmark",    "Compare multiple architectures head-to-head")
    cli_opt("3 / c", "Chat",         "Interactive multi-turn generation session")
    cli_opt("4 / p", "Perplexity",   "Evaluate model perplexity on a text file")
    cli_opt("5 / m", "Stats",        "Print parameter counts and layer breakdown")
    cli_opt("6 / a", "Tokens",       "Analyse tokenization of a text file")
    cli_opt("7 / e", "Export",       "Export to TorchScript / ONNX / quantized int8")
    cli_opt("8 / h", "Sweep",        "Hyperparameter grid search over a single model")
    cli_opt("9 / v", "Speed",        "Measure throughput in tokens / second")
    cli_blank_row()
    print(f"  {_c(_CY, '└' + '─' * (W - 2) + '┘')}\n")

    cont = prompt_str("Selection").lower().strip()
    
    if cont in ("train","t","0"):
        # Training Mode
        resume = False
        if os.path.exists(CONFIG_PATH) and os.path.exists(CHECKPOINT_PATH):
            if prompt_str("Resume previous run? (y/n): ").lower() == "y":
                resume = True

        if resume:
            cfg = load_json(CONFIG_PATH)
            # 1. Load Vocab & Save it immediately to prevent sync issues
            vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
            cfg["vocab_tokens"] = getattr(vocab, "tokens", None)
            save_json(CONFIG_PATH, cfg) 
            
            dataset, valid = build_datasets(cfg, vocab)
            model = build_model(cfg, vocab.size)
            model.to(DEVICE)
            #if torch.__version__.startswith("2."):
            #    model = wrap_model_with_compile(model, cfg)
            
            print(f"[Resume] Loading checkpoint {CHECKPOINT_PATH}...")
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        else:
            cfg = build_config_new()
            # 1. Load Vocab & Save it immediately
            vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
            cfg["vocab_tokens"] = getattr(vocab, "tokens", None)
            save_json(CONFIG_PATH, cfg)

            dataset, valid = build_datasets(cfg, vocab)
            model = build_model(cfg, vocab.size)
            model.to(DEVICE)
            #if torch.__version__.startswith("2."):
            #    model = wrap_model_with_compile(model, cfg)
        
        # Print model stats
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model: {MODEL_NAMES.get(cfg['model_selection'], '?')} | Params: {readable_num(total_params)} | Vocab: {vocab.size}")
        
        model.to(DEVICE)
        opt = build_optimizer(model, cfg)
        
        train_loop(cfg, model, opt, dataset, valid, vocab, cfg["dataset_type"]==1)

    elif cont in ("sample","s","1"):
        # Sampling Mode
        if not os.path.exists(CONFIG_PATH):
            print("No config found. Train first.")
            return

        cfg = load_json(CONFIG_PATH)
        vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
        
        # Build model with vocab size from config (now correctly reconstructed)
        model = build_model(cfg, vocab.size)
        model.to(DEVICE)
        #if torch.__version__.startswith("2."):
        #    model = wrap_model_with_compile(model, cfg)
        
        print(f"[Sample] Loading {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

        model.to(DEVICE)
        run_sampling_ui(cfg, model, vocab)

    elif cont in ("benchmark","b","2"):
        run_benchmark()
    
    elif cont in ("chat","c","3"):
        if not os.path.exists(CONFIG_PATH):
            print("No config found. Train first."); return
        cfg = load_json(CONFIG_PATH)
        vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
        model = build_model(cfg, vocab.size); model.to(DEVICE)
        print(f"Loading {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        run_interactive_chat(cfg, model, vocab)
    
    elif cont in ("perplexity","p","4"):
        run_perplexity_eval()
    
    elif cont in ("stats","m","5"):
        run_model_stats()
    
    elif cont in ("tokens","a","6"):
        run_token_analysis()
    
    elif cont in ("export","e","7"):
        run_export()
    
    elif cont in ("sweep","h","8"):
        run_hyperparam_sweep()
    
    elif cont in ("speed","v","9"):
        run_speed_benchmark()
    
    else:
        print("Unknown choice.")


def get_prompt_batch(cfg, vocab: CharVocab):
    line_mode = (cfg["dataset_type"]==1)
    cli_section("Prompt", 64)
    print(f"  │  {_c(_DIM, 'The prompt seeds generation — the model continues from where it left off.')}")
    print(f"  │")
    cli_opt(0, "Random token",  "Pick a random token from the vocabulary as the seed")
    cli_opt(1, "'BEGIN'",       "Use the literal text 'BEGIN' as the prompt")
    cli_opt(2, "File",          "Load prompts line-by-line from a text file")
    cli_opt(3, "Custom",        "Type your own prompt text (or hex if byte mode)")
    print(f"  │")
    cli_section_end(64)
    smode = prompt_int("Prompt mode", valid={0,1,2,3})
    prompts: List[str] = []
    if smode == 0:
        prompts = [BOS_TOKEN] if line_mode else [random.choice(vocab.tokens)]
    elif smode == 1:
        prompts = ["BEGIN"]
        if line_mode: prompts = [BOS_TOKEN + prompts[0]]
    elif smode == 2:
        f = prompt_str("Path to prompt file")
        if not os.path.exists(f): pwarn("File not found."); return []
        if line_mode:
            with open(f,"r",encoding="utf-8") as r: prompts = [BOS_TOKEN + ln.rstrip("\n") for ln in r.readlines()]
        else:
            with open(f,"r",encoding="utf-8") as r: prompts = [r.read()]
    else:
        p = prompt_str("Custom prompt")
        prompts = [BOS_TOKEN + p] if line_mode else [p]
    return prompts

def _visible_decode_prompt(vocab, ids: List[int], line_mode: bool) -> str:
    """Decode but hide BOS if line mode."""
    if hasattr(vocab, "bos_id") and line_mode and vocab.bos_id is not None:
        ids = [i for i in ids if i != vocab.bos_id]
    return vocab.decode(ids)

def run_sampling_ui(cfg, model, vocab):
    model.eval()
    if int(cfg.get("tokenizer_mode",1)) == 0:
        ensure_filegen_clean()

    cli_banner("Sample", "Generate text from the trained model", width=64)
    cli_section("Generation Settings", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'How many independent samples to generate in this run.')}")
    count = prompt_int("Number of samples", default=1)

    line_mode = (cfg["dataset_type"]==1)
    byte_text = bool(cfg.get("byte_output_text", False))
    if not line_mode:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Maximum tokens to generate per sample.')}")
        print(f"  │  {_c(_DIM, 'In line mode this is set automatically from the model config.')}")
        max_len = prompt_int("Max generated tokens", default=200)
    else:
        max_len = cfg["seq_len"]

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Temperature — scales the logits before sampling.')}")
    print(f"  │  {_c(_DIM, '  0   = greedy (always picks the top token, fully deterministic)')}")
    print(f"  │  {_c(_DIM, '  1.0 = unmodified distribution')}")
    print(f"  │  {_c(_DIM, '  >1  = more random / creative,  <1 = more focused / repetitive')}")
    temp = prompt_float("Temperature  (0 = greedy)", default=cfg.get("temperature", 1.0))
    cfg["temperature"] = temp

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Top-k filtering — keep only the k most likely tokens at each step.')}")
    print(f"  │  {_c(_DIM, 'Prevents sampling very unlikely tokens. 0 = disabled.')}")
    print(f"  │  {_c(_DIM, 'Typical values: 20–200.')}")
    top_k = prompt_int("Top-k  (0 = off)", default=0)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Top-p (nucleus) — keep the smallest set of tokens whose cumulative')}")
    print(f"  │  {_c(_DIM, 'probability exceeds p, then sample from that set only.')}")
    print(f"  │  {_c(_DIM, 'Adapts dynamically to confidence. 0.0 = disabled. Typical: 0.9.')}")
    top_p = prompt_float("Top-p  (0.0 = off)", default=0.0)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Repetition penalty — divides logits of recently seen tokens,')}")
    print(f"  │  {_c(_DIM, 'making the model less likely to repeat itself. 1.0 = off.')}")
    print(f"  │  {_c(_DIM, 'Values 1.1–1.3 reduce loops without distorting output much.')}")
    rep_penalty = prompt_float("Repetition penalty  (1.0 = off)", default=1.0)
    cfg["_top_k"] = top_k
    cfg["_top_p"] = top_p
    cfg["_rep_penalty"] = rep_penalty

    want_capture = False
    if isinstance(model, BuiltinRNNWrapper):
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Activation capture — saves per-timestep hidden state vectors to')}")
        print(f"  │  {_c(_DIM, 'binary .bin files in FileGen/ for offline visualisation.')}")
        yn = prompt_str("Record per-timestep activations?  (y/n)", default="n").lower()
        want_capture = yn in ("y", "yes", "1")
    cli_section_end(64)

    prompts = get_prompt_batch(cfg, vocab)
    if not prompts: return

    outputs = []
    tmode = int(cfg.get("tokenizer_mode",1))
    out_dir = pathlib.Path("FileGen")

    for i in range(count):
        prompt = prompts[i % len(prompts)]
        p_ids = vocab.encode(prompt)

        if tmode == 0:
            p_bytes = prompt if isinstance(prompt, (bytes, bytearray, memoryview)) else bytes(str(prompt), "latin1", "ignore")
            print(bold(f"--- PROMPT (hex) ---\n{p_bytes.hex()}"))
        else:
            vis = _visible_decode_prompt(vocab, p_ids, line_mode)
            print(bold(f"--- PROMPT ---\n{vis}"))

        # ===== NEW: begin capture (builtins only) =====
        if want_capture:
            model.start_capture()

        if line_mode:
            # Force stream=False if capturing, so we have the ids for saving
            out_ids = generate_line_mode(model, cfg, vocab, p_ids, limit_len=max_len)
            # Trim leading BOS for readable text output
            if tmode == 0:
                data = vocab.to_bytes(out_ids) if hasattr(vocab, "to_bytes") else bytes()
                if byte_text:
                    print(f"--- sample {i+1} ---\n{vocab.decode(out_ids[1:]) if len(out_ids)>1 else ''}")
                else:
                    (out_dir / f"sample_{i+1:03d}").write_bytes(data)
                    print(f"[saved] FileGen/sample_{i+1:03d} ({len(data)} bytes)")
            else:
                text = vocab.decode(out_ids[1:]) if len(out_ids)>1 else ""
                outputs.append(text)
        else:
            # Classic
            if tmode == 0:
                out_ids = generate_classic(model, cfg, vocab, p_ids, max_len=max_len, stream=False)
                if byte_text:
                    print(f"--- sample {i+1} ---\n{vocab.decode(out_ids)}")
                else:
                    data = vocab.to_bytes(out_ids) if hasattr(vocab, "to_bytes") else bytes()
                    (out_dir / f"sample_{i+1:03d}").write_bytes(data)
                    print(f"[saved] FileGen/sample_{i+1:03d} ({len(data)} bytes)")
            else:
                out_ids = generate_classic(model, cfg, vocab, p_ids, max_len=max_len, stream=True)
                sys.stdout.write(f"--- sample {i+1} ---\n")
                sys.stdout.write(vocab.decode(out_ids))
                sys.stdout.write("\n")

        # ===== NEW: finish + dump capture =====
        if want_capture:
            model.stop_capture()
            cap = model.get_captured()  # list of [1, S, H_l] or None

            # Build per-step token strings aligned to S steps (exclude prompt)
            # For both classic & line modes, sampling loops generate exactly K new tokens.
            # In out_ids, the number of newly generated tokens is:
            #   - classic: len(out_ids) - len(p_ids)
            #   - line:    len(out_ids) - len(p_ids)
            gen_only = out_ids[len(p_ids):]
            # visible per-step tokens (hide BOS visually in line mode)
            step_tokens = []
            for tok_id in gen_only:
                if line_mode and hasattr(vocab, "bos_id") and tok_id == getattr(vocab, "bos_id", None):
                    step_tokens.append("")  # hide BOS
                else:
                    step_tokens.append(vocab.decode([tok_id]))

            # Ensure FileGen/ exists
            out_dir.mkdir(parents=True, exist_ok=True)
            bin_path = out_dir / f"activations_{i+1:03d}.bin"
            save_activation_capture_bin(str(bin_path), step_tokens, cap)
            print(f"[saved] {bin_path} (activations + tokens)")


    if line_mode and tmode != 0:
        print("==== BATCH OUTPUTS ====")
        for i,o in enumerate(outputs, 1):
            print(f"--- sample {i} ---\n{o}\n")


def train_for_iterations(cfg, model, optimizer, dataset, valid_ds, vocab, line_mode, iters_total, loss_window=100):
    criterion = nn.CrossEntropyLoss()
    model.train()
    iters = 0
    losses = []

    # simple minibatch fetcher (no TBPTT in benchmarks)
    def fetch():
        return dataset.get_batch(cfg["batch_size"])

    # Build a nice tqdm description
    name = None
    try:
        name = MODEL_NAMES.get(cfg["model_selection"], None)  # optional, if available
    except Exception:
        pass
    model_sel = cfg.get("model_selection", "?")
    desc = f"[bench] {name if name else f'model {model_sel}'}"


    msel = cfg["model_selection"]
    with tqdm(total=iters_total, desc=desc, ncols=100, leave=False) as pbar:
        while iters < iters_total:
            x, y = fetch()
            if msel in RNN_MODEL_IDS:
                logits = model(x, None)[0]
            elif msel in SCAN_MODEL_IDS:
                out = model(x); logits = out[0] if isinstance(out, tuple) else out
            else:
                logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                iters += 1; pbar.update(1)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # track losses
            l = float(loss.item())
            losses.append(l)
            if len(losses) > loss_window:
                losses.pop(0)

            # update progress bar with current + rolling avg loss
            avg = sum(losses) / max(1, len(losses))
            pbar.set_postfix(loss=f"{l:.5f}", avg=f"{avg:.5f}")
            pbar.update(1)

            iters += 1

    # Decide score
    use_valid = False
    if line_mode and valid_ds is not None and cfg.get("valid_examples", 0) > 1000:
        use_valid = True
    if (not line_mode) and valid_ds is not None and bool(cfg.get("classic_val_path", "")):
        use_valid = True

    if use_valid:
        vloss = eval_valid_loss(model, cfg, valid_ds, vocab, line_mode=line_mode, max_samples=1000)
        return float(vloss) if vloss is not None else (sum(losses) / max(1, len(losses)))
    else:
        return sum(losses) / max(1, len(losses))

# ==============================================================================
#  MODEL NAMES MAPPING
# ==============================================================================
# ==============================================================================
#  MODEL DEFINITIONS (User Spec)
# ==============================================================================

MODEL_NAMES = {
    # ==== MLPs ====
    0: "MLP (one-hot encoding window instead of embeddings, Mish activation)",
    1: "MLP (MLP Block from Transformer, no attention, Mish activation)",
    
    # ==== RNNs ====
    2: "Recurrent Neural Network (Vanilla Tanh)",
    3: "Recurrent Neural Network (ReLU)",
    4: "Gated Recurrent Unit",
    5: "Long Short-Term Memory",
    6: "Independently Recurrent Neural Network (IndRNN)",
    7: "IndyGRU",
    8: "ATanU activated LSTM",
    18: "JANET (forget-gate LSTM)",
    23: "Liquid Neural Network (LTC)",

    # ==== Non-recurrents (Transformers/Convs) ====
    9: "Temporal ConvNet, Mish activation",
    10: "GPT-2 Decoder-only Transformer",
    19: "HyperMixer",
    21: "gMLP, causal",
    22: "aMLP, causal (gMLP with TinyAttention)",
    24: "MLPMixer (Causal)",
    25: "Modern Transformer (Llama3 Decoder-only Transformer)",
    31: "MEGABYTE",
    34: "KAN-Transformer (Chebyshev)",
    37: "DCT-Former",

    # ==== xLSTM ====
    11: "xLSTM (sLSTM only)",
    12: "xLSTM (mLSTM only)",
    13: "xLSTM (mixed m:s)",

    # ==== Space State Machines / Linear Recurrence ====
    14: "Mamba (selective scan)",
    15: "minGRU (Parallelized GRU)",
    16: "minLSTM (Parallelized LSTM)",
    17: "RWKV (scan)",
    20: "GateLoop (scan)",
    26: "MinRNN (Parallelized Vanilla RNN)",
    27: "Griffin (RG-LRU)",
    28: "DeltaNet",
    29: "RetNet",
    30: "HGRN",
    32: "MinIndRNN (Parallelized IndRNN)",
    33: "MinJANET (Parallelized JANET)",
    35: "Linear Transformer (Recurrent)",
    36: "H3 (Hungry Hungry Hippos)",
    38: "MinIndyGRU (Parallelized IndyGRU)",
    39: "MinIndyLSTM (Parallelized IndyLSTM)"
}

# Used to trigger the "RNN Specific Settings" menu (residuals, norms, etc)
# These are models that are likely implemented via the BuiltinRNNWrapper or similar loops
def build_config_benchmark():
    dataset_path = prompt_str("Dataset location (text file path): ")
    dataset_type = prompt_int("Dataset type: standard/0 or line/1? ", valid={0,1})
    tokenizer_mode = prompt_int("Tokenizer (-1=binary, 0=byte, 1=char, 2=word): ", valid={-1,0,1,2})

    print(activation_menu_text())
    names = activation_names()
    a_idx = prompt_int("Activation #: ", valid=set(range(len(names))))
    activation_name = names[a_idx]

    embed_dim  = prompt_int("Embed dim (also hidden dim / TCN channels / Transformer d_model): ")
    layer_count = prompt_int("Layer count: ")

    # Only ask if this model actually uses attention heads
    head_count = 4
    if msel in ATTN_MODEL_IDS:
        head_count = prompt_int("Head count: ", default=4)

    if dataset_type == 0:
        seq_len = prompt_int("Seq len (classic corpus): ")
        classic_val_path = prompt_str("Optional validation file path (enter to skip): ", default="")
    else:
        seq_len = 0
        classic_val_path = prompt_str("Optional validation file path (enter to skip): ", default="")

    batch_size  = prompt_int("Batch size: ")
    learning_rate = prompt_float("Learning rate (Adam): ")
    iters_total = prompt_int("Benchmark iteration count (total SGD steps per model): ")
    loss_window = prompt_int("Average-of-last-N loss window (used if no validation): ", default=200)

    cfg = RunConfig(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        model_selection=0,
        activation_name=activation_name,
        embed_dim=embed_dim,
        head_count=head_count,
        layer_count=layer_count,
        seq_len=seq_len,
        epoch_count=1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tokenizer_mode=tokenizer_mode
    ).to_dict()

    if classic_val_path:
        cfg["classic_val_path"] = classic_val_path

    cfg["use_tbptt"] = False
    cfg["bptt_window"] = 0
    cfg["tbptt_total_len"] = 0

    cfg["_bench_iters"] = iters_total
    cfg["_bench_loss_window"] = loss_window
    return cfg


# ==============================================================================
#  BENCHMARKING SUITE (Updated)
# ==============================================================================

STRICT_RNN_IDS = {2, 3, 4, 5, 6, 7, 8, 18, 23} 

# Group Definitions for the Menu
BENCH_GROUPS = {
    # Feedforwards / Attention / Convs
    2: [0, 1, 9, 10, 19, 21, 22, 24, 25, 31, 34, 37], 
    
    # Strict RNNs (The ones that usually crash without careful initialization)
    3: [2, 3, 4, 5, 6, 7, 8, 18, 23],
    
    # Scan / Linear Recurrence / SSMs / xLSTM (Modern Sequence Models)
    4: [11, 12, 13, 14, 15, 16, 17, 20, 26, 27, 28, 29, 30, 32, 33, 35, 36, 38, 39],
    
    # "Standard Only" (Excludes the very experimental custom RNNs and xLSTMs)
    # Keeping MLP, GPT, Standard RNNs, Mamba, RWKV
    5: [0, 1, 2, 3, 4, 5, 9, 10, 24, 21, 22, 19, 25, 31, 34, 14, 17, 26, 15, 16, 32, 38, 39, 33] 
}
# Group 0 and 1 are dynamic (ALL)

def get_bench_model_list():
    W = 64
    print(f"\n  {_c(_CY, _B, '┌─')} {_c(_WH, _B, 'Benchmark Group')} {_c(_CY, '─' * (W - 22) + '┐')}")
    cli_opt(0, "All models",         "Every architecture; ask once for MinRNN/IndRNN activation")
    cli_opt(1, "Ultra (all × acts)", "All models + every activation variant for MinRNN/IndRNN")
    cli_opt(2, "Feedforwards",        "MLPs, Transformers, TCN, Mixer variants only")
    cli_opt(3, "Classic RNNs",        "Vanilla RNN, GRU, LSTM, IndRNN, JANET, LTC…")
    cli_opt(4, "Scan / SSM",          "Mamba, RWKV, minGRU/LSTM, GateLoop, RetNet…")
    cli_opt(5, "Standard set",        "Excludes custom RNNs and xLSTM variants")
    cli_opt(6, "Custom list",         "Enter comma-separated model IDs manually")
    cli_blank_row()
    print(f"  {_c(_CY, '└' + '─' * (W - 2) + '┘')}\n")

    choice = prompt_int("Group", valid={0,1,2,3,4,5,6})

    minrnn_acts   = [0]
    minindrnn_acts = [0]
    tasks = []

    if choice in [0, 1]:
        base_ids = list(set(BENCH_GROUPS[2] + BENCH_GROUPS[3] + BENCH_GROUPS[4]))
    elif choice in BENCH_GROUPS:
        base_ids = BENCH_GROUPS[choice]
    elif choice == 6:
        print_model_menu()
        raw = prompt_str("Model IDs  (comma-separated, e.g. 0,10,14)")
        base_ids = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    else:
        base_ids = []

    test_all_acts = (choice == 1)

    if choice == 6 and (26 in base_ids or 32 in base_ids):
        test_all_acts = (prompt_int("Test all activations for MinRNN/IndRNN?  (1=yes 0=no)", valid={0,1}) == 1)

    if test_all_acts:
        minrnn_range    = range(6)
        minindrnn_range = range(18)
    else:
        if 26 in base_ids:
            cli_section("MinRNN Activation", 64)
            cli_opt(0,"Tanh"); cli_opt(1,"ReLU"); cli_opt(2,"SiLU")
            cli_opt(3,"GELU"); cli_opt(4,"Sigmoid"); cli_opt(5,"g_act")
            cli_section_end(64)
            a = prompt_int("Select for MinRNN", valid=set(range(6)))
            minrnn_range = [a]
        else:
            minrnn_range = [0]

        if 32 in base_ids:
            cli_section("MinIndRNN Activation", 64)
            for idx, name in enumerate(["Tanh","ReLU","SiLU","PReLU 0","PReLU def",
                                        "LReLU 0.2","LReLU 0.01","GELU","BentId",
                                        "Sine","Cosine","Snake","StepSine","StepCos",
                                        "Mish","Cone","ReLU²","g_act"]):
                cli_opt(idx, name, kw=3, lw=14)
            cli_section_end(64)
            a = prompt_int("Select for MinIndRNN", valid=set(range(18)))
            minindrnn_range = [a]
        else:
            minindrnn_range = [0]

    for mid in base_ids:
        if mid == 26:
            for act in minrnn_range:
                tasks.append((mid, {"minrnn_act": act}))
        elif mid == 32:
            for act in minindrnn_range:
                tasks.append((mid, {"minrnn_act": act}))
        else:
            tasks.append((mid, {}))

    return tasks

def bench_train_loop(cfg, model, optimizer, train_ds, valid_ds, vocab, line_mode, total_iters, fitness_mode, nan_skip):
    """
    Returns: (score, best_step, status_string)
    status_string is "OK" or "NaN"
    """
    # === FIX START: Handle None pad_id explicitly ===
    pad_id = getattr(vocab, "pad_id", -100)
    if pad_id is None: 
        pad_id = -100
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    # === FIX END ===
    model.train()
    
    losses = []
    best_valid_loss = float('inf')
    best_valid_step = 0
    
    # Validation settings from config
    val_freq = cfg.get("_val_freq", 100)
    
    # Simple batch fetcher
    def get_batch(ds):
        if hasattr(ds, 'get_batch'): return ds.get_batch(cfg["batch_size"])
        # Fallback for old style datasets if any
        return ds[0], ds[1] # dummy

    # Progress bar
    pbar_desc = f"[bench] {MODEL_NAMES.get(cfg['model_selection'], cfg['model_selection'])}"
    if "minrnn_act" in cfg: pbar_desc += f" (act={cfg['minrnn_act']})"
    
    with tqdm(total=total_iters, desc=pbar_desc, leave=False) as pbar:
        for i in range(1, total_iters + 1):
            try:
                x, y = get_batch(train_ds)
                
                # Forward
                if cfg["model_selection"] in RNN_MODEL_IDS:
                    out = model(x, None)
                    logits = out[0]
                elif cfg["model_selection"] in SCAN_MODEL_IDS:
                    out = model(x) # ScanLM handles None state internally usually
                    logits = out[0] if isinstance(out, tuple) else out
                else:
                    logits = model(x)

                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                
                # NaN Check
                if torch.isnan(loss) or torch.isinf(loss):
                    if nan_skip:
                        return float('inf'), i, "NaN"
                    else:
                        # If not skipping, we must zero grad and maybe try to recover, 
                        # but usually optimization is broken. We'll just log high loss.
                        loss = torch.tensor(100.0, device=DEVICE, requires_grad=True)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Record
                l_val = loss.item()
                losses.append(l_val)
                if len(losses) > 100: losses.pop(0)
                
                # Update pbar
                avg_train = sum(losses)/len(losses)
                pbar.set_postfix(loss=f"{l_val:.4f}", avg=f"{avg_train:.4f}")
                pbar.update(1)

                # Validation Logic (Fitness Mode 2)
                if fitness_mode == 2 and valid_ds is not None:
                    if i % val_freq == 0 or i == total_iters:
                        vloss = eval_valid_loss(model, cfg, valid_ds, vocab, line_mode, max_samples=cfg.get("_val_samples", 1000))
                        if vloss is not None:
                            if vloss < best_valid_loss:
                                best_valid_loss = vloss
                                best_valid_step = i
                            # pbar.write(f"   Step {i}: Valid Loss {vloss:.4f} (Best: {best_valid_loss:.4f} @ {best_valid_step})")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    return float('inf'), i, "OOM"
                if nan_skip:
                    return float('inf'), i, "Crash"
                raise e

    # === Final Scoring ===
    score = float('inf')
    
    if fitness_mode == 0: # Sliding Avg
        score = sum(losses) / max(1, len(losses))
        
    elif fitness_mode == 1: # Final Train Eval
        # Eval on 10 batches
        model.eval()
        with torch.no_grad():
            tmp_loss = 0
            count = 0
            for _ in range(10):
                x, y = get_batch(train_ds)
                out = model(x, None) if cfg["model_selection"] in RNN_MODEL_IDS else model(x)
                logits = out[0] if isinstance(out, tuple) else out
                l = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                tmp_loss += l.item()
                count += 1
        score = tmp_loss / count
        
    elif fitness_mode == 2: # Best Valid Loss
        # Perform one last check if not just done
        if total_iters % val_freq != 0:
            vloss = eval_valid_loss(model, cfg, valid_ds, vocab, line_mode, max_samples=cfg.get("_val_samples", 1000))
            if vloss is not None and vloss < best_valid_loss:
                best_valid_loss = vloss
                best_valid_step = total_iters
        
        score = best_valid_loss
        if score == float('inf'): score = 1000.0 # Fallback if validation failed entirely

    return score, best_valid_step, "OK"

def run_benchmark():
    cli_banner("Benchmark", "Compare architectures on the same dataset", width=64)

    # ── Dataset & tokenizer ────────────────────────────────────────────────────
    cli_section("Dataset", 64)
    print(f"  │")
    dataset_path = prompt_str("Dataset file path")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Dataset type:')}")
    cli_opt(0, "Standard (corpus)", "Sliding-window over a continuous text stream")
    cli_opt(1, "Line mode",         "One example per line with BOS/EOS padding")
    print(f"  │")
    dataset_type = prompt_int("Dataset type", valid={0,1})
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Tokenizer:')}")
    cli_opt(-1, "Binary"); cli_opt(0, "Byte"); cli_opt(1, "Char")
    cli_opt( 2, "Word");   cli_opt(3, "Tiktoken"); cli_opt(4, "BPE")
    print(f"  │")
    tokenizer_mode = prompt_int("Tokenizer", valid={-1,0,1,2,3,4})
    cli_section_end(64)

    # Load vocab
    cfg_dummy = {"dataset_type": dataset_type, "tokenizer_mode": tokenizer_mode, "custom_bpe_size": 4096}
    if tokenizer_mode == 3: cfg_dummy["tiktoken_encoding"] = "cl100k_base"
    if tokenizer_mode in [-1, 0, 3]: texts = []
    else: texts = read_dataset(dataset_path, dataset_type)
    vocab = load_or_make_vocab(cfg_dummy, dataset_path)

    # ── Model group ────────────────────────────────────────────────────────────
    tasks    = get_bench_model_list()
    has_rnns = any(t[0] in RNN_MODEL_IDS for t in tasks)
    has_attn = any(t[0] in ATTN_MODEL_IDS for t in tasks)

    # ── Architecture defaults ──────────────────────────────────────────────────
    cli_section("Architecture Defaults", 64)
    print(f"  │  {_c(_DIM, 'Applied to every model in the run.')}")
    print(f"  │")
    embed_dim   = prompt_int("Embedding / hidden dim", default=256)
    layer_count = prompt_int("Layer count", default=4)
    head_count  = 4
    if has_attn:
        head_count = prompt_int("Attention / xLSTM head count", default=4)
    if dataset_type == 0:
        seq_len = prompt_int("Sequence length", default=128)
    else:
        seq_len = 0
    batch_size  = prompt_int("Batch size", default=32)
    lr          = prompt_float("Learning rate", default=1e-3)
    total_iters = prompt_int("Iterations per model", default=500)
    cli_section_end(64)

    # ── RNN-specific ───────────────────────────────────────────────────────────
    rnn_cfg = {}
    if has_rnns:
        cli_section("RNN Structure Options", 64)
        print(f"  │  {_c(_DIM, 'Applied only to classic RNN models in the run.')}")
        print(f"  │")
        rnn_cfg["res_every"] = prompt_int("Residual every N layers  (0 = off)", default=0)
        if rnn_cfg["res_every"] > 0:
            print(f"  │")
            cli_opt(0, "Add"); cli_opt(1, "Concat+proj")
            cli_opt(2, "ReZero scalar"); cli_opt(3, "ReZero vector")
            print(f"  │")
            rnn_cfg["res_type"] = prompt_int("Residual type", default=0, valid={0,1,2,3})
        else:
            rnn_cfg["res_type"] = 0
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Norm:')}")
        cli_opt(0,"None"); cli_opt(1,"BN"); cli_opt(2,"LN")
        cli_opt(3,"RMS");  cli_opt(4,"TTanh"); cli_opt(5,"ETTanh"); cli_opt(6,"DyT")
        print(f"  │")
        rnn_cfg["use_norm"] = prompt_int("Norm type", default=2, valid={0,1,2,3,4,5,6})
        rnn_cfg["dropout"]  = prompt_float("Inter-layer dropout  (0.0 = off)", default=0.0)
        print(f"  │")
        if prompt_int("Configure advanced RNN init?  (1=yes 0=no)", default=0) == 1:
            rnn_cfg["tanh_spectral_radius"] = prompt_float("Tanh spectral radius", default=0.99)
            rnn_cfg["relu_identity_scale"]  = prompt_float("ReLU identity scale", default=1.0)
        else:
            rnn_cfg["tanh_spectral_radius"] = 0.99
            rnn_cfg["relu_identity_scale"]  = 1.0
        cli_section_end(64)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    cli_section("Evaluation", 64)
    print(f"  │")
    nan_skip = (prompt_int("Skip model on NaN / crash  (1=yes 0=no)", valid={0,1}) == 1)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Fitness metric:')}")
    cli_opt(0, "Sliding train loss",  "Rolling average of last 100 training steps")
    cli_opt(1, "Final train eval",    "Evaluate 1 000 random training samples at the end")
    cli_opt(2, "Validation loss",     "Best held-out loss tracked throughout training")
    print(f"  │")
    fitness_mode = prompt_int("Fitness metric", valid={0,1,2})
    cli_section_end(64)

    valid_ds = None
    train_ds = None

    cfg_base = {
        "dataset_path": dataset_path,
        "dataset_type": dataset_type,
        "seq_len": seq_len,
        "vocab_tokens": getattr(vocab, "tokens", None),
        "val_split": 0.0,
        "valid_examples": 0
    }

    if fitness_mode == 2:
        if dataset_type == 0:
            vpath = prompt_str("Validation file  (blank to auto-split)", default="")
            if vpath:
                cfg_base["classic_val_path"] = vpath
            else:
                cfg_base["val_split"] = prompt_float("Validation split fraction", default=0.1)
        else:
            cfg_base["val_split"] = prompt_float("Validation split fraction", default=0.1)
        cfg_base["_val_freq"]    = prompt_int("Validation frequency  (steps)", default=100)
        cfg_base["_val_samples"] = prompt_int("Validation samples per check", default=1000)

    train_ds, valid_ds = build_datasets(cfg_base, vocab)

    if dataset_type == 1:
        cfg_base["seq_len"] = min(train_ds.max_len, 2048)
        pinfo(f"Auto-set seq_len → {cfg_base['seq_len']}")

    # ── Run ────────────────────────────────────────────────────────────────────
    results = []
    metric_name = ["Sliding Loss","Final Train Eval","Best Valid Loss"][fitness_mode]
    print(f"\n  {_c(_GR, _B, '▸')} Starting benchmark — {_c(_WH, len(tasks))} models · metric: {_c(_YL, metric_name)}\n")
    
    for msel, overrides in tasks:
        cfg = cfg_base.copy()
        # Apply Base settings
        cfg.update({
            "model_selection": msel,
            "embed_dim": embed_dim,
            "layer_count": layer_count,
            "head_count": head_count,
            "batch_size": batch_size,
            "learning_rate": lr,
            "activation_name": "gelu",
            "dropout": 0.0,
            "tokenizer_mode": tokenizer_mode
        })
        # Apply RNN settings (if any)
        if has_rnns and msel in RNN_MODEL_IDS:
            cfg.update(rnn_cfg)
            
        # Apply Task specific overrides (e.g. minRNN activation)
        cfg.update(overrides)
        
        # Construct Name
        model_name = MODEL_NAMES.get(msel, f"Model {msel}")
        if "minrnn_act" in overrides:
            model_name += f" (act={overrides['minrnn_act']})"
        if msel in RNN_MODEL_IDS and has_rnns:
            norm_name = ["None","BN","LN","RMS","TT","ETT","DyT"][cfg.get("use_norm",0)]
            res_str = f"|Res{cfg.get('res_every',0)}" if cfg.get("res_every",0)>0 else ""
            model_name += f" [{norm_name}{res_str}]"

        try:
            model = build_model(cfg, vocab.size)
            model.to(DEVICE)
            optimizer = build_optimizer(model, cfg)

            score, best_step, status = bench_train_loop(
                cfg, model, optimizer, train_ds, valid_ds, vocab,
                (dataset_type==1), total_iters, fitness_mode, nan_skip
            )

            results.append({"name": model_name, "score": score, "best_step": best_step, "status": status})

            del model, optimizer
            torch.cuda.empty_cache()

            score_fmt = f"{score:.5f}" if score != float('inf') else "∞"
            best_str  = f"  {_c(_DIM, f'best @ step {best_step}')}" if fitness_mode==2 else ""
            pok(f"{_c(_WH, model_name)}  →  {_c(_GR, _B, score_fmt)}{best_str}")

        except Exception as e:
            pwarn(f"{_c(_WH, model_name)}  →  {_c(_RD, 'CRASH:')} {e}")
            results.append({"name": model_name, "score": float('inf'), "best_step": -1, "status": "CRASH"})

    # ── Results table ──────────────────────────────────────────────────────────
    results.sort(key=lambda x: (x["score"], 0 if x["status"]=="OK" else 1))

    W = 82
    metric_lbl = ["Sliding Loss","Final Train Eval","Best Valid Loss"][fitness_mode]
    print(f"\n")
    cli_banner(f"Benchmark Results  ·  {metric_lbl}", width=W)
    hdr = f"  {'Rank':<4}   {'Model':<52}   {'Score':<10}   Status"
    print(f"  {_c(_DIM, hdr)}")
    cli_rule(W - 4)
    for i, r in enumerate(results):
        score_str = f"{r['score']:.5f}" if r['score'] != float('inf') else "∞"
        rank_col  = _c(_YL, _B, f"  {i+1:<4}") if i == 0 else f"  {i+1:<4}"
        name_col  = _c(_WH, f"{r['name']:<52}") if i == 0 else f"{r['name']:<52}"
        scr_col   = _c(_GR, _B, f"{score_str:<10}") if i == 0 else (_c(_RD, f"{score_str:<10}") if r["status"] != "OK" else f"{score_str:<10}")
        stat_col  = _c(_GR, r["status"]) if r["status"] == "OK" else _c(_RD, r["status"])
        print(f"  {rank_col}   {name_col}   {scr_col}   {stat_col}")
    cli_rule(W - 4)
    if results and results[0]["score"] != float('inf'):
        best = results[0]
        winner_score = f"{best['score']:.5f}"
        print(f"\n  {_c(_GR, '★')} Winner: {_c(_WH, _B, best['name'])}  ·  score {_c(_GR, _B, winner_score)}\n")

# Replace the old `run_benchmark` call in __main__ with this one.


if __name__ == "__main__":
    try:
        interactive_train()
    except Exception as e:
        print("Fatal error:", repr(e))
        raise
