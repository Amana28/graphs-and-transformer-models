"""
GPT model modified to use relative positional encoding.
Based on the original GPT implementation but with relative position embeddings.
"""

import math
import inspect
from dataclasses import dataclass

import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RelativePositionEncoding(nn.Module):
    """
    Implements relative positional encoding for transformers.
    """
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # Create relative position embeddings
        # We'll use 2*max_len-1 possible relative positions (-max_len+1 to max_len-1)
        self.rel_pos_embeddings = nn.Parameter(torch.zeros(2 * max_len - 1, d_model))
        
        # Initialize embeddings
        nn.init.normal_(self.rel_pos_embeddings, mean=0.0, std=0.02)

    def forward(self, seq_len):
        # Generate all relative position indices for the sequence length
        # For each position i, we compute its relative positions to all positions j
        # This gives us a matrix of shape [seq_len, seq_len]
        positions = torch.arange(seq_len, device=self.rel_pos_embeddings.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Shift indices to be non-negative (from 0 to 2*max_len-2)
        relative_positions += self.max_len - 1
        
        # Clamp to ensure we don't go out of bounds for sequences longer than max_len
        relative_positions = torch.clamp(relative_positions, 0, 2 * self.max_len - 2)
        
        # Return the embeddings for the relative positions
        return self.rel_pos_embeddings[relative_positions]

class RelativeCausalSelfAttention(nn.Module):
    """
    Self-attention layer with relative positional encoding.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Relative position encoding
        self.rel_pos_enc = RelativePositionEncoding(config.block_size, config.n_embd // config.n_head)
        
        # Flash attention for efficiency if available
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        head_size = C // self.n_head

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)

        # Get relative position encodings
        rel_pos = self.rel_pos_enc(T)  # [T, T, head_size]
        
        # Compute attention with relative positions
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            # Note: Flash attention doesn't directly support relative positions,
            # so we apply a simple workaround by adding it to the key or value
            
            # Add relative position bias to keys
            k_with_pos = k.clone()
            for i in range(T):
                k_with_pos[:, :, i, :] = k[:, :, i, :] + rel_pos[i, :, :].view(1, 1, T, head_size)
            
            # Use flash attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k_with_pos, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual implementation with relative positions
            # Standard attention calculation: QK^T / sqrt(d_k)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # Add relative position bias
            rel_pos_bias = torch.zeros_like(att)
            for i in range(self.n_head):
                # Compute position bias for each head
                for j in range(T):
                    # q_j · rel_pos_j,k for all k
                    q_head = q[:, i, :, :]  # [B, T, head_size]
                    rel_pos_j = rel_pos[j, :, :]  # [T, head_size]
                    
                    # Matrix multiplication for all batches
                    # [B, T, head_size] @ [head_size, T] -> [B, T, T]
                    pos_bias = q_head @ rel_pos_j.transpose(0, 1)
                    
                    # Add to the jth row of attention for all batches and this head
                    rel_pos_bias[:, i, j, :] = pos_bias[:, j, :]
            
            # Add the relative position bias
            att = att + rel_pos_bias
            
            # Apply causal mask
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            
            # Apply softmax
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            # Apply attention to values
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = RelativeCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # No positional embeddings - using relative positions in attention
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        
        # No position embeddings - we're using relative positions
        x = self.transformer.drop(tok_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # Using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # Model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        
        # Update relative position encoding max_len
        for block in self.transformer.h:
            if hasattr(block.attn, 'rel_pos_enc'):
                block.attn.rel_pos_enc.max_len = block_size
                # Reinitialize relative position embeddings for new size
                block.attn.rel_pos_enc.rel_pos_embeddings = nn.Parameter(
                    torch.zeros(2 * block_size - 1, block.attn.rel_pos_enc.d_model))
                nn.init.normal_(block.attn.rel_pos_enc.rel_pos_embeddings, mean=0.0, std=0.02)
            
            # Update attention mask
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    # Methods for model configuration, optimization, and generation would remain the same
    # Only including necessary methods for brevity
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Optimizer configuration (same as original)
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' %