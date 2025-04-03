"""
Simplified Transformer model as described in the paper:
1) Attention weight only on the target node (second token)
2) No layer normalization
3) Simplified FFN: FFN(X) = XWM instead of two-layer with activation
4) No residual connections in the typical way
5) Transformer(X) = FFN(X) + MHA(X)
6) Token embedding and output matrices set to identity
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class SimplifiedGPTConfig:
    block_size: int = 32
    vocab_size: int = 10  # For a 10-node graph plus padding
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 10     # Set to be the same as vocab_size as per paper
    dropout: float = 0.0
    bias: bool = False

class TargetOnlyAttention(nn.Module):
    """
    Self-attention that only focuses on the target node (second token).
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.dropout = config.dropout
        
        # Just need value projection (WV in the paper)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Optional dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Only compute value vectors
        v = self.value(x)
        
        # Create a mask that only attends to the second token (target node)
        # Every row in the attention matrix is set to a one-hot vector with second coord being 1
        # This is index 1 in zero-indexed Python
        if T > 1:  # Make sure the sequence is long enough
            # Create manually set attention weights as described in the paper
            # a one-hot vector with the second coordinate being 1
            attn_mask = torch.zeros(B, T, T, device=x.device)
            attn_mask[:, :, 1] = 1.0  # Set all attention to second token
            
            # Apply the mask to create the output
            y = torch.bmm(attn_mask, v)
        else:
            # If there's only one token, just return the value vectors
            y = v
            
        return y

class SimpleFFN(nn.Module):
    """
    Simplified FFN that just does a single matrix multiplication: FFN(X) = XWM
    """
    def __init__(self, config):
        super().__init__()
        self.wm = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.wm(x))

class SimplifiedBlock(nn.Module):
    """
    Simplified transformer block with no layer norm and FFN(X) + MHA(X)
    """
    def __init__(self, config):
        super().__init__()
        self.attn = TargetOnlyAttention(config)
        self.ffn = SimpleFFN(config)

    def forward(self, x):
        # No residuals within the block, just attention and FFN
        a = self.attn(x)
        f = self.ffn(x)
        # Transformer(X) = FFN(X) + MHA(X) as in the paper
        return f + a

class SimplifiedGPT(nn.Module):
    """
    The simplified transformer model as described in the paper.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings (set to identity instead of learnable in a real implementation)
        # However, we'll keep the embedding layer for API compatibility
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # No position embeddings (Wp = 0 in paper)
            h = nn.ModuleList([SimplifiedBlock(config) for _ in range(config.n_layer)]),
        ))
        
        # Output projection (set to identity in real implementation)
        # Again, keeping for API compatibility
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # For a true identity embedding, we could set the weights directly:
        # However, this would only work if vocab_size == n_embd
        if config.vocab_size == config.n_embd:
            with torch.no_grad():
                # Initialize token embeddings to identity
                identity = torch.eye(config.n_embd)
                self.transformer.wte.weight.copy_(identity)
                # Initialize output projection to identity
                self.lm_head.weight.copy_(identity)
        
        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Get token embeddings
        x = self.transformer.wte(idx)
        
        # Forward through all transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        else:
            # Only get logits for last position during inference
            logits = logits[:, [-1], :] 
            loss = None
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens given a context.
        """
        for _ in range(max_new_tokens):
            # If sequence is too long, truncate
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward to get logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx