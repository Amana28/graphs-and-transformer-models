import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class AlpineConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # Will be used as embedding size as well
    bias: bool = False  # No bias as per simplified model

class FixedTargetAttention(nn.Module):
    """
    Attention mechanism that always attends to the second token (target node).
    This implements the simplified attention from the ALPINE paper.
    """
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        # Value projection - the only projection we need
        self.c_proj = nn.Linear(config.vocab_size, config.vocab_size, bias=config.bias)
        # We'll use this to enforce attention on the second token only
        self.register_buffer("fixed_attention", torch.zeros(1, 1, config.block_size, config.block_size))
        # Set the second column to 1 for all rows
        self.fixed_attention[:, :, :, 1] = 1.0

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dim
        
        # Fixed attention to second token (target node)
        # Instead of computing attention scores, we use our fixed pattern
        att = self.fixed_attention[:, :, :T, :T]
        
        # Get the second token's embedding (target node) for each sequence in batch
        # This is equivalent to attending only to the second token
        target_token = x[:, 1:2, :]  # Shape: (B, 1, C)
        
        # Apply value projection
        v = self.c_proj(target_token)  # Shape: (B, 1, C)
        
        # Repeat for each position in the sequence
        # This is equivalent to the value being the same for all positions
        y = v.repeat(1, T, 1)  # Shape: (B, T, C)
        
        return y

class SimplifiedMLP(nn.Module):
    """
    Simplified MLP that just applies a linear transformation: FFN(X) = XWM
    """
    def __init__(self, config):
        super().__init__()
        self.c_proj = nn.Linear(config.vocab_size, config.vocab_size, bias=config.bias)

    def forward(self, x):
        return self.c_proj(x)

class SimplifiedBlock(nn.Module):
    """
    Simplified Transformer block that implements: Transformer(X) = FFN(X) + MHA(X)
    """
    def __init__(self, config):
        super().__init__()
        self.attn = FixedTargetAttention(config)
        self.mlp = SimplifiedMLP(config)

    def forward(self, x):
        # No layer norm, just: Transformer(X) = FFN(X) + MHA(X)
        return self.mlp(x) + self.attn(x)

class AlpineModel(nn.Module):
    """
    Implementation of the simplified model from the ALPINE paper
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        # Identity token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.vocab_size)
        # Initialize as identity matrix
        with torch.no_grad():
            self.wte.weight.copy_(torch.eye(config.vocab_size))
        
        # Zero positional embeddings
        self.register_buffer("zero_pos_emb", torch.zeros(config.block_size, config.vocab_size))
        
        # Single transformer block
        self.block = SimplifiedBlock(config)
        
        # Output projection (identity)
        self.lm_head = nn.Linear(config.vocab_size, config.vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.eye(config.vocab_size))
        
        # Weight tying
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Get token embeddings (identity mapping)
        tok_emb = self.wte(idx)
        
        # No positional embeddings (all zeros)
        pos_emb = self.zero_pos_emb[:t].unsqueeze(0).expand(b, t, -1)
        
        # Input is just token embeddings
        x = tok_emb + pos_emb
        
        # Apply simplified transformer block
        x = self.block(x)
        
        # Get logits
        if targets is not None:
            # Training mode - compute all logits
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode - only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

def create_alpine_model(vocab_size, block_size=1024):
    """
    Create a simplified Transformer model as described in the ALPINE paper
    """
    config = AlpineConfig(
        block_size=block_size,
        vocab_size=vocab_size
    )
    return AlpineModel(config)