"""
Third update of model.py without MLP layer
"""

import math
import inspect
from dataclasses import dataclass

import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
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

class CausalSelfAttention(nn.Module):

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
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = False
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
        #     # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# Removed MLP class entirely

class Block(nn.Module):
    """Transformer block without MLP - attention only"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        # Removed MLP components:
        # self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # self.mlp = MLP(config)
        self.drop_resid = nn.Dropout(config.dropout)

    def forward(self, x):
        # Only attention path, no MLP
        shortcut = x
        x = self.ln_1(x)
        x = self.attn(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Single residual connection
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_identity_embeddings: bool = False  # Whether to use identity matrix for token embeddings
    use_fixed_positions: bool = False  # Whether to use fixed positional embeddings (identity matrix)

class IdentityEmbedding(nn.Module):
    """Identity embedding layer - creates one-hot vectors for input tokens"""
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        
        # Initialize projection matrix at creation time
        if n_embd != vocab_size:
            if n_embd < vocab_size:
                # Truncate: take only first n_embd dimensions
                projection = torch.zeros(vocab_size, n_embd)
                projection[:n_embd, :] = torch.eye(n_embd)
            else:
                # Pad: use identity for first vocab_size dimensions, zeros for rest
                projection = torch.zeros(vocab_size, n_embd)
                projection[:, :vocab_size] = torch.eye(vocab_size)
            
            self.register_buffer('projection', projection)
        
    def forward(self, idx):
        # Create one-hot vectors for each input token
        # For efficient implementation, we create an identity matrix and use it as a lookup table
        B, T = idx.size()
        device = idx.device
        
        # If embedding dimension matches vocabulary size, use pure one-hot encoding
        if self.n_embd == self.vocab_size:
            # Use one-hot encoding (sparse to dense conversion)
            one_hot = torch.zeros(B, T, self.vocab_size, device=device)
            one_hot.scatter_(2, idx.unsqueeze(-1), 1)
            return one_hot
        else:
            # If n_embd < vocab_size, we need to project down the one-hot vectors
            # or if n_embd > vocab_size, we need to pad the one-hot vectors
            # For simplicity, we'll use a fixed projection matrix
            # This is just a basic approach - there are more sophisticated ways to handle this
            one_hot = torch.zeros(B, T, self.vocab_size, device=device)
            one_hot.scatter_(2, idx.unsqueeze(-1), 1)
            
            # Apply the projection to get embeddings of the right dimension
            return one_hot @ self.projection

class FixedPositionalEmbedding(nn.Module):
    """Fixed positional embedding using identity matrix (one-hot encoding)"""
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        
        # Create fixed identity matrix - this won't be learned
        # Each position gets a unique one-hot vector
        identity_matrix = torch.eye(block_size)  # (block_size, block_size)
        self.register_buffer('position_embeddings', identity_matrix)
        
    def forward(self, seq_len):
        # Return the first seq_len rows of the identity matrix
        # Shape: (seq_len, block_size)
        return self.position_embeddings[:seq_len]

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Calculate effective embedding dimension
        # If using fixed positions, we need to account for concatenated positional info
        if config.use_fixed_positions:
            # Token embeddings: n_embd - block_size, Positional: block_size
            token_emb_dim = config.n_embd - config.block_size
            if token_emb_dim <= 0:
                raise ValueError(f"n_embd ({config.n_embd}) must be larger than block_size ({config.block_size}) when using fixed positions")
        else:
            # Use full n_embd for both token and learned positional embeddings
            token_emb_dim = config.n_embd

        self.transformer = nn.ModuleDict(dict(
            # Token embeddings with adjusted dimension
            wte = (nn.Embedding(config.vocab_size, token_emb_dim) 
                   if not config.use_identity_embeddings 
                   else IdentityEmbedding(config.vocab_size, token_emb_dim)),
            
            # Positional embeddings - either fixed or learned
            wpe = (FixedPositionalEmbedding(config.block_size) 
                   if config.use_fixed_positions 
                   else nn.Embedding(config.block_size, config.n_embd)),  # LEARNED positional embeddings
            
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Output head uses full n_embd dimension
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying - only if we're not using identity embeddings and not using fixed positions
        # Note: Weight tying becomes complex with concatenated embeddings due to dimension mismatch
        if not config.use_identity_embeddings and not config.use_fixed_positions:
            self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        Note: Fixed positional embeddings have no learnable parameters to subtract.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_fixed_positions:
            # Subtract learned positional embedding parameters
            n_params -= self.transformer.wpe.weight.numel()
        # Fixed positional embeddings have no learnable parameters to subtract
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
        
        # Get token embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, token_emb_dim or n_embd)
        
        if self.config.use_fixed_positions:
            # Fixed positional embeddings (concatenated)
            pos_emb = self.transformer.wpe(t)  # (t, block_size)
            pos_emb = pos_emb.unsqueeze(0).expand(b, -1, -1)  # (b, t, block_size)
            x = torch.cat([tok_emb, pos_emb], dim=-1)  # (b, t, n_embd)
        else:
            # Learned positional embeddings (added, like original GPT)
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
            pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
            x = tok_emb + pos_emb  # (b, t, n_embd) - ADDITION like original GPT
        
        x = self.transformer.drop(x)
        
        # Process through transformer blocks (now without MLP)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        
        if self.config.use_fixed_positions:
            # Update the fixed positional embedding buffer
            new_identity = torch.eye(block_size)
            self.transformer.wpe.register_buffer('position_embeddings', new_identity)
        else:
            # Update learned positional embeddings
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k in ['dropout', 'use_identity_embeddings', 'use_fixed_positions'] for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        
        # Set default values for our new parameters
        config_args['use_identity_embeddings'] = False
        config_args['use_fixed_positions'] = False
        
        # Handle overrides
        for k, v in override_args.items():
            config_args[k] = v
            print(f"overriding {k} to {v}")
            
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        
        # For identity embeddings, we don't need to load the token embeddings
        if config_args['use_identity_embeddings']:
            print("Using identity embeddings, not loading token embeddings from pretrained model")
        
        # For fixed positions, we don't need to load positional embeddings
        if config_args['use_fixed_positions']:
            print("Using fixed positions, not loading positional embeddings from pretrained model")
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        
        # Don't load positional embeddings if we're using fixed positions
        if config_args['use_fixed_positions']:
            print("Not using learnable positional embeddings, removing them from state dict keys")
            sd_keys = [k for k in sd_keys if not k.startswith('transformer.wpe')]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        
        # Skip token embeddings if identity embeddings are used
        if config_args['use_identity_embeddings']:
            sd_keys_hf = [k for k in sd_keys_hf if not k.startswith('transformer.wte')]
            sd_keys = [k for k in sd_keys if not k.startswith('transformer.wte')]
        
        # Skip positional embeddings if using fixed positions
        if config_args['use_fixed_positions']:
            sd_keys_hf = [k for k in sd_keys_hf if not k.startswith('transformer.wpe')]
        
        # Remove MLP-related keys from pretrained model
        sd_keys_hf = [k for k in sd_keys_hf if not any(mlp_key in k for mlp_key in ['mlp.c_fc', 'mlp.c_proj', 'ln_2'])]
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # Filter keys to match what's available
        common_keys = set(sd_keys).intersection(set(sd_keys_hf))
        sd_keys = [k for k in sd_keys if k in common_keys]
        sd_keys_hf = [k for k in sd_keys_hf if k in common_keys]
        
        # Ensure the common keys are in the same order
        sd_keys.sort()
        sd_keys_hf.sort()
        
        print(f"Loading {len(sd_keys)} matching parameters from pretrained model")
        
        for k_model, k_hf in zip(sd_keys, sd_keys_hf):
            if any(k_model.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k_hf].shape[::-1] == sd[k_model].shape
                with torch.no_grad():
                    sd[k_model].copy_(sd_hf[k_hf].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k_hf].shape == sd[k_model].shape, f"Shape mismatch: {sd_hf[k_hf].shape} vs {sd[k_model].shape} for {k_model} and {k_hf}"
                with torch.no_grad():
                    sd[k_model].copy_(sd_hf[k_hf])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding, IdentityEmbedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        if 'lm_head.weight' in decay and not self.config.use_identity_embeddings and not self.config.use_fixed_positions:
            decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
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