"""
Fifth update of model.py without MLP layer + partial weight Tying (wte - lm_head) with positional dims zeroed out
Updated to support learned concatenation and removed identity embedding/fixed position components
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

class IdentityProjection(nn.Module):
    """Identity projection layer that acts like nn.Linear but with fixed identity weights"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create identity matrix (or truncated/padded version)
        if in_features == out_features:
            # Perfect identity
            identity_weight = torch.eye(out_features, in_features)
        elif in_features > out_features:
            # Truncate: take first out_features columns
            identity_weight = torch.eye(out_features, in_features)
        else:
            # Pad: identity for first in_features, zeros for rest
            identity_weight = torch.zeros(out_features, in_features)
            identity_weight[:in_features, :] = torch.eye(in_features)
        
        # Register as buffer (non-trainable)
        self.register_buffer('weight', identity_weight)
        
        # Bias handling
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection - either learned or identity
        if config.use_identity_output_projection:
            self.c_proj = IdentityProjection(config.n_embd, config.n_embd, bias=config.bias)
        else:
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_identity_V = config.use_identity_V
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

        # Apply identity V if configured
        if self.use_identity_V:
            # Replace v with identity-based values
            # v should be the input x transformed to match the attention head dimensions
            x_reshaped = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = x_reshaped

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
    # REMOVED: use_identity_embeddings and use_fixed_positions
    use_identity_output_projection: bool = False  # Whether to use identity matrix for attention output projection
    use_identity_V: bool = False  # Whether to use identity matrix for V projection
    # NEW: Concatenation support
    use_concat: bool = False  # Whether to concatenate embeddings instead of adding them
    token_emb_dim: int = None  # Dimension for token embeddings (if None, calculated automatically)
    pos_emb_dim: int = None    # Dimension for positional embeddings (if None, calculated automatically)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Calculate embedding dimensions
        if config.use_concat:
            # User can specify dimensions or we'll split evenly
            if config.token_emb_dim is None or config.pos_emb_dim is None:
                # Split n_embd evenly (or close to evenly)
                config.token_emb_dim = config.n_embd // 2
                config.pos_emb_dim = config.n_embd - config.token_emb_dim
            else:
                assert config.token_emb_dim + config.pos_emb_dim == config.n_embd, \
                    f"token_emb_dim ({config.token_emb_dim}) + pos_emb_dim ({config.pos_emb_dim}) must equal n_embd ({config.n_embd})"
            
            token_emb_dim = config.token_emb_dim
            pos_emb_dim = config.pos_emb_dim
            print(f"Using learned concatenation: {token_emb_dim} token dims + {pos_emb_dim} pos dims = {config.n_embd} total")
        else:
            # Standard addition mode - both use full n_embd
            token_emb_dim = config.n_embd
            pos_emb_dim = config.n_embd
            print(f"Using addition mode: {token_emb_dim} token dims + {pos_emb_dim} pos dims (added) = {config.n_embd} total")

        self.transformer = nn.ModuleDict(dict(
            # Token embeddings - always learned
            wte = nn.Embedding(config.vocab_size, token_emb_dim),
            
            # Positional embeddings - always learned
            wpe = nn.Embedding(config.block_size, pos_emb_dim),
            
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Output head - always learned linear layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying logic
        if config.use_concat:
            # For concatenation: copy token weights and zero positional portion
            # We'll keep them synchronized in the forward pass instead of parameter sharing
            with torch.no_grad():
                # Copy token embeddings to first token_emb_dim columns  
                self.lm_head.weight[:, :config.token_emb_dim].copy_(self.transformer.wte.weight)
                # Zero out the positional portion
                self.lm_head.weight[:, config.token_emb_dim:].zero_()
            
            # Store dimensions for forward pass synchronization
            self.token_emb_dim = config.token_emb_dim
            self.pos_emb_dim = config.pos_emb_dim
            
            print(f"Weight tying: On (token portion synchronized, {config.pos_emb_dim} positional dims zeroed)")
        else:
            # Standard full weight tying
            self.transformer.wte.weight = self.lm_head.weight
            print("Weight tying: On (full)")

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
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract learned positional embedding parameters
            n_params -= self.transformer.wpe.weight.numel()
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
        tok_emb = self.transformer.wte(idx)  # (b, t, token_emb_dim)
        
        # Get positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        pos_emb = self.transformer.wpe(pos)  # (1, t, pos_emb_dim)
        
        if self.config.use_concat:
            # Concatenate token and positional embeddings
            x = torch.cat([tok_emb, pos_emb.expand(b, -1, -1)], dim=-1)  # (b, t, n_embd)
        else:
            # Standard addition mode (both embeddings have same dimension)
            x = tok_emb + pos_emb  # (b, t, n_embd) - ADDITION like original GPT
        
        x = self.transformer.drop(x)
        
        # Process through transformer blocks (now without MLP)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Keep weights synchronized and positional columns zeroed for concat mode
        if self.config.use_concat and hasattr(self, 'token_emb_dim'):
            with torch.no_grad():
                # Sync token embeddings: copy from wte to lm_head
                self.lm_head.weight[:, :self.token_emb_dim].copy_(self.transformer.wte.weight)
                # Keep positional columns zeroed
                self.lm_head.weight[:, self.token_emb_dim:].zero_()

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
        assert all(k in ['dropout', 'use_identity_output_projection', 'use_identity_V', 'use_concat', 'token_emb_dim', 'pos_emb_dim'] for k in override_args)
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
        
        # Set default values for our parameters
        config_args['use_identity_output_projection'] = False
        config_args['use_identity_V'] = False
        config_args['use_concat'] = False
        config_args['token_emb_dim'] = None
        config_args['pos_emb_dim'] = None
        
        # Handle overrides
        for k, v in override_args.items():
            config_args[k] = v
            print(f"overriding {k} to {v}")
            
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        
        # For identity output projection, we don't need to load c_proj weights
        if config_args['use_identity_output_projection']:
            print("Using identity output projection, not loading attention output projection weights from pretrained model")
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        
        # Skip attention output projection if using identity
        if config_args['use_identity_output_projection']:
            sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.c_proj.weight') and not k.endswith('attn.c_proj.bias')]
            sd_keys = [k for k in sd_keys if not k.endswith('attn.c_proj.weight') and not k.endswith('attn.c_proj.bias')]
        
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
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding, IdentityProjection)
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

        # Handle weight tying: remove lm_head.weight from decay if it exists and weight tying is on
        if 'lm_head.weight' in decay and not self.config.use_concat:
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