"""
Training script for the simplified transformer model as described in the paper.
This script is adapted from the original training script to work with the ALPINE_simplified_model.py.

To run on a single GPU, example:
$ python train_simplified.py --batch_size=32 --compile=False
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import argparse
import inspect

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import networkx as nx
import re

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simplified model instead of the standard GPT
from ALPINE_simplified_model import SimplifiedGPTConfig, SimplifiedGPT
from logger import get_logger
import logging

# -----------------------------------------------------------------------------
# input parameters

parser = argparse.ArgumentParser(description='Training of the Simplified Transformer (ALPINE model).')

parser.add_argument('--dataset', type=str, default='simple_graph', help='Name of the dataset to use')  
parser.add_argument('--n_layer', type=int, default=1, help='Number of layers (default: 1)')  
parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads (default: 1)')  
parser.add_argument('--n_embd', type=int, default=10, help='Size of the embeddings (default: 10)')
parser.add_argument('--max_iters', type=int, default=10000, help='Number of Iterations (default: 10000)')
parser.add_argument('--num_nodes', type=int, default=10, help='Number of Nodes (default: 10)')
parser.add_argument('--num_of_paths', type=int, default=20, help='Number of Paths (default: 20)')
parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size (default: 1024)')
parser.add_argument('--val_batch_size', type=int, default=64, help='Validation batch size (default: 64)')
parser.add_argument('--compile', type=bool, default=False, help='Whether to use torch.compile (default: False)')
parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
parser.add_argument('--dtype', type=str, default='float32', help='Data type to use (default: float32)')

args = parser.parse_args()

dataset = args.dataset
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
train_batch_size = args.batch_size
val_batch_size = args.val_batch_size
device = args.device
dtype = args.dtype
compile_model = args.compile

# Data and output directories
data_dir = os.path.join('data', f'{dataset}/{num_nodes}')
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
    
stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']

# Update output directory to reflect simplified model
out_dir = f'out/ALPINE_{dataset}_{n_layer}_{n_head}_{n_embd}_{num_nodes}'

# -----------------------------------------------------------------------------
# default config values
eval_interval = max_iters // 10
log_interval = max_iters // 100
eval_iters = max_iters // 10

eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume'

dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside Linear layers?

# optimizer settings
learning_rate = 5e-4  # max learning rate 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = max_iters//20  # how many steps to warm up for
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = learning_rate/10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system settings
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# collect all config parameters
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# -----------------------------------------------------------------------------
# I/O setup
if os.path.exists(out_dir):
    print(f"Output directory {out_dir} already exists")
else:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory {out_dir}")

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# logger setup
if num_of_paths == 0:
    logger = get_logger(os.path.join(out_dir, "train.log"))
    log_file_name = os.path.join(out_dir, "train.log")
else:
    logger = get_logger(os.path.join(out_dir, f'train_{num_of_paths}.log'))
    log_file_name = os.path.join(out_dir, f"train_{num_of_paths}.log")

# data loader setup
if num_of_paths == 0:
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
else:
    train_data = np.memmap(os.path.join(data_dir, f'train_{num_of_paths}.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

def get_batch(split):
    """Get a random batch of data."""
    data = train_data if split == 'train' else val_data
    batch_size = train_batch_size if split == 'train' else val_batch_size
    
    data_size = block_size + 1
    
    # Make sure we have enough data
    if (len(data) - data_size) // data_size <= 0:
        raise ValueError(f"Not enough data for batch. Data size: {len(data)}, block size: {block_size}")
    
    # Select random starting indices
    ix = torch.randint((len(data) - data_size) // data_size, (batch_size,)) * data_size
    
    # Get input and target sequences
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Move to device
    x, y = x.to(device), y.to(device)
    return x, y

def open_and_append(filename, text):
    """Append text to a file."""
    with open(filename, 'a') as file:
        file.write(text + '\n')

# -----------------------------------------------------------------------------
# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
    dropout=dropout
)

if init_from == 'scratch':
    print("Initializing a new simplified model from scratch")
    # Create the simplified model configuration
    gptconf = SimplifiedGPTConfig(**model_args)
    model = SimplifiedGPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # Resume training from a checkpoint
    if num_of_paths == 0:
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    else:
        ckpt_path = os.path.join(out_dir, f'ckpt_{num_of_paths}.pt')
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    # Force these config attributes to be equal
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    
    # Create the model
    gptconf = SimplifiedGPTConfig(**model_args)
    model = SimplifiedGPT(gptconf)
    
    # Load the model weights
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Load state dict with less strict matching due to architecture differences
    model.load_state_dict(state_dict, strict=False)
    
    # Restore training state
    iter_num = checkpoint.get('iter_num', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
else:
    raise ValueError(f"Invalid init_from: {init_from}")

# Move model to device
model.to(device)

# Initialize variables for training loop
iter_num = 0 if 'iter_num' not in locals() else iter_num
best_val_loss = float('inf') if 'best_val_loss' not in locals() else best_val_loss

# GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
# Create a standard optimizer instead of using the model's configure_optimizers
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if 'weight' in name and 'ln' not in name and 'wte' not in name:
        decay_params.append(param)
    else:
        no_decay_params.append(param)

optim_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
]

use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)

# Compile the model if requested and supported
if compile_model and hasattr(torch, 'compile'):
    print("Compiling the model (this may take a minute)...")
    unoptimized_model = model
    model = torch.compile(model)

# Learning rate scheduler
def get_lr(it):
    # Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# Evaluation helper
@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# training loop
print(f"Beginning training with {model_args}")
logger.info(f"Beginning training with {model_args}")

X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0

while True:
    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        open_and_append(log_file_name, f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint if validation loss has improved or if always_save_checkpoint is True
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"Saving checkpoint to {out_dir}")
                logger.info(f"Saving checkpoint to {out_dir}")
                if num_of_paths == 0:
                    torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_ckpt.pt'))
                else:
                    torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_ckpt_{num_of_paths}.pt'))
    
    # Exit if eval_only flag is set
    if iter_num == 0 and eval_only:
        break
    
    # Forward and backward pass
    with ctx:
        logits, loss = model(X, Y)
    X, Y = get_batch('train')  # fetch the next batch
    
    # Backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()
    
    # Clip gradients if specified
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Step the optimizer and scaler
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0:
        lossf = loss.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        logger.info(f"iter {iter_num}: loss {lossf:.4f}")
        open_and_append(log_file_name, f"iter {iter_num}: loss {lossf:.4f}")
    
    iter_num += 1
    local_iter_num += 1
    
    # Check if we've reached the maximum iterations
    if iter_num > max_iters:
        break

print(f"Training complete after {iter_num} iterations.")