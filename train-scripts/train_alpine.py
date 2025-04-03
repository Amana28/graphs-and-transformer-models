"""
Training script for the simplified Alpine model for path-finding tasks.
This script trains a simplified transformer model based on the ALPINE paper.

Example usage:
$ python train_alpine.py --dataset=simple_graph --num_nodes=100 --max_iters=3000
"""
import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import argparse
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import networkx as nx

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simplified Alpine model
from model.alpine_model import AlpineModel, AlpineConfig, create_alpine_model
from logger import get_logger
import logging

# -----------------------------------------------------------------------------
# Command line arguments
parser = argparse.ArgumentParser(description='Training of the simplified Alpine model for path-finding.')
parser.add_argument('--dataset', type=str, default='simple_graph', help='Name of the dataset to use')  
parser.add_argument('--max_iters', type=int, default=3000, help='Number of iterations (default: 3000)')
parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph (default: 100)')
parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths per source-target pair (default: 20)')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate (default: 5e-4)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
args = parser.parse_args()

# Extract arguments
dataset = args.dataset
max_iters = args.max_iters
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
learning_rate = args.learning_rate
weight_decay = args.weight_decay

# Data directories and output paths
data_dir = os.path.join('data', f'{dataset}/{num_nodes}')
out_dir = f'out/alpine_{dataset}_{num_nodes}'

# Load metadata
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
    
stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']
vocab_size = meta['vocab_size']

# -----------------------------------------------------------------------------
# Training configuration
eval_interval = max_iters // 10
log_interval = max_iters // 100
eval_iters = max_iters // 10
train_batch_size = 32
val_batch_size = 32
grad_clip = 1.0

# Optimization
warmup_iters = max_iters // 20
lr_decay_iters = max_iters
min_lr = learning_rate / 10
beta1 = 0.9
beta2 = 0.95

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = True and torch.cuda.is_available()

# Logging
if num_of_paths == 0:
    log_file_name = os.path.join(out_dir, "alpine_train.log")
else:
    log_file_name = os.path.join(out_dir, f"alpine_train_{num_of_paths}.log")

os.makedirs(out_dir, exist_ok=True)
logger = get_logger(log_file_name)

# -----------------------------------------------------------------------------
# Initialize distributed training if needed
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# Set random seed for reproducibility
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Load data
if num_of_paths == 0:
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
else:
    train_data = np.memmap(os.path.join(data_dir, f'train_{num_of_paths}.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Data loader function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_size = train_batch_size if split == 'train' else val_batch_size
    
    data_size = block_size + 1
    ix = torch.randint((len(data) - data_size) // data_size, (batch_size,)) * data_size
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Initialize model and optimizer
print(f"Initializing Alpine model with vocab_size={vocab_size}")
model = create_alpine_model(vocab_size, block_size)
model.to(device)

# For float16 precision
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Simplified optimizer for Alpine model - just standard Adam with fixed parameters
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay
)

# Compile the model if possible
if compile and torch.__version__ >= '2.0.0':
    print("Compiling the model (requires PyTorch 2.0+)")
    model = torch.compile(model)

# Wrap model in DDP if using distributed training
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# Evaluation function
@torch.no_grad()
def estimate_loss():
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

# Learning rate scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# Helper function for logging
def log_to_file(filename, text):
    with open(filename, 'a') as file:
        file.write(text + '\n')

# -----------------------------------------------------------------------------
# Training loop
iter_num = 0
best_val_loss = float('inf')
X, Y = get_batch('train')  # Get the first batch
t0 = time.time()
raw_model = model.module if ddp else model  # Unwrap DDP container if needed

print(f"Beginning training for {max_iters} iterations")
logger.info(f"Beginning training for {max_iters} iterations")

while iter_num < max_iters:
    # Set learning rate for this iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Evaluate the model periodically
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        log_message = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}"
        print(log_message)
        logger.info(log_message)
        
        # Save checkpoint if validation loss improves
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'vocab_size': vocab_size,
                    'block_size': block_size,
                }
                checkpoint_path = os.path.join(out_dir, f'alpine_ckpt_{iter_num}.pt')
                print(f"Saving checkpoint to {checkpoint_path}")
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)
    
    # Forward and backward pass
    with ctx:
        logits, loss = model(X, Y)
    
    # Get next batch (for next iteration)
    X, Y = get_batch('train')
    
    # Backward pass with gradient scaling if using float16
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    
    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Update weights
    scaler.step(optimizer)
    scaler.update()
    
    # Logging
    if iter_num % log_interval == 0 and master_process:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        log_message = f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms"
        print(log_message)
        logger.info(log_message)
    
    iter_num += 1

# Final evaluation and checkpoint
if master_process:
    losses = estimate_loss()
    log_message = f"Final step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    print(log_message)
    logger.info(log_message)
    
    # Save final model
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'vocab_size': vocab_size,
        'block_size': block_size,
    }
    checkpoint_path = os.path.join(out_dir, 'alpine_ckpt_final.pt')
    print(f"Saving final checkpoint to {checkpoint_path}")
    logger.info(f"Saving final checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)

print("Training completed!")
logger.info("Training completed!")

# Clean up distributed training resources
if ddp:
    destroy_process_group()