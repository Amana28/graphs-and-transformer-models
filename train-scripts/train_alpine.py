"""
Training script for the ALPINE model (simplified Transformer) for path-finding tasks.

This script is adapted from the original train.py, but uses the simplified
AlpineModel architecture instead of the full GPT model.

To run on a single GPU, example:
$ python train_alpine.py --batch_size=32 --compile=False
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
import re

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AlpineModel instead of GPT
from model.alpine_model import AlpineConfig, AlpineModel, create_alpine_model
from logger import get_logger
import logging

# -----------------------------------------------------------------------------
# the input parameters

parser = argparse.ArgumentParser(description='Training of the AlpineModel for path-finding.')

parser.add_argument('--dataset', type=str, default='simple_graph', help='Name of the dataset to use')  
parser.add_argument('--n_layer', type=int, default=1, help='Number of layers (default: 1)')  
parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads (default: 1)')  
parser.add_argument('--n_embd', type=int, default=120, help='Size of the embeddings (default: 120)')
parser.add_argument('--max_iters', type=int, default=10000, help='Number of Iterations (default: 10000)')
parser.add_argument('--num_nodes', type=int, default=100, help='Number of Nodes (default: 100)')
parser.add_argument('--num_of_paths', type=int, default=20, help='Number of Paths (default: 1)')

args = parser.parse_args()

dataset = args.dataset
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths

data_dir = os.path.join('data', f'{dataset}/{num_nodes}')
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
    
stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']

out_dir = f'out_alpine/{dataset}_{n_layer}_{n_head}_{n_embd}_{num_nodes}'

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
eval_interval = max_iters // 10
log_interval = max_iters // 100
eval_iters = max_iters // 10

eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'alpine' # 'run' + str(time.time())

# data
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
train_batch_size = 1024 # if gradient_accumulation_steps > 1, this is the micro-batch size
val_batch_size = 64
batch_size = train_batch_size

# model
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 5e-4 # max learning rate 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = max_iters//20 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
if(num_of_paths == 0):
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
else:
    train_data = np.memmap(os.path.join(data_dir, f'train_{num_of_paths}.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_size = train_batch_size if split == 'train' else val_batch_size
    
    data_size = block_size + 1
    data = train_data if split == 'train' else val_data
    ix = torch.randint( (len(data) - data_size)//data_size , (batch_size,)) * data_size
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# logger
if(num_of_paths == 0):
    logger = get_logger(os.path.join(out_dir, "no_output_train.log"))
    log_file_name = os.path.join(out_dir, "train.log")
else:
    logger = get_logger(os.path.join(out_dir, f'no_output_train_{num_of_paths}.log'))
    log_file_name = os.path.join(out_dir, f"train_{num_of_paths}.log")

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

def get_shortest(p_graph):
    shortest_paths = {}
    for i in p_graph.nodes:
        for j in p_graph.nodes:
            try:
                shortest_paths[(i,j)] = list(nx.all_shortest_paths(p_graph, i, j))
            except:
                shortest_paths[(i,j)] = []
    return shortest_paths

if dataset == 'reasoning':
    p_graph_path = os.path.join(data_dir, 'fixed_model.graphml')
    p_graph = nx.read_graphml(p_graph_path)
    shortest_paths = get_shortest(p_graph)
    
stoi, itos = meta['stoi'], meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

# model init
if init_from == 'scratch':
    print("Initializing a new ALPINE model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of 50304 (50257 rounded up for efficiency)")
    vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
    model = create_alpine_model(vocab_size, block_size)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    vocab_size = checkpoint['vocab_size']
    model = create_alpine_model(vocab_size, block_size)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item() 
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def open_and_append(filename, text):
    with open(filename, 'a') as file:
        file.write(text + '\n')

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
accuracy = []
corrects = []
totals = []

while True:
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        open_and_append(log_file_name, f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'vocab_size': meta_vocab_size,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                logger.info(f"saving checkpoint to {out_dir}")
                open_and_append(log_file_name, f"saving checkpoint to {out_dir}")
                if(num_of_paths == 0):
                    torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_ckpt.pt'))
                else:
                    torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_ckpt_{num_of_paths}.pt'))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            # Note: Alpine model doesn't have estimate_mfu method, so we skip MFU calculation
            running_mfu = -1.0
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        logger.info(f"iter {iter_num}: loss {lossf:.4f}")
        open_and_append(log_file_name, f"iter {iter_num}: loss {lossf:.4f}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

torch.save(torch.tensor(corrects).cpu(), os.path.join(out_dir, f'corrects.pt'))
torch.save(torch.tensor(totals).cpu(), os.path.join(out_dir, f'totals.pt'))

if ddp:
    destroy_process_group()