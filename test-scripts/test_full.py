import os

# Add the parent directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.simplified_model import GPTConfig, GPT 

import numpy as np
import networkx as nx
import argparse
import pickle
import re
import torch
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_iter', type=int, default=10000)
    parser.add_argument('--config', type=str, default='1_1_120')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--num_of_paths', type=int, default=20)
    parser.add_argument('--embedding_config', type=str, default='', help='Optional suffix for embedding config (e.g., _identity_nopos)')
    return parser.parse_args()

args = parse_args()
dataset = 'simple_graph'
ckpt_iter = args.ckpt_iter
device = args.device
temperature = args.temperature
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
config = args.config
embedding_config = args.embedding_config
full_config = f'{config}_{embedding_config}'


data_path = f'data/{dataset}/{num_nodes}'
meta_path = f'{data_path}/meta.pkl'

print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
max_new_tokens = meta['block_size']
top_k = len(itos)
simple_format = meta['simple_format']

out_dir = f'out/{dataset}_{full_config}_{num_nodes}/'

if(num_of_paths == 0):
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt.pt')
else:
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt_{num_of_paths}.pt')

print(f"Loading checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

# Print model configuration
print(f"Model configuration:")
print(f"- Layers: {gptconf.n_layer}")
print(f"- Heads: {gptconf.n_head}")
print(f"- Embedding dim: {gptconf.n_embd}")
print(f"- Using identity embeddings: {getattr(gptconf, 'use_identity_embeddings', False)}")
print(f"- Using positional embeddings: {getattr(gptconf, 'use_positional_embeddings', True)}")

model.eval()
model.to(device)

path_graph = f'{data_path}/path_graph.graphml'
path_graph = nx.read_graphml(path_graph)

def find_third_number_position(number_string):  
    numbers = number_string.split()  
    third_number_index = 2 
    position = sum(len(num) for num in numbers[:third_number_index]) + third_number_index-1 
    return position 

def encode(s):
    ss = s.split(" ")
    encoded_string = [stoi[ch] for ch in ss]
    return encoded_string

def decode(l):
    dec = ""
    for i in l:
        dec = dec + itos[i] + " "
    return dec[:-1]

def check_path(G, gen_str):
    path = re.findall(r'\d+', gen_str)
    if len(path) < 4:
        return 'wrong syntax'

    for node in path:
        if int(node) > len(itos) or int(node) < 0:
            return 'wrong syntax'
    
    if path[2] != path[0] or path[-1] != path[1]:
        return 'incorrect start/end'
        
    for i in range(2, len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return f'non-existence path {path[i], path[i + 1]}'
            
    return ''

def check_path_unreachable(G, gen_str, gt):
    path = re.findall(r'\d+|x', gen_str)
    if 'x' in path and len(path) < 4:
        return 0 if 'x' in gt else 1
        
    if 'x' in gt and 'x' not in gen_str:
        return 1

    return check_path(G, gen_str)

typedata = 'test'
f = open(f'{data_path}/{typedata}.txt', encoding='gbk')
texts = []
encode_texts = []
ground_truth = []

for line in f:
    if not simple_format:
        texts.append(line.split(':')[0] + ':')
        encode_texts.append(encode(line.split(':')[0] + ':'))
    else:
        pos = find_third_number_position(line)
        if(line[:pos] != ''):
            texts.append(line[:pos])
            encode_texts.append(encode(line[:pos]))
        
    ground_truth.append(line)
    
ground_truth = np.array(ground_truth)
encode_texts = torch.tensor(encode_texts, dtype=torch.long, device=device)
    
from tqdm import tqdm

batch_size = 1000
total_samples = len(encode_texts)
print(f"Total number of test samples: {total_samples}")

# Use smaller batch size if encode_texts is smaller than batch_size
if total_samples < batch_size:
    batch_size = total_samples
    ix = torch.arange(total_samples)
else:
    ix = torch.randint(total_samples, (batch_size,))

# Create output file with timestamp for uniqueness
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = out_dir + f'pred_{typedata}_{ckpt_iter}_{timestamp}.txt'
with open(output_file, 'w') as f:
    pass

wrong = 0
total = 0
errors_by_type = {
    "wrong syntax": 0,
    "incorrect start/end": 0,
    "non-existence path": 0
}

for i in tqdm(range(10)):
    x = encode_texts[ix]
    x_gt = ground_truth[ix]

    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    y_pred = [decode(y[t].tolist()).split('\n')[0] for t in range(batch_size)]

    with open(output_file, 'a') as f:
        for t, item in enumerate(y_pred):
            symbol = check_path(path_graph, item)
            total += 1
            
            # Track error types
            if symbol != "":
                wrong += 1
                if symbol == "wrong syntax":
                    errors_by_type["wrong syntax"] += 1
                elif symbol == "incorrect start/end":
                    errors_by_type["incorrect start/end"] += 1
                elif "non-existence path" in symbol:
                    errors_by_type["non-existence path"] += 1
                    
            f.write(item + " " + symbol + '\n')

# Calculate accuracy metrics
accuracy = (total - wrong) / total * 100
error_rate = wrong / total * 100

print(f"\nTest Results:")
print(f"Total samples: {total}")
print(f"Correct paths: {total - wrong}")
print(f"Incorrect paths: {wrong}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Error Rate: {error_rate:.2f}%")
print("\nError breakdown:")
for error_type, count in errors_by_type.items():
    percentage = count / total * 100 if total > 0 else 0
    print(f"- {error_type}: {count} ({percentage:.2f}%)")

# Add accuracy metrics to the bottom of the file
with open(output_file, 'a') as f:
    f.write("\n" + "-"*50 + "\n")
    f.write(f"MODEL CONFIGURATION:\n")
    f.write(f"- Checkpoint: {ckpt_iter}\n")
    f.write(f"- Layers: {gptconf.n_layer}\n")
    f.write(f"- Heads: {gptconf.n_head}\n")
    f.write(f"- Embedding dim: {gptconf.n_embd}\n")
    f.write(f"- Using identity embeddings: {getattr(gptconf, 'use_identity_embeddings', False)}\n")
    f.write(f"- Using positional embeddings: {getattr(gptconf, 'use_positional_embeddings', True)}\n")
    f.write("\n")
    f.write(f"TEST RESULTS:\n")
    f.write(f"Total samples: {total}\n")
    f.write(f"Correct paths: {total - wrong}\n")
    f.write(f"Incorrect paths: {wrong}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
    f.write(f"Error Rate: {error_rate:.2f}%\n")
    f.write("\nError breakdown:\n")
    for error_type, count in errors_by_type.items():
        percentage = count / total * 100 if total > 0 else 0
        f.write(f"- {error_type}: {count} ({percentage:.2f}%)\n")

print(f"\nResults saved to: {output_file}")