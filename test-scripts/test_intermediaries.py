import os
from collections import Counter
import tiktoken
import matplotlib.pyplot as plt
from model.model import GPTConfig, GPT
import numpy as np
import networkx as nx
import argparse
import pickle
import re
import torch
from gpt_utils.utils_final import (
    AttentionVisualizer
)

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_iter', type=int, default=10000)
parser.add_argument('--graph_type', type=str, default='simple_graph')
parser.add_argument('--config', type=str, default='1_1_120')
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--num_nodes', type=int, default=100)
parser.add_argument('--num_of_paths', type=int, default=20)
parser.add_argument("--problem", type=str, default="path", help="Which algorithmic problem (path/cut)")

args = parser.parse_args()
dataset = args.graph_type
ckpt_iter = args.ckpt_iter
problem = args.problem
device = args.device
temperature = args.temperature
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
config = args.config

data_path = f'data/{dataset}/{num_nodes}_{problem}'
meta_path = f'{data_path}/meta.pkl'

print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
max_new_tokens = meta['block_size']
top_k = len(itos)
simple_format = meta['simple_format']

out_dir = f'out/{dataset}_{config}_{num_nodes}_{problem}/'

if (num_of_paths == 0):
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt.pt')
else:
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt_{num_of_paths}.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")
print(type(out_dir))
viz = AttentionVisualizer(model, tokenizer, out_dir = out_dir, test_path=f'{data_path}/test.txt', meta_path=meta_path)
viz.infer_and_visualize_attention( heads=[0], layers = [0], input_text="21 44 21 23 30 32 44", problem = "path")


path_graph = f'{data_path}/path_graph.graphml'
path_graph = nx.read_graphml(path_graph)


def find_fourth_number_position(number_string):
    numbers = number_string.split()
    fourth_number_index = 3
    position = sum(len(num) for num in numbers[:fourth_number_index]) + fourth_number_index - 1
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
    print(gen_str)
    if len(path) < 6:
        return 'wrong syntax'

    if path[3] != path[0] or path[-1] != path[2]:
        return 'incorrect start/end'

    for i in range(3, len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return f'non-existent path {path[i], path[i + 1]}'

    if path[1] not in path[3:-1]:  # Ensure the intermediate node appears in the path
        return 'missing intermediate node'

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
        pos = find_fourth_number_position(line)
        if (line[:pos] != ''):
            texts.append(line[:pos])
            #print(line)
            #print(line[:pos])
            encode_texts.append(encode(line[:pos]))

    ground_truth.append(line)

ground_truth = np.array(ground_truth)
encode_texts = torch.tensor(encode_texts, dtype=torch.long, device=device)

from tqdm import tqdm

batch_size = 1000
ix = torch.randint(len(encode_texts), (batch_size,))

with open(out_dir + f'pred_{typedata}_{ckpt_iter}.txt', 'w') as f:
    pass

wrong = 0
intermed_wrong = 0
for i in tqdm(range(10)):
    x = encode_texts[ix]
    x_gt = ground_truth[ix]
    # x = (torch.tensor(text, dtype=torch.long, device=device))
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    y_pred = [decode(y[t].tolist()).split('\n')[0] for t in range(batch_size)]

    # Lists to store path lengths
    correct_lengths = []
    incorrect_lengths = []

    with open(out_dir + f'pred_{typedata}_{ckpt_iter}.txt', 'a') as f:
        for t, item in enumerate(y_pred):
            symbol = check_path(path_graph, item)
            path_len = len(re.findall(r'\d+', item))
            if (symbol != ""):
                incorrect_lengths.append(path_len)
                wrong = wrong + 1
                if symbol == 'missing intermediate node':
                    intermed_wrong += 1
            else:
                correct_lengths.append(path_len)
            f.write(item + " % " + symbol + '\n')
        f.write(f"Number of wrongs: {wrong}")
        f.write(f"Intermediate node missing: {intermed_wrong}")

    # Plotting
correct_counts = Counter(correct_lengths)
incorrect_counts = Counter(incorrect_lengths)

# Get all unique path lengths
all_lengths = set(correct_counts.keys()).union(incorrect_counts.keys())

# Compute proportion of incorrect paths per length
proportions = {
    length: correct_counts[length] / (correct_counts[length] + incorrect_counts[length])
    for length in all_lengths
}

# Sorting lengths for plotting
sorted_lengths = sorted(proportions.keys())
sorted_proportions = [proportions[length] for length in sorted_lengths]

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(sorted_lengths, sorted_proportions, color="green", alpha=0.7)
plt.xlabel("Path Length")
plt.ylabel("Proportion of Correct Paths")
plt.title("Proportion of Correct Paths per Path Length")
plt.xticks(sorted_lengths)
plt.ylim(0, 1)  # Proportion ranges from 0 to 1
plt.savefig(out_dir + f'pathlen_proportion.png', dpi=400)