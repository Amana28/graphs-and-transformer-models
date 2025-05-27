import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.updated_model import GPTConfig, GPT

import numpy as np
import networkx as nx
import argparse
import pickle
import re
import torch
from tqdm import tqdm
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Test STS model on graph path completion')
    parser.add_argument('--ckpt_iter', type=int, default=10000, help='Checkpoint iteration to load')
    parser.add_argument('--config', type=str, default='1_1_120', help='Model configuration')
    parser.add_argument('--temperature', type=float, default=1, help='Generation temperature')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in graph')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths used in training')
    parser.add_argument('--task_type', type=str, default='sts', choices=['st', 'sts'], help='Task type (st or sts)')
    parser.add_argument('--test_samples', type=int, default=-1, help='Number of test samples (-1 for all)')
    return parser.parse_args()

def encode(s, stoi):
    """Encode string to token IDs"""
    ss = s.split(" ")
    encoded_string = [stoi[ch] for ch in ss]
    return encoded_string

def decode(l, itos):
    """Decode token IDs to string"""
    dec = ""
    for i in l:
        dec = dec + itos[i] + " "
    return dec[:-1]

def check_sts_path(G, gen_str, itos):
    """
    Check if generated STS path is valid + if output it is a palindrome
    """
    path = re.findall(r'\d+', gen_str)
    
    # Basic checks
    if len(path) < 4:
        return 'incorrect path'

    for node in path:
        if int(node) > len(itos) or int(node) < 0:
            return 'incorrect path'
    
    source = path[0]
    target = path[1]
    
    # Check STS format and ending
    if path[2] != source or path[-1] != source:
        return 'incorrect path'
    
    # Extract path part (skip s t s)
    path_part = path[3:]
    
    if len(path_part) == 0:
        return 'no path'
    
    # Find target position
    target_pos = -1
    for i, node in enumerate(path_part):
        if node == target:
            target_pos = i
            break
    
    if target_pos == -1:
        return 'target not found'
    
    # Extract forward path and check edges
    forward_path = path_part[:target_pos + 1]
    for i in range(len(forward_path) - 1):
        if not G.has_edge(forward_path[i], forward_path[i + 1]):
            return 'invalid path'
    
    # Check if path_part is a palindrome
    n = len(path_part)
    for i in range(n // 2):
        if path_part[i] != path_part[n - 1 - i]:
            return 'incorrect reversal'
    
    return ''  # Valid path

def extract_prompt_from_sts_line(line):
    """
    Extract prompt from STS test line
    Input: "8 22 8 8 9 12 14 18 21 22 21 18 14 12 9 8"
    Output: "8 22 8" (s t s format)
    """
    tokens = line.strip().split()
    if len(tokens) >= 3:
        return f"{tokens[0]} {tokens[1]} {tokens[2]}"
    return line.strip()

def main():
    args = parse_args()
    
    print(f"Testing STS model on graph path completion")
    print(f"Task type: {args.task_type.upper()}")
    print(f"Checkpoint: {args.ckpt_iter}")
    print("="*60)
    
    # Setup paths
    dataset = 'simple_graph'
    # Always include task_type in path since we're testing STS
    data_path = f'/content/graphs-and-transformer-models/data/{dataset}/{args.task_type}/{args.num_nodes}'
    out_dir = f'out/{dataset}_{args.task_type}_{args.config}_{args.num_nodes}'
    
    meta_path = f'{data_path}/meta.pkl'
    
    print(f"Loading metadata from: {meta_path}")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    max_new_tokens = meta['block_size']
    top_k = len(itos)
    
    print(f"Vocabulary size: {len(stoi)}")
    print(f"Block size: {max_new_tokens}")
    
    # Load model
    os.makedirs(out_dir, exist_ok=True)
    
    if args.num_of_paths == 0:
        ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt.pt')
    else:
        ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt_{args.num_of_paths}.pt')
    
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Clean state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)
    
    # Print model configuration
    print(f"\nModel Configuration:")
    print(f"- Layers: {gptconf.n_layer}")
    print(f"- Heads: {gptconf.n_head}")
    print(f"- Embedding dim: {gptconf.n_embd}")
    print(f"- Using identity embeddings: {getattr(gptconf, 'use_identity_embeddings', False)}")
    print(f"- Using positional embeddings: {getattr(gptconf, 'use_positional_embeddings', True)}")
    
    # Load graph
    graph_path = f'{data_path}/path_graph.graphml'
    print(f"Loading graph from: {graph_path}")
    path_graph = nx.read_graphml(graph_path)
    print(f"Graph: {path_graph.number_of_nodes()} nodes, {path_graph.number_of_edges()} edges")
    
    # Load test data
    if args.task_type == 'sts':
        test_file = f'{data_path}/test_sts.txt'
    else:
        test_file = f'{data_path}/test.txt'
    
    print(f"Loading test data from: {test_file}")
    
    test_lines = []
    test_prompts = []
    
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                test_lines.append(line)
                # Extract prompt (s t s) from the full STS line
                prompt = extract_prompt_from_sts_line(line)
                test_prompts.append(prompt)
    
    print(f"Loaded {len(test_lines)} test examples")
    
    # Limit test samples if specified
    if args.test_samples > 0 and args.test_samples < len(test_lines):
        import random
        random.seed(42)
        indices = random.sample(range(len(test_lines)), args.test_samples)
        test_lines = [test_lines[i] for i in indices]
        test_prompts = [test_prompts[i] for i in indices]
        print(f"Limited testing to {args.test_samples} randomly sampled examples")
    
    # Encode prompts
    encoded_prompts = [encode(prompt, stoi) for prompt in test_prompts]
    encoded_texts = [torch.tensor(p, dtype=torch.long, device=args.device) for p in encoded_prompts]
    
    # Create output file
    output_file = os.path.join(out_dir, f'sts_test_results_{args.ckpt_iter}.txt')
    with open(output_file, 'w') as f:
        pass  # Create empty file
    
    # Statistics tracking
    total = 0
    correct = 0
    errors_by_type = {}
    
    print(f"\nTesting on {len(test_lines)} examples...")
    print("="*60)
    
    # Process each test case individually
    for i in tqdm(range(len(encoded_texts)), desc="Testing"):
        x = encoded_texts[i].unsqueeze(0)
        
        # Generate completion
        y = model.generate(x, max_new_tokens, temperature=args.temperature, top_k=top_k)
        generated = decode(y[0].tolist(), itos).split('\n')[0]
        
        # Check path validity
        error = check_sts_path(path_graph, generated, itos)
        total += 1
        
        if error == '':
            correct += 1
            error_msg = ""
        else:
            error_msg = f"  {error}"
            # Track error types
            error_type = error.split(' - ')[0].split(':')[0]
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        # Write result (line by line format)
        with open(output_file, 'a') as f:
            f.write(f"{generated}{error_msg}\n")
    
    # Calculate final statistics
    accuracy = (correct / total * 100) if total > 0 else 0
    error_rate = ((total - correct) / total * 100) if total > 0 else 0
    
    print(f"\nFINAL RESULTS:")
    print(f"="*60)
    print(f"Total samples tested: {total}")
    print(f"Correct paths: {correct}")
    print(f"Incorrect paths: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Error rate: {error_rate:.2f}%")
    
    if errors_by_type:
        print(f"\nError breakdown:")
        for error_type, count in sorted(errors_by_type.items()):
            percentage = (count / total) * 100
            print(f"- {error_type}: {count} ({percentage:.2f}%)")
    
    # Write summary to file
    with open(output_file, 'a') as f:
        f.write("\n" + "="*60 + "\n")
        f.write("SUMMARY:\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Correct paths: {correct}\n")
        f.write(f"Incorrect paths: {total - correct}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Error rate: {error_rate:.2f}%\n")
        
        if errors_by_type:
            f.write(f"\nError breakdown:\n")
            for error_type, count in sorted(errors_by_type.items()):
                percentage = (count / total) * 100
                f.write(f"- {error_type}: {count} ({percentage:.2f}%)\n")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()