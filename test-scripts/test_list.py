import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.updated_model import GPTConfig, GPT 

import numpy as np
import argparse
import pickle
import re
import torch
from tqdm import tqdm
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_iter', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='list', help='Should be "list" for list reversal task')
    parser.add_argument('--config', type=str, default='1_1_120')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--min_value', type=int, default=0)
    parser.add_argument('--max_value', type=int, default=100)
    parser.add_argument('--is_sorted', type=str, default="True", help='Whether lists are sorted')
    parser.add_argument('--num_list_copies', type=int, default=5)
    parser.add_argument('--embedding_config', type=str, default='', help='Optional suffix for embedding config')
    parser.add_argument('--test_samples', type=int, default=-1, help='Number of test samples to evaluate (-1 for all)')
    return parser.parse_args()

def encode(s, stoi):
    ss = s.split(" ")
    encoded_string = [stoi[ch] for ch in ss]
    return encoded_string

def decode(l, itos):
    dec = ""
    for i in l:
        dec = dec + itos[i] + " "
    return dec[:-1]

def check_reversed_list(generated):
    """Check if the generated reversed list is correct"""
    
    # Check if '%' exists in the generated output
    if '%' not in generated:
        return "wrong syntax", -1
    
    # Split at '%'
    parts = generated.split('%')
    if len(parts) != 2:
        return "wrong syntax", -1
    
    input_list = parts[0].strip()
    output_list = parts[1].strip()
    
    # Split into tokens
    input_tokens = input_list.split()
    output_tokens = output_list.split()
    
    # Check if the length matches (input = output)
    if len(input_tokens) != len(output_tokens):
        return "wrong length", -1
    
    # Check if each token in output is the reverse of input
    expected_output = list(reversed(input_tokens))
    
    for i, (expected, actual) in enumerate(zip(expected_output, output_tokens)):
        if expected != actual:
            return f"failed at {i}", i
    
    return "", -1  # Success

def main():
    args = parse_args()
    
    # Determine list type directory
    list_type = "sorted" if args.is_sorted == "True" else "unsorted"
    embedding_config = args.embedding_config
    full_config = f'{args.config}_{embedding_config}' if embedding_config else args.config
    
    # Define paths
    data_path = f'data/{args.dataset}/{list_type}/{args.min_value}-{args.max_value}'
    meta_path = f'{data_path}/meta.pkl'
    
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    max_new_tokens = meta['block_size']
    top_k = len(stoi)
    
    out_dir = f'out/{args.dataset}_{list_type}_{full_config}_{args.min_value}-{args.max_value}'
    os.makedirs(out_dir, exist_ok=True)
    
    if args.num_list_copies == 0:
        ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt.pt')
    else:
        ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt_{args.num_list_copies}.pt')
    
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
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
    model.to(args.device)
    
    # Load test data
    typedata = 'test'
    test_file = f'{data_path}/{typedata}.txt'
    print(f"Loading test data from {test_file}...")
    
    test_prompts = []
    
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '%' in line:
                parts = line.split('%')
                prefix = parts[0].strip() + ' %'  # Include % as prompt ending
                test_prompts.append(prefix)

    # Limit test samples if specified
    if args.test_samples > 0 and args.test_samples < len(test_prompts):
        import random
        # Set seed for reproducibility
        random.seed(42)
        # Randomly sample a subset
        indices = random.sample(range(len(test_prompts)), args.test_samples)
        test_prompts = [test_prompts[i] for i in indices]
        print(f"Limited testing to {args.test_samples} randomly sampled examples")
    
    # Convert prompts to tensors individually (no padding needed - slower)
    encoded_prompts = [encode(p, stoi) for p in test_prompts]
    encoded_texts = [torch.tensor(p, dtype=torch.long, device=args.device) for p in encoded_prompts]
    
    # Create output file
    output_file = os.path.join(out_dir, f'list_reversal_{args.ckpt_iter}.txt')
    with open(output_file, 'w') as f:
        pass
    
    total = 0
    wrong = 0
    errors_by_type = {
        "wrong syntax": 0,
        "wrong length": 0,
        "failed at": {}  # Dictionary to track failures at each position
    }
    
    # Use appropriate batch size for tracking purposes
    batch_size = min(100, len(encoded_texts))
    total_samples = len(encoded_texts)
    
    print(f"Testing on {total_samples} examples")
    
    # Process in batches (modified for individual tensors)
    for i in tqdm(range(0, total_samples, batch_size)):
        # Get batch indices
        end_idx = min(i + batch_size, total_samples)
        batch_size_actual = end_idx - i
        
        # Process each example in the batch
        y_pred = []
        for j in range(i, end_idx):
            # Add batch dimension
            x = encoded_texts[j].unsqueeze(0)
            
            # Generate completion
            y = model.generate(x, max_new_tokens, temperature=args.temperature, top_k=top_k)
            
            # Decode and save
            y_pred.append(decode(y[0].tolist(), itos).split('\n')[0])
        
        # Evaluate predictions
        with open(output_file, 'a') as f:
            for j, item in enumerate(y_pred):
                # The input prompt
                prompt = test_prompts[i + j]
                
                # Evaluate
                error_type, error_idx = check_reversed_list(item)
                total += 1
                
                # Track error types
                if error_type != "":
                    wrong += 1
                    if error_type == "wrong syntax":
                        errors_by_type["wrong syntax"] += 1
                    elif error_type == "wrong length":
                        errors_by_type["wrong length"] += 1
                    elif "failed at" in error_type:
                        if error_idx not in errors_by_type["failed at"]:
                            errors_by_type["failed at"][error_idx] = 0
                        errors_by_type["failed at"][error_idx] += 1
                
                # Write to output file
                f.write(f"Input: {prompt}\n")
                f.write(f"Output: {item}\n")
                f.write(f"Error: {error_type}\n\n")
    
    # Calculate accuracy
    accuracy = (total - wrong) / total * 100 if total > 0 else 0
    error_rate = wrong / total * 100 if total > 0 else 0
    
    print(f"\nTest Results:")
    print(f"Total samples: {total}")
    print(f"Correct reversals: {total - wrong}")
    print(f"Incorrect reversals: {wrong}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Error Rate: {error_rate:.2f}%")
    
    print("\nError breakdown:")
    print(f"- wrong syntax: {errors_by_type['wrong syntax']}")
    print(f"- wrong length: {errors_by_type['wrong length']}")
    print("- failed at position:")
    for pos, count in sorted(errors_by_type["failed at"].items()):
        percentage = (count / total) * 100
        print(f"  - position {pos}: {count} ({percentage:.2f}%)")
    
    # Add summary to the output file
    with open(output_file, 'a') as f:
        f.write("\n" + "-"*50 + "\n")
        f.write("SUMMARY OF TEST RESULTS:\n")
        f.write(f"MODEL CONFIGURATION:\n")
        f.write(f"- Checkpoint: {args.ckpt_iter}\n")
        f.write(f"- Layers: {gptconf.n_layer}\n")
        f.write(f"- Heads: {gptconf.n_head}\n")
        f.write(f"- Embedding dim: {gptconf.n_embd}\n")
        f.write(f"- Using identity embeddings: {getattr(gptconf, 'use_identity_embeddings', False)}\n")
        f.write(f"- Using positional embeddings: {getattr(gptconf, 'use_positional_embeddings', True)}\n")
        f.write("\n")
        f.write(f"TEST RESULTS:\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Correct reversals: {total - wrong}\n")
        f.write(f"Incorrect reversals: {wrong}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Error Rate: {error_rate:.2f}%\n")
        f.write("\nError breakdown:\n")
        f.write(f"- wrong syntax: {errors_by_type['wrong syntax']}\n")
        f.write(f"- wrong length: {errors_by_type['wrong length']}\n")
        f.write("- failed at position:\n")
        for pos, count in sorted(errors_by_type["failed at"].items()):
            percentage = (count / total) * 100
            f.write(f"  - position {pos}: {count} ({percentage:.2f}%)\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()