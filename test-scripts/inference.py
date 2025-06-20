import os
import sys
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.updated_model_4 import GPTConfig, GPT

def parse_args():
    parser = argparse.ArgumentParser(description='Inference script for transformer models on list tasks')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (PT file)')
    parser.add_argument('--meta_path', type=str, required=True, help='Path to the meta.pkl file')
    parser.add_argument('--input_list', type=str, nargs='+', help='List of input prompts (e.g., "123%", "456%")')
    parser.add_argument('--input_file', type=str, help='File containing input prompts, one per line')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--max_new_tokens', type=int, default=None, 
                        help='Maximum number of tokens to generate (defaults to model block_size if not specified)')
    return parser.parse_args()

def encode(s, stoi):
    """Encode a string into token IDs using the vocabulary mapping"""
    ss = s.split(" ")
    encoded_string = [stoi[ch] for ch in ss]
    return encoded_string

def decode(l, itos):
    """Decode a list of token IDs back into a string"""
    dec = ""
    for i in l:
        dec = dec + itos[i] + " "
    return dec[:-1]  # Remove trailing space

def load_model(checkpoint_path, device):
    """Load a model from a checkpoint file"""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model configuration and initialize model
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Handle potential prefix in state dict keys
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Load weights
    model.load_state_dict(state_dict)
    
    # Print model configuration
    print(f"Model configuration:")
    print(f"- Layers: {gptconf.n_layer}")
    print(f"- Heads: {gptconf.n_head}")
    print(f"- Embedding dim: {gptconf.n_embd}")
    print(f"- Using identity embeddings: {getattr(gptconf, 'use_identity_embeddings', False)}")
    print(f"- Using positional embeddings: {getattr(gptconf, 'use_positional_embeddings', True)}")
    
    return model.to(device), gptconf

def load_tokenizer(meta_path):
    """Load the vocabulary mappings from the meta.pkl file"""
    print(f"Loading vocabulary from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    
    return stoi, itos, block_size

def get_inputs(args):
    """Get input prompts from either command line arguments or an input file"""
    if args.input_list:
        return args.input_list
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            return [line.strip() for line in f]
    else:
        raise ValueError("Either --input_list or --input_file must be provided")

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, gptconf = load_model(args.checkpoint, args.device)
    stoi, itos, block_size = load_tokenizer(args.meta_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get input prompts
    input_prompts = get_inputs(args)
    print(f"Running inference on {len(input_prompts)} prompts")
    
    # Set max_new_tokens if not specified
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else block_size
    
    # Run inference on each prompt
    results = []
    for prompt in tqdm(input_prompts):
        # Encode the prompt
        encoded_prompt = encode(prompt, stoi)
        input_tensor = torch.tensor(encoded_prompt, dtype=torch.long, device=args.device).unsqueeze(0)
        
        # Generate output
        with torch.no_grad():
            output_tensor = model.generate(
                input_tensor, 
                max_new_tokens=max_new_tokens, 
                temperature=args.temperature,
                top_k=len(stoi)  # Use all tokens in the vocabulary
            )
        
        # Decode the output
        output_text = decode(output_tensor[0].tolist(), itos)
        results.append((prompt, output_text))
    
    # Print results
    print("\nInference Results:")
    print("=" * 60)
    for prompt, output in results:
        print(f"Input: {prompt}")
        print(f"Input+Output: {output}")
        print("-" * 60)

if __name__ == "__main__":
    main()