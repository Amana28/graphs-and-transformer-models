"""
Attention Pattern Visualization Script for updated_model_4 (no MLP)
Visualizes how the model attends to different positions for a given input.

Usage:
    python visualize_attention.py --checkpoint_path <path> --meta_path <path> --input "0 1 0 1 % ?"

Example:
    python visualization-scripts/visualize_attention.py \
        --checkpoint_path out/list_unsorted_variable_custom_1_2_16_cmaes_0-1/500_ckpt_1.pt \
        --meta_path data/list/unsorted/variable/0-1/custom/meta.pkl \
        --input "0 1 0 % % % % % ? 0" \
        --heads 0 1
"""
import os
import sys
import argparse
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.updated_model_4 import GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize attention patterns')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--meta_path', type=str, required=True,
                        help='Path to meta.pkl file')
    parser.add_argument('--input', type=str, default=None,
                        help='Single input sequence (space-separated tokens)')
    parser.add_argument('--inputs_file', type=str, default=None,
                        help='Path to test file with multiple inputs (one per line)')
    parser.add_argument('--custom_inputs', type=str, nargs='+', default=None,
                        help='List of custom input sequences to average over')
    parser.add_argument('--target_length', type=int, default=None,
                        help='Filter inputs to this sequence length (for averaging)')
    parser.add_argument('--heads', type=int, nargs='+', default=[0],
                        help='Attention heads to visualize (default: 0)')
    parser.add_argument('--layers', type=int, nargs='+', default=[0],
                        help='Transformer layers to visualize (default: 0)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save visualization (default: show inline)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (default: cpu)')
    return parser.parse_args()


class ModelWithAttention(GPT):
    """Wrapper class that extends GPT to return attention weights"""
    
    def forward(self, idx, targets=None, return_attn_weights=False):
        device = idx.device
        b, t = idx.size()
        
        # Get embeddings
        tok_emb = self.transformer.wte(idx)
        
        if self.config.use_fixed_positions:
            if self.config.use_identity_embeddings:
                x = tok_emb
            else:
                pos_emb = self.transformer.wpe(t)
                pos_emb = pos_emb.unsqueeze(0).expand(b, -1, -1)
                x = torch.cat([tok_emb, pos_emb], dim=-1)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = tok_emb + pos_emb
        
        x = self.transformer.drop(x)
        
        attention_weights = []
        
        if return_attn_weights:
            for block in self.transformer.h:
                normalized = block.ln_1(x)
                
                # Extract Q, K, V
                q, k, v = block.attn.c_attn(normalized).split(self.config.n_embd, dim=2)
                
                B, T, C = q.size()
                n_head = block.attn.n_head
                head_size = C // n_head
                
                q = q.view(B, T, n_head, head_size).transpose(1, 2)
                k = k.view(B, T, n_head, head_size).transpose(1, 2)
                v = v.view(B, T, n_head, head_size).transpose(1, 2)
                
                # Calculate attention scores
                att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
                
                # Apply causal mask
                mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                # Softmax
                att = torch.nn.functional.softmax(att, dim=-1)
                attention_weights.append(att.detach())
                
                # Continue forward pass (no MLP in model_4)
                x = x + block.attn(normalized)
            
            x = self.transformer.ln_f(x)
            
            if targets is not None:
                logits = self.lm_head(x)
                import torch.nn.functional as F
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None
            
            return logits, loss, attention_weights
        else:
            for block in self.transformer.h:
                x = x + block.attn(block.ln_1(x))
            
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, [-1], :]) if targets is None else self.lm_head(x)
            loss = None
            return logits, loss


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = ModelWithAttention(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, gptconf


def visualize_attention(model, input_tokens, token_labels, heads, layers, config, save_path=None):
    """Visualize attention patterns"""
    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        _, _, attention_weights = model(input_tensor, return_attn_weights=True)
    
    # Create subplots: rows = layers, cols = heads
    n_layers = len(layers)
    n_heads = len(heads)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(6*n_heads, 5*n_layers), squeeze=False)
    
    for row, layer_idx in enumerate(layers):
        if layer_idx >= len(attention_weights):
            print(f"Warning: Layer {layer_idx} doesn't exist, skipping")
            continue
            
        layer_weights = attention_weights[layer_idx]
        
        for col, head_idx in enumerate(heads):
            if head_idx >= layer_weights.shape[1]:
                print(f"Warning: Head {head_idx} doesn't exist, skipping")
                continue
            
            ax = axes[row, col]
            attn_matrix = layer_weights[0, head_idx].cpu().numpy()
            
            sns.heatmap(attn_matrix, cmap="viridis", cbar=True, square=True, 
                       annot=False, ax=ax, vmin=0, vmax=1)
            
            ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=12)
            ax.set_xticks(np.arange(len(token_labels)) + 0.5)
            ax.set_yticks(np.arange(len(token_labels)) + 0.5)
            ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(token_labels, fontsize=9)
            ax.set_xlabel('Key (attending to)', fontsize=10)
            ax.set_ylabel('Query (from)', fontsize=10)
    
    plt.suptitle(f'Attention Patterns\nInput: {" ".join(token_labels)}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_average_attention(model, input_list, stoi, heads, layers, config, target_length=None, save_path=None):
    """Visualize average attention patterns across multiple inputs"""
    device = next(model.parameters()).device
    
    # Filter by length if specified
    if target_length:
        input_list = [inp for inp in input_list if len(inp.split()) == target_length]
        print(f"Filtered to {len(input_list)} inputs with length {target_length}")
    
    if not input_list:
        print("No valid inputs found!")
        return
    
    # Determine sequence length from first input
    seq_length = len(input_list[0].split())
    
    # Initialize attention accumulators
    attention_sums = {}
    valid_count = 0
    
    for i, input_text in enumerate(input_list):
        tokens = input_text.split()
        if len(tokens) != seq_length:
            continue  # Skip inputs with different lengths
            
        # Encode tokens
        try:
            input_tokens = [stoi[t] for t in tokens]
        except KeyError as e:
            print(f"Skipping input {i}: token {e} not in vocabulary")
            continue
        
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            _, _, attention_weights = model(input_tensor, return_attn_weights=True)
        
        # Accumulate attention weights
        for layer_idx in layers:
            if layer_idx < len(attention_weights):
                layer_weights = attention_weights[layer_idx]
                for head_idx in heads:
                    if head_idx < layer_weights.shape[1]:
                        key = (layer_idx, head_idx)
                        attn = layer_weights[0, head_idx].cpu().numpy()
                        if key not in attention_sums:
                            attention_sums[key] = np.zeros_like(attn)
                        attention_sums[key] += attn
        
        valid_count += 1
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(input_list)} inputs")
    
    print(f"Averaged over {valid_count} valid inputs")
    
    if valid_count == 0:
        print("No valid inputs processed!")
        return
    
    # Compute averages and plot
    n_layers = len(layers)
    n_heads = len(heads)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(6*n_heads, 5*n_layers), squeeze=False)
    
    # Create position labels
    position_labels = [str(i+1) for i in range(seq_length)]
    
    for row, layer_idx in enumerate(layers):
        for col, head_idx in enumerate(heads):
            key = (layer_idx, head_idx)
            if key in attention_sums:
                ax = axes[row, col]
                avg_attn = attention_sums[key] / valid_count
                
                sns.heatmap(avg_attn, cmap="viridis", cbar=True, square=True,
                           annot=False, ax=ax, vmin=0, vmax=1)
                
                ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=12)
                ax.set_xticks(np.arange(seq_length) + 0.5)
                ax.set_yticks(np.arange(seq_length) + 0.5)
                ax.set_xticklabels(position_labels, fontsize=9)
                ax.set_yticklabels(position_labels, fontsize=9)
                ax.set_xlabel('Key Position', fontsize=10)
                ax.set_ylabel('Query Position', fontsize=10)
    
    plt.suptitle(f'Average Attention Patterns ({valid_count} samples, length {seq_length})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return attention_sums, valid_count


def main():
    args = parse_args()
    
    # Validate input arguments
    if not args.input and not args.inputs_file and not args.custom_inputs:
        print("Error: Must provide one of --input, --inputs_file, or --custom_inputs")
        return
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.meta_path}...")
    with open(args.meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    
    # Load model
    model, config = load_model(args.checkpoint_path, args.device)
    
    print(f"\nModel: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd")
    print(f"Visualizing heads: {args.heads}, layers: {args.layers}")
    
    # Mode 1: Single input visualization
    if args.input:
        print(f"Single input mode: {args.input}")
        token_labels = args.input.split()
        input_tokens = []
        for token in token_labels:
            if token in stoi:
                input_tokens.append(stoi[token])
            else:
                print(f"Warning: Token '{token}' not in vocabulary, skipping")
        
        if not input_tokens:
            print("Error: No valid tokens found")
            return
        
        visualize_attention(model, input_tokens, token_labels, args.heads, args.layers, 
                           config, save_path=args.save_path)
    
    # Mode 2: Average from test file
    elif args.inputs_file:
        print(f"Inputs file mode: {args.inputs_file}")
        with open(args.inputs_file, 'r') as f:
            input_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(input_list)} inputs from file")
        
        visualize_average_attention(model, input_list, stoi, args.heads, args.layers,
                                   config, target_length=args.target_length, 
                                   save_path=args.save_path)
    
    # Mode 3: Average from custom inputs list
    elif args.custom_inputs:
        print(f"Custom inputs mode: {len(args.custom_inputs)} inputs")
        visualize_average_attention(model, args.custom_inputs, stoi, args.heads, args.layers,
                                   config, target_length=args.target_length,
                                   save_path=args.save_path)


if __name__ == '__main__':
    main()