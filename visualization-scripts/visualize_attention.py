# Partly Adapted from: https://github.com/rmaestre/transformers_path_search?tab=readme-ov-file.

import os
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import tempfile
import traceback

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model classes
from model.updated_model import GPTConfig, GPT
from gpt_utils.utils_final import AttentionVisualizer

class ModelWithAttention(GPT):
    """Wrapper class that extends the GPT model to return attention weights"""
    
    def forward(self, idx, targets=None, return_attn_weights=False):
        device = idx.device
        b, t = idx.size()
        
        # Get token embeddings
        tok_emb = self.transformer.wte(idx)
        
        # Add positional embeddings if enabled
        if self.config.use_positional_embeddings:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        
        # If we need to return attention weights, capture them during the forward pass
        attention_weights = []
        if return_attn_weights:
            # Process through transformer blocks
            for block in self.transformer.h:
                normalized = block.ln_1(x)
                
                # Extract attention weights by running attention manually
                q, k, v = block.attn.c_attn(normalized).split(self.config.n_embd, dim=2)
                
                # Reshape for attention
                B, T, C = q.size()
                n_head = block.attn.n_head
                head_size = C // n_head
                
                q = q.view(B, T, n_head, head_size).transpose(1, 2)
                k = k.view(B, T, n_head, head_size).transpose(1, 2)
                v = v.view(B, T, n_head, head_size).transpose(1, 2)
                
                # Calculate attention scores
                att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float)))
                
                # Apply causal mask
                mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                # Apply softmax
                att = torch.nn.functional.softmax(att, dim=-1)
                
                # Store attention weights
                attention_weights.append(att.detach())
                
                # Continue with normal forward pass
                x = x + block.attn(normalized)
                x = x + block.mlp(block.ln_2(x))
            
            # Final layer norm
            x = self.transformer.ln_f(x)
            
            # Calculate logits
            if targets is not None:
                # Training mode - compute all logits
                logits = self.lm_head(x)
                # Need to import F for cross_entropy
                import torch.nn.functional as F
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            else:
                # Inference mode - compute only last position
                logits = self.lm_head(x[:, [-1], :])
                loss = None
                
            return logits, loss, attention_weights
            
        else:
            # Normal forward pass without capturing attention
            # Process through transformer blocks
            for block in self.transformer.h:
                x = x + block.attn(block.ln_1(x))
                x = x + block.mlp(block.ln_2(x))
                
            x = self.transformer.ln_f(x)
            
            if targets is not None:
                logits = self.lm_head(x)
                # Need to import F for cross_entropy
                import torch.nn.functional as F
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None
                
            return logits, loss

def visualize_attention_inline(model_path, meta_path, input_text, heads=[0], layers=[0], device='cuda'):
    """
    Main function to visualize attention patterns inline in a notebook for a single input.
    """
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model from checkpoint  
    if 'model_args' in checkpoint:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = ModelWithAttention(gptconf)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    else:
        raise ValueError("Checkpoint format not recognized")
    
    # Clean up state dict if needed
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Load vocabulary
    print(f"Loading vocabulary from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    
    # Encode the input
    tokens = []
    for token in input_text.split():
        if token in stoi:
            tokens.append(stoi[token])
        else:
            print(f"Warning: Token '{token}' not found in vocabulary")
    
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Run model with attention weights
    with torch.no_grad():
        _, _, attention_weights = model(input_tensor, return_attn_weights=True)
    
    # Visualize attention patterns
    for layer_idx in layers:
        if layer_idx < len(attention_weights):
            layer_weights = attention_weights[layer_idx]
            for head_idx in heads:
                if head_idx < layer_weights.shape[1]:
                    # Create plot
                    plt.figure(figsize=(12, 10))
                    attn_matrix = layer_weights[0, head_idx].cpu().numpy()
                    
                    # Get sequence length and tokens from input_text
                    tokens_list = input_text.split()
                    seq_length = len(tokens_list)
                    
                    # Create the heatmap
                    sns.heatmap(attn_matrix, cmap="viridis", cbar=True, square=True, annot=False)
                    
                    # Update the title to show sequence length
                    plt.title(f"Attention Plot - Sequence length {seq_length}", fontsize=14)
                    
                    # Set the ticks to the actual tokens
                    plt.xticks(np.arange(len(tokens_list)) + 0.5, tokens_list, rotation=45)
                    plt.yticks(np.arange(len(tokens_list)) + 0.5, tokens_list)
                    
                    plt.xlabel("Key Tokens", fontsize=12)
                    plt.ylabel("Query Tokens", fontsize=12)
                    
                    # Display inline
                    display(plt.gcf())
                    plt.close()

def visualize_average_attention_inline(model_path, meta_path, test_file_path, test_length, heads=[0], layers=[0], device='cuda'):
    """
    Main function to visualize average attention patterns for inputs of a specific length from a test file.
    
    Args:
        model_path: Path to the model checkpoint
        meta_path: Path to the meta.pkl file with vocabulary
        test_file_path: Path to the test file containing multiple inputs
        test_length: Target sequence length to filter and average over
        heads: List of attention heads to visualize
        layers: List of transformer layers to visualize
        device: Device to run on ('cuda' or 'cpu')
    """
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model from checkpoint
    if 'model_args' in checkpoint:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = ModelWithAttention(gptconf)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    else:
        raise ValueError("Checkpoint format not recognized")
    
    # Clean up state dict if needed
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Load vocabulary
    print(f"Loading vocabulary from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    
    # Load and filter test inputs by length
    print(f"Loading test inputs from {test_file_path}...")
    test_inputs = []
    with open(test_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split()
                if len(tokens) == test_length:
                    test_inputs.append(line)
    
    print(f"Found {len(test_inputs)} inputs with length {test_length}")
    
    if len(test_inputs) == 0:
        print(f"No inputs found with length {test_length}")
        return
    
    # Initialize attention accumulators
    attention_accumulators = {}
    
    # Process each test input
    print("Processing test inputs...")
    for i, input_text in enumerate(test_inputs):
        if i % 10 == 0:
            print(f"Processing input {i+1}/{len(test_inputs)}")
        
        # Encode the input
        tokens = []
        valid_input = True
        for token in input_text.split():
            if token in stoi:
                tokens.append(stoi[token])
            else:
                print(f"Warning: Token '{token}' not found in vocabulary, skipping input")
                valid_input = False
                break
        
        if not valid_input:
            continue
        
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Run model with attention weights
        with torch.no_grad():
            _, _, attention_weights = model(input_tensor, return_attn_weights=True)
        
        # Accumulate attention weights
        for layer_idx in layers:
            if layer_idx < len(attention_weights):
                layer_weights = attention_weights[layer_idx]
                for head_idx in heads:
                    if head_idx < layer_weights.shape[1]:
                        key = (layer_idx, head_idx)
                        attn_matrix = layer_weights[0, head_idx].cpu().numpy()
                        
                        if key not in attention_accumulators:
                            attention_accumulators[key] = np.zeros_like(attn_matrix)
                        
                        attention_accumulators[key] += attn_matrix
    
    # Calculate and visualize average attention
    num_valid_inputs = len([inp for inp in test_inputs if all(token in stoi for token in inp.split())])
    print(f"Averaging over {num_valid_inputs} valid inputs")
    
    for (layer_idx, head_idx), accumulated_attention in attention_accumulators.items():
        # Calculate average
        avg_attention = accumulated_attention / num_valid_inputs
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Create position labels (1-indexed)
        position_labels = [str(i+1) for i in range(test_length)]
        
        # Create the heatmap
        sns.heatmap(avg_attention, cmap="viridis", cbar=True, square=True, annot=False)
        
        # Update the title
        plt.title(f"Average Attention - Layer {layer_idx}, Head {head_idx} - Sequence Length {test_length} ({num_valid_inputs} samples)", fontsize=14)
        
        # Set the ticks to position numbers
        plt.xticks(np.arange(test_length) + 0.5, position_labels)
        plt.yticks(np.arange(test_length) + 0.5, position_labels)
        
        plt.xlabel("Key Position", fontsize=12)
        plt.ylabel("Query Position", fontsize=12)
        
        # Display inline
        display(plt.gcf())
        plt.close()

# Usage functions for convenience
def visualize_single_attention(model_path, meta_path, input_text, heads=[0], layers=[0], device='cuda'):
    """Convenience function for single input visualization"""
    return visualize_attention_inline(model_path, meta_path, input_text, heads, layers, device)

def visualize_average_attention(model_path, meta_path, test_file_path, test_length, heads=[0], layers=[0], device='cuda'):
    """Convenience function for average attention visualization"""
    return visualize_average_attention_inline(model_path, meta_path, test_file_path, test_length, heads, layers, device)