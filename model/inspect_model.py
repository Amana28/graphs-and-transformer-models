import os
import argparse
import pickle
import torch
import sys

# Add the parent directory to the Python path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import model
from updated_model import GPTConfig, GPT

def main():
    parser = argparse.ArgumentParser(description="Inspect model architecture")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint file')
    args = parser.parse_args()
    
    # Load the model checkpoint
    print(f"Loading model from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Clean up state dict if needed
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Print model configuration
    print(f"Model configuration: {gptconf}")
    
    # Print model structure
    print("\nModel Structure:")
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {module.__class__.__name__}, Parameters: {params}")
        
        # Print parameter shapes for this module
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                print(f"  {param_name}: {param.shape}")
    
    # Specifically examine the attention module structure
    print("\nDetailed Attention Structure:")
    if hasattr(model.transformer, 'h') and len(model.transformer.h) > 0:
        attn_block = model.transformer.h[0].attn
        print(f"Attention module: {attn_block.__class__.__name__}")
        
        # List all attributes
        print("Attributes:")
        for attr_name in dir(attn_block):
            if not attr_name.startswith('_') and not callable(getattr(attn_block, attr_name)):
                attr = getattr(attn_block, attr_name)
                if isinstance(attr, torch.nn.Module):
                    print(f"  {attr_name}: {attr.__class__.__name__}")
                    for sub_name, param in attr.named_parameters():
                        print(f"    {sub_name}: {param.shape}")
                elif isinstance(attr, (int, float, bool, str)):
                    print(f"  {attr_name}: {attr}")
        
        # Print forward method source if possible
        print("\nForward Method:")
        if hasattr(attn_block.__class__, 'forward'):
            import inspect
            try:
                forward_source = inspect.getsource(attn_block.__class__.forward)
                print(forward_source)
            except Exception as e:
                print(f"Could not get source: {e}")
    
    print("\nDone inspecting model.")

if __name__ == "__main__":
    main()