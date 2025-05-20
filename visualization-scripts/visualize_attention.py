# This file is copied verbatim from: https://github.com/rmaestre/transformers_path_search?tab=readme-ov-file.
# No authorship is claimed


import argparse
import matplotlib.pyplot as plt
import tiktoken
import torch
import os
from utils_final import (
    GPTModel,
    AttentionVisualizer,
    load_config,
)


def main():
    """
    Visualize attention weights of a pre-trained GPT model using input text.
    """
    parser = argparse.ArgumentParser(
        description="Visualize attention weights of a pre-trained GPT model."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/gpt_config.json",
        help="Path to the model configuration file. Default is 'config/gpt_config.json'.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="models/gpt_model.pth",
        help="Path to the pre-trained model file. Default is 'models/gpt_model.pth'.",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="19 94 19 27 34 46 58 65 67 82 87 94",
        help="Input text to analyze. Default is '19 94 19 27 34 46 58 65 67 82 87 94'.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=[0],
        help="List of attention heads to visualize. Default is [0].",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0],
        help="List of transformer layers to visualize. Default is [0].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on. Options are 'cpu' or 'cuda'. Default is 'cpu'.",
    )
    parser.add_argument(
        "--use_power_scale",
        action="store_true",
        help="Enable power scale normalization for the attention visualization.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma value for power scale normalization. Default is 0.5.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode for detailed output.",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    if args.verbose:
        print(f"Using device: {device}")

    # Load the model configuration
    GPT_CONFIG = load_config(args.config_file)

    # Initialize the model and load pre-trained weights
    model = GPTModel(GPT_CONFIG)
    model.load_state_dict(torch.load(args.model_file, map_location=device))

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    if args.verbose:
        print(f"Model and tokenizer loaded successfully.")
        print(f"Input text: {args.input_text}")

    # Extract the base name of the config file (without extension) for saving the image
    config_base_name = os.path.splitext(os.path.basename(args.config_file))[0]
    save_path = f"../img/{config_base_name}_attention.png"

    if args.verbose:
        print(f"Attention visualization will be saved to: {save_path}")

    # Initialize the AttentionVisualizer
    visualizer = AttentionVisualizer(model, tokenizer)
    visualizer.infer_and_visualize_attention(
        input_text=args.input_text,
        heads=args.heads,
        layers=args.layers,
        save_path=save_path,
        use_power_scale=args.use_power_scale,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()

# Example run:
# python visualize_attention.py --config_file config/gpt_config.json --model_file models/gpt_model.pth --input_text "19 94 19 27 34 46 58 65 67 82 87 94" --heads 0 1 2 --layers 0 1 --verbose