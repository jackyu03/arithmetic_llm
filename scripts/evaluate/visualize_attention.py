#!/usr/bin/env python3
"""Script to visualize attention weights of the arithmetic transformer."""

import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

from core.model.transformer import ArithmeticTransformer
from core.data.tokenizer import ArithmeticBPETokenizer


def load_model(checkpoint_path: str, model_config_path: str, device: str) -> tuple[ArithmeticTransformer, ArithmeticBPETokenizer]:
    """Load the model and tokenizer from checkpoint."""
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    # Checkpoint contains tokenizer info, but we need the actual tokenizer object
    # For simplicity, we assume the tokenizer is in the same directory or standard location
    tokenizer_dir = os.path.dirname(model_config_path)
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_dir)

    model = ArithmeticTransformer(**model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def generate_and_get_attentions(model: ArithmeticTransformer, tokenizer: ArithmeticBPETokenizer, prompt: str, device: str):
    """Generate a response and capture the attention weights."""
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated_ids, attentions = model.generate(
            input_ids,
            max_length=256,
            temperature=0.1,  # Low temperature for greedy-like deterministic generation
            eos_token_id=tokenizer.token2id.get('<eos>', None),
            output_attentions=True
        )
        
    generated_text = tokenizer.decode(generated_ids[0].cpu().numpy().tolist())
    
    return generated_ids, attentions, generated_text


def plot_attention_heatmaps(
    tokens: List[str], 
    attentions: torch.Tensor, 
    layer_idx: int, 
    head_idx: Optional[int] = None,
    save_path: Optional[str] = None
):
    """Plot attention heatmap for a specific layer and optionally a specific head.
    
    Args:
        tokens: List of string tokens
        attentions: Tensor of shape (num_layers, batch_size, num_heads, seq_len, seq_len)
                   Since we generated one by one, the shape from generation might be different.
                   Let's assume we pass in the full attention matrix for the final generated sequence.
        layer_idx: Which layer's attention to plot
        head_idx: Which head to plot. If None, average across all heads.
        save_path: Where to save the figure
    """
    # Assuming attentions is a tuple of tuples from Generation:
    # generation length steps -> layers -> (batch, heads, seq_len, seq_len)
    
    # Let's just do a single forward pass with the entire sequence to get the full NxN attention matrix easily
    pass


def main():
    parser = argparse.ArgumentParser(description="Visualize Transformer Attention Weights")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model_config.json")
    parser.add_argument("--prompt", type=str, default="2 + 3 = <think>", help="Arithmetic problem prompt")
    parser.add_argument("--output-dir", type=str, default="attention_maps", help="Directory to save heatmaps")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to visualize (default: last layer)")
    parser.add_argument("--head", type=int, default=None, help="Head index to visualize (default: average across heads)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model(args.checkpoint, args.model_config, device)
    
    print(f"Prompt: {args.prompt}")
    print("Generating response and capturing attentions...")
    
    # 1. Generate the full sequence
    generated_ids, _, generated_text = generate_and_get_attentions(model, tokenizer, args.prompt, device)
    print(f"Generated text: {generated_text}")
    
    # 2. To get a clean NxN matrix for all tokens, do one forward pass with the full sequence
    print("Performing full forward pass to extract complete NxN attention matrices...")
    with torch.no_grad():
        _, attentions = model(generated_ids, output_attentions=True)
        
    # attentions is a tuple of length num_layers
    # Each element is (batch_size, num_heads, seq_len, seq_len)
    
    tokens = [tokenizer.id2token.get(idx, f"<unk:{idx}>") for idx in generated_ids[0].cpu().numpy().tolist()]
    
    layer_idx = args.layer if args.layer >= 0 else len(attentions) + args.layer
    layer_attention = attentions[layer_idx][0]  # Take batch index 0. Shape: (num_heads, seq_len, seq_len)
    
    if args.head is not None:
        attn_matrix = layer_attention[args.head].cpu().numpy()
        title = f"Attention Weights: Layer {layer_idx}, Head {args.head}"
        filename = f"attention_l{layer_idx}_h{args.head}.png"
    else:
        attn_matrix = layer_attention.mean(dim=0).cpu().numpy()
        title = f"Attention Weights: Layer {layer_idx}, Averaged Heads"
        filename = f"attention_l{layer_idx}_avg.png"
        
    # Set up matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Create seaborn heatmap
    sns.heatmap(
        attn_matrix, 
        xticklabels=tokens, 
        yticklabels=tokens,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": .8}
    )
    
    plt.title(title, pad=20)
    plt.xlabel("Key Tokens (Attended To)")
    plt.ylabel("Query Tokens (Attending From)")
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    save_path = os.path.join(args.output_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved attention heatmap to: {save_path}")
    
    # Also save a text version of the tokens for reference
    with open(os.path.join(args.output_dir, "tokens.txt"), 'w') as f:
        for i, token in enumerate(tokens):
            f.write(f"{i}: {token}\n")

if __name__ == "__main__":
    main()