#!/usr/bin/env python3
"""Mind Reader Attention Visualizer.

A command-line visualization of where the arithmetic model's attention is focused
during each generation step. It uses ANSI color codes to highlight tokens in the
context based on their attention weights.
"""

import argparse
import time
import sys
import os
import torch
import torch.nn.functional as F

from core.inference.interactive import InteractiveArithmeticSolver

def get_color_ansi(weight: float) -> str:
    """Map a weight [0, 1] to a thermal ANSI background color (black -> red -> yellow -> white)."""
    # Exaggerate small weights for visibility
    w = weight ** 0.5
    
    # Simple color ramp: 
    # 0.0 -> Black (0, 0, 0)
    # 0.33 -> Red (255, 0, 0)
    # 0.66 -> Yellow (255, 255, 0)
    # 1.0 -> White (255, 255, 255)
    
    if w < 0.33:
        r = int((w / 0.33) * 255)
        g, b = 0, 0
    elif w < 0.66:
        r = 255
        g = int(((w - 0.33) / 0.33) * 255)
        b = 0
    else:
        r = 255
        g = 255
        b = int(((w - 0.66) / 0.34) * 255)
        
    return f"\033[48;2;{r};{g};{b}m\033[38;2;255;255;255m"

RESET_ANSI = "\033[0m"

class MindReader(InteractiveArithmeticSolver):
    
    def render_attention(self, tokens: list[str], attentions: torch.Tensor, current_token: str):
        """Render the tokens colored by their attention weight."""
        # attentions shape: (seq_len,)
        sys.stdout.write("\033[H\033[2J") # Clear screen
        print("=" * 60)
        print("""
        ▗▖  ▗▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄  ▗▄▄▖ ▗▄▄▄▖ ▗▄▖ ▗▄▄▄  ▗▄▄▄▖▗▄▄▖ 
        ▐▛▚▞▜▌  █  ▐▛▚▖▐▌▐▌  █ ▐▌ ▐▌▐▌   ▐▌ ▐▌▐▌  █ ▐▌   ▐▌ ▐▌
        ▐▌  ▐▌  █  ▐▌ ▝▜▌▐▌  █ ▐▛▀▚▖▐▛▀▀▘▐▛▀▜▌▐▌  █ ▐▛▀▀▘▐▛▀▚▖
        ▐▌  ▐▌▗▄█▄▖▐▌  ▐▌▐▙▄▄▀ ▐▌ ▐▌▐▙▄▄▖▐▌ ▐▌▐▙▄▄▀ ▐▙▄▄▖▐▌ ▐▌                                       
         ▗▄▖ ▗▄▄▖ ▗▄▄▄▖▗▄▄▄▖▗▖ ▗▖▗▖  ▗▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖ ▗▄▄▖   
        ▐▌ ▐▌▐▌ ▐▌  █    █  ▐▌ ▐▌▐▛▚▞▜▌▐▌     █    █  ▐▌      
        ▐▛▀▜▌▐▛▀▚▖  █    █  ▐▛▀▜▌▐▌  ▐▌▐▛▀▀▘  █    █  ▐▌      
        ▐▌ ▐▌▐▌ ▐▌▗▄█▄▖  █  ▐▌ ▐▌▐▌  ▐▌▐▙▄▄▖  █  ▗▄█▄▖▝▚▄▄▖                                                                                             
        ▗▖  ▗▖▗▄▄▄▖ ▗▄▄▖▗▖ ▗▖ ▗▄▖ ▗▖   ▗▄▄▄▖▗▄▄▄▄▖▗▄▄▄▖▗▄▄▖   
        ▐▌  ▐▌  █  ▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌     █     ▗▞▘▐▌   ▐▌ ▐▌  
        ▐▌  ▐▌  █   ▝▀▚▖▐▌ ▐▌▐▛▀▜▌▐▌     █   ▗▞▘  ▐▛▀▀▘▐▛▀▚▖  
         ▝▚▞▘ ▗▄█▄▖▗▄▄▞▘▝▚▄▞▘▐▌ ▐▌▐▙▄▄▖▗▄█▄▖▐▙▄▄▄▖▐▙▄▄▖▐▌ ▐▌ 
        
        """)
        print("=" * 60)
        print("\nWatching model attention in real-time...\n")
        
        output = ""
        for token, weight in zip(tokens, attentions.tolist()):
            color = get_color_ansi(weight)
            # Remove special markers for raw printing
            clean_tok = token.replace("</w>", "")
            output += f"{color}{clean_tok}{RESET_ANSI}"
            
        print("Context:")
        print(output)
        print(f"\nCurrently generating: -> {current_token} <-")
        print("\n" + "=" * 60)
        sys.stdout.flush()

    def solve_with_visualization(self, expression: str, delay: float = 0.1):
        prompt = f"Evaluate: {expression} <think>"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        eos_token_id = self.tokenizer.token2id.get('<eos>', None)
        if eos_token_id is not None and input_ids and input_ids[-1] == eos_token_id:
            input_ids = input_ids[:-1]
            
        generated = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        self.model.eval()
        
        is_digit = hasattr(self.tokenizer, 'scaffolding_tokens')
        temp = 0.1 if is_digit else 0.8
        
        with torch.no_grad():
            while generated.shape[1] < 2048:
                # Forward pass
                logits, attentions = self.model(generated, output_attentions=True)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temp
                
                # Simple greedy/top-1 decoding for stable visualization
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
                
                # Extract attention from the VERY LAST layer (attentions[-1])
                # shape: (batch_size, num_heads, seq_len, seq_len)
                last_layer_attn = attentions[-1][0] # (num_heads, seq_len, seq_len)
                
                # We care about the attention from the *current* token (the one we just generated from, index -2)
                # to all previous tokens (indices 0 to -2). 
                # Note: the input length was N. We generated the N+1th token. 
                # The attention matrix is size NxN. 
                # We want the last row of the attention matrix.
                current_step_attn = last_layer_attn[:, -1, :] # (num_heads, seq_len)
                avg_attn = current_step_attn.mean(dim=0) # (seq_len)
                
                # Decode all tokens up to the one we just processed
                current_seq_tokens = [self.tokenizer.id2token.get(idx, f"<unk:{idx}>") for idx in generated[0, :-1].tolist()]
                new_token_str = self.tokenizer.id2token.get(next_token.item(), "")
                
                self.render_attention(current_seq_tokens, avg_attn, new_token_str)
                time.sleep(delay)
                
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
        
        final_text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        print("\nFinal Output:")
        print(self.format_output(final_text))

    def run(self) -> None:
        print("\n" + "=" * 60)
        print("MIND READER ARITHMETIC VISUALIZER ")
        print("=" * 60)
        print("\nEnter an expression to watch the model's attention as it solves.")
        print("Commands: 'exit' or 'quit' to exit.")
        print("=" * 60 + "\n")
        
        while True:
            try:
                expression = input("Enter expression: ").strip()
                if expression.lower() in ['exit', 'quit', 'q']:
                    break
                if not expression:
                    continue
                
                try:
                    self.solve_with_visualization(expression)
                    input("\nPress Enter to continue...")
                except Exception as e:
                    print(f"\nError: {str(e)}")
            except (KeyboardInterrupt, EOFError):
                break

def main():
    parser = argparse.ArgumentParser(description='Mind Reader Attention Visualizer')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer-path', type=str, required=True, help='Path to tokenizer directory')
    parser.add_argument('--device', type=str, default='auto', help='Device for inference (cuda, mps, cpu, or auto)')
    parser.add_argument('--tokenizer-type', type=str, default='digit', choices=['digit', 'bpe'])
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
        
    reader = MindReader(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        tokenizer_type=args.tokenizer_type,
        device=device
    )
    reader.run()

if __name__ == '__main__':
    main()
