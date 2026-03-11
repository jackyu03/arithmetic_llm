#!/usr/bin/env python3
"""Command-line interface for tokenizer training."""

import argparse
import os
from core.data.tokenizer import ArithmeticBPETokenizer, ArithmeticDigitTokenizer


def main():
    """Train BPE tokenizer from command line."""
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer on arithmetic corpus"
    )
    
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path to training corpus file"
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Target vocabulary size (default: 1000)"
    )

    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="digit",
        choices=["digit", "bpe"],
        help="Tokenizer type to train: 'digit' or 'bpe' (default: digit)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tokenizer",
        help="Directory to save trained tokenizer (default: data/tokenizer)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.corpus_path):
        parser.error(f"Corpus file not found: {args.corpus_path}")
    
    if args.vocab_size <= 0:
        parser.error("vocab-size must be positive")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train tokenizer
    if args.tokenizer_type == "bpe":
        print(f"Training BPE tokenizer with vocabulary size {args.vocab_size}...")
        print(f"Corpus: {args.corpus_path}")
        tokenizer = ArithmeticBPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(args.corpus_path)
    elif args.tokenizer_type == "digit":
        print(f"Training digit tokenizer")
        tokenizer = ArithmeticDigitTokenizer()
        tokenizer.train(args.corpus_path)
    else:
        parser.error(f"Invalid tokenizer type: {args.tokenizer_type}")
    
    
    print(f"Saving tokenizer to: {args.output_dir}")
    tokenizer.save(args.output_dir)
    
    
    # Test tokenizer with a sample expression
    for test_expr in ["5 + 10 - 3", "12 - (4 + 2)", "((7+3)-(3+5))"]:
        encoded = tokenizer.encode(test_expr, add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        print("\nTest encoding:")
        print(f"  Input: {test_expr}")
        print(f"  Encoded (with BOS/EOS): {encoded}")
        print(f"  Decoded: {decoded}")
        
        # Test without special tokens
        encoded_no_special = tokenizer.encode(test_expr, add_special_tokens=False)
        print(f"  Encoded (without BOS/EOS): {encoded_no_special}")
    
    print("\nTokenizer training complete!")


if __name__ == "__main__":
    main()


