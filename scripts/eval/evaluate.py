#!/usr/bin/env python3
"""Command-line interface for model evaluation."""

import random
import argparse
from datetime import datetime
from core.eval.evaluator import ModelEvaluator


def main():
    """Evaluate trained model from command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained arithmetic LLM model"
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to tokenizer directory"
    )

    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="digit",
        choices=["digit", "bpe"],
        help="Tokenizer type to load (default: digit)"
    )

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        help="Path to base model checkpoint when evaluating a LoRA adapter"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of test expressions to generate (default: 1000)"
    )

    parser.add_argument(
        "--min-depth",
        type=int,
        default=1,
        help="Minimum depth of test expressions (default: 1)"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum depth of test expressions (default: 5)"
    )
    
    parser.add_argument(
        "--num-range",
        type=int,
        nargs=2,
        default=[1, 20],
        metavar=("MIN", "MAX"),
        help="Range of numbers in test expressions (default: 1 20)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results (default: evaluation_results)"
    )
    
    parser.add_argument(
        "--log-all-questions",
        action="store_true",
        help="Log detailed results for all questions"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: 'cuda', 'mps', 'cpu', or 'auto' (default: auto)"
    )
    
    parser.add_argument(
        "--max-gen-length",
        type=int,
        default=3072,
        help="Maximum generation length in tokens (default: 2048)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic evaluation generation"
    )

    parser.add_argument(
        "--max-sample-attempts",
        type=int,
        default=3,
        help="Adaptive sampling: if think not followed by Step 1, resample up to this many times (default: 1 = disabled)"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        import torch
        device = (
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
    else:
        device = args.device

    # Enforce deterministic random seeds for identically reproducible datasets
    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Display configuration
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"\nModel: {args.model_path}")
    print(f"Tokenizer ({args.tokenizer_type}): {args.tokenizer_path}")
    print(f"Device: {device}")
    print("\nEvaluation Configuration:")
    print(f"  Test samples: {args.num_samples}")
    print(f"  Min depth: {args.min_depth}")
    print(f"  Max depth: {args.max_depth}")
    print(f"  Number range: {args.num_range[0]} to {args.num_range[1]}")
    print(f"  Max generation length: {args.max_gen_length}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Log all questions: {args.log_all_questions}")
    print(f"  Max sample attempts (adaptive): {args.max_sample_attempts}")
    print("=" * 60 + "\n")
    
    # Create evaluator
    try:
        print("Loading model and tokenizer...")
        evaluator = ModelEvaluator(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            tokenizer_type=args.tokenizer_type,
            base_checkpoint_path=args.base_checkpoint,
            device=device
        )
        print("Model loaded successfully!\n")
        
        # Run evaluation
        print("Starting evaluation...")
        import os
        os.makedirs(args.output_dir, exist_ok=True)

        # Save evaluation arguments
        import json
        
        args_dict = vars(args)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args_path = os.path.join(args.output_dir, f'evaluation_args_{timestamp}.json')
        with open(args_path, 'w') as f:
            json.dump(args_dict, f, indent=2)

        metrics = evaluator.evaluate(
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            min_depth=args.min_depth,
            num_range=tuple(args.num_range),
            output_dir=args.output_dir,
            max_gen_length=args.max_gen_length,
            log_all_questions=args.log_all_questions,
            max_sample_attempts=args.max_sample_attempts,
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nTotal Samples: {metrics['total_samples']}")
        print(f"Correct Samples: {metrics['correct_samples']}")
        print(f"Parseable Samples: {metrics['parseable_samples']}")
        print(f"\nExact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%")
        print(f"Parse Success Rate: {metrics['parse_success_rate']:.2f}%")
        print(f"Expression Now Consistent: {metrics.get('expression_now_consistent_rate', 0):.2f}%")
        print(f"Avg Generation Length: {metrics['avg_generation_length']:.2f} tokens")
        print("=" * 60)
        
        # Provide interpretation
        print("\nInterpretation:")
        if metrics['exact_match_accuracy'] >= 80:
            print("  ✓ Excellent performance!")
        elif metrics['exact_match_accuracy'] >= 60:
            print("  ✓ Good performance")
        elif metrics['exact_match_accuracy'] >= 40:
            print("  ~ Moderate performance - consider more training")
        else:
            print("  ✗ Poor performance - model needs more training or debugging")
        
        if metrics['parse_success_rate'] < 90:
            print("  ! Low parse success rate - model may need better instruction tuning")
        
        print("=" * 60 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("EVALUATION FAILED!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()

