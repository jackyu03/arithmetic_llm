#!/usr/bin/env python3
"""Compare contrastive vs instruction-only by expression depth (1-5).

Evaluates two checkpoints (instruction baseline and instruction+contrastive) at each
depth 1..5 with 200 samples per depth. Records exact match rate, expression-now
consistent rate, and each-step-correct rate. No temperature/weight sweep.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.eval.evaluator import ModelEvaluator


def _get_device(device_str: str) -> str:
    if device_str != "auto":
        return device_str
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def evaluate_model_at_depth(
    model_path: str,
    tokenizer_path: str,
    tokenizer_type: str,
    device: str,
    depth: int,
    num_samples: int = 200,
    base_checkpoint_path: Optional[str] = None,
    num_range: Tuple[int, int] = (1, 20),
    max_gen_length: int = 3072,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float]:
    """Evaluate a model at a single depth (min_depth=max_depth=depth). Returns (metrics, eval_sec)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    evaluator = ModelEvaluator(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        tokenizer_type=tokenizer_type,
        base_checkpoint_path=base_checkpoint_path,
        device=device,
    )
    t0 = time.perf_counter()
    metrics = evaluator.evaluate(
        num_samples=num_samples,
        min_depth=depth,
        max_depth=depth,
        num_range=num_range,
        output_dir=None,
        max_gen_length=max_gen_length,
        log_all_questions=False,
    )
    eval_sec = time.perf_counter() - t0
    return metrics, eval_sec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare instruction vs contrastive by depth (1-5), 200 samples per depth"
    )
    parser.add_argument("--instruction-baseline-path", type=str, required=True,
                        help="Path to instruction-only checkpoint (e.g. best_model.pt)")
    parser.add_argument("--contrastive-path", type=str, required=True,
                        help="Path to instruction+contrastive checkpoint")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path to tokenizer directory")
    parser.add_argument("--tokenizer-type", type=str, default="digit", choices=["digit", "bpe"])
    parser.add_argument("--base-checkpoint", type=str, default=None,
                        help="Base model checkpoint (only if evaluating LoRA adapters)")
    parser.add_argument("--output-dir", type=str, default="contrastive_depth_sweep_results",
                        help="Directory for results (default: contrastive_depth_sweep_results)")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Samples per depth (default: 200)")
    parser.add_argument("--num-range", type=int, nargs=2, default=[1, 20], metavar=("MIN", "MAX"))
    parser.add_argument("--max-gen-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    device = _get_device(args.device)
    num_range = tuple(args.num_range)
    os.makedirs(args.output_dir, exist_ok=True)

    depths = [1, 2, 3, 4, 5]
    results: List[Dict[str, Any]] = []

    for model_name, model_path in [
        ("instruction_baseline", args.instruction_baseline_path),
        ("contrastive", args.contrastive_path),
    ]:
        if not os.path.exists(model_path):
            print(f"Skip {model_name}: path not found: {model_path}")
            continue
        print(f"\nEvaluating {model_name}: {model_path}")
        for depth in depths:
            print(f"  Depth {depth} ({args.num_samples} samples)...", end=" ", flush=True)
            try:
                metrics, eval_sec = evaluate_model_at_depth(
                    model_path=model_path,
                    tokenizer_path=args.tokenizer_path,
                    tokenizer_type=args.tokenizer_type,
                    device=device,
                    depth=depth,
                    num_samples=args.num_samples,
                    base_checkpoint_path=args.base_checkpoint,
                    num_range=num_range,
                    max_gen_length=args.max_gen_length,
                    seed=args.seed,
                )
                rec = {
                    "model": model_name,
                    "depth": depth,
                    "exact_match_accuracy": metrics["exact_match_accuracy"],
                    "expression_now_consistent_rate": metrics.get("expression_now_consistent_rate", 0.0),
                    "steps_all_correct_rate": metrics.get("steps_all_correct_rate", 0.0),
                    "eval_time_sec": round(eval_sec, 2),
                    "num_samples": metrics["total_samples"],
                }
                results.append(rec)
                print(f"exact={rec['exact_match_accuracy']:.1f}%  expr_now={rec['expression_now_consistent_rate']:.1f}%  steps_ok={rec['steps_all_correct_rate']:.1f}%  ({eval_sec:.1f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "model": model_name,
                    "depth": depth,
                    "error": str(e),
                    "exact_match_accuracy": None,
                    "expression_now_consistent_rate": None,
                    "steps_all_correct_rate": None,
                })

    # Summary table
    print("\n" + "=" * 100)
    print("CONTRASTIVE VS INSTRUCTION BY DEPTH (exact match %  |  expr-now consistent %  |  steps all correct %)")
    print("=" * 100)
    print(f"{'Depth':<6} | {'Instruction (exact)':<18} {'Instruction (expr_now)':<22} {'Instruction (steps_ok)':<22} | {'Contrastive (exact)':<18} {'Contrastive (expr_now)':<22} {'Contrastive (steps_ok)':<22}")
    print("-" * 100)

    base_rows = {r["depth"]: r for r in results if r.get("model") == "instruction_baseline"}
    cont_rows = {r["depth"]: r for r in results if r.get("model") == "contrastive"}
    for d in depths:
        b = base_rows.get(d, {})
        c = cont_rows.get(d, {})
        b_exact = f"{b.get('exact_match_accuracy') or 0:.1f}%" if b.get("exact_match_accuracy") is not None else "N/A"
        b_expr = f"{b.get('expression_now_consistent_rate') or 0:.1f}%" if b.get("expression_now_consistent_rate") is not None else "N/A"
        b_steps = f"{b.get('steps_all_correct_rate') or 0:.1f}%" if b.get("steps_all_correct_rate") is not None else "N/A"
        c_exact = f"{c.get('exact_match_accuracy') or 0:.1f}%" if c.get("exact_match_accuracy") is not None else "N/A"
        c_expr = f"{c.get('expression_now_consistent_rate') or 0:.1f}%" if c.get("expression_now_consistent_rate") is not None else "N/A"
        c_steps = f"{c.get('steps_all_correct_rate') or 0:.1f}%" if c.get("steps_all_correct_rate") is not None else "N/A"
        print(f"{d:<6} | {b_exact:<18} {b_expr:<22} {b_steps:<22} | {c_exact:<18} {c_expr:<22} {c_steps:<22}")
    print("=" * 100)

    out = {
        "timestamp": datetime.now().isoformat(),
        "instruction_baseline_path": args.instruction_baseline_path,
        "contrastive_path": args.contrastive_path,
        "tokenizer_path": args.tokenizer_path,
        "num_samples_per_depth": args.num_samples,
        "depths": depths,
        "results": results,
    }
    out_path = os.path.join(args.output_dir, "contrastive_depth_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
