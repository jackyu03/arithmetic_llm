#!/usr/bin/env python3
"""Sweep LoRA configurations (e.g. rank r) and compare instruction performance with full instruction tuning.

Trains instruction LoRA for each config, evaluates on the same test set, and optionally
trains/evaluates a full instruction model as baseline. Reports accuracy and efficiency
(training time, trainable params, inference throughput). Outputs a comparison table and JSON.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import torch

from core.model.lora.config import LoRAConfig
from core.training.config import TrainingConfig
from core.training.instruction_lora import train_instruction_model_lora
from core.training.instruction import train_instruction_model
from core.eval.evaluator import ModelEvaluator


def _get_device(device_str: str) -> str:
    if device_str != "auto":
        return device_str
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def _reset_cuda_peak_memory() -> None:
    """Reset CUDA peak memory stats so next _get_cuda_peak_memory_mb() is per-run."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _get_cuda_peak_memory_mb() -> Tuple[Optional[float], Optional[float]]:
    """Return (peak_allocated_mb, peak_reserved_mb) since last reset. (None, None) if not CUDA."""
    if not torch.cuda.is_available():
        return None, None
    torch.cuda.synchronize()
    alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    return round(alloc, 2), round(reserved, 2)


def count_trainable_params_lora(model_config: Dict[str, Any], rank: int) -> int:
    """Approximate LoRA trainable parameters (attention only: Q,K,V,O per layer)."""
    d_model = model_config.get("d_model", 256)
    num_layers = model_config.get("num_layers", 6)
    # Per attention matrix: lora_A (rank, d_model) + lora_B (d_model, rank) = 2*rank*d_model
    # 4 matrices per layer (Q,K,V,O)
    return num_layers * 4 * 2 * rank * d_model


def count_total_params(model_config: Dict[str, Any]) -> int:
    """Approximate total transformer parameters (embedding + layers)."""
    vocab_size = model_config.get("vocab_size", 128)
    d_model = model_config.get("d_model", 256)
    num_layers = model_config.get("num_layers", 6)
    dim_feedforward = model_config.get("dim_feedforward", 1024)
    # Embedding + output projection
    emb = vocab_size * d_model + vocab_size * d_model  # token emb + output
    # Per layer: attention 4 * d_model^2, ffn 2 * d_model * dim_feedforward
    per_layer = 4 * (d_model * d_model) + 2 * (d_model * dim_feedforward)
    return emb + num_layers * per_layer


def run_lora_training(
    instruction_corpus_path: str,
    tokenizer_path: str,
    tokenizer_type: str,
    foundational_checkpoint: str,
    output_dir: str,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, Optional[float], Optional[float]]:
    """Train instruction LoRA. Return (adapter_path, training_time_sec, vram_alloc_mb, vram_reserved_mb)."""
    os.makedirs(output_dir, exist_ok=True)
    _reset_cuda_peak_memory()
    t0 = time.perf_counter()
    path = train_instruction_model_lora(
        instruction_corpus_path=instruction_corpus_path,
        tokenizer_path=tokenizer_path,
        tokenizer_type=tokenizer_type,
        foundational_checkpoint=foundational_checkpoint,
        output_dir=output_dir,
        config=training_config,
        lora_config=lora_config,
        model_config=model_config,
        save_merged_model=False,
    )
    elapsed = time.perf_counter() - t0
    vram_alloc, vram_reserved = _get_cuda_peak_memory_mb()
    return path, elapsed, vram_alloc, vram_reserved


def run_full_instruction_training(
    instruction_corpus_path: str,
    tokenizer_path: str,
    tokenizer_type: str,
    foundational_checkpoint: str,
    output_dir: str,
    training_config: TrainingConfig,
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, Optional[float], Optional[float]]:
    """Train full instruction model. Return (checkpoint_path, training_time_sec, vram_alloc_mb, vram_reserved_mb)."""
    os.makedirs(output_dir, exist_ok=True)
    _reset_cuda_peak_memory()
    t0 = time.perf_counter()
    path = train_instruction_model(
        instruction_corpus_path=instruction_corpus_path,
        tokenizer_path=tokenizer_path,
        foundational_checkpoint=foundational_checkpoint,
        output_dir=output_dir,
        config=training_config,
        model_config=model_config,
        tokenizer_type=tokenizer_type,
    )
    elapsed = time.perf_counter() - t0
    vram_alloc, vram_reserved = _get_cuda_peak_memory_mb()
    return path, elapsed, vram_alloc, vram_reserved


def evaluate_model(
    model_path: str,
    tokenizer_path: str,
    tokenizer_type: str,
    device: str,
    base_checkpoint_path: Optional[str] = None,
    num_samples: int = 500,
    min_depth: int = 1,
    max_depth: int = 5,
    num_range: tuple = (1, 20),
    max_gen_length: int = 3072,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float, Optional[float], Optional[float]]:
    """Evaluate a model. Returns (metrics, eval_wall_sec, vram_alloc_mb, vram_reserved_mb)."""
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
    _reset_cuda_peak_memory()
    t0 = time.perf_counter()
    metrics = evaluator.evaluate(
        num_samples=num_samples,
        min_depth=min_depth,
        max_depth=max_depth,
        num_range=num_range,
        output_dir=None,
        max_gen_length=max_gen_length,
        log_all_questions=False,
    )
    eval_sec = time.perf_counter() - t0
    vram_alloc, vram_reserved = _get_cuda_peak_memory_mb()
    return metrics, eval_sec, vram_alloc, vram_reserved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep LoRA configs (e.g. rank r) and compare with full instruction tuning"
    )
    # Paths
    parser.add_argument("--instruction-corpus-path", type=str, required=True,
                        help="Path to instruction corpus")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path to tokenizer directory")
    parser.add_argument("--foundational-checkpoint", type=str, required=True,
                        help="Path to foundational model checkpoint")
    parser.add_argument("--output-dir", type=str, default="lora_sweep_results",
                        help="Base directory for all runs and results (default: lora_sweep_results)")
    parser.add_argument("--tokenizer-type", type=str, default="digit", choices=["digit", "bpe"])

    # What to run
    parser.add_argument("--lora-ranks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64],
                        help="LoRA ranks to sweep (default: 2 4 8 16 32 64)")
    parser.add_argument("--run-full-instruction", action="store_true",
                        help="Also train and evaluate full instruction model as baseline")
    parser.add_argument("--instruction-model-path", type=str, default=None,
                        help="Use existing full instruction checkpoint as baseline (skip training)")

    # Training (shared; align with scripts/train/instruction_lora.py and instruction.py where possible)
    parser.add_argument("--num-epochs", type=int, default=3, help="Epochs per run (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (default: off for sweep)")

    # LoRA options (alpha scale)
    parser.add_argument("--lora-alpha-scale", type=float, default=2.0,
                        help="LoRA alpha = lora_alpha_scale * rank (default: 2.0)")

    # Evaluation
    parser.add_argument("--num-eval-samples", type=int, default=500,
                        help="Test samples per evaluation (default: 500)")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--max-gen-length", type=int, default=3072)

    # Model config (optional)
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model config JSON")

    parser.add_argument("--num-workers", type=int, default=0,
                        help="Dataloader workers (default: 0 for sweep reliability; use 8 to match foundational and speed up if stable)")

    args = parser.parse_args()

    device = _get_device(args.device)
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    # Training config shared by all runs (consistent with standalone instruction_lora.py / instruction.py)
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_clip=1.0,
        save_every=1000,
        device=device,
        use_wandb=args.wandb,
        use_contrastive=False,
        contrastive_weight=0.0,
        contrastive_temperature=0.0,
        use_curriculum=False,
        curriculum_steps=10000,
        num_workers=args.num_workers,
    )

    model_config = None
    if args.model_config:
        with open(args.model_config, "r") as f:
            model_config = json.load(f)
    # Default model config for param counting when not provided
    _mc = model_config or {
        "vocab_size": 128,
        "d_model": 256,
        "num_layers": 6,
        "dim_feedforward": 1024,
    }

    num_range = (1, 20)
    results: List[Dict[str, Any]] = []

    # --- Full instruction baseline ---
    if args.run_full_instruction:
        print("\n" + "=" * 60)
        print("Training FULL INSTRUCTION model (baseline)")
        print("=" * 60)
        full_dir = os.path.join(base_dir, "full_instruction")
        try:
            full_checkpoint, train_sec, vram_train_alloc, vram_train_reserved = run_full_instruction_training(
                instruction_corpus_path=args.instruction_corpus_path,
                tokenizer_path=args.tokenizer_path,
                tokenizer_type=args.tokenizer_type,
                foundational_checkpoint=args.foundational_checkpoint,
                output_dir=full_dir,
                training_config=training_config,
                model_config=model_config,
            )
            # best_model.pt is the usual name; instruction returns final path
            full_model_path = os.path.join(full_dir, "best_model.pt")
            if not os.path.exists(full_model_path):
                full_model_path = full_checkpoint
            metrics, eval_sec, vram_eval_alloc, vram_eval_reserved = evaluate_model(
                model_path=full_model_path,
                tokenizer_path=args.tokenizer_path,
                tokenizer_type=args.tokenizer_type,
                device=device,
                base_checkpoint_path=None,
                num_samples=args.num_eval_samples,
                min_depth=1,
                max_depth=5,
                num_range=num_range,
                max_gen_length=args.max_gen_length,
                seed=args.eval_seed,
            )
            trainable_params = count_total_params(_mc)
            total_tokens = metrics.get("total_generated_tokens", 0) or int(metrics["avg_generation_length"] * metrics["total_samples"])
            results.append({
                "name": "full_instruction",
                "lora_rank": None,
                "lora_alpha": None,
                "model_path": full_model_path,
                "metrics": metrics,
                "efficiency": {
                    "training_time_sec": train_sec,
                    "eval_time_sec": eval_sec,
                    "trainable_params": trainable_params,
                    "samples_per_sec": metrics["total_samples"] / eval_sec if eval_sec > 0 else 0,
                    "tokens_per_sec": total_tokens / eval_sec if eval_sec > 0 else 0,
                    "vram_train_allocated_mb": vram_train_alloc,
                    "vram_train_reserved_mb": vram_train_reserved,
                    "vram_eval_allocated_mb": vram_eval_alloc,
                    "vram_eval_reserved_mb": vram_eval_reserved,
                },
            })
            vram_str = f"  train VRAM: {vram_train_alloc}/{vram_train_reserved} MB  eval VRAM: {vram_eval_alloc}/{vram_eval_reserved} MB" if vram_train_alloc is not None else ""
            print(f"  Full instruction accuracy: {metrics['exact_match_accuracy']:.2f}%  train: {train_sec:.0f}s  eval: {eval_sec:.1f}s{vram_str}")
        except Exception as e:
            print(f"  Full instruction failed: {e}")
            results.append({
                "name": "full_instruction",
                "error": str(e),
                "metrics": None,
            })

    if args.instruction_model_path:
        print("\n" + "=" * 60)
        print("Evaluating existing FULL INSTRUCTION model (baseline)")
        print("=" * 60)
        try:
            metrics, eval_sec, vram_eval_alloc, vram_eval_reserved = evaluate_model(
                model_path=args.instruction_model_path,
                tokenizer_path=args.tokenizer_path,
                tokenizer_type=args.tokenizer_type,
                device=device,
                base_checkpoint_path=None,
                num_samples=args.num_eval_samples,
                min_depth=1,
                max_depth=5,
                num_range=num_range,
                max_gen_length=args.max_gen_length,
                seed=args.eval_seed,
            )
            trainable_params = count_total_params(_mc)
            total_tokens = metrics.get("total_generated_tokens", 0) or int(metrics["avg_generation_length"] * metrics["total_samples"])
            results.append({
                "name": "full_instruction (pre-trained)",
                "lora_rank": None,
                "lora_alpha": None,
                "model_path": args.instruction_model_path,
                "metrics": metrics,
                "efficiency": {
                    "training_time_sec": None,
                    "eval_time_sec": eval_sec,
                    "trainable_params": trainable_params,
                    "samples_per_sec": metrics["total_samples"] / eval_sec if eval_sec > 0 else 0,
                    "tokens_per_sec": total_tokens / eval_sec if eval_sec > 0 else 0,
                    "vram_train_allocated_mb": None,
                    "vram_train_reserved_mb": None,
                    "vram_eval_allocated_mb": vram_eval_alloc,
                    "vram_eval_reserved_mb": vram_eval_reserved,
                },
            })
            vram_str = f"  eval VRAM: {vram_eval_alloc} / {vram_eval_reserved} MB" if vram_eval_alloc is not None else ""
            print(f"  Full instruction accuracy: {metrics['exact_match_accuracy']:.2f}%  eval: {eval_sec:.1f}s{vram_str}")
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            results.append({
                "name": "full_instruction (pre-trained)",
                "error": str(e),
                "metrics": None,
            })

    # --- LoRA sweep ---
    for r in args.lora_ranks:
        alpha = args.lora_alpha_scale * r
        lora_config = LoRAConfig(
            rank=r,
            alpha=float(alpha),
            target_modules=["attention"],
            dropout=0.0,
        )
        lora_config.validate()

        run_name = f"lora_r{r}"
        run_dir = os.path.join(base_dir, run_name)
        print("\n" + "=" * 60)
        print(f"Training LoRA rank={r} alpha={alpha}")
        print("=" * 60)
        try:
            adapter_path, train_sec, vram_train_alloc, vram_train_reserved = run_lora_training(
                instruction_corpus_path=args.instruction_corpus_path,
                tokenizer_path=args.tokenizer_path,
                tokenizer_type=args.tokenizer_type,
                foundational_checkpoint=args.foundational_checkpoint,
                output_dir=run_dir,
                lora_config=lora_config,
                training_config=training_config,
                model_config=model_config,
            )
            metrics, eval_sec, vram_eval_alloc, vram_eval_reserved = evaluate_model(
                model_path=adapter_path,
                tokenizer_path=args.tokenizer_path,
                tokenizer_type=args.tokenizer_type,
                device=device,
                base_checkpoint_path=args.foundational_checkpoint,
                num_samples=args.num_eval_samples,
                min_depth=1,
                max_depth=5,
                num_range=num_range,
                max_gen_length=args.max_gen_length,
                seed=args.eval_seed,
            )
            trainable_params = count_trainable_params_lora(_mc, r)
            total_tokens = metrics.get("total_generated_tokens", 0) or int(metrics["avg_generation_length"] * metrics["total_samples"])
            results.append({
                "name": run_name,
                "lora_rank": r,
                "lora_alpha": alpha,
                "model_path": adapter_path,
                "metrics": metrics,
                "efficiency": {
                    "training_time_sec": train_sec,
                    "eval_time_sec": eval_sec,
                    "trainable_params": trainable_params,
                    "samples_per_sec": metrics["total_samples"] / eval_sec if eval_sec > 0 else 0,
                    "tokens_per_sec": total_tokens / eval_sec if eval_sec > 0 else 0,
                    "vram_train_allocated_mb": vram_train_alloc,
                    "vram_train_reserved_mb": vram_train_reserved,
                    "vram_eval_allocated_mb": vram_eval_alloc,
                    "vram_eval_reserved_mb": vram_eval_reserved,
                },
            })
            vram_str = f"  train VRAM: {vram_train_alloc}/{vram_train_reserved} MB" if vram_train_alloc is not None else ""
            print(f"  LoRA r={r} accuracy: {metrics['exact_match_accuracy']:.2f}%  train: {train_sec:.0f}s  eval: {eval_sec:.1f}s  params: {trainable_params:,}{vram_str}")
        except Exception as e:
            print(f"  LoRA r={r} failed: {e}")
            results.append({
                "name": run_name,
                "lora_rank": r,
                "lora_alpha": alpha,
                "error": str(e),
                "metrics": None,
            })

    # --- Summary table and save ---
    print("\n" + "=" * 60)
    print("COMPARISON: Instruction performance (accuracy)")
    print("=" * 60)

    row_len = 72
    headers = ["Config", "Exact %", "Parse %", "Expr-now %", "N"]
    print(f"\n{headers[0]:<26} {headers[1]:>8} {headers[2]:>8} {headers[3]:>10} {headers[4]:>6}")
    print("-" * row_len)

    for rec in results:
        name = rec["name"]
        if rec.get("metrics") is None:
            err = rec.get("error", "?")
            print(f"{name:<26} {'FAIL':>8} {err[:24]}")
            continue
        m = rec["metrics"]
        exact = m["exact_match_accuracy"]
        parse = m["parse_success_rate"]
        expr = m.get("expression_now_consistent_rate", 0.0)
        n = m["total_samples"]
        print(f"{name:<26} {exact:>8.2f} {parse:>8.2f} {expr:>10.2f} {n:>6}")

    print("-" * row_len)

    # Efficiency table
    print("\n" + "=" * 60)
    print("COMPARISON: Efficiency")
    print("=" * 60)
    eff_headers = ["Config", "Train(s)", "Eval(s)", "Trainable", "Samp/s", "Tok/s", "TrAlloc", "TrRes", "EvAlloc", "EvRes"]
    print(f"\n{eff_headers[0]:<26} {eff_headers[1]:>8} {eff_headers[2]:>8} {eff_headers[3]:>12} {eff_headers[4]:>8} {eff_headers[5]:>10} {eff_headers[6]:>8} {eff_headers[7]:>8} {eff_headers[8]:>8} {eff_headers[9]:>8}")
    print("-" * 100)

    for rec in results:
        name = rec["name"]
        eff = rec.get("efficiency")
        if eff is None:
            print(f"{name:<26} {'—':>8} {'—':>8} {'—':>12} {'—':>8} {'—':>10} {'—':>8} {'—':>8} {'—':>8} {'—':>8}")
            continue
        train_s = eff.get("training_time_sec")
        eval_s = eff.get("eval_time_sec") or 0
        params = eff.get("trainable_params") or 0
        samp_s = eff.get("samples_per_sec") or 0
        tok_s = eff.get("tokens_per_sec") or 0
        tr_alloc = eff.get("vram_train_allocated_mb")
        tr_res = eff.get("vram_train_reserved_mb")
        ev_alloc = eff.get("vram_eval_allocated_mb")
        ev_res = eff.get("vram_eval_reserved_mb")
        train_str = f"{train_s:.0f}" if train_s is not None else "—"
        tr_alloc_s = f"{tr_alloc:.0f}" if tr_alloc is not None else "—"
        tr_res_s = f"{tr_res:.0f}" if tr_res is not None else "—"
        ev_alloc_s = f"{ev_alloc:.0f}" if ev_alloc is not None else "—"
        ev_res_s = f"{ev_res:.0f}" if ev_res is not None else "—"
        print(f"{name:<26} {train_str:>8} {eval_s:>8.1f} {params:>12,} {samp_s:>8.2f} {tok_s:>10.1f} {tr_alloc_s:>8} {tr_res_s:>8} {ev_alloc_s:>8} {ev_res_s:>8}")

    print("-" * 100)
    print("  (VRAM columns in MB: TrAlloc/TrRes = peak allocated/reserved during training; EvAlloc/EvRes = during evaluation; — = N/A e.g. CPU)")

    # Save JSON (include efficiency in each result)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": [
            {
                "name": r["name"],
                "lora_rank": r.get("lora_rank"),
                "lora_alpha": r.get("lora_alpha"),
                "model_path": r.get("model_path"),
                "error": r.get("error"),
                "metrics": r.get("metrics"),
                "efficiency": r.get("efficiency"),
            }
            for r in results
        ],
    }
    out_path = os.path.join(base_dir, "lora_sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
