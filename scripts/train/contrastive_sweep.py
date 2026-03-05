#!/usr/bin/env python3
"""Compare full instruction learning vs full instruction + contrastive learning.

Runs:
  1. Baseline: full instruction fine-tuning (no contrastive).
  2. Full instruction + contrastive with a grid of contrastive_weight and
     contrastive_temperature.

Optionally evaluates each model on the same test set. Outputs a comparison
table and JSON (training metrics, evaluation metrics, timing).
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.training.config import TrainingConfig
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _get_cuda_peak_memory_mb() -> Tuple[Optional[float], Optional[float]]:
    if not torch.cuda.is_available():
        return None, None
    torch.cuda.synchronize()
    alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    return round(alloc, 2), round(reserved, 2)


def _set_training_seed(seed: int) -> None:
    """Set random and torch seeds for reproducible training."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_full_instruction_training(
    instruction_corpus_path: str,
    tokenizer_path: str,
    tokenizer_type: str,
    foundational_checkpoint: str,
    output_dir: str,
    training_config: TrainingConfig,
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, Optional[float], Optional[float]]:
    """Train full instruction model. Returns (checkpoint_path, training_time_sec, vram_alloc_mb, vram_reserved_mb)."""
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


def load_training_summary(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load training_summary.json from a run if it exists."""
    path = os.path.join(output_dir, "training_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


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
    import random
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
        description="Compare full instruction vs full instruction + contrastive (weight/temperature sweep)"
    )
    # Paths
    parser.add_argument("--instruction-corpus-path", type=str, required=True,
                        help="Path to instruction corpus")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path to tokenizer directory")
    parser.add_argument("--foundational-checkpoint", type=str, required=True,
                        help="Path to foundational model checkpoint")
    parser.add_argument("--output-dir", type=str, default="contrastive_sweep_results",
                        help="Base directory for all runs and results (default: contrastive_sweep_results)")
    parser.add_argument("--tokenizer-type", type=str, default="digit", choices=["digit", "bpe"])

    # Training (shared)
    parser.add_argument("--num-epochs", type=int, default=3, help="Epochs per run (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Dataloader workers (default: 0 for sweep reliability)")

    # Contrastive sweep grid
    parser.add_argument("--contrastive-weights", type=float, nargs="+", default=[0.01, 0.03, 0.05],
                        help="Contrastive loss weights to sweep (default: 0.01, 0.03, 0.05)")
    parser.add_argument("--contrastive-temperatures", type=float, nargs="+", default=[0.05, 0.1],
                        help="Contrastive temperatures to sweep (default: 0.05, 0.1; higher = gentler)")
    parser.add_argument("--contrastive-warmup-steps", type=int, default=0,
                        help="CE-only steps before adding contrastive loss (0 = no warmup, default: 0)")
    parser.add_argument("--contrastive-warmup-epochs", type=float, default=0,
                        help="If > 0, CE-only epochs before contrastive (overrides warmup-steps; e.g. 3 = first 3 epochs like baseline)")
    parser.add_argument("--contrastive-num-epochs", type=int, default=None,
                        help="Epochs for contrastive runs only (default: same as --num-epochs). E.g. 5 = 3 warmup + 2 contrastive when used with --contrastive-warmup-epochs 3")
    parser.add_argument("--contrastive-learning-rate", type=float, default=None,
                        help="Learning rate for contrastive runs only (default: use --learning-rate)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline (instruction-only) run")

    # Evaluation
    parser.add_argument("--eval", action="store_true", help="Evaluate each model on test set after training")
    parser.add_argument("--num-eval-samples", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--max-gen-length", type=int, default=3072)

    parser.add_argument("--model-config", type=str, default=None, help="Path to model config JSON")
    parser.add_argument("--train-seed", type=int, default=42,
                        help="Random seed for training (data order, init, etc.) (default: 42)")

    args = parser.parse_args()

    # Set training seed for reproducibility
    _set_training_seed(args.train_seed)
    print(f"Training seed: {args.train_seed}")

    device = _get_device(args.device)
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    model_config = None
    if args.model_config:
        with open(args.model_config, "r") as f:
            model_config = json.load(f)

    num_range = (1, 20)
    results: List[Dict[str, Any]] = []

    def _steps_per_epoch(corpus_path: str, batch_size: int, train_split: float = 0.9) -> int:
        """Approximate training steps per epoch from corpus line count."""
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            num_lines = sum(1 for _ in f)
        train_size = int(train_split * num_lines)
        return max(1, (train_size + batch_size - 1) // batch_size)

    def make_base_config(
        use_contrastive: bool,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.05,
        contrastive_warmup_steps: int = 0,
        learning_rate_override: Optional[float] = None,
        num_epochs_override: Optional[int] = None,
    ) -> TrainingConfig:
        lr = learning_rate_override if learning_rate_override is not None else args.learning_rate
        num_epochs = num_epochs_override if num_epochs_override is not None else args.num_epochs
        return TrainingConfig(
            learning_rate=lr,
            batch_size=args.batch_size,
            num_epochs=num_epochs,
            warmup_steps=args.warmup_steps,
            gradient_clip=1.0,
            save_every=1000,
            device=device,
            use_wandb=args.wandb,
            use_contrastive=use_contrastive,
            contrastive_weight=contrastive_weight,
            contrastive_temperature=contrastive_temperature,
            contrastive_warmup_steps=contrastive_warmup_steps,
            use_curriculum=False,
            curriculum_steps=10000,
            num_workers=args.num_workers,
        )

    # --- Baseline: full instruction only ---
    if not args.skip_baseline:
        _set_training_seed(args.train_seed)
        print("\n" + "=" * 60)
        print("Training FULL INSTRUCTION (baseline, no contrastive)")
        print("=" * 60)
        baseline_dir = os.path.join(base_dir, "instruction_baseline")
        try:
            checkpoint_path, train_sec, vram_train_alloc, vram_train_reserved = run_full_instruction_training(
                instruction_corpus_path=args.instruction_corpus_path,
                tokenizer_path=args.tokenizer_path,
                tokenizer_type=args.tokenizer_type,
                foundational_checkpoint=args.foundational_checkpoint,
                output_dir=baseline_dir,
                training_config=make_base_config(use_contrastive=False),
                model_config=model_config,
            )
            best_model_path = os.path.join(baseline_dir, "best_model.pt")
            if not os.path.exists(best_model_path):
                best_model_path = checkpoint_path

            summary = load_training_summary(baseline_dir)
            rec: Dict[str, Any] = {
                "name": "instruction_baseline",
                "contrastive": False,
                "contrastive_weight": None,
                "contrastive_temperature": None,
                "model_path": best_model_path,
                "training_summary": summary,
                "efficiency": {
                    "training_time_sec": train_sec,
                    "eval_time_sec": None,
                    "vram_train_allocated_mb": vram_train_alloc,
                    "vram_train_reserved_mb": vram_train_reserved,
                    "vram_eval_allocated_mb": None,
                    "vram_eval_reserved_mb": None,
                },
                "metrics": None,
            }

            if args.eval:
                metrics, eval_sec, vram_eval_alloc, vram_eval_reserved = evaluate_model(
                    model_path=best_model_path,
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
                rec["metrics"] = metrics
                rec["efficiency"]["eval_time_sec"] = eval_sec
                rec["efficiency"]["vram_eval_allocated_mb"] = vram_eval_alloc
                rec["efficiency"]["vram_eval_reserved_mb"] = vram_eval_reserved
                vram_str = f"  eval VRAM: {vram_eval_alloc}/{vram_eval_reserved} MB" if vram_eval_alloc else ""
                print(f"  Baseline accuracy: {metrics['exact_match_accuracy']:.2f}%  train: {train_sec:.0f}s  eval: {eval_sec:.1f}s{vram_str}")
            else:
                best_val = summary.get("best_val_loss") if summary else None
                print(f"  Baseline  best_val_loss: {best_val}  train: {train_sec:.0f}s")
            results.append(rec)
        except Exception as e:
            print(f"  Baseline failed: {e}")
            results.append({
                "name": "instruction_baseline",
                "contrastive": False,
                "error": str(e),
                "metrics": None,
                "training_summary": None,
            })

    # --- Full instruction + contrastive (grid of weight × temperature) ---
    # Resolve warmup steps for contrastive: --contrastive-warmup-epochs overrides --contrastive-warmup-steps when > 0
    contrastive_warmup_steps_resolved = args.contrastive_warmup_steps
    if getattr(args, "contrastive_warmup_epochs", 0) > 0:
        steps_per_epoch = _steps_per_epoch(
            args.instruction_corpus_path, args.batch_size
        )
        contrastive_warmup_steps_resolved = int(
            getattr(args, "contrastive_warmup_epochs") * steps_per_epoch
        )
        print(f"Contrastive warmup: {getattr(args, 'contrastive_warmup_epochs')} epochs = {contrastive_warmup_steps_resolved} steps (steps_per_epoch={steps_per_epoch})")
    contrastive_num_epochs = getattr(args, "contrastive_num_epochs", None)

    for cw in args.contrastive_weights:
        for ct in args.contrastive_temperatures:
            _set_training_seed(args.train_seed)
            run_name = f"contrastive_w{cw}_t{ct}"
            run_dir = os.path.join(base_dir, run_name)
            print("\n" + "=" * 60)
            print(f"Training FULL INSTRUCTION + CONTRASTIVE  weight={cw}  temperature={ct}")
            print("=" * 60)
            try:
                config = make_base_config(
                    use_contrastive=True,
                    contrastive_weight=cw,
                    contrastive_temperature=ct,
                    contrastive_warmup_steps=contrastive_warmup_steps_resolved,
                    learning_rate_override=args.contrastive_learning_rate,
                    num_epochs_override=contrastive_num_epochs,
                )
                checkpoint_path, train_sec, vram_train_alloc, vram_train_reserved = run_full_instruction_training(
                    instruction_corpus_path=args.instruction_corpus_path,
                    tokenizer_path=args.tokenizer_path,
                    tokenizer_type=args.tokenizer_type,
                    foundational_checkpoint=args.foundational_checkpoint,
                    output_dir=run_dir,
                    training_config=config,
                    model_config=model_config,
                )
                best_model_path = os.path.join(run_dir, "best_model.pt")
                if not os.path.exists(best_model_path):
                    best_model_path = checkpoint_path

                summary = load_training_summary(run_dir)
                rec = {
                    "name": run_name,
                    "contrastive": True,
                    "contrastive_weight": cw,
                    "contrastive_temperature": ct,
                    "model_path": best_model_path,
                    "training_summary": summary,
                    "efficiency": {
                        "training_time_sec": train_sec,
                        "eval_time_sec": None,
                        "vram_train_allocated_mb": vram_train_alloc,
                        "vram_train_reserved_mb": vram_train_reserved,
                        "vram_eval_allocated_mb": None,
                        "vram_eval_reserved_mb": None,
                    },
                    "metrics": None,
                }

                if args.eval:
                    metrics, eval_sec, vram_eval_alloc, vram_eval_reserved = evaluate_model(
                        model_path=best_model_path,
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
                    rec["metrics"] = metrics
                    rec["efficiency"]["eval_time_sec"] = eval_sec
                    rec["efficiency"]["vram_eval_allocated_mb"] = vram_eval_alloc
                    rec["efficiency"]["vram_eval_reserved_mb"] = vram_eval_reserved
                    vram_str = f"  eval VRAM: {vram_eval_alloc}/{vram_eval_reserved} MB" if vram_eval_alloc else ""
                    print(f"  w={cw} t={ct}  accuracy: {metrics['exact_match_accuracy']:.2f}%  train: {train_sec:.0f}s  eval: {eval_sec:.1f}s{vram_str}")
                else:
                    best_val = summary.get("best_val_loss") if summary else None
                    print(f"  w={cw} t={ct}  best_val_loss: {best_val}  train: {train_sec:.0f}s")
                results.append(rec)
            except Exception as e:
                print(f"  {run_name} failed: {e}")
                results.append({
                    "name": run_name,
                    "contrastive": True,
                    "contrastive_weight": cw,
                    "contrastive_temperature": ct,
                    "error": str(e),
                    "metrics": None,
                    "training_summary": None,
                })

    # --- Comparison table: training metrics ---
    print("\n" + "=" * 60)
    print("COMPARISON: Training (best_val_loss, final_train_loss)")
    print("=" * 60)
    row_len = 80
    headers = ["Config", "contrastive", "weight", "temp", "best_val_loss", "final_train_loss"]
    print(f"\n{headers[0]:<28} {headers[1]:>10} {headers[2]:>8} {headers[3]:>8} {headers[4]:>14} {headers[5]:>16}")
    print("-" * row_len)

    for rec in results:
        name = rec["name"]
        c = rec.get("contrastive", False)
        w = rec.get("contrastive_weight")
        t = rec.get("contrastive_temperature")
        w_str = f"{w}" if w is not None else "—"
        t_str = f"{t}" if t is not None else "—"
        if rec.get("error"):
            print(f"{name:<28} {str(c):>10} {w_str:>8} {t_str:>8} {'FAIL':>14} {rec['error'][:20]:>16}")
            continue
        summary = rec.get("training_summary") or {}
        best_val = summary.get("best_val_loss")
        final_train = summary.get("final_train_loss")
        best_str = f"{best_val:.4f}" if best_val is not None else "—"
        train_str = f"{final_train:.4f}" if final_train is not None else "—"
        print(f"{name:<28} {str(c):>10} {w_str:>8} {t_str:>8} {best_str:>14} {train_str:>16}")
    print("-" * row_len)

    # --- Comparison table: evaluation (if --eval) ---
    if args.eval:
        print("\n" + "=" * 60)
        print("COMPARISON: Evaluation (Exact %, Parse %, Expression-now consistent %)")
        print("=" * 60)
        headers = ["Config", "Exact %", "Parse %", "Expr-now %", "N"]
        print(f"\n{headers[0]:<28} {headers[1]:>8} {headers[2]:>8} {headers[3]:>10} {headers[4]:>6}")
        print("-" * row_len)
        for rec in results:
            name = rec["name"]
            if rec.get("metrics") is None:
                err = rec.get("error", "?")
                print(f"{name:<28} {'FAIL':>8} {err[:24]}")
                continue
            m = rec["metrics"]
            exact = m["exact_match_accuracy"]
            parse = m["parse_success_rate"]
            expr = m.get("expression_now_consistent_rate", 0.0)
            n = m["total_samples"]
            print(f"{name:<28} {exact:>8.2f} {parse:>8.2f} {expr:>10.2f} {n:>6}")
        print("-" * row_len)
        print("  (Expr-now % = expression_now_consistent_rate)")

    # --- Efficiency table ---
    print("\n" + "=" * 60)
    print("COMPARISON: Efficiency (time, VRAM)")
    print("=" * 60)
    eff_headers = ["Config", "Train(s)", "Eval(s)", "TrAlloc", "TrRes", "EvAlloc", "EvRes"]
    print(f"\n{eff_headers[0]:<28} {eff_headers[1]:>8} {eff_headers[2]:>8} {eff_headers[3]:>8} {eff_headers[4]:>8} {eff_headers[5]:>8} {eff_headers[6]:>8}")
    print("-" * 90)
    for rec in results:
        name = rec["name"]
        eff = rec.get("efficiency")
        if eff is None:
            print(f"{name:<28} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8}")
            continue
        train_s = eff.get("training_time_sec")
        eval_s = eff.get("eval_time_sec")
        tr_alloc = eff.get("vram_train_allocated_mb")
        tr_res = eff.get("vram_train_reserved_mb")
        ev_alloc = eff.get("vram_eval_allocated_mb")
        ev_res = eff.get("vram_eval_reserved_mb")
        train_str = f"{train_s:.0f}" if train_s is not None else "—"
        eval_str = f"{eval_s:.1f}" if eval_s is not None else "—"
        tr_alloc_s = f"{tr_alloc:.0f}" if tr_alloc is not None else "—"
        tr_res_s = f"{tr_res:.0f}" if tr_res is not None else "—"
        ev_alloc_s = f"{ev_alloc:.0f}" if ev_alloc is not None else "—"
        ev_res_s = f"{ev_res:.0f}" if ev_res is not None else "—"
        print(f"{name:<28} {train_str:>8} {eval_str:>8} {tr_alloc_s:>8} {tr_res_s:>8} {ev_alloc_s:>8} {ev_res_s:>8}")
    print("-" * 90)
    print("  (VRAM in MB: Tr = training, Ev = evaluation; — = N/A e.g. CPU)")

    # --- Save JSON (include explicit eval comparison: exact_match, parse, expression_now_consistent) ---
    def _result_for_json(r: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "name": r["name"],
            "contrastive": r.get("contrastive"),
            "contrastive_weight": r.get("contrastive_weight"),
            "contrastive_temperature": r.get("contrastive_temperature"),
            "model_path": r.get("model_path"),
            "error": r.get("error"),
            "training_summary": r.get("training_summary"),
            "metrics": r.get("metrics"),
            "efficiency": r.get("efficiency"),
        }
        # Explicit final comparison fields for easy parsing (incl. expression_now_consistent_rate)
        m = r.get("metrics")
        if m is not None:
            out["exact_match_accuracy"] = m.get("exact_match_accuracy")
            out["parse_success_rate"] = m.get("parse_success_rate")
            out["expression_now_consistent_rate"] = m.get("expression_now_consistent_rate")
        return out

    out_summary = {
        "timestamp": datetime.now().isoformat(),
        "train_seed": args.train_seed,
        "eval_seed": getattr(args, "eval_seed", None),
        "args": vars(args),
        "results": [_result_for_json(r) for r in results],
    }
    out_path = os.path.join(base_dir, "contrastive_sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
