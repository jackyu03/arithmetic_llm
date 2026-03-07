#!/usr/bin/env python3
"""Run DPO (Direct Preference Optimization) after instruction tuning.

Loads an instruction-tuned checkpoint as both reference (frozen) and policy (trainable),
and trains on preference pairs (chosen=correct solution, rejected=make_wrong_solution).
"""

import argparse
import json
import os
from datetime import datetime

import torch

from core.model.transformer import ArithmeticTransformer
from core.data.tokenizer import ArithmeticBPETokenizer, ArithmeticDigitTokenizer
from core.data.loader import DPOPreferenceDataset, collate_dpo_fn
from core.training.dpo import train_dpo_epoch
from core.training.foundational import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DPO: optimize policy to prefer correct over wrong solutions (run after instruction)"
    )
    parser.add_argument("--instruction-checkpoint", type=str, required=True,
                        help="Path to instruction-tuned checkpoint (e.g. best_model.pt)")
    parser.add_argument("--instruction-corpus-path", type=str, required=True,
                        help="Path to instruction JSONL corpus (same format as instruction training)")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path to tokenizer directory")
    parser.add_argument("--tokenizer-type", type=str, default="digit", choices=["digit", "bpe"])
    parser.add_argument("--output-dir", type=str, default="models/dpo",
                        help="Output directory for DPO checkpoints")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of DPO epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                        help="Learning rate for policy (default: 5e-6)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (implicit reward temperature, default: 0.1)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length (default: 2048)")
    parser.add_argument("--allow-drop-subtree", action="store_true", default=True,
                        help="Allow drop-subtree in wrong solutions (default: True)")
    parser.add_argument("--no-allow-drop-subtree", action="store_false", dest="allow_drop_subtree")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4; use 0 on Windows if spawn issues)")
    parser.add_argument("--wrong-examples", type=int, default=20,
                        help="Number of make_wrong_solution examples to save to wrong_examples.txt (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"dpo_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("Loading tokenizer...")
    if args.tokenizer_type == "digit":
        tokenizer = ArithmeticDigitTokenizer()
    else:
        tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(args.tokenizer_path)
    vocab_size = len(tokenizer.token2id)

    print("Loading checkpoint and model config...")
    checkpoint = torch.load(args.instruction_checkpoint, map_location="cpu")
    model_config = checkpoint.get("model_config", {
        "vocab_size": vocab_size,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "max_seq_length": args.max_length,
    })
    if model_config.get("vocab_size") != vocab_size:
        model_config["vocab_size"] = vocab_size

    print("Building policy and reference models...")
    policy_model = ArithmeticTransformer(**model_config)
    load_checkpoint(args.instruction_checkpoint, policy_model)
    policy_model = policy_model.to(device)

    ref_model = ArithmeticTransformer(**model_config)
    load_checkpoint(args.instruction_checkpoint, ref_model)
    ref_model = ref_model.to(device)
    for p in ref_model.parameters():
        p.requires_grad = False

    print("Creating DPO dataset and dataloader...")
    dpo_dataset = DPOPreferenceDataset(
        corpus_path=args.instruction_corpus_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        allow_drop_subtree=args.allow_drop_subtree,
    )
    if len(dpo_dataset) == 0:
        raise ValueError("DPO dataset is empty; check instruction corpus path and format")

    # Save wrong-solution examples to txt for inspection
    if args.wrong_examples > 0:
        n_save = min(args.wrong_examples, len(dpo_dataset))
        wrong_path = os.path.join(run_dir, "wrong_examples.txt")
        with open(wrong_path, "w", encoding="utf-8") as f:
            f.write("# DPO wrong (rejected) examples: prompt | chosen (correct) | rejected (make_wrong_solution)\n")
            f.write("# " + "=" * 80 + "\n\n")
            for i in range(n_save):
                prompt, chosen_text, rejected_text = dpo_dataset.get_example_texts(i)
                f.write(f"--- Example {i + 1} ---\n")
                f.write("PROMPT:\n")
                f.write(prompt + "\n\n")
                f.write("CHOSEN (correct):\n")
                f.write(chosen_text + "\n\n")
                f.write("REJECTED (wrong):\n")
                f.write(rejected_text + "\n\n")
        print(f"Saved {n_save} wrong examples to {wrong_path}")

    train_loader = torch.utils.data.DataLoader(
        dpo_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_dpo_fn,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    print(f"DPO training: {args.num_epochs} epochs, {len(train_loader)} batches/epoch, beta={args.beta}")
    print("=" * 60)

    best_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_dpo_epoch(
            policy_model=policy_model,
            ref_model=ref_model,
            train_dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=args.beta,
        )
        print(f"Epoch {epoch}/{args.num_epochs}  DPO loss: {train_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(run_dir, "best_model.pt")
            save_config = {
                "vocab_size": policy_model.vocab_size,
                "d_model": policy_model.d_model,
                "nhead": policy_model.nhead,
                "num_layers": policy_model.num_layers,
                "dim_feedforward": policy_model.dim_feedforward,
                "dropout": policy_model.dropout,
                "max_seq_length": getattr(policy_model, "max_seq_length", args.max_length),
            }
            torch.save({
                "model_state_dict": policy_model.state_dict(),
                "model_config": save_config,
                "tokenizer_vocab_size": vocab_size,
            }, best_path)
            print(f"  Saved best model to {best_path}")

    final_path = os.path.join(run_dir, "final_model.pt")
    save_config = {
        "vocab_size": policy_model.vocab_size,
        "d_model": policy_model.d_model,
        "nhead": policy_model.nhead,
        "num_layers": policy_model.num_layers,
        "dim_feedforward": policy_model.dim_feedforward,
        "dropout": policy_model.dropout,
        "max_seq_length": getattr(policy_model, "max_seq_length", args.max_length),
    }
    torch.save({
        "model_state_dict": policy_model.state_dict(),
        "model_config": save_config,
        "tokenizer_vocab_size": vocab_size,
    }, final_path)
    print(f"Final model saved to {final_path}")

    with open(os.path.join(run_dir, "dpo_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
