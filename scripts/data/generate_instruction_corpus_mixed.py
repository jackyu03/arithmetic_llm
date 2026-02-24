#!/usr/bin/env python3
"""Generate mixed instruction corpus (valid + invalid) without subprocesses."""

import argparse
import os
import random
import tempfile
from typing import List, Tuple

from core.data.corpus import CorpusGenerator


def _generate_instruction_corpus(
    num_samples: int,
    target_tokens: int,
    max_depth: int,
    num_range: Tuple[int, int],
    invalid_rate: float,
    output_path: str,
    tokenizer_type: str = "digit",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generator = CorpusGenerator(
        target_tokens=target_tokens,
        num_samples=num_samples,
        max_depth=max_depth,
        num_range=num_range,
        invalid_rate=invalid_rate,
        output_path=output_path,
        tokenizer_type=tokenizer_type,
    )
    generator.generate_instruction_corpus(output_path)


def _read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line for line in f if line.strip()]


def _write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a mixed instruction corpus with valid and invalid samples."
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Legacy sample count")
    parser.add_argument("--target-tokens", type=int, default=None, help="Target total tokens to generate")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--num-range", type=int, nargs=2, default=[1, 20])
    parser.add_argument("--invalid-rate", type=float, default=0.1)
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="digit",
        choices=["digit", "bpe"],
        help="Tokenizer type to use for counting target tokens (default: digit)"
    )
    parser.add_argument(
        "--output-mixed",
        type=str,
        default="data/instruction_corpus.txt",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.num_samples is None and args.target_tokens is None:
        parser.error("Must provide either --num-samples or --target-tokens")
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("num-samples must be positive")
    if args.target_tokens is not None and args.target_tokens <= 0:
        parser.error("target-tokens must be positive")
    if args.max_depth <= 0:
        parser.error("max-depth must be positive")
    if args.num_range[0] >= args.num_range[1]:
        parser.error("num-range MIN must be less than MAX")
    if not 0.0 <= args.invalid_rate <= 1.0:
        parser.error("invalid-rate must be between 0.0 and 1.0")

    num_range = (args.num_range[0], args.num_range[1])

    # Apportion targets
    error_samples = int(args.num_samples * args.invalid_rate) if args.num_samples else None
    correct_samples = int(args.num_samples * (1 - args.invalid_rate)) if args.num_samples else None
    
    error_tokens = int(args.target_tokens * args.invalid_rate) if args.target_tokens else None
    correct_tokens = int(args.target_tokens * (1 - args.invalid_rate)) if args.target_tokens else None

    # Generate to temp files to avoid persisting intermediate corpora
    with tempfile.TemporaryDirectory() as tmpdir:
        error_path = os.path.join(tmpdir, "instruction_corpus_error.txt")
        correct_path = os.path.join(tmpdir, "instruction_corpus_correct.txt")

        if args.invalid_rate > 0.0:
            _generate_instruction_corpus(
                num_samples=error_samples,
                target_tokens=error_tokens,
                max_depth=args.max_depth,
                num_range=num_range,
                invalid_rate=1.0, # All errors
                output_path=error_path,
                tokenizer_type=args.tokenizer_type,
            )

        _generate_instruction_corpus(
            num_samples=correct_samples,
            target_tokens=correct_tokens,
            max_depth=args.max_depth,
            num_range=num_range,
            invalid_rate=0.0,
            output_path=correct_path,
            tokenizer_type=args.tokenizer_type,
        )

        lines = _read_lines(correct_path)
        if args.invalid_rate > 0.0:
            lines += _read_lines(error_path)

    # Shuffle and write mixed
    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(lines)
    _write_lines(args.output_mixed, lines)


if __name__ == "__main__":
    main()


