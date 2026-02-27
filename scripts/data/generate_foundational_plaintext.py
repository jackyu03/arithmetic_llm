#!/usr/bin/env python3
"""Generate foundational corpus and output shuffled plain-text lines."""

import argparse
import json
import os
import random
import tempfile
from typing import List, Tuple

from core.data.corpus import CorpusGenerator


def _normalize_line(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def _read_jsonl_lines(
    path: str,
    one_sequence_per_sample: bool = True,
    valid_only: bool = False,
) -> List[str]:
    """Read JSONL and return training lines.
    
    If one_sequence_per_sample is True (default): each line is "problem + ' ' + solution"
    so the foundational model learns to continue from "Evaluate: X" with " <think> ... </think> Final Result: N".
    If False (legacy): problem and solution are separate lines (model never sees problem->solution).
    If valid_only is True: skip entries where answer is "ERROR" (no invalid expression solutions).
    """
    lines: List[str] = []
    with open(path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if valid_only and obj.get("answer") == "ERROR":
                continue
            problem = _normalize_line(str(obj.get("problem", "")))
            solution = _normalize_line(str(obj.get("solution", "")))
            if one_sequence_per_sample:
                if problem and solution:
                    lines.append(problem + " " + solution)
            else:
                if problem:
                    lines.append(problem)
                if solution:
                    lines.append(solution)
    return lines


def _write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate foundational corpus and save as shuffled plain text."
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Legacy sample count")
    parser.add_argument("--target-tokens", type=int, default=None, help="Target total tokens to generate")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--num-range", type=int, nargs=2, default=[1, 20])
    parser.add_argument("--invalid-rate", type=float, default=0.1)
    parser.add_argument(
        "--output-txt",
        type=str,
        default="data/foundational_corpus.txt",
        help="Path to save shuffled plain-text corpus",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Exclude ERROR solutions (only valid problem-solution pairs)",
    )
    parser.add_argument(
        "--legacy-separate-lines",
        action="store_true",
        help="Write problem and solution as separate lines (old behavior; not recommended)",
    )
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

    num_range: Tuple[int, int] = (args.num_range[0], args.num_range[1])

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "foundational_corpus.jsonl")
        generator = CorpusGenerator(
            target_tokens=args.target_tokens,
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            num_range=num_range,
            invalid_rate=args.invalid_rate,
            output_path=jsonl_path,
        )
        generator.generate_corpus()

        lines = _read_jsonl_lines(
            jsonl_path,
            one_sequence_per_sample=not args.legacy_separate_lines,
            valid_only=args.valid_only,
        )

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(lines)
    _write_lines(args.output_txt, lines)


if __name__ == "__main__":
    main()


