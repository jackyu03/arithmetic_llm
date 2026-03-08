#!/usr/bin/env python3
"""Generate a few wrong-answer samples via make_wrong_solution for inspection.

Usage:
  # Use built-in examples (no corpus needed)
  python scripts/utils/generate_wrong_samples.py

  # From instruction JSONL corpus
  python scripts/utils/generate_wrong_samples.py --corpus path/to/instruction.jsonl --num 5

  # Save to file
  python scripts/utils/generate_wrong_samples.py --output wrong_samples.txt
"""

import argparse
import json
import os
import sys

# Project root = parent of scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.training.contrastive import make_wrong_solution


# Built-in examples (no corpus file needed)
BUILTIN_EXAMPLES = [
    {
        "problem": "Evaluate: (4 + 3) + 8",
        "solution": """<think>
Step 1: 4 + 3 = 7
Expression now: (7 + 8)
Step 2: 7 + 8 = 15
Expression now: 15
Final Result: 15
</think>""",
        "answer": 15,
    },
    {
        "problem": "Evaluate: ((7 + 14) + 8)",
        "solution": """<think>
Step 1: 7 + 14 = 21
Expression now: 21 + 8
Step 2: 21 + 8 = 29
Expression now: 29
Final Result: 29
</think>""",
        "answer": 29,
    },
    {
        "problem": "Evaluate: (5 - 13) + (18 - 16) + ((6 + 11) - (7 + 11)) + (17 + ((12 - 10) - (17 - 17)))",
        "solution": """<think>
Step 1: 5 - 13 = -8
Expression now: (-8 - (18 - 16)) - ((6 + 11) - (7 + 11)) + (17 + ((12 - 10) - (17 - 17)))
Step 2: 18 - 16 = 2
Expression now: (-8 - 2) - ((6 + 11) - (7 + 11)) + (17 + ((12 - 10) - (17 - 17)))
Step 3: -8 - 2 = -10
Expression now: (-10 - ((6 + 11) - (7 + 11))) + (17 + ((12 - 10) - (17 - 17)))
Step 4: 6 + 11 = 17
Expression now: (-10 - (17 - (7 + 11))) + (17 + ((12 - 10) - (17 - 17)))
Step 5: 7 + 11 = 18
Expression now: (-10 - (17 - 18)) + (17 + ((12 - 10) - (17 - 17)))
Step 6: 17 - 18 = -1
Expression now: (-10 - -1) + (17 + ((12 - 10) - (17 - 17)))
Step 7: -10 - -1 = -9
Expression now: -9 + (17 + ((12 - 10) - (17 - 17)))
Step 8: 12 - 10 = 2
Expression now: -9 + (17 + (2 - (17 - 17)))
Step 9: 17 - 17 = 0
Expression now: -9 + (17 + (2 - 0))
Step 10: 2 - 0 = 2
Expression now: -9 + (17 + 2)
Step 11: 17 + 2 = 19
Expression now: -9 + 19
Step 12: -9 + 19 = 10
Expression now: 10
Final Result: 10
</think>""",
        "answer": 10,
    },
]


def load_from_corpus(path: str, num: int) -> list:
    """Load (problem, solution, answer) from instruction JSONL."""
    examples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(examples) >= num:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                problem = entry.get("problem")
                solution = entry.get("solution")
                answer = entry.get("answer")
                if problem is None or solution is None or answer is None or answer == "ERROR":
                    continue
                examples.append({
                    "problem": problem,
                    "solution": solution.strip(),
                    "answer": int(answer),
                })
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate wrong-answer samples with make_wrong_solution for inspection."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Instruction JSONL corpus path (optional; if not set, use built-in examples)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=3,
        help="Number of samples to generate (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set, write results to this txt file instead of stdout",
    )
    parser.add_argument(
        "--no-drop-subtree",
        action="store_true",
        help="Disable drop-subtree corruption type (only step wrong + propagate)",
    )
    args = parser.parse_args()

    if args.corpus:
        examples = load_from_corpus(args.corpus, args.num)
        if not examples:
            print("No valid examples loaded from corpus.", file=sys.stderr)
            sys.exit(1)
    else:
        examples = BUILTIN_EXAMPLES[: args.num]

    lines = []
    lines.append("# Wrong-answer samples (correct vs wrong)")
    lines.append("# " + "=" * 76)
    lines.append("")

    for i, ex in enumerate(examples):
        problem = ex["problem"]
        solution = ex["solution"]
        answer = ex["answer"]
        wrong = make_wrong_solution(
            solution,
            answer,
            seed=i,
            allow_drop_subtree=not args.no_drop_subtree,
        )
        # Strip <think> / </think> for display if present
        def strip_tags(t: str) -> str:
            t = t.strip()
            if t.startswith("<think>"):
                t = t[len("<think>"):].lstrip()
            if t.endswith("</think>"):
                t = t[: -len("</think>")].rstrip()
            return t

        sol_display = strip_tags(solution)
        wrong_display = strip_tags(wrong)

        lines.append(f"--- Example {i + 1} ---")
        lines.append("PROMPT: " + problem)
        lines.append("")
        lines.append("CORRECT:")
        lines.append(sol_display)
        lines.append("")
        lines.append("WRONG:")
        lines.append(wrong_display)
        lines.append("")

    text = "\n".join(lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {len(examples)} samples to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
