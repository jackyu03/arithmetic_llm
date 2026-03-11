"""Contrastive learning without propagation: wrong = one step or final only (delta / tens-digit).

Same as contrastive.py but:
- Wrong solutions do NOT propagate: only one step's result is wrong (delta or tens-digit),
  and/or Final Result is wrong (delta or tens-digit). Rest of solution unchanged.
- Loss is still step-level but only at the corrupted positions: we record only the
  step(s) that were changed and the final answer, not all steps. So L_correct and
  L_wrong are computed over the same set of positions (the corrupted step + final).
"""

import re
import random
from typing import Optional, Tuple, List, Any

from core.training.contrastive import (
    _STEP_PATTERN,
    _FINAL_RESULT_PATTERN,
    _tens_digit_wrong_value,
    _result_spans_in_solution,
    compute_contrastive_loss,
)


def make_wrong_solution_no_prop(
    solution: str,
    correct_answer: int,
    seed: Optional[int] = None,
) -> Tuple[str, int, bool]:
    """Create a wrong solution without propagation: one step and/or final wrong (delta or tens-digit).

    Error modes:
    - One step wrong: pick a step, set result to result+delta or tens-digit wrong; set Final wrong too.
    - Final only: set only Final Result to correct_answer+delta or tens-digit wrong.

    Args:
        solution: Full solution text including steps and "Final Result: N"
        correct_answer: The correct final numeric answer
        seed: Optional seed for reproducibility

    Returns:
        (wrong_solution_str, corrupted_step_index_0based, corrupted_final)
        corrupted_step_index is -1 if only final was corrupted.
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    delta = rng.choice([-3, -2, -1, 1, 2, 3])

    out = solution
    steps = list(_STEP_PATTERN.finditer(out))
    # 0 = one step + final wrong, 1 = final only wrong
    choice = rng.choice([0, 1]) if steps else 1

    corrupted_step = -1
    corrupted_final = True  # we always corrupt final in both modes

    if choice == 0 and steps:
        # Corrupt one step (delta or tens-digit) and final
        idx = rng.randrange(len(steps))
        m = steps[idx]
        try:
            result_val = int(m.group(2))
            if abs(result_val) >= 10 and rng.random() < 0.5:
                wrong_val = _tens_digit_wrong_value(result_val, rng)
                wrong_result = wrong_val if wrong_val is not None else result_val + delta
            else:
                wrong_result = result_val + delta
                if wrong_result == result_val:
                    wrong_result = result_val + (1 if result_val >= 0 else -1)
            wrong_str = str(wrong_result)
            out = out[: m.start(2)] + wrong_str + out[m.end(2) :]
            corrupted_step = idx
        except ValueError:
            pass

    # Final result: delta or tens-digit
    if abs(correct_answer) >= 10 and rng.random() < 0.5:
        wrong_final = _tens_digit_wrong_value(correct_answer, rng)
        wrong_final = wrong_final if wrong_final is not None else correct_answer + delta
    else:
        wrong_final = correct_answer + delta
    wrong_final_str = str(wrong_final)
    pattern = re.compile(r"Final\s+Result\s*:\s*[+-]?\s*\d+", flags=re.IGNORECASE)
    if pattern.search(out):
        out = pattern.sub(f"Final Result: {wrong_final_str}", out, count=1)

    return (out, corrupted_step, corrupted_final)


def get_result_token_mask_for_positions(
    full_text: str,
    solution_start_char: int,
    target_length: int,
    tokenizer: Any,
    step_indices_0based: List[int],
    include_final: bool,
) -> List[int]:
    """Build a 0/1 mask of length target_length: 1 only at selected step result positions and optionally final.

    step_indices_0based: which step results to include (0-based; e.g. [0, 2] = first and third step).
    include_final: whether to include the Final Result position.

    Convention: same as get_result_token_mask in contrastive.py (logits[i] predicts token i+1).
    """
    solution = full_text[solution_start_char:]
    spans = _result_spans_in_solution(solution)
    if not spans:
        return [0] * target_length
    # spans = [step0_span, step1_span, ..., final_span]; n_steps = len(spans)-1
    n_steps = len(spans) - 1
    selected = []
    for i in step_indices_0based:
        if 0 <= i < n_steps:
            selected.append(spans[i])
    if include_final:
        selected.append(spans[-1])

    mask = [0] * target_length
    for (sol_start, sol_end) in selected:
        full_start = solution_start_char + sol_start
        full_end = solution_start_char + sol_end
        if full_end <= 0:
            continue
        ids_before_span = tokenizer.encode(full_text[:full_start], add_special_tokens=True)
        ids_through_span = tokenizer.encode(full_text[:full_end], add_special_tokens=True)
        start_token = len(ids_before_span)
        end_token = len(ids_through_span) - 1
        for token_index in range(start_token, end_token + 1):
            target_index = token_index - 1
            if 0 <= target_index < target_length:
                mask[target_index] = 1
    return mask


def get_result_token_mask_only_corrupted(
    full_text: str,
    solution_start_char: int,
    target_length: int,
    tokenizer: Any,
    corrupted_step_index: int,
    corrupted_final: bool,
) -> List[int]:
    """Mask for wrong sequence: only the corrupted step (if any) and the final result."""
    step_indices = [corrupted_step_index] if corrupted_step_index >= 0 else []
    return get_result_token_mask_for_positions(
        full_text,
        solution_start_char,
        target_length,
        tokenizer,
        step_indices,
        include_final=corrupted_final,
    )


def get_result_token_mask_correct_at_same_positions(
    full_text: str,
    solution_start_char: int,
    target_length: int,
    tokenizer: Any,
    corrupted_step_index: int,
    corrupted_final: bool,
) -> List[int]:
    """Mask for correct sequence at the same positions as we corrupted in the wrong sample."""
    step_indices = [corrupted_step_index] if corrupted_step_index >= 0 else []
    return get_result_token_mask_for_positions(
        full_text,
        solution_start_char,
        target_length,
        tokenizer,
        step_indices,
        include_final=corrupted_final,
    )


# Re-export so callers can use one module
__all__ = [
    "make_wrong_solution_no_prop",
    "get_result_token_mask_for_positions",
    "get_result_token_mask_only_corrupted",
    "get_result_token_mask_correct_at_same_positions",
    "compute_contrastive_loss",
]
