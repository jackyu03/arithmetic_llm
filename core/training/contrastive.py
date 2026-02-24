"""Contrastive learning utilities for arithmetic LLM instruction tuning.

Trains the model to assign higher likelihood to correct solutions and lower
to incorrect ones (e.g. wrong step, wrong final answer, or dropped expression
subtrees), improving discrimination.
"""

import re
import torch
import torch.nn.functional as F
import random
from typing import Optional, Tuple, List


# Step pattern: "Step N: A op B = R"
_STEP_PATTERN = re.compile(
    r"(Step\s+\d+\s*:\s*-?\d+\s*[+-]\s*-?\d+\s*=\s*)(-?\d+)",
    flags=re.IGNORECASE,
)

# "Expression now: EXPR" line (capture EXPR; . does not match newline)
_EXPR_NOW_PATTERN = re.compile(
    r"(Expression\s+now\s*:\s*)(.+?)(?=\n|$)",
    re.IGNORECASE,
)


def _top_level_terms(expr: str) -> List[Tuple[str, Optional[str]]]:
    """Split expression into top-level terms and following separator.
    
    Returns list of (term, sep_after), e.g. [(t0, ' + '), (t1, ' + '), (t2, None)].
    sep_after is the run of spaces + operator + spaces after each term, or None for the last.
    """
    expr = expr.strip()
    if not expr:
        return []
    depth = 0
    terms: List[Tuple[str, Optional[str]]] = []
    start = 0
    i = 0
    while i < len(expr):
        c = expr[i]
        if c == "(":
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            i += 1
        elif depth == 0 and c in "+-":
            term = expr[start:i].strip()
            if term:  # binary operator: we have a term before this +/-
                j = i
                while j < len(expr) and expr[j].isspace():
                    j += 1
                if j < len(expr) and expr[j] in "+-":
                    op_end = j + 1
                    while op_end < len(expr) and expr[op_end].isspace():
                        op_end += 1
                    sep = expr[i:op_end]
                    terms.append((term, sep))
                    start = op_end
                    i = op_end
                    continue
            i += 1
        else:
            i += 1
    if start < len(expr):
        terms.append((expr[start:].strip(), None))
    return terms


def _drop_one_subtree(expr: str, rng: random.Random) -> Optional[str]:
    """Return expression with one top-level term dropped, or None if not possible.
    
    E.g. "(1+9) + (3+(5-4)) + 1" -> "(1+9) + (3+(5-4))" or "(1+9) + 1" or "(3+(5-4)) + 1".
    """
    parts = _top_level_terms(expr)
    if len(parts) < 2:
        return None
    idx = rng.randrange(len(parts))
    kept = [(parts[i][0], parts[i][1]) for i in range(len(parts)) if i != idx]
    result = ""
    for term, sep in kept:
        result += term
        if sep is not None:
            result += sep
    return result.strip()


def make_wrong_solution(solution: str, correct_answer: int, seed: Optional[int] = None) -> str:
    """Create a wrong solution by corrupting final result, a step, or dropping an expression subtree.

    Uses three types of negatives:
    - Type A: Replace "Final Result: N" with N+delta.
    - Type B: Replace one step's result (e.g. "Step 2: 5 + 3 = 8" -> "= 9").
    - Type C: Drop one top-level subtree in a random "Expression now: EXPR" line
      (e.g. "(1+9) + (3+(5-4)) + 1" -> "(1+9) + (3+(5-4))" or "(1+9) + 1") so the
      model learns not to drop subtrees.

    Args:
        solution: Full solution text including steps and "Final Result: N"
        correct_answer: The correct final numeric answer
        seed: Optional seed (e.g. sample index) to pick corruption type

    Returns:
        Solution string with one type of corruption applied
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    delta = rng.choice([-3, -2, -1, 1, 2, 3])

    # Choose corruption type: 0 = step + final, 1 = final only, 2 = drop subtree
    corruption_type = (seed % 3) if seed is not None else rng.randint(0, 2)

    out = solution

    if corruption_type == 2:
        # Type C: Drop one subtree in an "Expression now: EXPR" line
        expr_now_matches = list(_EXPR_NOW_PATTERN.finditer(out))
        if expr_now_matches:
            match = rng.choice(expr_now_matches)
            prefix = match.group(1)   # "Expression now: "
            expr = match.group(2).strip()
            dropped = _drop_one_subtree(expr, rng)
            if dropped is not None:
                new_line = prefix + dropped
                out = out[: match.start()] + new_line + out[match.end() :]
        # Also corrupt final so the negative is clearly wrong
        wrong_answer = correct_answer + delta
        wrong_str = str(wrong_answer)
        pattern = re.compile(r"Final Result\s*:\s*[+-]?\s*\d+", flags=re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(f"Final Result: {wrong_str}", out, count=1)
        return out

    # Type A/B: step corruption and/or final
    use_step_corruption = corruption_type == 0
    if use_step_corruption:
        steps = list(_STEP_PATTERN.finditer(out))
        if steps:
            idx = rng.randrange(len(steps))
            match = steps[idx]
            try:
                result_val = int(match.group(2))
                wrong_result = result_val + delta
                if wrong_result != result_val:
                    wrong_str = str(wrong_result)
                    out = out[: match.start(2)] + wrong_str + out[match.end(2) :]
            except ValueError:
                pass

    wrong_answer = correct_answer + delta
    if wrong_answer != correct_answer:
        wrong_str = str(wrong_answer)
        pattern = re.compile(r"Final Result\s*:\s*[+-]?\s*\d+", flags=re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(f"Final Result: {wrong_str}", out, count=1)
    return out


def compute_contrastive_loss(
    logits_correct: torch.Tensor,
    labels_correct: torch.Tensor,
    logits_wrong: torch.Tensor,
    labels_wrong: torch.Tensor,
    completion_mask_correct: torch.Tensor,
    completion_mask_wrong: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute contrastive loss so correct completion is preferred over wrong.

    Per-sample margins then mean loss to avoid cancellation across the batch:
    For each sample i: L_correct_i = mean log P(correct completion), L_wrong_i = mean log P(wrong).
    loss_i = softplus(-(L_correct_i - L_wrong_i) / temperature)
    loss = mean(loss_i)

    Args:
        logits_correct: (batch, seq_len, vocab) from model on correct sequences
        labels_correct: (batch, seq_len) next-token targets, -100 on prompt
        logits_wrong: (batch, seq_len, vocab) from model on wrong sequences
        labels_wrong: (batch, seq_len) next-token targets for wrong, -100 on prompt
        completion_mask_correct: (batch, seq_len) 1 where we score correct
        completion_mask_wrong: (batch, seq_len) 1 where we score wrong
        temperature: Scale for the margin (smaller = stronger push)

    Returns:
        Scalar contrastive loss (mean over samples)
    """
    log_p_correct = F.log_softmax(logits_correct, dim=-1)
    log_p_wrong = F.log_softmax(logits_wrong, dim=-1)

    targets_c = labels_correct.unsqueeze(-1).clamp(min=0)
    targets_w = labels_wrong.unsqueeze(-1).clamp(min=0)
    token_log_p_c = log_p_correct.gather(dim=-1, index=targets_c).squeeze(-1)
    token_log_p_w = log_p_wrong.gather(dim=-1, index=targets_w).squeeze(-1)

    token_log_p_c = token_log_p_c.masked_fill(~completion_mask_correct.bool(), 0.0)
    token_log_p_w = token_log_p_w.masked_fill(~completion_mask_wrong.bool(), 0.0)

    # Per-sample: sum over seq, divide by count for that sample
    n_c_per_sample = completion_mask_correct.sum(dim=1).clamp(min=1)
    n_w_per_sample = completion_mask_wrong.sum(dim=1).clamp(min=1)
    L_correct_per = token_log_p_c.sum(dim=1) / n_c_per_sample
    L_wrong_per = token_log_p_w.sum(dim=1) / n_w_per_sample

    margin_per = (L_correct_per - L_wrong_per) / temperature
    loss_per = F.softplus(-margin_per)
    return loss_per.mean()
