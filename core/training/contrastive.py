"""Contrastive learning utilities for arithmetic LLM instruction tuning.

Trains the model to assign higher likelihood to correct solutions and lower
to incorrect ones (e.g. wrong step, wrong final answer, or dropped expression
subtrees), improving discrimination.

Supports:
- Completion-level contrastive (default): margin over full completion logP.
- Result-token contrastive: margin only at "Step ... = <result>" and
  "Final Result: <answer>" positions, so the signal is not diluted by sequence length.
"""

import re
import torch
import torch.nn.functional as F
import random
from typing import Optional, Tuple, List, Any


# Step pattern: "Step N: A op B = R"
_STEP_PATTERN = re.compile(
    r"(Step\s+\d+\s*:\s*-?\d+\s*[+-]\s*-?\d+\s*=\s*)(-?\d+)",
    flags=re.IGNORECASE,
)

# Final Result: N (capture the number)
_FINAL_RESULT_PATTERN = re.compile(
    r"Final\s+Result\s*:\s*([+-]?\s*\d+)",
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


def _result_spans_in_solution(solution: str) -> List[Tuple[int, int]]:
    """Return (start, end) character spans of each result in solution (relative to solution string).
    Order: step results first (by occurrence), then final result.
    """
    spans: List[Tuple[int, int]] = []
    for m in _STEP_PATTERN.finditer(solution):
        spans.append((m.start(2), m.end(2)))
    fm = _FINAL_RESULT_PATTERN.search(solution)
    if fm:
        spans.append((fm.start(1), fm.end(1)))
    return spans


def get_result_token_mask(
    full_text: str,
    solution_start_char: int,
    target_length: int,
    tokenizer: Any,
) -> List[int]:
    """Build a 0/1 mask of length target_length: 1 only at positions that predict a result token.

    Result positions are: the number after "= " in "Step N: ... = <result>", and
    the number in "Final Result: <answer>". Used for result-token contrastive loss
    so the margin is not diluted by sequence length.

    Args:
        full_text: Full string that was tokenized (prompt + ' ' + solution).
        solution_start_char: Character index in full_text where solution starts.
        target_length: Length of target sequence (len(token_ids) - 1).
        tokenizer: Tokenizer with encode(..., add_special_tokens=True).

    Returns:
        List of 0/1 of length target_length. 1 at positions that predict a result token.
    """
    solution = full_text[solution_start_char:]
    spans = _result_spans_in_solution(solution)
    mask = [0] * target_length
    for (sol_start, sol_end) in spans:
        full_start = solution_start_char + sol_start
        full_end = solution_start_char + sol_end
        if full_end <= 0:
            continue
        prefix = full_text[:full_end]
        ids = tokenizer.encode(prefix, add_special_tokens=True)
        # Token index (0-based) of the token containing the end of the result
        token_index = len(ids) - 1
        # Target position that predicts this token
        target_index = token_index - 1
        if 0 <= target_index < target_length:
            mask[target_index] = 1
    return mask


def compute_contrastive_loss(
    logits_correct: torch.Tensor,
    labels_correct: torch.Tensor,
    logits_wrong: torch.Tensor,
    labels_wrong: torch.Tensor,
    completion_mask_correct: torch.Tensor,
    completion_mask_wrong: torch.Tensor,
    temperature: float = 0.1,
    result_token_mask_correct: Optional[torch.Tensor] = None,
    result_token_mask_wrong: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute contrastive loss so correct completion is preferred over wrong.

    Per-sample margins then mean loss to avoid cancellation across the batch:
    For each sample i: L_correct_i = mean log P over mask, L_wrong_i = mean log P over mask.
    loss_i = softplus(-(L_correct_i - L_wrong_i) / temperature)
    loss = mean(loss_i)

    When result_token_mask_* are provided, only positions of "Step ... = <result>"
    and "Final Result: <answer>" are used (result-token contrastive), so the
    margin is not diluted by sequence length. Otherwise uses full completion mask.

    Args:
        logits_correct: (batch, seq_len, vocab) from model on correct sequences
        labels_correct: (batch, seq_len) next-token targets, -100 on prompt
        logits_wrong: (batch, seq_len, vocab) from model on wrong sequences
        labels_wrong: (batch, seq_len) next-token targets for wrong, -100 on prompt
        completion_mask_correct: (batch, seq_len) 1 where we score correct (fallback)
        completion_mask_wrong: (batch, seq_len) 1 where we score wrong (fallback)
        temperature: Scale for the margin (smaller = stronger push)
        result_token_mask_correct: (batch, seq_len) 1 only at result-token positions; if set, used for correct
        result_token_mask_wrong: (batch, seq_len) 1 only at result-token positions; if set, used for wrong

    Returns:
        Scalar contrastive loss (mean over samples)
    """
    log_p_correct = F.log_softmax(logits_correct, dim=-1)
    log_p_wrong = F.log_softmax(logits_wrong, dim=-1)

    targets_c = labels_correct.unsqueeze(-1).clamp(min=0)
    targets_w = labels_wrong.unsqueeze(-1).clamp(min=0)
    token_log_p_c = log_p_correct.gather(dim=-1, index=targets_c).squeeze(-1)
    token_log_p_w = log_p_wrong.gather(dim=-1, index=targets_w).squeeze(-1)

    use_result_token = (
        result_token_mask_correct is not None and result_token_mask_wrong is not None
    )
    if use_result_token:
        mask_c = result_token_mask_correct.bool()
        mask_w = result_token_mask_wrong.bool()
    else:
        mask_c = completion_mask_correct.bool()
        mask_w = completion_mask_wrong.bool()

    token_log_p_c = token_log_p_c.masked_fill(~mask_c, 0.0)
    token_log_p_w = token_log_p_w.masked_fill(~mask_w, 0.0)

    n_c_per_sample = mask_c.sum(dim=1).clamp(min=1)
    n_w_per_sample = mask_w.sum(dim=1).clamp(min=1)
    L_correct_per = token_log_p_c.sum(dim=1) / n_c_per_sample
    L_wrong_per = token_log_p_w.sum(dim=1) / n_w_per_sample

    margin_per = (L_correct_per - L_wrong_per) / temperature
    loss_per = F.softplus(-margin_per)
    return loss_per.mean()


def compute_expression_now_consistency_loss(
    logits: torch.Tensor,
    expr_now_spans: List[Tuple[int, int]],
    expr_now_correct_ids: List[List[int]],
) -> torch.Tensor:
    """Cross-entropy over "Expression now: EXPR" spans to enforce correct EXPR (no drop subtree).

    Use when you have ground-truth EXPR token spans and correct token ids (e.g. from
    eval_expression). For batch, call per sample and mean, or stack and loop.

    Args:
        logits: (seq_len, vocab) for one sample
        expr_now_spans: List of (start, end) target indices for each EXPR span
        expr_now_correct_ids: List of correct token id lists for each EXPR

    Returns:
        Scalar CE loss (mean over all EXPR tokens in this sample).
    """
    if not expr_now_spans or not expr_now_correct_ids:
        return logits.new_zeros(1).squeeze()
    device = logits.device
    losses = []
    for (start, end), correct_ids in zip(expr_now_spans, expr_now_correct_ids):
        if start >= end or end > logits.size(0) or not correct_ids:
            continue
        length = min(end - start, len(correct_ids))
        logits_slice = logits[start : start + length].contiguous()
        targets = torch.tensor(correct_ids[:length], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits_slice, targets, reduction="mean"))
    if not losses:
        return logits.new_zeros(1).squeeze()
    return torch.stack(losses).mean()
