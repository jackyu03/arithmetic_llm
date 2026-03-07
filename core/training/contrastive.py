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


# Step pattern: "Step N: A op B = R" (group 2 = result R)
_STEP_PATTERN = re.compile(
    r"(Step\s+\d+\s*:\s*-?\d+\s*[+-]\s*-?\d+\s*=\s*)(-?\d+)",
    flags=re.IGNORECASE,
)
# Full step: capture A, op, B, R for recomputation (groups 1-4)
_STEP_FULL_PATTERN = re.compile(
    r"Step\s+\d+\s*:\s*(-?\d+)\s*([+-])\s*(-?\d+)\s*=\s*(-?\d+)",
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


def _tens_digit_wrong_value(val: int, rng: random.Random) -> Optional[int]:
    """Return a value that differs only in the tens digit (e.g. 20 -> 10 or 30), or None if not applicable.
    Only applies when abs(val) >= 10. Single-digit numbers return None.
    """
    if abs(val) < 10:
        return None
    sign = 1 if val >= 0 else -1
    abs_val = abs(val)
    tens = (abs_val // 10) % 10
    ones = abs_val % 10
    wrong_tens = tens + rng.choice([-1, 1])
    if wrong_tens < 0:
        wrong_tens = 0
    if wrong_tens > 9:
        wrong_tens = 9
    new_abs = wrong_tens * 10 + ones
    return sign * new_abs if new_abs != abs_val else None


def _replace_number_whole_word_after(text: str, after_start: int, old_val: int, new_val: str) -> str:
    """Replace every whole-word occurrence of old_val with new_val in text[after_start:].
    Avoids replacing digits inside other numbers (e.g. 12 inside 212).
    """
    rest = text[after_start:]
    # Match the number as a standalone token: not preceded/followed by digit (or minus for negative)
    old_str = str(old_val)
    pattern = re.compile(r"(?<![0-9])" + re.escape(old_str) + r"(?![0-9])")
    rest = pattern.sub(new_val, rest)
    return text[:after_start] + rest


def _recompute_step_results_after(text: str, corrupt_idx: int) -> Tuple[str, Optional[int]]:
    """Recompute and replace the result R of every step after index corrupt_idx so that R = A op B.
    Returns (updated_text, last_step_result). Only one error (at corrupt_idx) then propagates to final.
    """
    steps_full = list(_STEP_FULL_PATTERN.finditer(text))
    if corrupt_idx + 1 >= len(steps_full):
        return text, None
    out = text
    last_R: Optional[int] = None
    # Re-find steps each time so positions stay valid after each replacement
    for i in range(corrupt_idx + 1, len(steps_full)):
        matches = list(_STEP_FULL_PATTERN.finditer(out))
        if i >= len(matches):
            break
        m = matches[i]
        try:
            a, b = int(m.group(1)), int(m.group(3))
            op = m.group(2)
            new_R = (a + b) if op == "+" else (a - b)
            last_R = new_R
            out = out[: m.start(4)] + str(new_R) + out[m.end(4) :]
        except ValueError:
            pass
    return out, last_R


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


def make_wrong_solution(
    solution: str,
    correct_answer: int,
    seed: Optional[int] = None,
    allow_drop_subtree: bool = True,
) -> str:
    """Create a wrong solution: one step tens-digit wrong, then propagate to final (no second error).

    - Main type: Pick one step with result >= 10, flip tens digit (e.g. 20->10 or 30); replace that number
      everywhere after; then recompute every later step's "= R" so R = A op B. Final Result = last step result.
      So only one error, propagated consistently to the end (no extra tens-digit error that could cancel).
    - Type C (small prob, only when allow_drop_subtree): Drop one top-level subtree in an "Expression now" line.

    Type A/B (final-only or arbitrary step+final) are removed so negatives match real model errors.

    Args:
        solution: Full solution text including steps and "Final Result: N"
        correct_answer: The correct final numeric answer
        seed: Optional seed (e.g. sample index) to pick corruption type
        allow_drop_subtree: If False, never use Type C (only wrong step/final)

    Returns:
        Solution string with one type of corruption applied
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    delta = rng.choice([-3, -2, -1, 1, 2, 3])

    # 0 = tens-digit wrong + propagate (main), 1 = C drop subtree (small prob)
    if allow_drop_subtree:
        corruption_type = rng.choices([0, 1], weights=[0.92, 0.08], k=1)[0]
    else:
        corruption_type = 0

    out = solution

    if corruption_type == 1:
        # Type C: Drop one subtree in an "Expression now: EXPR" line (small prob)
        expr_now_matches = list(_EXPR_NOW_PATTERN.finditer(out))
        if expr_now_matches:
            match = rng.choice(expr_now_matches)
            prefix = match.group(1)
            expr = match.group(2).strip()
            dropped = _drop_one_subtree(expr, rng)
            if dropped is not None:
                new_line = prefix + dropped
                out = out[: match.start()] + new_line + out[match.end() :]
        wrong_answer = correct_answer + delta
        wrong_str = str(wrong_answer)
        pattern = re.compile(r"Final Result\s*:\s*[+-]?\s*\d+", flags=re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(f"Final Result: {wrong_str}", out, count=1)
        return out

    # Main: one step tens-digit wrong, then propagate (replace that number everywhere after), then
    # recompute every later step's "= R" so only one error propagates to final (no second tens-digit error).
    steps = list(_STEP_PATTERN.finditer(out))
    candidates: List[Tuple[int, Any, int, int]] = []
    for idx, m in enumerate(steps):
        try:
            result_val = int(m.group(2))
            if abs(result_val) >= 10:
                wrong_val = _tens_digit_wrong_value(result_val, rng)
                if wrong_val is not None:
                    candidates.append((idx, m, result_val, wrong_val))
        except ValueError:
            pass
    corrupt_idx: Optional[int] = None
    if candidates:
        corrupt_idx, match, result_val, wrong_result = rng.choice(candidates)
        wrong_str = str(wrong_result)
        out = out[: match.start(2)] + wrong_str + out[match.end(2) :]
        after_end = match.end(2)
        out = _replace_number_whole_word_after(out, after_end, result_val, wrong_str)
    else:
        # No step with tens digit: corrupt one step by delta and propagate
        if steps:
            idx = rng.randrange(len(steps))
            corrupt_idx = idx
            match = steps[idx]
            try:
                result_val = int(match.group(2))
                wrong_result = result_val + delta
                if wrong_result == result_val:
                    wrong_result = result_val + (1 if result_val >= 0 else -1)
                wrong_str = str(wrong_result)
                out = out[: match.start(2)] + wrong_str + out[match.end(2) :]
                after_end = match.end(2)
                out = _replace_number_whole_word_after(out, after_end, result_val, wrong_str)
            except ValueError:
                corrupt_idx = None
    # Recompute every step after the corrupted one so R = A op B (one error, propagate to final)
    if corrupt_idx is not None:
        out, last_R = _recompute_step_results_after(out, corrupt_idx)
        if last_R is not None:
            pattern = re.compile(r"Final\s+Result\s*:\s*[+-]?\s*\d+", flags=re.IGNORECASE)
            if pattern.search(out):
                out = pattern.sub(f"Final Result: {last_R}", out, count=1)
            return out
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

    Multi-token results: If the result number is tokenized as multiple tokens
    (e.g. digit tokenizer gives "12" -> "1", "2"), we mark every position that
    predicts any of those tokens. Otherwise contrastive signal only on the last
    digit would miss the tens-digit errors (which happen at earlier tokens).

    Convention: logits[i] predicts input_ids[i+1], so the position that predicts
    token at index k is target_index = k - 1.

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
        # Token range for this result span: ids for full_text[:full_start] end at index
        # start_token; ids for full_text[:full_end] end at index end_token (inclusive last
        # token of result). So result tokens are at indices [start_token, end_token].
        ids_before_span = tokenizer.encode(full_text[:full_start], add_special_tokens=True)
        ids_through_span = tokenizer.encode(full_text[:full_end], add_special_tokens=True)
        start_token = len(ids_before_span)
        end_token = len(ids_through_span) - 1
        # Positions that predict result tokens: logits[k-1] predicts token k
        for token_index in range(start_token, end_token + 1):
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
    margin_max: Optional[float] = None,
    hard_ratio: float = 1.0,
) -> torch.Tensor:
    """Compute contrastive loss so correct completion is preferred over wrong.

    Per-sample margins then mean loss to avoid cancellation across the batch:
    For each sample i: L_correct_i = mean log P over mask, L_wrong_i = mean log P over mask.
    loss_i = softplus(-(L_correct_i - L_wrong_i) / temperature)
    loss = mean(loss_i) over selected samples.

    Hard-pair filtering (often the key to beat baseline when it's already high):
    - margin_max: only backprop on samples where (Lc-Lw) < margin_max in raw log-prob space.
      With result-token contrastive, (Lc-Lw) is often 0.5~2+ once model is decent; use 0.5~2.0 or try hard_ratio instead.
    - hard_ratio: only backprop on top hard_ratio*100%% samples by loss (e.g. 0.3 = top 30%%).
    When both are set, margin filter is applied first, then top hard_ratio of those.

    When result_token_mask_* are provided, only positions of "Step ... = <result>"
    and "Final Result: <answer>" are used (result-token contrastive).

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
        margin_max: If set, only samples with (Lc-Lw) < margin_max contribute (raw margin; e.g. 0.2~0.5)
        hard_ratio: If < 1.0, only top hard_ratio fraction by loss contribute (e.g. 0.3 = top 30%%)

    Returns:
        Scalar contrastive loss (mean over selected samples; 0 if none selected)
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

    margin_raw = L_correct_per - L_wrong_per
    margin_per = margin_raw / temperature
    loss_per = F.softplus(-margin_per)

    # Hard-pair filter: only backprop on hard samples (little extra cost, often big gain)
    hard_mask = torch.ones_like(loss_per, dtype=torch.bool, device=loss_per.device)
    if margin_max is not None:
        hard_mask = hard_mask & (margin_raw < margin_max)
    if hard_ratio < 1.0:
        b = loss_per.size(0)
        k = max(1, int(b * hard_ratio))
        _, top_idx = torch.topk(loss_per, k, dim=0)
        topk_mask = torch.zeros_like(loss_per, dtype=torch.bool, device=loss_per.device)
        topk_mask.scatter_(0, top_idx, True)
        hard_mask = hard_mask & topk_mask

    n_hard = hard_mask.sum()
    if n_hard == 0:
        return loss_per.mean() * 0.0
    loss = (loss_per * hard_mask.float()).sum() / n_hard
    return loss


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
