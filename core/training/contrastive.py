"""Contrastive learning utilities for arithmetic LLM instruction tuning.

Trains the model to assign higher likelihood to correct solutions and lower
to incorrect ones (e.g. wrong step or wrong final answer), improving discrimination.
"""

import re
import torch
import torch.nn.functional as F
import random
from typing import Optional, Tuple


# Step pattern: "Step N: A op B = R"
_STEP_PATTERN = re.compile(
    r"(Step\s+\d+\s*:\s*-?\d+\s*[+-]\s*-?\d+\s*=\s*)(-?\d+)",
    flags=re.IGNORECASE,
)


def make_wrong_solution(solution: str, correct_answer: int, seed: Optional[int] = None) -> str:
    """Create a wrong solution by corrupting the final result and/or a step.

    Uses two types of negatives for a stronger contrastive signal:
    - Type A: Replace "Final Result: N" with N+1 (or N-1).
    - Type B: Replace one step's result (e.g. "Step 2: 5 + 3 = 8" -> "= 9") so
      many more tokens differ and the model learns to prefer correct reasoning.

    Args:
        solution: Full solution text including steps and "Final Result: N"
        correct_answer: The correct final numeric answer
        seed: Optional seed (e.g. sample index) to pick corruption type and step

    Returns:
        Solution string with a wrong result (final and/or one step)
    """
    use_step_corruption = True
    step_index = 0
    if seed is not None:
        # Deterministic per sample: ~50% corrupt a step, 50% only final
        use_step_corruption = (seed % 2) == 0
        step_index = seed % 10  # which step to corrupt if multiple

        rng = random.Random(seed)

        # use_step_corruption = rng.random() < 0.5
    out = solution
    delta = rng.choice([-3, -2, -1, 1, 2, 3])
    # Corrupt one step's result so wrong solution diverges earlier (stronger signal)
    if use_step_corruption:
        steps = list(_STEP_PATTERN.finditer(out))
        if steps:
            # idx = step_index % len(steps)
            idx = rng.randrange(len(steps))
            match = steps[idx]
            try:
                result_val = int(match.group(2))
                wrong_result = result_val + delta
                assert wrong_result != result_val
                wrong_str = str(wrong_result)
                out = out[: match.start(2)] + wrong_str + out[match.end(2) :]
            except ValueError:
                pass

    # Always corrupt final result so the negative has wrong answer
    wrong_answer = correct_answer + delta
    assert wrong_answer != correct_answer
    wrong_str = str(wrong_answer)
    pattern = re.compile(
        r"Final Result\s*:\s*[+-]?\s*\d+",
        flags=re.IGNORECASE,
    )
    if pattern.search(out):
        replacement = f"Final Result: {wrong_str}"
        out = pattern.sub(replacement, out, count=1)
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
