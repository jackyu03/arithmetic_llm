"""Contrastive learning utilities for arithmetic LLM instruction tuning.

Trains the model to assign higher likelihood to correct solutions and lower
to incorrect ones (e.g. wrong final answer), improving discrimination.
"""

import re
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def make_wrong_solution(solution: str, correct_answer: int) -> str:
    """Create a wrong solution by corrupting the final result.

    Replaces "Final Result: N" with a different integer so the completion
    is a valid negative example for contrastive learning.

    Args:
        solution: Full solution text including steps and "Final Result: N"
        correct_answer: The correct final numeric answer

    Returns:
        Solution string with a wrong final result (e.g. correct_answer + 1)
    """
    # Avoid trivial wrong answer: use +1 or -1, or +2 if 0
    if correct_answer == 0:
        wrong_answer = 1
    else:
        wrong_answer = correct_answer + 1
    wrong_str = str(wrong_answer)
    # Match "Final Result: N" (with optional spaces, optional minus)
    pattern = re.compile(
        r"Final Result\s*:\s*[+-]?\s*\d+",
        flags=re.IGNORECASE,
    )
    if not pattern.search(solution):
        return solution
    replacement = f"Final Result: {wrong_str}"
    return pattern.sub(replacement, solution, count=1)


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

    L_correct = mean log P(correct completion tokens)
    L_wrong = mean log P(wrong completion tokens)
    Loss = -log sigmoid((L_correct - L_wrong) / temperature)

    Args:
        logits_correct: (batch, seq_len, vocab) from model on correct sequences
        labels_correct: (batch, seq_len) next-token targets, -100 on prompt
        logits_wrong: (batch, seq_len, vocab) from model on wrong sequences
        labels_wrong: (batch, seq_len) next-token targets for wrong, -100 on prompt
        completion_mask_correct: (batch, seq_len) 1 where we score correct
        completion_mask_wrong: (batch, seq_len) 1 where we score wrong
        temperature: Scale for the margin (smaller = stronger push)

    Returns:
        Scalar contrastive loss
    """
    # Log probs: (batch, seq_len, vocab) -> gather at target positions
    log_p_correct = F.log_softmax(logits_correct, dim=-1)
    log_p_wrong = F.log_softmax(logits_wrong, dim=-1)

    # Gather log P(target token) at each position
    # labels: (B, L), we need (B, L, 1) to gather from (B, L, V)
    targets_c = labels_correct.unsqueeze(-1).clamp(min=0)
    targets_w = labels_wrong.unsqueeze(-1).clamp(min=0)
    token_log_p_c = log_p_correct.gather(dim=-1, index=targets_c).squeeze(-1)
    token_log_p_w = log_p_wrong.gather(dim=-1, index=targets_w).squeeze(-1)

    # Mask: only completion positions (labels != -100)
    token_log_p_c = token_log_p_c.masked_fill(~completion_mask_correct.bool(), 0.0)
    token_log_p_w = token_log_p_w.masked_fill(~completion_mask_wrong.bool(), 0.0)

    n_c = completion_mask_correct.sum().clamp(min=1)
    n_w = completion_mask_wrong.sum().clamp(min=1)
    L_correct = token_log_p_c.sum() / n_c
    L_wrong = token_log_p_w.sum() / n_w

    # -log sigmoid((L_correct - L_wrong)/tau) = log(1 + exp(-(L_correct - L_wrong)/tau))
    margin = (L_correct - L_wrong) / temperature
    loss = F.softplus(-margin)
    return loss
