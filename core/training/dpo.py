"""Direct Preference Optimization (DPO) for arithmetic LLM.

Trains the policy to prefer chosen (correct) completions over rejected (wrong)
completions relative to a frozen reference model. No reward model or RL.
Run after instruction tuning: load instruction checkpoint as both ref and policy init.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def _completion_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    """Per-sample sum of log P(token_t | context) over completion tokens only.

    logits: (batch, seq_len, vocab_size), from model(input_ids)
    input_ids: (batch, seq_len)
    attention_mask: (batch, seq_len), 1 = real token, 0 = pad
    prompt_lengths: (batch,) int, number of prompt tokens (BOS + prompt) per sample

    Returns:
        (batch,) tensor of sum of log probs over completion positions (non-padding).
    """
    batch, seq_len, vocab_size = logits.shape
    device = logits.device
    # logits[:, t, :] predicts input_ids[:, t+1]
    log_p = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:].unsqueeze(-1)
    token_log_p = log_p.gather(dim=-1, index=target).squeeze(-1)
    # Mask: 1 only where we have a completion token (position t+1 >= prompt_length) and non-pad
    positions = torch.arange(seq_len - 1, device=device).unsqueeze(0).expand(batch, -1)
    completion_mask = (positions + 1 >= prompt_lengths.unsqueeze(1)) & (attention_mask[:, 1:] > 0)
    token_log_p = token_log_p.masked_fill(~completion_mask, 0.0)
    return token_log_p.sum(dim=1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO loss: -log σ(β * (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l))).

    Args:
        policy_chosen_logps: (batch,) log π_θ(chosen | x)
        policy_rejected_logps: (batch,) log π_θ(rejected | x)
        ref_chosen_logps: (batch,) log π_ref(chosen | x)
        ref_rejected_logps: (batch,) log π_ref(rejected | x)
        beta: Temperature for the implicit reward (higher = stronger preference signal)

    Returns:
        Scalar loss (mean over batch).
    """
    log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
    log_ratio_rejected = policy_rejected_logps - ref_rejected_logps
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -F.logsigmoid(logits).mean()
    return loss


def run_dpo_step(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    chosen_input_ids: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 0.1,
) -> float:
    """One optimizer step of DPO. Policy is updated; ref is frozen.

    chosen/rejected_input_ids: (batch, seq_len) including BOS + prompt + completion.
    prompt_lengths: (batch,) number of tokens in BOS + prompt.
    """
    policy_model.train()
    ref_model.eval()

    chosen_input_ids = chosen_input_ids.to(device)
    chosen_attention_mask = chosen_attention_mask.to(device)
    rejected_input_ids = rejected_input_ids.to(device)
    rejected_attention_mask = rejected_attention_mask.to(device)
    prompt_lengths = prompt_lengths.to(device)

    with torch.no_grad():
        ref_logits_chosen = ref_model(chosen_input_ids, attention_mask=chosen_attention_mask)
        ref_logits_rejected = ref_model(rejected_input_ids, attention_mask=rejected_attention_mask)
        ref_chosen_logps = _completion_log_probs(
            ref_logits_chosen, chosen_input_ids, chosen_attention_mask, prompt_lengths
        )
        ref_rejected_logps = _completion_log_probs(
            ref_logits_rejected, rejected_input_ids, rejected_attention_mask, prompt_lengths
        )

    policy_logits_chosen = policy_model(chosen_input_ids, attention_mask=chosen_attention_mask)
    policy_logits_rejected = policy_model(rejected_input_ids, attention_mask=rejected_attention_mask)
    policy_chosen_logps = _completion_log_probs(
        policy_logits_chosen, chosen_input_ids, chosen_attention_mask, prompt_lengths
    )
    policy_rejected_logps = _completion_log_probs(
        policy_logits_rejected, rejected_input_ids, rejected_attention_mask, prompt_lengths
    )

    loss = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def train_dpo_epoch(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 0.1,
) -> float:
    """Run one epoch of DPO training. Returns mean loss."""
    policy_model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in train_dataloader:
        (chosen_ids, chosen_mask, rejected_ids, rejected_mask, prompt_lengths) = batch
        loss = run_dpo_step(
            policy_model=policy_model,
            ref_model=ref_model,
            chosen_input_ids=chosen_ids,
            chosen_attention_mask=chosen_mask,
            rejected_input_ids=rejected_ids,
            rejected_attention_mask=rejected_mask,
            prompt_lengths=prompt_lengths,
            optimizer=optimizer,
            device=device,
            beta=beta,
        )
        total_loss += loss
        n_batches += 1
    return total_loss / max(1, n_batches)
