"""Verifier for arithmetic solutions used in GRPO training."""

from typing import Optional
import re


class ArithmeticVerifier:
    """Verifier for arithmetic solutions.

    Extracts final results from generated text and compares
    with ground truth to compute rewards.
    """

    def __init__(
        self,
        reward_format: bool = False,
        format_weight: float = 0.2,
        reward_length_penalty: bool = False,
        length_penalty_weight: float = 0.001,
        reward_equation_steps: bool = False,
        step_weight: float = 0.1
    ):
        self.reward_format = reward_format
        self.format_weight = format_weight
        self.reward_length_penalty = reward_length_penalty
        self.length_penalty_weight = length_penalty_weight
        self.reward_equation_steps = reward_equation_steps
        self.step_weight = step_weight

    def extract_final_result(self, generated_text: str) -> Optional[int]:
        """Extract final numeric result from generated text.

        Args:
            generated_text: Generated solution text

        Returns:
            Extracted integer result, or None if not found/parseable
        """
        error_match = re.search(
            r"Final Result\s*:\s*ERROR\b",
            generated_text,
            flags=re.IGNORECASE
        )
        if error_match:
            return None

        match = re.search(
            r"Final Result\s*:\s*([+-]?\s*\d+)",
            generated_text,
            flags=re.IGNORECASE
        )
        if match:
            raw_value = match.group(1).replace(" ", "")
            try:
                return int(raw_value)
            except ValueError:
                return None

        return None
        
    def check_format(self, generated_text: str) -> float:
        """Reward if output follows <think>...</think> Final Result: format."""
        if re.search(r"<think>.*?</think>\s*Final Result:", generated_text, re.DOTALL | re.IGNORECASE):
            return self.format_weight
        return 0.0

    def check_length_penalty(self, generated_text: str) -> float:
        """Penalize extremely long generations based on character count."""
        return - (self.length_penalty_weight * len(generated_text))

    def check_equation_steps(self, generated_text: str) -> float:
        """Parse intermediate steps and reward valid logic, penalize invalid logic."""
        reward = 0.0
        # Match "Step X: A + B = C"
        step_matches = re.finditer(r"Step\s*\d+:\s*([+-]?\d+)\s*([+-])\s*([+-]?\d+)\s*=\s*([+-]?\d+)", generated_text, re.IGNORECASE)
        for match in step_matches:
            try:
                A = int(match.group(1).replace(" ", ""))
                op = match.group(2).strip()
                B = int(match.group(3).replace(" ", ""))
                C_model = int(match.group(4).replace(" ", ""))
                
                C_true = (A + B) if op == '+' else (A - B)
                if C_model == C_true:
                    reward += self.step_weight
                else:
                    reward -= self.step_weight * 2.0  # Double penalty for logically false steps
            except ValueError:
                pass
        return reward

    def compute_reward(self, generated_text: str, ground_truth: int) -> float:
        """Compute reward for generated response dynamically incorporating baseline and granular features.

        Args:
            generated_text: Generated solution text
            ground_truth: Correct answer

        Returns:
            Reward value representing total score (base + components)
        """
        result = self.extract_final_result(generated_text)
        
        # Base reward for absolute correctness
        total_reward = 1.0 if result == ground_truth else 0.0
        
        # Add granular metrics
        if self.reward_format:
            total_reward += self.check_format(generated_text)
            
        if self.reward_length_penalty:
            total_reward += self.check_length_penalty(generated_text)
            
        if self.reward_equation_steps:
            total_reward += self.check_equation_steps(generated_text)
            
        return total_reward
