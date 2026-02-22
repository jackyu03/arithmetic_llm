#!/usr/bin/env python3
"""
Untracked test script to demonstrate Curriculum Learning sampling distributions
using the project's authentic Abstract Syntax Tree expression generator.
"""

import numpy as np
import collections
from core.data.loader import CurriculumSampler
from core.inference.generator import ExpressionGenerator
from core.eval.evaluator import eval_expression

def main():
    print("=" * 80)
    print("1. GENERATING AUTHENTIC AST TRAINING DATA (MICRO-SAMPLE)")
    print("=" * 80)
    
    # Instantiate the real AST generator used during corpus building
    generator = ExpressionGenerator(max_depth=4, num_range=(1, 50), invalid_rate=0.0)
    
    # Generate a small pool of 20 authentic expressions
    num_samples = 20
    dataset_prompts = []
    complexities = []
    
    # We generate a variety of depths manually to guarantee a spread of difficulty
    for _ in range(num_samples):
        # We enforce a random mix of depths to demonstrate the curriculum spread
        target_depth = np.random.choice([1, 2, 3, 4])
        generator.max_depth = target_depth
        
        expr = generator.generate()
        result = eval_expression(expr)
        
        # Instruction format
        prompt = result['problem'] + " <think> " + result['solution']
        
        # Foundational format (the "original")
        foundational = result['expression'] + " = " + str(result['answer'])
        
        dataset_prompts.append((foundational, prompt))
        
        # In `loader.py`, complexity is proxied by length
        complexities.append(float(len(prompt)))

    # Sort the generated pool strictly by complexity/length
    sorted_pairs = sorted(zip(complexities, dataset_prompts), key=lambda x: x[0])
    complexities = [c for c, p in sorted_pairs]
    dataset_prompts = [p for c, p in sorted_pairs]

    print("\n[Real Training Examples Ranked by Complexity (Length)]\n")
    for i, (comp, formats) in enumerate(zip(complexities, dataset_prompts)):
        foundational, prompt = formats
        
        # Truncate prompt for terminal readability
        display_prompt = prompt if len(prompt) < 80 else prompt[:77] + "..."
        display_foundational = foundational if len(foundational) < 60 else foundational[:57] + "..."
        
        print(f"Index {i:02d} | Length: {int(comp):03d}")
        print(f"  -> Foundational: {display_foundational}")
        print(f"  -> Instruction : {display_prompt}\n")

    print("\n" + "=" * 80)
    print("2. BASELINE: UNIFORM RANDOM SAMPLING (No Curriculum)")
    print("=" * 80)
    
    # Standard PyTorch RandomSampler is uniform
    simulation_draws = 1000
    uniform_weights = np.ones(num_samples) / num_samples
    uniform_indices = np.random.choice(num_samples, size=simulation_draws, replace=True, p=uniform_weights)
    uniform_counts = collections.Counter(uniform_indices)
    
    for i in range(num_samples):
        pct = (uniform_counts[i] / simulation_draws) * 100
        print(f"Index {i:02d} [Ln {int(complexities[i]):03d}]: {pct:.1f}% selection probability")

    print("\n" + "=" * 80)
    print("3. CURRICULUM SAMPLING: PROGRESSIVE DIFFICULTY ANNEALING")
    print("=" * 80)

    total_curriculum_steps = 10000
    sampler = CurriculumSampler(
        dataset_complexities=complexities,
        batch_size=2,
        total_steps=total_curriculum_steps
    )

    # Let's observe the distribution at 0%, 50%, and 100% of training
    stages = [0, 5000, 10000]
    
    for step in stages:
        sampler.current_step = step
        progress_pct = (step / total_curriculum_steps) * 100
        print(f"\n--- Training Stage: {int(progress_pct)}% (Step {step}/{total_curriculum_steps}) ---")
        
        # We manually peek into the CurriculumSampler's probability distribution generator
        progress = min(1.0, sampler.current_step / max(1, sampler.total_steps))
        
        if progress >= 1.0:
            weights = np.ones(sampler.num_samples)
        else:
            target_c = progress
            sigma = 0.1 + (progress * 0.4)
            dist = np.abs(sampler.norm_complexities - target_c)
            weights = np.exp(-(dist**2) / (2 * sigma**2))
            weights = np.clip(weights, a_min=0.01, a_max=None)
            
        weights = weights / weights.sum()
        
        sampled_indices = np.random.choice(num_samples, size=simulation_draws, replace=True, p=weights)
        counts = collections.Counter(sampled_indices)
        
            # Display the probabilities and prompt an example of what the model actually receives
        for i in range(num_samples):
            pct = (counts[i] / simulation_draws) * 100
            
            _, display_prompt = dataset_prompts[i]
            display_str = display_prompt if len(display_prompt) < 60 else display_prompt[:57] + "..."
            
            # Draw attention to highly favored samples
            focus_marker = "<-- HIGH DRAW PROBABILITY" if pct > (100.0 / num_samples) * 2.0 else ""
            
            print(f"idx {i:02d} [Ln {int(complexities[i]):03d}]: {pct:>4.1f}% | Ex: {display_str} {focus_marker}")

if __name__ == "__main__":
    main()
