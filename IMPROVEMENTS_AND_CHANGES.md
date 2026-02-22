# Arithmetic LLM: Technical Improvements and Research Log

## Phase: Curriculum Learning, GRPO, and Interpretability

This document serves as an ongoing technical report detailing the architectural evolution, training strategies, and interpretability mechanisms introduced to elevate the Arithmetic LLM from a simple pattern matcher to a robust logical reasoning engine.

### 1. Scaling the Dataset & Multithreading Optimization (Curriculum Learning)
*Status: Completed*

**Motivation & Chinchilla Scaling:**
Large Language Models exhibit predictable scaling laws. A pivotal finding from the **Chinchilla paper (Hoffmann et al., 2022)** "Training Compute-Optimal Large Language Models" is that models are often significantly undertrained rather than under-parameterized. The optimal compute/data balance suggests training on approximately 20 tokens per model parameter.

Given our core model operates with **4,940,032 parameters**, our target dataset size was meticulously scaled to approximately **98.8 million tokens**.

**Implementation Updates:**
Instead of manually guessing the `num-samples` required to hit this threshold, we retrofitted the data generation pipeline (`generate_instruction_corpus_mixed.py`, `generate_foundational_plaintext.py`) to actively tokenize mathematical expressions during generation. 
The generation scripts were upgraded to utilize multiprocessing (`concurrent.futures`), achieving dramatic speedups for million-scale token generation.
Furthermore, a custom `CurriculumSampler` was built into the `core/data/loader.py`, allowing the PyTorch `DataLoader` to dynamically implement Curriculum Learning by annealing the sampling distribution from simple to complex curricula over the course of training.

**AST Generation Robustness:**
To ensure the curriculum is meaningful from step 0, we identified and patched an edge case in the underlying `ExpressionGenerator` Abstract Syntax Tree (AST) logic where zero-depth recursions could randomly terminate into single integers, forcing the model to occasionally train on trivial echoes (e.g. evaluating `14` into `14`). 

*Before Logic (Allowed Trivial Echoes):*
```python
def generate(self, current_depth=0):
    # If we reached max depth or a random chance, return a number
    if current_depth >= self.max_depth or random.random() < 0.3:
        return str(random.randint(*self.num_range))
    # ...
```

*After Logic (Enforces $\ge 1$ Operation):*
```python
def generate(self, current_depth=0):
    # Require current_depth > 0 before allowing a random short-circuit
    if current_depth >= self.max_depth or (current_depth > 0 and random.random() < 0.3):
        return str(random.randint(*self.num_range))
    # ...
```

### 2. Group Relative Policy Optimization (GRPO)
*Status: Planned*

**The Theoretical Shift:**
Supervised Fine-Tuning (SFT) often leads to models blindly memorizing dataset templates. To instill genuine logical deduction, we are adopting Group Relative Policy Optimization (GRPO), a technique recently proven highly effective in the **DeepSeekMath paper (Shao et al., 2024)**.

GRPO represents a significant algorithmic leap over standard Proximal Policy Optimization (PPO). Standard PPO requires maintaining a separate "Critic" model (often the same size as the policy model itself) to estimate the value baseline of a given state, effectively doubling the VRAM requirement. GRPO eliminates this constraint by deriving a baseline from within the generated group itself.

**The Self-Taught Critic & Reward Sculpting:**
For a given arithmetic query, the model will sample a group of $G$ diverse reasoning paths. We will actively sculpt the model's logic through dense, contrastive rewards:
1.  **Step-wise Verification:** Rather than a sparse "1 or 0" depending on if the final answer is right, we parse the logic chain. The model is penalized precisely at the mathematical junction where it deviates from valid rules.
2.  **Occam's Razor Penalty:** A proportional length penalty is applied. This aggressively curbs the common LLM failure mode of rambling verbose loops, forcing the most direct and mathematically sound proof.

By comparing the $G$ responses against each other using these reward signals, the network isolates its own logical fallacies.

#### 2.1 Design of the Ablation Study

An ablation study systematically disables components of a system to understand their individual contribution to overall performance. To analyze the exact causal relationship between these granular rewards and the emergence of robust logic versus "nonsense", we added `--reward-format`, `--reward-length-penalty`, and `--reward-equation-steps` toggles into the `scripts/train/grpo.py` CLI to support the following configurations:

| Configuration | Baseline (Binary) | Format Reward | Length Penalty | Step-wise Verify | Expected Insight |
|---------------|:---:|:---:|:---:|:---:|---|
| **Baseline** | ✅ | ❌ | ❌ | ❌ | Establishes base accuracy & verbosity. |
| **Experiment 1** | ✅ | ✅ | ❌ | ❌ | Does early structural reinforcement speed up convergence? |
| **Experiment 2** | ✅ | ✅ | ✅ | ❌ | Does length penalty reduce generation time without degrading accuracy? |
| **Experiment 3** | ✅ | ✅ | ❌ | ✅ | Does step-verification eliminate mathematically invalid "hallucinated" proofs? |
| **Experiment 4 (Full)** | ✅ | ✅ | ✅ | ✅ | The theoretical optimal setup. |

##### Evaluation Metrics for Analysis
During evaluation, we will track not only traditional metrics but also:
1. **Average Generation Length**: Used to prove the length penalty's effectiveness.
2. **Formatting Failure Rate**: Tracks how often the model fails to produce the `<think>` block.
3. **Logic Deviation Rate**: The proportion of generated solutions where the final result is correct, but an intermediate math step was technically false (a critical measure of robust vs rote memorization).

### 3. Checkpointing Flexibility
*Status: Completed*

To maximize disk efficiency while preserving critical milestones, model persistence is now parameterized across all training environments:
- **Pretraining**: Checkpoints are strictly keyed to occur every **25,000 steps** by default.
- **Fine-Tuning/LoRA**: Checkpoints save every **1,000 steps** to capture rapid policy shifts during RLHF and instructional tuning.
This is fully configurable via the `--save-steps` argument linked dynamically to the underlying `TrainingConfig`.

### 4. The "Mind Reader" Attention Visualizer
*Status: Completed*

**Implementation:**
To mathematically prove the model isn't "cheating" by relying on dataset bias, we implemented a "Mind Reader" toolset (`scripts/evaluate/visualize_attention.py`).
We modified the core `ArithmeticTransformer`'s forward pass to expose its internal multi-head attention weight matrices ($A = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})$) explicitly via an `output_attentions` flag.

By rendering these matrices as a 2D heatmap utilizing Seaborn/Matplotlib during generation, we visually trace token-to-token cross and self-attention. This explicitly proves the model's attention heads are correctly locked onto the target operands in the prompt while sequentially evaluating logic chains.
