# Arithmetic LLM: Technical Improvements and Research Log

## Phase: Curriculum Learning, GRPO, and Interpretability

This document serves as an ongoing technical report detailing the architectural evolution, training strategies, and interpretability mechanisms introduced to elevate the Arithmetic LLM from a simple pattern matcher to a robust logical reasoning engine.

### 1. Scaling the Dataset & Multithreading Optimization (Curriculum Learning)
*Status: Planned*

**Motivation & Chinchilla Scaling:**
Large Language Models exhibit predictable scaling laws. A pivotal finding from the **Chinchilla paper (Hoffmann et al., 2022)** "Training Compute-Optimal Large Language Models" is that models are often significantly undertrained rather than under-parameterized. The optimal compute/data balance suggests training on approximately 20 tokens per model parameter.

Given our core model operates with **4,940,032 parameters**, our target dataset size must scale to approximately **98.8 million tokens**.

Instead of manually guessing the `num-samples` required to hit this threshold, we are retrofitting the data generation pipeline (`generate_instruction_corpus_mixed.py`, `generate_foundational_plaintext.py`) to actively tokenize mathematical expressions during generation. The generator will continually synthesize data across varying levels of arithmetic complexity (e.g., Level 1 simple addition to Level N multi-step operations) until the 100M token threshold is explicitly met.

**Performance Bottlenecks:**
To handle the generation of ~100M tokens efficiently, the generation scripts are being upgraded to utilize multiprocessing (`concurrent.futures`). Furthermore, the PyTorch `DataLoader` will be optimized with multithreaded batch preparation (`num_workers > 0`) to ensure the GPU is never bottlenecked waiting for data. This dataloader will dynamically implement Curriculum Learning, annealing the sampling distribution from simple to complex curricula over the course of training.

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

### 3. Checkpointing Flexibility
*Status: Planned*

To maximize disk efficiency while preserving critical milestones, model persistence is being parameterized:
- **Pretraining**: Given the massive volume of tokens (100M), checkpoints will strictly occur every **25,000 steps** by default.
- **Fine-Tuning/LoRA**: Checkpoints will be saved every **1,000 steps** to capture rapid policy shifts during RLHF and instructional tuning.

### 4. The "Mind Reader" Attention Visualizer
*Status: Planned*

**Motivation:**
How do we mathematically prove the model isn't "cheating" by relying on dataset bias? Enter the "Mind Reader".

We will modify the core Transformer's forward pass to expose its internal multi-head attention weight matrices ($A = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})$). By rendering these matrices as a 2D heatmap during inference, we can visually trace token-to-token cross and self-attention. When the model generates the mathematical logic "10 - 5 = 5", the visualization will explicitly prove its attention heads are locked onto the specific target operands in the prompt.

*To be updated as implementation progresses...*
