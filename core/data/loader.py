"""Data loading utilities for arithmetic LLM training."""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from typing import List, Tuple, Optional, Iterator, Union
from core.data.tokenizer import (
    ArithmeticBPETokenizer, ArithmeticDigitTokenizer
)
from core.eval.evaluator import eval_expression
from core.training.contrastive import make_wrong_solution

class CurriculumSampler(Sampler):
    """
    Samples elements progressively from easier to harder based on a curriculum schedule.
    """
    def __init__(self, dataset_complexities: List[float], batch_size: int, total_steps: int = 10000):
        self.complexities = np.array(dataset_complexities)
        self.num_samples = len(self.complexities)
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.current_step = 0
        
        # Normalize complexities to 0-1 for easier weighting
        min_c = self.complexities.min()
        max_c = self.complexities.max()
        if max_c > min_c:
            self.norm_complexities = (self.complexities - min_c) / (max_c - min_c)
        else:
            self.norm_complexities = np.zeros_like(self.complexities)
            
    def step(self, steps: int = 1):
        """Advance the curriculum step."""
        self.current_step = min(self.total_steps, self.current_step + steps)
        
    def __iter__(self) -> Iterator[int]:
         # Yield indices iteratively, updating weights based on the current step.
        # We calculate weights once per batch to avoid massive overhead.
        for i in range(0, self.num_samples, self.batch_size):
            progress = min(1.0, self.current_step / max(1, self.total_steps))
            batch_size_current = min(self.batch_size, self.num_samples - i)
            
            if progress >= 1.0:
                batch_indices = np.random.choice(self.num_samples, size=batch_size_current, replace=True).tolist()
            else:
                target_c = progress
                sigma = 0.1 + (progress * 0.4) 
                
                dist = np.abs(self.norm_complexities - target_c)
                weights = np.exp(-(dist**2) / (2 * sigma**2))
                
                weights = np.clip(weights, a_min=1e-5, a_max=None)
                weights = weights / weights.sum()
                
                # IMPORTANT: We MUST use replace=True.
                # Using replace=False on a massive array means items with lower weights 
                # are forced to the end of the list with 100% probability, causing the 
                # end of the epoch to consist entirely of the 'wrong' complexities.
                batch_indices = np.random.choice(
                    self.num_samples, 
                    size=batch_size_current, 
                    replace=True, 
                    p=weights
                ).tolist()
                
            for idx in batch_indices:
                yield idx

    def __len__(self) -> int:
        return self.num_samples

class ArithmeticDataset(Dataset):
    """Dataset for arithmetic expressions and evaluations.
    
    Foundational mode loads plain text (one instance per line).
    Instruction mode loads JSONL entries for training.
    """
     
    def __init__(
        self,
        corpus_path: str,
        tokenizer: Union[ArithmeticBPETokenizer, ArithmeticDigitTokenizer],
        max_length: int = 512,
        mode: str = "foundational",
        use_contrastive: bool = False,
    ):
        """Initialize dataset.
        
        Args:
            corpus_path: Path to corpus file (text for foundational, JSONL for instruction)
            tokenizer: Trained tokenizer instance
            max_length: Maximum sequence length
            mode: Training mode - "foundational" or "instruction"
            use_contrastive: If True (instruction only), __getitem__ returns wrong_* for contrastive learning
        """
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.use_contrastive = use_contrastive and (mode == "instruction")
        self.entries = []
        self.prompt_lengths = []  # Store prompt lengths for instruction mode
        self.complexities = []
        self.prompts: List[str] = []
        self.solutions: List[str] = []
        self.answers: List[int] = []
        
        # Load corpus
        self._load_corpus()

    def get_instruction_pairs(self, validate_expressions: bool = False) -> List[dict]:
        """Return instruction prompts and ground-truth answers.

        Each item has keys: 'prompt' and 'ground_truth'. This uses the
        instruction JSONL entries loaded in the dataset.
        """
        pairs: List[dict] = []
        if self.mode != "instruction":
            return pairs

        with open(self.corpus_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                problem = entry.get("problem")
                answer = entry.get("answer")
                if problem is None or answer is None:
                    continue
                if answer == "ERROR":
                    continue

                prompt = problem + " <think>"
                try:
                    ground_truth = int(answer)
                except (TypeError, ValueError):
                    continue

                if validate_expressions:
                    expression = entry.get("expression")
                    if not expression and isinstance(problem, str):
                        prefix = "Evaluate:"
                        if problem.startswith(prefix):
                            expression = problem[len(prefix):].strip()
                    if not expression:
                        continue
                    expected = eval_expression(expression).get("answer")
                    if expected == "ERROR":
                        continue
                    try:
                        expected_value = int(expected)
                    except (TypeError, ValueError):
                        continue
                    if expected_value != ground_truth:
                        continue

                pairs.append({"prompt": prompt, "ground_truth": ground_truth})

        return pairs
    
    def _load_corpus(self) -> None:
        """Load corpus.
        
        Foundational mode: treat each non-empty line as a standalone text instance.
        Instruction mode: each line is a JSON object with keys:
        - expression: The arithmetic expression
        - problem: "Evaluate: [expression]"
        - solution: The full solution with <think> tags
        - answer: The final numeric answer or "ERROR"
        """
        with open(self.corpus_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if self.mode == "foundational":
                    # Foundational: raw text per line
                    self.entries.append(line)
                    self.prompt_lengths.append(0)
                    self.complexities.append(float(len(line))) # Use raw text length as proxy
                    continue

                try:
                    entry = json.loads(line)

                    # Instruction mode formatting
                    # prompt is problem + "<think>", target is solution
                    prompt = entry['problem'] + ' <think>'
                    target = entry['solution']

                    # Strip leading '<think>' from target if present
                    target = target.strip()
                    if target.startswith('<think>'):
                        # Remove it so we don't duplicate it with the prompt's <think>
                        target = target[len('<think>'):].lstrip()

                    # Concatenate prompt and target for training
                    text = prompt + ' ' + target

                    # Calculate prompt length in tokens (without BOS/EOS)
                    prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                    prompt_length = len(prompt_tokens)

                    self.entries.append(text)
                    self.prompt_lengths.append(prompt_length)
                    self.complexities.append(float(len(text))) # Total sequence length as complexity proxy
                    try:
                        ans = entry['answer']
                        self.prompts.append(prompt)
                        self.solutions.append(entry['solution'])
                        self.answers.append(int(ans) if ans != "ERROR" else 0)
                    except (TypeError, ValueError, KeyError):
                        self.prompts.append(prompt)
                        self.solutions.append(entry['solution'])
                        self.answers.append(0)

                except json.JSONDecodeError:
                    # Skip malformed JSON lines in instruction mode
                    continue
    
    def __len__(self) -> int:
        """Return number of entries in dataset."""
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> dict:
        """Get tokenized entry at index.
        
        Args:
            idx: Index of entry
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask', 'length', and 'prompt_length'
        """
        entry = self.entries[idx]
        prompt_length = self.prompt_lengths[idx]
        
        # Tokenize entry (BOS and EOS will be added automatically)
        token_ids = self.tokenizer.encode(entry, add_special_tokens=True)
        
        # Adjust prompt_length to account for BOS token
        # BOS is at position 0, so all positions shift by 1
        if prompt_length > 0:
            prompt_length += 1  # Account for BOS token
        
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            # Adjust prompt_length if truncated
            if prompt_length > self.max_length:
                prompt_length = self.max_length
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        out = {
            'input_ids': token_ids, # prompt + solution
            'attention_mask': attention_mask,
            'length': len(token_ids),
            'prompt_length': prompt_length
        }
        
        if self.use_contrastive and idx < len(self.prompts):
            prompt = self.prompts[idx]
            solution = self.solutions[idx]
            answer = self.answers[idx]
            wrong_solution = make_wrong_solution(solution, answer, seed=idx)
            full_wrong = prompt + ' ' + wrong_solution
            wrong_ids = self.tokenizer.encode(full_wrong, add_special_tokens=True)
            if len(wrong_ids) > self.max_length:
                wrong_ids = wrong_ids[:self.max_length]
            wrong_plen = prompt_length  # same as correct (BOS + prompt)
            # Same convention as correct: labels[i] = token at position i (masked for prompt);
            # training uses targets = labels[:, 1:], so target at position t is labels[t+1] = ids[t+1]
            wrong_labels = [-100] * wrong_plen + wrong_ids[wrong_plen:]
            out['wrong_input_ids'] = wrong_ids
            out['wrong_attention_mask'] = [1] * len(wrong_ids)
            out['wrong_length'] = len(wrong_ids)
            out['wrong_labels'] = wrong_labels
        return out


def collate_fn(
    batch: List[dict],
    pad_token_id: int = 0,
    mode: str = "foundational",
    use_contrastive: bool = False,
) -> Tuple:
    """Collate function for batching with padding.
    
    Args:
        batch: List of dictionaries from dataset
        pad_token_id: Token ID to use for padding
        mode: Training mode - "foundational" or "instruction"
        use_contrastive: If True and batch has wrong_*, return wrong tensors too
        
    Returns:
        Tuple of (input_ids, attention_mask, labels) tensors;
        if use_contrastive and batch has wrong_*, then
        (input_ids, attention_mask, labels, wrong_input_ids, wrong_attention_mask, wrong_labels)
        For instruction mode, labels have -100 for prompt tokens (ignored in loss)
    """
    # Find maximum length in batch
    max_length = max(item['length'] for item in batch)
    
    # Pad sequences
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        # Pad input_ids
        padded_ids = item['input_ids'] + [pad_token_id] * (
            max_length - item['length']
        )
        input_ids.append(padded_ids)
        
        # Pad attention_mask
        padded_mask = item['attention_mask'] + [0] * (
            max_length - item['length']
        )
        attention_masks.append(padded_mask)
        
        # Create labels for loss computation
        if mode == "instruction":
            # For instruction mode: mask prompt tokens with -100
            prompt_length = item['prompt_length']
            # Labels are shifted by 1 (predict next token)
            # Mask prompt tokens (including the shift)
            label_ids = [-100] * prompt_length + item['input_ids'][prompt_length:]
            # Pad labels
            label_ids = label_ids + [-100] * (max_length - item['length'])
        else:
            # For foundational mode: use all tokens
            label_ids = item['input_ids'] + [-100] * (max_length - item['length'])
        
        labels.append(label_ids)
    
    # Convert to tensors
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    attention_masks_t = torch.tensor(attention_masks, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    
    if use_contrastive and "wrong_input_ids" in batch[0]:
        max_wrong = max(item["wrong_length"] for item in batch)
        wrong_input_ids = []
        wrong_attention_masks = []
        wrong_labels = []
        for item in batch:
            wlen = item["wrong_length"]
            wrong_input_ids.append(
                item["wrong_input_ids"] + [pad_token_id] * (max_wrong - wlen)
            )
            wrong_attention_masks.append(
                item["wrong_attention_mask"] + [0] * (max_wrong - wlen)
            )
            wl = item["wrong_labels"] + [-100] * (max_wrong - wlen)
            wrong_labels.append(wl)
        wrong_input_ids_t = torch.tensor(wrong_input_ids, dtype=torch.long)
        wrong_attention_masks_t = torch.tensor(wrong_attention_masks, dtype=torch.long)
        wrong_labels_t = torch.tensor(wrong_labels, dtype=torch.long)
        return (
            input_ids_t,
            attention_masks_t,
            labels_t,
            wrong_input_ids_t,
            wrong_attention_masks_t,
            wrong_labels_t,
        )
    return input_ids_t, attention_masks_t, labels_t


def create_dataloaders(
    corpus_path: str,
    tokenizer: Union[ArithmeticBPETokenizer, ArithmeticDigitTokenizer],
    batch_size: int = 32,
    max_length: int = 512,
    train_split: float = 0.9,
    shuffle: bool = True,
    num_workers: int = 4,  # Increased default for parallelism
    mode: str = "foundational",
    use_curriculum: bool = False,
    curriculum_steps: int = 10000,
    use_contrastive: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[CurriculumSampler]]:
    """Create train and validation dataloaders.
    
    Args:
        corpus_path: Path to corpus file (text for foundational, JSONL for instruction)
        tokenizer: Trained tokenizer instance
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle training data
        num_workers: Number of worker processes for data loading
        mode: Training mode - "foundational" or "instruction"
        use_curriculum: If True, uses the CurriculumSampler for the training set.
        curriculum_steps: Total steps over which curriculum anneals to uniform random.
        use_contrastive: If True and batch has wrong_*, return wrong tensors too

    Returns:
        Tuple of (train_dataloader, val_dataloader, train_sampler)
    """
    # Create full dataset
    full_dataset = ArithmeticDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mode=mode,
        use_contrastive=use_contrastive,
    )
    
    # Split into train and validation
    # Ensure at least 1 sample in each split if dataset is large enough
    dataset_size = len(full_dataset)
    
    if dataset_size == 0:
        raise ValueError("Dataset is empty")
    
    if dataset_size == 1:
        # If only 1 sample, use it for both train and val
        train_size = 1
        val_size = 1
        train_dataset = full_dataset
        val_dataset = full_dataset
        # For a dataset of 1, split makes lengths tricky, so manually extract complexities
        train_complexities = full_dataset.complexities
    else:
        # Ensure at least 1 sample in validation set
        train_size = max(1, int(dataset_size * train_split))
        val_size = max(1, dataset_size - train_size)
        
        # Adjust train_size if needed to ensure both sets have at least 1 sample
        if train_size + val_size > dataset_size:
            train_size = dataset_size - 1
            val_size = 1
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size]
        )
        # Extract complexities using Subset indices to ensure mapping is correct
        train_complexities = [full_dataset.complexities[idx] for idx in train_dataset.indices]
    
    # Get pad token ID
    pad_token_id = tokenizer.token2id.get('<pad>', 0)
    
    # Create collate function with pad token ID, mode, and contrastive flag
    def collate_with_pad(batch):
        return collate_fn(
            batch,
            pad_token_id=pad_token_id,
            mode=mode,
            use_contrastive=use_contrastive,
        )
    
     # Setup sampler
    train_sampler = None
    if use_curriculum and shuffle:
        train_sampler = CurriculumSampler(
            dataset_complexities=train_complexities,
            batch_size=batch_size,
            total_steps=curriculum_steps
        )
        # DataLoader shouldn't use shuffle=True if it has a custom sampler
        shuffle = False
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_with_pad
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_pad
    )
    
    return train_dataloader, val_dataloader, train_sampler


