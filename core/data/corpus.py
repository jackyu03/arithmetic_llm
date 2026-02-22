"""Corpus generation module for arithmetic LLM training."""

import json
import concurrent.futures
import multiprocessing
from typing import Tuple, List, Dict
from core.inference.generator import ExpressionGenerator
from core.eval.evaluator import eval_expression

# Attempt to load a tokenizer for accurate token counting
try:
    from transformers import AutoTokenizer
    # GPT-2 tokenizer is a fast, good proxy for standard BPE LLM tokens
    _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
except ImportError:
    _TOKENIZER = None

def _count_tokens(text: str) -> int:
    """Helper to count tokens in a string."""
    if _TOKENIZER is not None:
        # Avoid creating the attention mask/tensors for speed
        return len(_TOKENIZER.encode(text, add_special_tokens=False))
    # Fallback to simple split if transformers isn't available
    return len(text.split())

def _generate_chunk(chunk_size: int, max_depth: int, num_range: Tuple[int, int], invalid_rate: float) -> Tuple[List[Dict], int]:
    """Generates a chunk of evaluate expressions and returns the entries and their total token count."""
    generator = ExpressionGenerator(
        max_depth=max_depth,
        num_range=num_range,
        invalid_rate=invalid_rate
    )
    entries = []
    token_count = 0
    
    for _ in range(chunk_size):
        expression = generator.generate()
        result = eval_expression(expression)
        
        entry = {
            'expression': result['expression'],
            'problem': result['problem'],
            'solution': result['solution'],
            'answer': result['answer']
        }
        
        # Count tokens for the problem and solution combined
        text_to_count = str(entry['problem']) + " " + str(entry['solution'])
        token_count += _count_tokens(text_to_count)
        entries.append(entry)
        
    return entries, token_count

class CorpusGenerator:
    """Generates training corpus of arithmetic expressions and evaluations using target token counts."""
    
    def __init__(
        self,
        target_tokens: int = None,
        num_samples: int = None, # kept for backward compatibility if needed temporarily
        max_depth: int = 5,
        num_range: Tuple[int, int] = (1, 20),
        invalid_rate: float = 0.1,
        output_path: str = "corpus.txt"
    ):
        """
        Initialize corpus generator.
        
        Args:
            target_tokens: Target total number of tokens to generate.
            num_samples: Legacy parameter (overridden if target_tokens is provided).
            max_depth: Maximum depth of expression trees
            num_range: Range of numbers to use in expressions
            invalid_rate: Fraction of invalid expressions to include
            output_path: Path to save generated corpus
        """
        self.target_tokens = target_tokens
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.num_range = num_range
        self.invalid_rate = invalid_rate
        self.output_path = output_path
        
        # Ensure at least one goal is set
        if self.target_tokens is None and self.num_samples is None:
            raise ValueError("Must provide either target_tokens or num_samples")

    def _generate_parallel(self, file_path: str) -> None:
        """Internal method to generate corpus using multiprocessing."""
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        # Choose a chunk size that isn't too small (overhead) or too large (overshoot target too much)
        chunk_samples = 1000 
        
        total_tokens = 0
        total_samples = 0
        
        with open(file_path, 'w') as f:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # We submit batches of tasks
                active_futures = set()
                
                # Initial fill of the worker pool
                for _ in range(num_workers * 2):
                    active_futures.add(executor.submit(
                        _generate_chunk, chunk_samples, self.max_depth, self.num_range, self.invalid_rate
                    ))
                
                while active_futures:
                    done, active_futures = concurrent.futures.wait(
                        active_futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        entries, chunk_tokens = future.result()
                        total_tokens += chunk_tokens
                        total_samples += len(entries)
                        
                        # Write the chunk to disk
                        for entry in entries:
                            f.write(json.dumps(entry) + '\n')
                            
                        # Check if we met our goal
                        goal_met = False
                        if self.target_tokens is not None and total_tokens >= self.target_tokens:
                            goal_met = True
                        elif self.num_samples is not None and total_samples >= self.num_samples:
                            goal_met = True
                            
                        # If we haven't met our goal, submit another job
                        if not goal_met:
                            active_futures.add(executor.submit(
                                _generate_chunk, chunk_samples, self.max_depth, self.num_range, self.invalid_rate
                            ))
                        else:
                            # Cancel remaining futures if goal met
                            for pending in active_futures:
                                pending.cancel()
                            active_futures.clear()
                            break

    def generate_corpus(self) -> None:
        """Generate foundational training corpus and save to disk in JSONL format."""
        self._generate_parallel(self.output_path)
    
    def generate_instruction_corpus(self, output_path: str) -> None:
        """Generate instruction-formatted corpus for fine-tuning in JSONL format."""
        self._generate_parallel(output_path)


