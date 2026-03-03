```bash

# 1. Generate training corpus (100,000 samples recommended)

# Generate foundational training corpus with 100K samples (plain text)
# This large corpus provides the base model with extensive arithmetic patterns
python scripts/data/generate_foundational_plaintext.py --target-tokens 100000000 --output-txt data/foundational_corpus.txt

# Generate mixed instruction corpus (valid + invalid)
# This creates a balanced dataset without writing intermediate files
python scripts/data/generate_instruction_corpus_mixed.py --target-tokens 10000000 --max-depth 5 --num-range 1 20 --invalid-rate 0 --output-mixed data/instruction_corpus.txt

# Generate separate test corpus for evaluation (10K samples, minimal errors)
# This provides a clean test set with only 1% invalid expressions
python scripts/data/generate_corpus.py --instruction-only --num-samples 1000 --max-depth 4 --output-instruction data/instruction_corpus_test.txt --num-range 1 20 --invalid-rate 0

#check line counts
python -c "import sys; [print(f'{sum(1 for _ in open(f))} {f}') for f in ['data/foundational_corpus.txt', 'data/instruction_corpus.txt', 'data/instruction_corpus_test.txt']]"

# 2. Train tokenizer
python scripts/data/train_tokenizer.py --corpus-path data/foundational_corpus.txt --output-dir data/tokenizer --tokenizer-type digit

# show tokenizer table
python scripts/utils/print_token_table.py --tokenizer_path data/tokenizer/tokenizer.pkl  > tokens.csv

# Analyze your instruction corpus
python scripts/utils/check_sequence_lengths.py --corpus-path data/instruction_corpus.txt --tokenizer-path data/tokenizer

# 3. Train foundational model
python scripts/train/foundational.py --corpus-path data/foundational_corpus.txt --output-dir models/foundational --tokenizer-path data/tokenizer --wandb

# 3.1 Evaluate the foundational model, performance would be bad
python scripts/eval/evaluate.py --model-path models/foundational/best_model.pt --tokenizer-path data/tokenizer --tokenizer-type digit --num-samples 1000


# 4. Fine-tune instruction model
python scripts/train/instruction.py --instruction-corpus-path data/instruction_corpus.txt --output-dir models/ --tokenizer-path data/tokenizer --foundational-checkpoint models/foundational/best_model.pt --contrastive --contrastive-weight 0.3 --contrastive-temperature 0.05 --wandb

# 4.1 Evaluate the model
python scripts/eval/evaluate.py --model-path models/instruction_20260220_122251_979267/best_model.pt --tokenizer-path data/tokenizer --max-gen-length 512 --batch-size 1 --num-samples 1000 --constrain-decoding

# 5 Fine-tune with LoRA adapters (optional)
python scripts/train/instruction_lora.py --instruction-corpus-path data/instruction_corpus.txt --output-dir models/ --tokenizer-path data/tokenizer --foundational-checkpoint models/foundational_20260220_103758_473709/best_model.pt --num-epochs 10 --lora-rank 8 --lora-alpha 16 --lora-target-modules attention --save-merged-model

# 5.1 Evaluate the LoRA merged model (optional)
python scripts/eval/evaluate.py --model-path models/instruction_lora_20260220_131932_609000/merged_model.pt --tokenizer-path data/tokenizer --max-gen-length 512 --batch-size 1 --num-samples 1000

# 6 GRPO training (optional)
python scripts/train/grpo.py --tokenizer data/tokenizer --sft-checkpoint models/instruction_20260221_044738_553100/best_model.pt --output-dir models/grpo --data-mode generated --log-every 1 --num-samples 1024 --num-epochs 3 --num-candidates 8 --max-gen-length 511 --temperature 0.8 --batch-size 1 --gradient-accumulation-steps 16 --kl-penalty-coef 0.05
 

# 6.1 eval GRPO model
python scripts/eval/evaluate.py   --model-path models/grpo/grpo_20260223_034718_367216/final_model.pt    --tokenizer-path data/tokenizer   --max-gen-length 512   --batch-size 1   --num-samples 1000



# LoRA sweep only (ranks 2, 4, 8, 16), 500 eval samples, 3 epochs:
python scripts/train/lora_sweep.py --instruction-corpus-path data/instruction_corpus.txt --tokenizer-path data/tokenizer_digit --foundational-checkpoint models/foundational/best_model.pt --output-dir lora_sweep_results --lora-ranks 2 4 8 16 --num-epochs 3 --num-eval-samples 500


# Include full instruction baseline (train + evaluate):
python scripts/train/lora_sweep.py --instruction-corpus-path data/instruction_corpus.txt --tokenizer-path data/tokenizer_digit --foundational-checkpoint models/foundational/best_model.pt --output-dir lora_sweep_results --run-full-instruction --lora-ranks 2 4 8 16 32 64 --num-epochs 3 --num-eval-samples 500

# Use an existing full-instruction checkpoint as baseline (no extra training):
python scripts/train/lora_sweep.py --instruction-corpus-path data/instruction_corpus.txt --tokenizer-path data/tokenizer_digit --foundational-checkpoint models/foundational/best_model.pt --instruction-model-path models/instruction/best_model.pt --output-dir lora_sweep_results --lora-ranks 2 4 8 16 32 --num-eval-samples 500



# Default: baseline + 9 contrastive (weights 0.1,0.3,0.5 × temps 0.05,0.1,0.2), no eval
python scripts/train/contrastive_sweep.py --instruction-corpus-path data/instruction_corpus.txt --tokenizer-path data/tokenizer_digit --foundational-checkpoint models/foundational/best_model.pt --output-dir contrastive_sweep_results

# With evaluation after each training run
python scripts/train/contrastive_sweep.py --instruction-corpus-path data/instruction_corpus.txt --tokenizer-path data/tokenizer_digit --foundational-checkpoint models/foundational/best_model.pt --eval --num-eval-samples 500

# Custom grid and skip baseline
python scripts/train/contrastive_sweep.py --instruction-corpus-path path/to/corpus.jsonl --tokenizer-path path/to/tokenizer --foundational-checkpoint path/to/foundational.pt --contrastive-weights 0.2 0.4 --contrastive-temperatures 0.05 0.15 --skip-baseline
```


