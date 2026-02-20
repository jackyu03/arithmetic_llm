```bash

# 1. Generate training corpus (100,000 samples recommended)

# Generate foundational training corpus with 100K samples (plain text)
# This large corpus provides the base model with extensive arithmetic patterns
python scripts/data/generate_foundational_plaintext.py --num-samples 100000 --max-depth 4 --num-range 1 20 --invalid-rate 0.05 --output-txt data/foundational_corpus.txt

# Generate mixed instruction corpus (valid + invalid)
# This creates a balanced dataset without writing intermediate files
python scripts/data/generate_instruction_corpus_mixed.py --num-samples 20000 --max-depth 4 --num-range 1 20 --invalid-rate 0 --output-mixed data/instruction_corpus.txt

# Generate separate test corpus for evaluation (10K samples, minimal errors)
# This provides a clean test set with only 1% invalid expressions
python scripts/data/generate_corpus.py --instruction-only --num-samples 1000 --max-depth 4 --output-instruction data/instruction_corpus_test.txt --num-range 1 20 --invalid-rate 0

#check line counts
python -c "import sys; [print(f'{sum(1 for _ in open(f))} {f}') for f in ['data/foundational_corpus.txt', 'data/instruction_corpus.txt', 'data/instruction_corpus_test.txt']]"

# 2. Train tokenizer
python scripts/data/train_tokenizer.py --corpus-path data/foundational_corpus.txt --output-dir data/tokenizer --vocab-size 1000

# show tokenizer table
python scripts/utils/print_token_table.py --tokenizer_path data/tokenizer/tokenizer.pkl  > tokens.csv

# Analyze your instruction corpus
python scripts/utils/check_sequence_lengths.py --corpus-path data/instruction_corpus.txt --tokenizer-path data/tokenizer

# 3. Train foundational model
python scripts/train/foundational.py --corpus-path data/foundational_corpus.txt --output-dir models/ --tokenizer-path data/tokenizer --num-epochs 10 --max-seq-length 512 --batch-size 16

# 3.1 Evaluate the foundational model, performance would be bad
python scripts/eval/evaluate.py --model-path models/foundational_20260220_103758_473709/best_model.pt --tokenizer-path data/tokenizer --max-gen-length 512 --num-samples 100 --batch-size 1


# 4. Fine-tune instruction model
python scripts/train/instruction.py --instruction-corpus-path data/instruction_corpus.txt --output-dir models/ --tokenizer-path data/tokenizer --foundational-checkpoint models/foundational_20260220_103758_473709/best_model.pt --num-epochs 10

# 4.1 Evaluate the model
python scripts/eval/evaluate.py --model-path models/instruction_20260220_122251_979267/best_model.pt --tokenizer-path data/tokenizer --max-gen-length 512 --batch-size 1 --num-samples 1000

# 5 Fine-tune with LoRA adapters (optional)
python scripts/train/instruction_lora.py --instruction-corpus-path data/instruction_corpus.txt --output-dir models/ --tokenizer-path data/tokenizer --foundational-checkpoint models/foundational_20260220_103758_473709/best_model.pt --num-epochs 10 --lora-rank 8 --lora-alpha 16 --lora-target-modules attention --save-merged-model

# 5.1 Evaluate the LoRA merged model (optional)
python scripts/eval/evaluate.py --model-path models/instruction_lora_20260220_131932_609000/merged_model.pt --tokenizer-path data/tokenizer --max-gen-length 512 --batch-size 1 --num-samples 1000

# 6 GRPO training (optional)
python scripts/train/grpo.py --tokenizer data/tokenizer --sft-checkpoint models/instruction_YYYYMMDD_HHMMSS/best_model.pt --output-dir models/grpo --data-mode generated --log-every 1 --num-samples 1024 --num-epochs 3 --num-candidates 8 --max-gen-length 511 --temperature 0.8 --batch-size 1 --gradient-accumulation-steps 16 --kl-penalty-coef 0.05
 

# 6.1 eval GRPO model
python scripts/eval/evaluate.py   --model-path models/grpo/grpo_YYYYMMMDD_HHMMSS/final_modelpt    --tokenizer-path data/tokenizer   --max-gen-length 512   --batch-size 1   --num-samples 1000

```