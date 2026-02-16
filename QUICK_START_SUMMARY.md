# Arithmetic LLM Training System - Quick Start Summary

## Quick Start Demo: Full Pipeline from Beginning to End

Follow these commands to reproduce the complete training pipeline:

### Step 1: Corpus Generation
```bash
python scripts/data/generate_corpus.py \
  --output-dir data/ \
  --foundational-samples 100000 \
  --instruction-samples 20000 \
  --test-samples 1000
```

### Step 2: Tokenizer Training
```bash
python scripts/data/train_tokenizer.py \
  --corpus-path data/foundational_corpus.txt \
  --output-dir data/tokenizer \
  --vocab-size 1000 \
  --special-tokens pad,unk,bos,eos,tool_call
```

### Step 3: Sequence Analysis
```bash
python scripts/utils/check_sequence_lengths.py \
  --corpus-path data/instruction_corpus.txt \
  --percentiles 90 95 99 \
  --output-dir analysis/
```

### Step 4: Foundational Model Training
```bash
python scripts/train/foundational.py \
  --corpus-path data/foundational_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --max-seq-length 512 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --num-epochs 5
```

### Step 5: Foundational Model Evaluation
```bash
python scripts/eval/evaluate.py \
  --model-path models/foundational_20260201_012912_173614/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 100 \
  --batch-size 1
```

### Step 6: Instruction Fine-tuning
```bash
python scripts/train/instruction.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_20260201_012912_173614/best_model.pt \
  --num-epochs 5 \
  --batch-size 16 \
  --learning-rate 0.0001
```

### Step 7: Instruction Model Evaluation
```bash
python scripts/eval/evaluate.py \
  --model-path models/instruction_20260201_042439_468735/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 1000 \
  --batch-size 1
```

### Step 8: LoRA Fine-tuning
```bash
python scripts/train/instruction_lora.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_20260201_012912_173614/best_model.pt \
  --num-epochs 3 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target-modules attention \
  --save-merged-model
```

### Step 9: LoRA Model Evaluation
```bash
python scripts/eval/evaluate.py \
  --model-path models/instruction_lora_20260201_053153_241537/lora_adapter.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 1000 \
  --batch-size 1
```

### Step 10: GRPO Training (Reinforcement Learning)
```bash
python scripts/train/grpo.py \
  --instruction-corpus data/instruction_corpus.txt \
  --tokenizer data/tokenizer \
  --sft-checkpoint models/instruction_20260201_042439_468735/best_model.pt \
  --output-dir models/grpo \
  --num-epochs 3 \
  --batch-size 8 \
  --num-candidates 4 \
  --temperature 0.8 \
  --kl-penalty-coef 0.05
```

### Step 11: GRPO Model Evaluation
```bash
python scripts/eval/evaluate.py \
  --model-path models/grpo/grpo_20260201_153650_018769/final_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 1000 \
  --batch-size 1
```

