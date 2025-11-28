# Tutorial 12: Checkpoint Management and Hub Integration

**Target Audience:** Beginner to Intermediate
**Duration:** 30 minutes
**Prerequisites:** None (standalone tutorial)

## Table of Contents
1. [Overview](#overview)
2. [Checkpoint Basics](#checkpoint-basics)
3. [Saving Strategies](#saving-strategies)
4. [Loading Checkpoints](#loading-checkpoints)
5. [HuggingFace Hub Integration](#huggingface-hub-integration)
6. [Hands-On Example](#hands-on-example)
7. [Summary](#summary)

---

## Overview

**Checkpoint management** is critical for long-running training jobs. Proper checkpointing enables:
- ✅ Recovery from failures
- ✅ Resume training
- ✅ Model versioning
- ✅ Sharing models with community

**What you'll learn:**
- Saving and loading checkpoints
- Checkpoint strategies (best, last, periodic)
- Pushing models to HuggingFace Hub
- Resuming interrupted training

---

## Checkpoint Basics

### What's in a Checkpoint?

```
checkpoint-1000/
├── config.json              # Model architecture config
├── model.safetensors        # Model weights
├── optimizer.pt             # Optimizer states
├── scheduler.pt             # LR scheduler state
├── rng_state.pth           # Random number generator state
├── trainer_state.json       # Training progress
└── training_args.bin        # Training hyperparameters
```

**Why save all this?**
- `model.safetensors`: The trained weights
- `optimizer.pt`: Continue training with same momentum
- `trainer_state.json`: Resume from exact step
- `rng_state.pth`: Reproducible data shuffling

### Checkpoint Size

**For 7B model:**
```
Model weights (FP16):      14 GB
Optimizer states (Adam):   28 GB (momentum + variance)
Other files:               ~100 MB
─────────────────────────
Total per checkpoint:      ~42 GB
```

**Tip:** With many checkpoints, disk usage grows fast!

---

## Saving Strategies

### Strategy 1: Save Best Model Only

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,  # Load best checkpoint at end
    metric_for_best_model="eval_loss",
    greater_is_better=False,  # Lower loss is better
    save_total_limit=3,  # Keep only 3 best checkpoints
)
```

**Result:**
- Saves checkpoint every 500 steps
- Keeps only 3 best (by eval_loss)
- Deletes older checkpoints automatically

### Strategy 2: Save Last Checkpoint Only

```python
training_args = TrainingArguments(
    output_dir="./output",
    save_strategy="epoch",  # Save at end of each epoch
    save_total_limit=1,  # Keep only last checkpoint
)
```

**Use case:** Limited disk space

### Strategy 3: Save All Checkpoints

```python
training_args = TrainingArguments(
    output_dir="./output",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=None,  # No limit
)
```

**Use case:** Post-training analysis (track model evolution)

### Recommended for DeepSeek R1

```python
training_args = TrainingArguments(
    output_dir="./output",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,  # Keep 5 recent checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
```

**Why:**
- Regular checkpoints (every 500 steps) for recovery
- Keep 5 recent (covers ~2500 steps of history)
- Automatically load best at end

---

## Loading Checkpoints

### Resume Training

```bash
# Training was interrupted at step 5000
# Checkpoint saved at checkpoint-5000/

# Resume from checkpoint
python src/open_r1/sft.py \
  --config config_distill.yaml \
  --resume_from_checkpoint ./output/checkpoint-5000

# Training continues from step 5001!
```

**What gets restored:**
- Model weights
- Optimizer state (momentum, variance)
- Learning rate scheduler
- Random seed (same data order)
- Training step counter

### Load for Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from checkpoint
model = AutoModelForCausalLM.from_pretrained("./output/checkpoint-5000")
tokenizer = AutoTokenizer.from_pretrained("./output/checkpoint-5000")

# Generate
inputs = tokenizer("What is 2+2?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Load Best Checkpoint

```python
# If using load_best_model_at_end=True
# Best model saved to output_dir (not in subdirectory)

model = AutoModelForCausalLM.from_pretrained("./output")
```

---

## HuggingFace Hub Integration

### Push to Hub During Training

```python
training_args = TrainingArguments(
    output_dir="./output",
    push_to_hub=True,  # Auto-push to Hub
    hub_model_id="username/deepseek-r1-7b-math",  # Hub repo name
    hub_strategy="checkpoint",  # Push each checkpoint
    hub_token="hf_xxx",  # Your Hub token (or use huggingface-cli login)
)
```

**What happens:**
- Checkpoint saved locally every `save_steps`
- Automatically pushed to HuggingFace Hub
- Public or private based on Hub settings

### Manual Push After Training

```bash
# Login to Hub (one-time)
huggingface-cli login

# Push model
python -c "
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('./output')
model.push_to_hub('username/deepseek-r1-7b-math')
"
```

### Push with Model Card

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./output")

model.push_to_hub(
    "username/deepseek-r1-7b-math",
    commit_message="Add SFT checkpoint after 10k steps",
    private=False,  # Public model
)

# Also push tokenizer
tokenizer.push_to_hub("username/deepseek-r1-7b-math")
```

**Result:** Model available at `https://huggingface.co/username/deepseek-r1-7b-math`

---

## Hands-On Example

### Example 1: Training with Checkpoints

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./sft_output",

    # Checkpoint strategy
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,

    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Training
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train
trainer.train()

# Best model automatically loaded and saved to ./sft_output
```

### Example 2: Resume Interrupted Training

```bash
# Training interrupted at step 2500
# Last checkpoint: checkpoint-2500

# Resume
python src/open_r1/sft.py \
  --config config_distill.yaml \
  --output_dir ./sft_output \
  --resume_from_checkpoint ./sft_output/checkpoint-2500

# Continues from step 2501
```

### Example 3: Push to Hub

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load best checkpoint
model = AutoModelForCausalLM.from_pretrained("./sft_output")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")

# Push to Hub
model.push_to_hub("username/qwen-7b-math-sft")
tokenizer.push_to_hub("username/qwen-7b-math-sft")

# Create model card
model_card = """
---
language: en
license: apache-2.0
tags:
- deepseek-r1
- math-reasoning
datasets:
- open-r1/OpenR1-Math-220k
---

# Qwen-7B Math SFT

Fine-tuned on OpenR1-Math-220k for math reasoning.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("username/qwen-7b-math-sft")
tokenizer = AutoTokenizer.from_pretrained("username/qwen-7b-math-sft")

inputs = tokenizer("Solve: 2x + 3 = 7", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Training Details

- Base model: Qwen/Qwen-7B
- Dataset: OpenR1-Math-220k
- Training steps: 10,000
- Learning rate: 5e-6
"""

# Save model card
with open("./sft_output/README.md", "w") as f:
    f.write(model_card)

# Push again with README
model.push_to_hub("username/qwen-7b-math-sft", commit_message="Add model card")
```

---

## Summary

**Key Takeaways:**

1. **Checkpoints enable recovery** from training failures
2. **Save strategies** control frequency and retention
3. **Resume training** from exact step with full state
4. **Push to Hub** for sharing and versioning
5. **Recommended:** Save every 500-1000 steps, keep 3-5 recent

**Configuration:**

```python
TrainingArguments(
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    load_best_model_at_end=True,
    push_to_hub=True,  # Optional
    hub_model_id="username/model-name",
)
```

**Workflow:**

```bash
# 1. Train with checkpointing
python src/open_r1/sft.py --config config.yaml

# 2. If interrupted, resume
python src/open_r1/sft.py --resume_from_checkpoint ./output/checkpoint-5000

# 3. Push to Hub
model.push_to_hub("username/model-name")
```

**Phase 4 COMPLETE!** All 12 foundational tutorials finished.

---

## Resources

- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
- [Checkpoint Management Guide](https://huggingface.co/docs/transformers/main_classes/trainer#checkpointing)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)

**Questions?** Open an issue on GitHub!
