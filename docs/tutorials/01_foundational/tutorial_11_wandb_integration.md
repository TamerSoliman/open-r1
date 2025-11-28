# Tutorial 11: WandB Integration and Experiment Tracking

**Target Audience:** Beginner to Intermediate
**Duration:** 25 minutes
**Prerequisites:** None (standalone tutorial)

## Table of Contents
1. [Overview](#overview)
2. [Setup WandB](#setup-wandb)
3. [Tracking Experiments](#tracking-experiments)
4. [Visualization](#visualization)
5. [Comparing Runs](#comparing-runs)
6. [Hands-On Example](#hands-on-example)
7. [Summary](#summary)

---

## Overview

**Weights & Biases (WandB)** is an experiment tracking platform that automatically logs metrics, hyperparameters, and artifacts during training.

**What you'll learn:**
- Setting up WandB
- Logging training metrics
- Visualizing experiments
- Comparing runs

**Benefits:**
- ✅ Automatic metric tracking
- ✅ Real-time visualization
- ✅ Compare experiments
- ✅ Reproduce results
- ✅ Team collaboration

---

## Setup WandB

### Installation

```bash
# Install wandb
pip install wandb

# Login (one-time setup)
wandb login
# Paste API key from https://wandb.ai/authorize
```

### Configuration

**In training config (`config_distill.yaml`):**

```yaml
# WandB settings
report_to: wandb
run_name: sft-qwen-7b-v1
logging_steps: 10

# Optional: WandB project
wandb_project: deepseek-r1
wandb_entity: your-team  # Your WandB username/team
```

---

## Tracking Experiments

### Automatic Tracking

**HuggingFace Trainer automatically logs:**
- Loss (training and validation)
- Learning rate
- Gradient norm
- Training speed (samples/sec)
- GPU memory usage

**No code changes needed!**

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    report_to="wandb",  # Enable WandB
    run_name="sft-experiment-1",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()  # Metrics automatically logged to WandB!
```

### Custom Metrics

```python
import wandb

# Log custom metrics
wandb.log({
    "custom/accuracy": 0.85,
    "custom/perplexity": 12.3,
    "step": step,
})
```

---

## Visualization

### Dashboard

**WandB automatically creates:**
- Loss curves
- Learning rate schedule
- System metrics (GPU, CPU, memory)
- Hyperparameters table

**Access:** https://wandb.ai/your-username/project-name

### Example Dashboard

```
┌─────────────────────────────────────┐
│ Loss vs Step                        │
│                                     │
│ 3.0 ┐                               │
│     │ \                             │
│ 2.0 ┤  \___                         │
│     │      \____                    │
│ 1.0 ┤           \______             │
│     │                  ────────     │
│ 0.0 └─────────────────────────────  │
│     0   5k  10k  15k  20k steps     │
└─────────────────────────────────────┘
```

---

## Comparing Runs

### Multiple Experiments

```bash
# Run 1: Baseline
python src/open_r1/sft.py \
  --config config_v1.yaml \
  --run_name sft-baseline

# Run 2: Higher LR
python src/open_r1/sft.py \
  --config config_v2.yaml \
  --learning_rate 1e-5 \
  --run_name sft-high-lr

# Run 3: Larger batch
python src/open_r1/sft.py \
  --config config_v3.yaml \
  --per_device_train_batch_size 8 \
  --run_name sft-large-batch
```

**Compare in WandB:**
- Side-by-side loss curves
- Hyperparameter differences highlighted
- Best run automatically identified

---

## Hands-On Example

### Example 1: Basic Tracking

```python
from transformers import Trainer, TrainingArguments
import wandb

# Initialize (optional, Trainer does this automatically)
wandb.init(
    project="deepseek-r1",
    name="sft-experiment-1",
    config={
        "model": "Qwen/Qwen-7B",
        "dataset": "OpenR1-Math-220k",
        "epochs": 3,
    }
)

# Training with WandB
training_args = TrainingArguments(
    output_dir="./output",
    report_to="wandb",
    run_name="sft-experiment-1",
    num_train_epochs=3,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Finish logging
wandb.finish()
```

### Example 2: Hyperparameter Sweep

```yaml
# sweep.yaml
program: src/open_r1/sft.py
method: bayes
metric:
  name: eval/loss
  goal: minimize

parameters:
  learning_rate:
    min: 1e-6
    max: 1e-4
  per_device_train_batch_size:
    values: [4, 8, 16]
  gradient_accumulation_steps:
    values: [4, 8, 16]
```

```bash
# Create sweep
wandb sweep sweep.yaml

# Run agents
wandb agent your-username/deepseek-r1/sweep-id
```

**WandB automatically:**
- Tries different hyperparameters
- Tracks all results
- Identifies best configuration

---

## Summary

**Key Takeaways:**

1. **WandB tracks experiments** automatically with HuggingFace Trainer
2. **No code changes** needed for basic tracking
3. **Real-time visualization** of metrics
4. **Compare runs** to find best hyperparameters
5. **Reproduce experiments** with logged configs

**Setup:**

```yaml
# In config.yaml
report_to: wandb
run_name: experiment-name
logging_steps: 10
```

**Workflow:**

```bash
# 1. Setup WandB
wandb login

# 2. Run training (automatic logging)
python src/open_r1/sft.py --config config.yaml

# 3. View results
# Visit: https://wandb.ai/your-username/project-name
```

**Next Tutorial:** Checkpoint Management

---

## Resources

- [WandB Documentation](https://docs.wandb.ai/)
- [WandB + HuggingFace](https://docs.wandb.ai/guides/integrations/huggingface)
- [Example Dashboard](https://wandb.ai/examples/deepseek-r1)

**Questions?** Open an issue on GitHub!
