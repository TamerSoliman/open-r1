# Tutorial 28: Long-Context Training (32K+ tokens)

**Target Audience:** Advanced
**Duration:** 30 minutes

## Overview

**Long-context training** extends model's context window from 2K to 32K+ tokens.

## Configuration

```yaml
# config_long_context.yaml
model_max_length: 32768  # 32K context

# Gradient checkpointing required!
gradient_checkpointing: true

# Smaller batch size (memory intensive)
per_device_train_batch_size: 1
gradient_accumulation_steps: 32

# Flash Attention 2 required
attn_implementation: flash_attention_2
```

## Memory Requirements

```
Standard (2K context): 40 GB per GPU
Long (32K context): 65 GB per GPU (with gradient checkpointing)

Required: A100 80GB or H100
```

## Training

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./long_context_output",
    max_seq_length=32768,
    gradient_checkpointing=True,
    bf16=True,
    attn_implementation="flash_attention_2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
)

trainer = Trainer(model=model, args=training_args, train_dataset=long_dataset)
trainer.train()
```

## Summary

- **32K context** requires gradient checkpointing
- **Flash Attention 2** essential
- **Smaller batch sizes** due to memory
- **80GB+ GPUs** recommended

**Next Tutorial:** Knowledge Distillation

## Resources
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Long Context Survey](https://arxiv.org/abs/2402.02244)
