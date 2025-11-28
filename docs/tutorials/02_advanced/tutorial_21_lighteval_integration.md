# Tutorial 21: LightEval Integration and Custom Tasks

**Target Audience:** Advanced
**Duration:** 35 minutes
**Prerequisites:** Annotated evaluation.py

## Overview

**LightEval** is a benchmark framework for evaluating language models on standard tasks (MMLU, GSM8K, HumanEval, etc.).

## Quick Start

```python
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.models.model_config import ModelConfig

# Configure model
model_config = ModelConfig(
    model="./my_checkpoint",
    accelerator="cuda",
)

# Run evaluation
from lighteval.main_accelerate import main

results = main(
    model_config=model_config,
    tasks=["mmlu", "gsm8k", "humaneval"],
    output_dir="./eval_results",
)

print(results)
```

## Custom Tasks

```python
# Define custom task
from lighteval.tasks import Task

custom_task = Task(
    name="my_math_task",
    prompt_function=lambda x: f"Solve: {x['problem']}",
    hf_repo="username/my-math-dataset",
    metric=["exact_match", "f1"],
)

# Add to task registry
from lighteval.tasks import TASKS_TABLE
TASKS_TABLE["my_math_task"] = custom_task
```

## Summary

- **LightEval** provides standardized benchmarks
- **Easy integration** with HuggingFace models
- **Custom tasks** supported
- **Metrics:** accuracy, F1, BLEU, etc.

**Next Tutorial:** Pass@K Metrics

## Resources
- [LightEval GitHub](https://github.com/huggingface/lighteval)
- [Annotated: evaluation.py](../../annotated_code/infrastructure/evaluation_ANNOTATED.py)
