# Tutorial 4: Dataset Preparation and Chat Templates

**Target Audience:** Beginner to Intermediate
**Duration:** 45 minutes
**Prerequisites:** Tutorial 1 (Three-Stage Pipeline)

## Table of Contents
1. [Overview](#overview)
2. [Dataset Structure](#dataset-structure)
3. [Chat Templates](#chat-templates)
4. [System Prompts](#system-prompts)
5. [Dataset Mixtures](#dataset-mixtures)
6. [Data Loading Pipeline](#data-loading-pipeline)
7. [Hands-On Example](#hands-on-example)
8. [Common Pitfalls](#common-pitfalls)
9. [Exercise](#exercise)

---

## Overview

Proper dataset preparation is critical for DeepSeek R1 training. The model needs:
- **Structured conversations** (system, user, assistant messages)
- **Proper formatting** (chat templates for different model architectures)
- **Quality data** (diverse problems, correct solutions)
- **Balanced mixtures** (multiple datasets with appropriate weights)

**What you'll learn:**
- How to structure datasets for chat models
- What chat templates are and why they matter
- How to write effective system prompts
- How to combine multiple datasets
- Practical dataset loading with HuggingFace

---

## Dataset Structure

### Raw Dataset Format

DeepSeek R1 expects datasets with **question-answer pairs** that will be converted to conversations.

**Example: Math Problem Dataset**

```json
{
  "problem": "Solve for x: 2x + 3 = 7",
  "solution": "<think>\nI need to isolate x.\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2\n</think>\n<answer>\nx = 2\n</answer>"
}
```

**Key Fields:**

- `problem` / `prompt` / `question`: The user's query
- `solution` / `answer` / `response`: The model's response
- `metadata` (optional): Additional context (difficulty, source, etc.)

### Conversation Format

Models don't directly consume question-answer pairs. They need **conversation format**:

```python
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Solve for x: 2x + 3 = 7"},
    {"role": "assistant", "content": "<think>\n...\n</think>\n<answer>\nx = 2\n</answer>"}
  ]
}
```

**Roles:**

- **system**: Instructions for the model (how to behave)
- **user**: The human's query
- **assistant**: The model's response

### Conversion Function

**From `src/open_r1/data.py`:**

```python
def convert_to_conversation(example, system_prompt, prompt_column="problem"):
    """
    WHAT: Converts dataset example to conversation format

    WHY: Chat models expect structured conversations

    HOW:
        1. Create messages list
        2. Add system prompt (if provided)
        3. Add user message with problem
        4. Add assistant message with solution (for SFT only)
    """
    messages = []

    # Add system prompt
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    # Add user query
    messages.append({
        "role": "user",
        "content": example[prompt_column]
    })

    # For SFT: Add assistant response
    if "solution" in example:
        messages.append({
            "role": "assistant",
            "content": example["solution"]
        })

    return {"messages": messages}
```

---

## Chat Templates

### What Are Chat Templates?

**Chat templates** convert conversations into model-specific text format.

**Different models use different formats:**

| Model Family | Chat Format |
|--------------|-------------|
| OpenAI GPT | ChatML |
| Llama 2/3 | `[INST]...[/INST]` format |
| Qwen | Special tokens `<|im_start|>` |
| Mistral | `[INST]...[/INST]` |

### ChatML Format (OpenAI/Qwen)

**Conversation:**

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"}
]
```

**ChatML Output:**

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi! How can I help?<|im_end|>
```

### Llama-2 Format

**Same Conversation:**

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"}
]
```

**Llama-2 Output:**

```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hello! [/INST] Hi! How can I help? </s>
```

### Applying Chat Templates

**HuggingFace Integration:**

```python
from transformers import AutoTokenizer
from trl.data_utils import apply_chat_template

# Load tokenizer (includes chat template)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")

# Dataset with conversations
dataset = load_dataset("...")

# Apply chat template
dataset = dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer}
)

# Result: Each example now has "text" field with formatted conversation
```

**What `apply_chat_template` does:**

1. Takes `messages` field from dataset
2. Applies tokenizer's chat template
3. Adds `text` field with formatted string
4. Model can now tokenize and train on `text`

### Why Chat Templates Matter

**❌ Wrong: Inconsistent format**

```python
# Training data
train_text = "<|im_start|>user\nHello<|im_end|>"

# Inference
inference_text = "Hello"  # No special tokens!

# Result: Model confused, poor performance
```

**✅ Correct: Consistent format**

```python
# Training data
train_text = tokenizer.apply_chat_template(messages)

# Inference
inference_text = tokenizer.apply_chat_template(messages)

# Result: Model sees same format at train and inference time
```

---

## System Prompts

### What Is a System Prompt?

The **system prompt** establishes the model's behavior and response format.

**For DeepSeek R1:**

```
You are Open-R1, a helpful AI assistant. Your role involves thoroughly
exploring questions through a systematic thinking process before providing
final solutions.

Please structure your response into two sections using this format:

<think>
[Your detailed reasoning process here]
</think>

<answer>
[Your final answer here]
</answer>

In the <think> section, show your work step-by-step. In the <answer>
section, provide only the final result.
```

### Key Components

**1. Identity:**
```
You are Open-R1, a helpful AI assistant.
```

**2. Task Description:**
```
Your role involves thoroughly exploring questions through a systematic
thinking process before providing final solutions.
```

**3. Format Instructions:**
```
Please structure your response into two sections using this format:

<think>
[Your detailed reasoning process here]
</think>

<answer>
[Your final answer here]
</answer>
```

**4. Clarifications:**
```
In the <think> section, show your work step-by-step. In the <answer>
section, provide only the final result.
```

### System Prompt in Configuration

**From `recipes/config_distill.yaml`:**

```yaml
system_prompt: |
  You are Open-R1, a helpful AI assistant. Your role involves thoroughly
  exploring questions through a systematic thinking process before providing
  final solutions.

  Please structure your response into two sections using this format:

  <think>
  [Your detailed reasoning process here]
  </think>

  <answer>
  [Your final answer here]
  </answer>

  In the <think> section, show your work step-by-step. In the <answer>
  section, provide only the final result.
```

**Usage in Code:**

```python
from open_r1.configs import GRPOConfig

# Load config
config = GRPOConfig.from_yaml("recipes/config_distill.yaml")

# System prompt automatically prepended to all conversations
system_prompt = config.system_prompt
```

### Why System Prompts Are Critical

**Without system prompt:**

```
User: Solve 2x = 10

Model: x = 5
```

No reasoning shown! ❌

**With system prompt:**

```
User: Solve 2x = 10

Model:
<think>
I need to solve 2x = 10
Dividing both sides by 2: x = 5
Verification: 2(5) = 10 ✓
</think>
<answer>
x = 5
</answer>
```

Structured reasoning! ✅

---

## Dataset Mixtures

### Why Mix Datasets?

Training on a **single dataset** leads to:
- ❌ Overfitting to specific problem types
- ❌ Poor generalization
- ❌ Narrow skill set

Training on **mixed datasets** provides:
- ✅ Diverse problem types
- ✅ Better generalization
- ✅ Broader capabilities

### Dataset Mixture Configuration

**From `config_distill.yaml`:**

```yaml
dataset_mixer:
  open-r1/OpenR1-Math-220k: 0.5      # 50% math problems
  open-r1/OpenR1-Code-100k: 0.3      # 30% code problems
  open-r1/OpenR1-Reasoning-50k: 0.2  # 20% general reasoning

# Total samples drawn according to weights
# Example: 10,000 training samples
#   - 5,000 from Math dataset
#   - 3,000 from Code dataset
#   - 2,000 from Reasoning dataset
```

### Weighted Sampling

**Implementation (from `src/open_r1/data.py`):**

```python
def load_mixed_dataset(dataset_mixer, split="train"):
    """
    WHAT: Loads and mixes multiple datasets with specified weights

    WHY: Provides diverse training data for better generalization

    HOW:
        1. Load each dataset from HuggingFace Hub
        2. Sample from each dataset according to weights
        3. Concatenate into single dataset
        4. Shuffle for randomness
    """
    datasets_list = []

    # Load each dataset
    for dataset_name, weight in dataset_mixer.items():
        ds = load_dataset(dataset_name, split=split)

        # Calculate number of samples from this dataset
        # weight is a fraction (e.g., 0.5 = 50%)
        num_samples = int(len(ds) * weight)

        # Sample with replacement if needed
        if num_samples > len(ds):
            # Oversample small datasets
            ds = ds.shuffle().select(range(len(ds)))
            ds = concatenate_datasets([ds] * (num_samples // len(ds) + 1))
            ds = ds.select(range(num_samples))
        else:
            # Regular sampling
            ds = ds.shuffle().select(range(num_samples))

        datasets_list.append(ds)

    # Combine all datasets
    mixed_dataset = concatenate_datasets(datasets_list)

    # Final shuffle
    mixed_dataset = mixed_dataset.shuffle(seed=42)

    return mixed_dataset
```

### Balancing Strategies

**Strategy 1: Equal Weight**

```yaml
dataset_mixer:
  dataset_A: 0.33
  dataset_B: 0.33
  dataset_C: 0.34
```

- Each dataset contributes equally
- Good for balanced skills

**Strategy 2: Difficulty-Based**

```yaml
dataset_mixer:
  easy_problems: 0.2
  medium_problems: 0.5
  hard_problems: 0.3
```

- Focus on medium difficulty
- Learn from hard problems
- Maintain easy baseline

**Strategy 3: Domain-Based**

```yaml
dataset_mixer:
  math: 0.6         # Primary skill
  code: 0.3         # Secondary skill
  general: 0.1      # Auxiliary skill
```

- Emphasize primary domain
- Maintain diverse capabilities

---

## Data Loading Pipeline

### Complete Pipeline (SFT)

**From `src/open_r1/sft.py`:**

```python
# Step 1: Load mixed dataset
from open_r1.data import load_datasets

dataset = load_datasets(
    dataset_mixer={
        "open-r1/OpenR1-Math-220k": 0.5,
        "open-r1/OpenR1-Code-100k": 0.5,
    },
    splits=["train", "test"]
)

# Step 2: Convert to conversation format
def make_conversation(example):
    messages = []

    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": example["problem"]})
    messages.append({"role": "assistant", "content": example["solution"]})

    return {"messages": messages}

dataset = dataset.map(make_conversation)

# Step 3: Apply chat template
from transformers import AutoTokenizer
from trl.data_utils import apply_chat_template

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer}
)

# Step 4: Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=32768,  # Long context support
    )

dataset = dataset.map(tokenize, batched=True)

# Step 5: Ready for training!
from transformers import Trainer

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    ...
)
```

### GRPO Pipeline Differences

**Key Difference:** GRPO doesn't include assistant responses in the dataset (generates them during training).

```python
# GRPO data loading
def make_conversation_grpo(example):
    messages = []

    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    # Only user message, no assistant response!
    messages.append({"role": "user", "content": example["problem"]})

    return {"messages": messages}

# No solution field needed in dataset
# Model generates completions during GRPO training
```

---

## Hands-On Example

### Example 1: Loading a Single Dataset

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.data_utils import apply_chat_template

# Load dataset
dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train[:100]")

# Define system prompt
system_prompt = """You are a helpful math tutor. Show your work step-by-step."""

# Convert to conversation
def to_conversation(example):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]}
        ]
    }

dataset = dataset.map(to_conversation)

# Apply chat template
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

# Inspect result
print(dataset[0]["text"])
```

**Output:**

```
<|im_start|>system
You are a helpful math tutor. Show your work step-by-step.<|im_end|>
<|im_start|>user
Solve for x: 2x + 3 = 7<|im_end|>
<|im_start|>assistant
<think>
I need to isolate x.
2x + 3 = 7
Subtract 3 from both sides:
2x = 4
Divide by 2:
x = 2
</think>
<answer>
x = 2
</answer><|im_end|>
```

### Example 2: Creating a Dataset Mixture

```python
from datasets import load_dataset, concatenate_datasets

# Define mixture
dataset_mixer = {
    "open-r1/OpenR1-Math-220k": 0.6,
    "open-r1/OpenR1-Code-100k": 0.4,
}

datasets_list = []

for dataset_name, weight in dataset_mixer.items():
    ds = load_dataset(dataset_name, split="train")

    # Sample according to weight
    num_samples = int(10000 * weight)  # 10k total samples
    ds = ds.shuffle(seed=42).select(range(num_samples))

    datasets_list.append(ds)

# Combine
mixed_dataset = concatenate_datasets(datasets_list)
mixed_dataset = mixed_dataset.shuffle(seed=42)

print(f"Total samples: {len(mixed_dataset)}")
print(f"Math samples: ~{10000 * 0.6}")
print(f"Code samples: ~{10000 * 0.4}")
```

**Output:**

```
Total samples: 10000
Math samples: ~6000.0
Code samples: ~4000.0
```

### Example 3: Custom Dataset

**Create your own dataset:**

```python
from datasets import Dataset

# Custom data
data = {
    "problem": [
        "What is 2 + 2?",
        "What is 5 * 6?",
        "Solve x^2 = 16",
    ],
    "solution": [
        "<think>\n2 + 2 = 4\n</think>\n<answer>\n4\n</answer>",
        "<think>\n5 * 6 = 30\n</think>\n<answer>\n30\n</answer>",
        "<think>\nx^2 = 16\nx = ±4\n</think>\n<answer>\nx = 4 or x = -4\n</answer>",
    ]
}

# Create dataset
dataset = Dataset.from_dict(data)

# Apply same pipeline
def to_conversation(example):
    return {
        "messages": [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]}
        ]
    }

dataset = dataset.map(to_conversation)

# Save to disk
dataset.save_to_disk("./my_custom_dataset")

# Or push to Hub
dataset.push_to_hub("username/my-custom-dataset")
```

---

## Common Pitfalls

### 1. **Inconsistent Chat Templates**

❌ **Wrong:** Different templates for training and inference

```python
# Training
train_template = "ChatML"

# Inference
# User forgets to use chat template!
input_text = "Solve 2x = 10"  # Raw text
```

✅ **Correct:** Always use same template

```python
# Training
messages = [{"role": "user", "content": "..."}]
text = tokenizer.apply_chat_template(messages)

# Inference
messages = [{"role": "user", "content": "..."}]
text = tokenizer.apply_chat_template(messages)
```

### 2. **Missing System Prompt**

❌ **Wrong:**

```yaml
system_prompt: null  # No system prompt!
```

**Result:** Model doesn't know to use `<think>/<answer>` format

✅ **Correct:**

```yaml
system_prompt: |
  You are Open-R1. Use <think> and <answer> tags.
```

### 3. **Unbalanced Dataset Mixture**

❌ **Wrong:**

```yaml
dataset_mixer:
  tiny_dataset_100_samples: 0.5   # Too much weight on tiny dataset!
  huge_dataset_1M_samples: 0.5
```

**Result:** Tiny dataset gets oversampled, model overfits

✅ **Correct:** Balance by dataset size or importance

```yaml
dataset_mixer:
  tiny_dataset_100_samples: 0.01   # Small weight for small dataset
  huge_dataset_1M_samples: 0.99
```

### 4. **Not Shuffling Data**

❌ **Wrong:**

```python
dataset = load_dataset(...)
# No shuffling! Model sees same order every epoch
```

**Result:** Model memorizes order, poor generalization

✅ **Correct:**

```python
dataset = load_dataset(...).shuffle(seed=42)
```

### 5. **Wrong Prompt Column Name**

❌ **Wrong:**

```python
dataset_prompt_column: "question"

# But dataset has field "problem"!
# Result: KeyError
```

✅ **Correct:** Check dataset schema first

```python
print(dataset.column_names)
# ['problem', 'solution', 'metadata']

dataset_prompt_column: "problem"  # Correct field name
```

---

## Exercise

### Task 1: Create Custom Dataset

```python
# 1. Create dataset with 10 math problems
data = {
    "problem": [
        "What is 10 + 5?",
        "What is 20 - 7?",
        # ... add 8 more
    ],
    "solution": [
        "<think>\n10 + 5 = 15\n</think>\n<answer>\n15\n</answer>",
        "<think>\n20 - 7 = 13\n</think>\n<answer>\n13\n</answer>",
        # ... add 8 more
    ]
}

dataset = Dataset.from_dict(data)
dataset.save_to_disk("./my_math_dataset")
```

### Task 2: Apply Chat Template

```python
# 2. Convert to conversation format with system prompt
system_prompt = "You are a math tutor."

def to_conversation(example):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]}
        ]
    }

dataset = dataset.map(to_conversation)

# 3. Apply ChatML template
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

# Inspect
print(dataset[0]["text"])
```

### Task 3: Create Dataset Mixture

```python
# 4. Mix with existing dataset
from datasets import concatenate_datasets, load_dataset

existing_dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train[:100]")

# Sample 50 from existing, 10 from custom
existing_sample = existing_dataset.shuffle(seed=42).select(range(50))
custom_sample = dataset

mixed = concatenate_datasets([existing_sample, custom_sample])
mixed = mixed.shuffle(seed=42)

print(f"Total: {len(mixed)}")  # Should be 60
```

### Task 4: Verify Chat Template Consistency

```python
# 5. Compare training vs inference format
# Training format
train_example = dataset[0]["text"]
print("Training:", train_example[:100])

# Inference format (manually created)
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2 + 2?"}
]
inference_text = tokenizer.apply_chat_template(messages, tokenize=False)
print("Inference:", inference_text[:100])

# Should have same structure (system + user format)
```

**Deliverable:**

- Custom dataset with 10 examples
- Mixed dataset combining custom + existing
- Comparison showing training and inference use same template

---

## Summary

**Key Takeaways:**

1. **Conversation format**: System + User + Assistant messages
2. **Chat templates**: Convert conversations to model-specific format
3. **System prompts**: Establish behavior and response format
4. **Dataset mixtures**: Combine multiple datasets with weights
5. **Consistency**: Use same template for training and inference

**Data Pipeline:**

```
Raw Dataset
    ↓
Convert to Conversation (add system prompt)
    ↓
Apply Chat Template (model-specific formatting)
    ↓
Tokenize
    ↓
Training
```

**Configuration Checklist:**

- ✅ Define system prompt with `<think>/<answer>` instructions
- ✅ Specify correct `dataset_prompt_column`
- ✅ Set up dataset mixture with appropriate weights
- ✅ Shuffle data before training
- ✅ Use same chat template for training and inference

**Next Tutorial:** vLLM Integration for Fast Inference

---

## Additional Resources

- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Chat Templates Guide](https://huggingface.co/docs/transformers/chat_templating)
- [Annotated Code: Data Loading](../../annotated_code/infrastructure/data_ANNOTATED.py)
- [Sample Datasets on HuggingFace Hub](https://huggingface.co/open-r1)

**Questions?** Open an issue on GitHub!
