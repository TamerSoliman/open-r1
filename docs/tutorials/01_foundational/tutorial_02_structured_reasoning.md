# Tutorial 2: Structured Reasoning with `<think>` and `<answer>` Tags

**Target Audience:** Beginner
**Duration:** 30 minutes
**Prerequisites:** Tutorial 1 (Three-Stage Pipeline)

## Table of Contents
1. [Overview](#overview)
2. [Why Structured Reasoning?](#why-structured-reasoning)
3. [The Think-Answer Format](#the-think-answer-format)
4. [Format Rewards](#format-rewards)
5. [Implementation Details](#implementation-details)
6. [Hands-On Example](#hands-on-example)
7. [Common Pitfalls](#common-pitfalls)
8. [Exercise](#exercise)

---

## Overview

Structured reasoning is the foundation of DeepSeek R1's capability. By separating the **thinking process** (`<think>`) from the **final answer** (`<answer>`), models learn to reason explicitly before responding.

**What you'll learn:**
- Why structure matters for reasoning
- How `<think>` and `<answer>` tags work
- How to enforce structure with format rewards
- Connection to Chain-of-Thought prompting

---

## Why Structured Reasoning?

### The Problem with Direct Answers

**Traditional Model (No Structure):**
```
User: What is the derivative of x² + 3x?

Model: 2x + 3
```

**Issues:**
- ❌ No visible reasoning process
- ❌ Can't debug wrong answers
- ❌ Harder to learn complex problems
- ❌ Can't verify intermediate steps

### The Solution: Explicit Reasoning

**Structured Model (With `<think>/<answer>`):**
```
User: What is the derivative of x² + 3x?

Model:
<think>
I need to find d/dx of (x² + 3x).
Using power rule: d/dx(xⁿ) = nxⁿ⁻¹

For x²: d/dx(x²) = 2x¹ = 2x
For 3x: d/dx(3x) = 3

Combining: 2x + 3
</think>
<answer>
2x + 3
</answer>
```

**Benefits:**
- ✅ Transparent reasoning process
- ✅ Can verify each step
- ✅ Easier to debug mistakes
- ✅ Better learning signal for training

---

## The Think-Answer Format

### Format Specification

```
<think>
[Step-by-step reasoning process]
[Can be multiple paragraphs]
[Show your work, verify answers, consider alternatives]
</think>
<answer>
[Final answer only]
[Concise and clear]
</answer>
```

### When to Use Each Section

**`<think>` Section (Internal Monologue):**
- Break down the problem
- Show intermediate calculations
- Consider multiple approaches
- Verify your reasoning
- Catch and correct mistakes

**`<answer>` Section (Final Output):**
- State the final answer clearly
- Include units if applicable
- Format for readability
- Be concise

### Example: Math Problem

```
Problem: A train travels 120 km in 2 hours. What is its average speed in m/s?

<think>
Given:
- Distance: 120 km
- Time: 2 hours

Step 1: Find speed in km/h
Speed = Distance / Time = 120 km / 2 h = 60 km/h

Step 2: Convert to m/s
60 km/h = 60 × 1000 m / 3600 s
       = 60000 / 3600 m/s
       = 16.67 m/s

Let me verify:
16.67 m/s × 3600 s/h ÷ 1000 m/km = 60 km/h ✓
</think>
<answer>
16.67 m/s
</answer>
```

### Example: Code Problem

```
Problem: Write a function to reverse a string in Python.

<think>
There are several approaches:
1. Slicing: s[::-1]
2. reversed() + join
3. Loop with temp array

Slicing is most Pythonic and efficient.
Need to handle edge cases:
- Empty string: "" → ""
- Single char: "a" → "a"

Let me write the function:
</think>
<answer>
```python
def reverse_string(s):
    return s[::-1]
```
</answer>
```

---

## Format Rewards

### Why Enforce Format?

During GRPO training, models might:
- Drop tags for brevity: ❌
- Mix tags incorrectly: ❌
- Skip reasoning: ❌

**Solution:** Format reward function

### Format Reward Implementation

**From `src/open_r1/rewards.py`:**

```python
def format_reward(completions, **kwargs):
    """
    Checks if completion has proper <think>...</think><answer>...</answer> format

    Returns:
        1.0 if valid format
        0.0 if invalid format
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    rewards = []

    for completion in completions:
        content = completion[0]["content"]  # Extract text
        if re.match(pattern, content, re.DOTALL):
            rewards.append(1.0)  # Valid format
        else:
            rewards.append(0.0)  # Invalid format

    return rewards
```

### Format Validation Rules

**Valid Examples:**

```python
# ✅ Correct
"<think>\nReasoning here\n</think>\n<answer>\nAnswer here\n</answer>"

# ✅ Also correct (multiline reasoning)
"<think>\nStep 1\nStep 2\nStep 3\n</think>\n<answer>\n42\n</answer>"
```

**Invalid Examples:**

```python
# ❌ Missing tags
"Just the answer without tags"

# ❌ Wrong order
"<answer>\n42\n</answer>\n<think>\nReasoning\n</think>"

# ❌ Missing newlines
"<think>Reasoning</think><answer>Answer</answer>"

# ❌ Extra text outside tags
"<think>\nReasoning\n</think>\n<answer>\nAnswer\n</answer>\nExtra text"
```

### Using Format Reward in GRPO

**Configuration (`config.yaml`):**

```yaml
reward_funcs:
  - accuracy      # Checks correctness
  - format        # Checks structure
  - tag_count     # Counts <think> tags

reward_weights:
  - 1.0          # Accuracy is primary
  - 0.1          # Format is important but secondary
  - 0.1          # Tag count encourages reasoning
```

**Effect:**
- Models learn to **always** use structured format
- Small penalty (0.1) keeps format consistent
- Doesn't override correctness (1.0 weight on accuracy)

---

## Implementation Details

### System Prompt Configuration

**From `config_distill.yaml`:**

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

### Chat Template Integration

**System prompt is automatically prepended:**

```python
# In SFT training
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Solve 2x + 3 = 7"},
    {"role": "assistant", "content": "<think>\n...\n</think>\n<answer>\nx = 2\n</answer>"}
]
```

### Loss Masking

**Key Detail:** During SFT, loss is computed **only on assistant tokens**

```python
# Pseudo-code
for token in sequence:
    if token.role == "assistant":
        loss += cross_entropy(predicted, actual)
    else:
        # User/system tokens don't contribute to loss
        pass
```

**Why:** Model learns to generate `<think>/<answer>` format, not just memorize it.

---

## Hands-On Example

### Testing Format Reward

```python
from open_r1.rewards import format_reward

# Test cases
test_completions = [
    # Valid
    [{"role": "assistant", "content": "<think>\nStep 1\n</think>\n<answer>\n42\n</answer>"}],
    # Invalid - missing tags
    [{"role": "assistant", "content": "Just 42"}],
    # Invalid - wrong order
    [{"role": "assistant", "content": "<answer>\n42\n</answer>\n<think>\nOops\n</think>"}],
]

for i, completion in enumerate(test_completions):
    reward = format_reward(completion)
    print(f"Test {i+1}: Reward = {reward[0]}")
```

**Expected Output:**
```
Test 1: Reward = 1.0
Test 2: Reward = 0.0
Test 3: Reward = 0.0
```

### Training with Format Rewards

```bash
# Configure GRPO with format reward
cat > config_format_demo.yaml << EOF
model_name_or_path: ./sft_checkpoint
dataset_name: open-r1/OpenR1-Math-220k
num_train_epochs: 1
max_steps: 100

reward_funcs:
  - accuracy
  - format

reward_weights:
  - 1.0    # Primary: correctness
  - 0.1    # Secondary: structure

output_dir: ./grpo_with_format
EOF

# Run GRPO
python src/open_r1/grpo.py --config config_format_demo.yaml
```

**Monitor Training:**
```
Step 10/100: reward_accuracy=0.65, reward_format=0.92
Step 50/100: reward_accuracy=0.72, reward_format=0.98
Step 100/100: reward_accuracy=0.78, reward_format=1.00
```

Notice: Format reward quickly reaches 100% as model learns structure.

---

## Common Pitfalls

### 1. **Conflicting Formats**

❌ **Wrong:** Using different formats in SFT and GRPO

```yaml
# SFT config
system_prompt: "Use <think> and <answer> tags"

# GRPO config
system_prompt: "Show your reasoning, then answer"  # Different!
```

✅ **Correct:** Use identical format across all stages

### 2. **No Format Enforcement**

❌ **Wrong:** Not using format reward in GRPO

```yaml
reward_funcs:
  - accuracy
# Missing format reward!
```

**Result:** Model drops structure for brevity

✅ **Correct:** Always include format reward

### 3. **Too High Format Weight**

❌ **Wrong:**
```yaml
reward_weights:
  - 1.0  # Accuracy
  - 2.0  # Format way too high!
```

**Result:** Model prioritizes format over correctness

✅ **Correct:** Keep format weight low (0.05-0.2)

### 4. **Inconsistent System Prompts**

❌ **Wrong:** Changing prompt between training and inference

✅ **Correct:** Use same system prompt always

### 5. **Missing Newlines**

❌ **Wrong:** `"<think>Reasoning</think><answer>Answer</answer>"`

✅ **Correct:** `"<think>\nReasoning\n</think>\n<answer>\nAnswer\n</answer>"`

---

## Exercise

**Goal:** Train a model to maintain structured format

**Tasks:**

1. **Create test dataset with mixed formats:**
   ```python
   test_cases = [
       "Solve: 2 + 2",  # Should produce <think>/<answer>
       "What is the capital of France?",  # Should produce <think>/<answer>
       "Write 'hello world' in Python",  # Should produce <think>/<answer>
   ]
   ```

2. **Test base model (before training):**
   - Does it use tags?
   - Is format consistent?

3. **Run SFT with format enforcement:**
   - Use `format` reward
   - Train for 100 steps

4. **Test SFT model:**
   - Does it always use tags?
   - Compare to base model

5. **Run GRPO with format reward:**
   - Weight: `accuracy=1.0`, `format=0.1`
   - Train for 50 steps

6. **Final test:**
   - Verify 100% format compliance
   - Check accuracy improvement

**Deliverable:**
- Script showing format compliance rates:
  - Base model: X%
  - SFT model: Y%
  - GRPO model: Z% (should be ~100%)

---

## Summary

**Key Takeaways:**
- `<think>` section shows reasoning, `<answer>` gives result
- Format rewards enforce structure during RL training
- System prompts establish format expectations
- Consistent format across training and inference is critical
- Low format weight (0.1) is sufficient for compliance

**Connection to Chain-of-Thought:**
- CoT uses prompts: "Let's think step by step"
- Structured reasoning uses tags: `<think>` and `<answer>`
- Tags enable programmatic validation and rewards

**Next Tutorial:** Group Relative Policy Optimization (GRPO) Explained

---

## Additional Resources

- [Annotated Code: Format Reward](../../annotated_code/core_training/rewards_ANNOTATED.py#L200-L250)
- [SFT Config Example](../../recipes/OpenR1-Distill-7B/sft/config_distill.yaml)
- [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903)
- [DeepSeek R1 Paper Section on Structured Reasoning](https://arxiv.org/abs/2401.xxxxx)

**Questions?** Open an issue on GitHub!
