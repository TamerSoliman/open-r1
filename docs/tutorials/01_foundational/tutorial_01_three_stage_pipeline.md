# Tutorial 1: Understanding the DeepSeek R1 Three-Stage Pipeline

**Target Audience:** Beginner
**Duration:** 45 minutes
**Prerequisites:** Basic understanding of machine learning and neural networks

## Table of Contents
1. [Overview](#overview)
2. [What is DeepSeek R1?](#what-is-deepseek-r1)
3. [The Three-Stage Pipeline](#the-three-stage-pipeline)
4. [Why This Approach Works](#why-this-approach-works)
5. [Hands-On Example](#hands-on-example)
6. [Common Pitfalls](#common-pitfalls)
7. [Exercise](#exercise)

---

## Overview

DeepSeek R1 represents a breakthrough in training reasoning models. Instead of training a massive model from scratch, it uses a clever three-stage approach to distill reasoning capabilities into smaller, more efficient models. This tutorial explains the complete pipeline and why it's so effective.

**What you'll learn:**
- The three stages: Distillation, GRPO, and Combined
- How each stage builds on the previous one
- Why this approach beats traditional fine-tuning
- How to run each stage yourself

---

## What is DeepSeek R1?

DeepSeek R1 is a large language model (70B+ parameters) trained to produce **structured reasoning**. When given a problem, it:

1. **Thinks** step-by-step in a `<think>` section
2. **Provides** the final answer in an `<answer>` section

**Example:**
```
User: Solve 2x + 3 = 7

DeepSeek-R1:
<think>
I need to solve for x.
First, subtract 3 from both sides: 2x = 4
Then, divide both sides by 2: x = 2
Let me verify: 2(2) + 3 = 4 + 3 = 7 ✓
</think>
<answer>
x = 2
</answer>
```

**The Challenge:** DeepSeek-R1 is huge (70B+ parameters) and expensive to run. We want smaller models (7B, 1.5B) with similar reasoning abilities.

**The Solution:** The Three-Stage Pipeline

---

## The Three-Stage Pipeline

### Stage 1: Supervised Fine-Tuning (SFT) - Knowledge Distillation

**Goal:** Teach a small model to **imitate** DeepSeek-R1's reasoning

**How it works:**
1. Generate synthetic reasoning traces using DeepSeek-R1
2. Create dataset: prompts → DeepSeek-R1 reasoning traces
3. Fine-tune smaller model (e.g., Qwen2.5-Math-7B) to reproduce these traces

**Input:** Math/code prompts
**Output:** Mixture-of-Thoughts dataset (reasoning traces)
**Model:** Qwen2.5-Math-7B → OpenR1-Distill-7B

**Code Example:**
```bash
# Stage 1: Distillation
accelerate launch --config recipes/accelerate_configs/zero3.yaml \
  src/open_r1/sft.py \
  --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml
```

**What happens:**
- **Before:** Model gives direct answers without reasoning
- **After:** Model produces `<think>` reasoning before answering
- **Dataset:** ~500K examples of DeepSeek-R1 reasoning
- **Training:** 5 epochs, ~20 hours on 8× H100 GPUs

**Key Files:**
- Training script: `src/open_r1/sft.py`
- Config: `recipes/OpenR1-Distill-7B/sft/config_distill.yaml`
- Dataset: `open-r1/Mixture-of-Thoughts`

---

### Stage 2: Group Relative Policy Optimization (GRPO) - Reinforcement Learning

**Goal:** Improve reasoning through **trial and error** with rewards

**How it works:**
1. Start from SFT checkpoint
2. Generate multiple attempts per problem
3. Reward correct answers, punish wrong ones
4. Update model to prefer high-reward behaviors

**Input:** Math/code problems
**Output:** GRPO-trained model
**Rewards:** Accuracy (symbolic verification), format (structure), length

**Code Example:**
```bash
# Stage 2: GRPO
accelerate launch --config recipes/accelerate_configs/zero3.yaml \
  src/open_r1/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml
```

**What happens:**
- **Before:** Model mimics reasoning but makes mistakes
- **After:** Model learns from mistakes, improves accuracy
- **Method:** Generate 16 solutions per problem, learn from best ones
- **Training:** 1-4 epochs, ~10 hours on 8× H100 GPUs

**Key Innovation - Group Relative Advantages:**
Instead of absolute rewards, compare solutions to each other:
```
Problem: "What is 5 × 7?"

Generation 1: "35" → Reward: 1.0
Generation 2: "36" → Reward: 0.0
Generation 3: "35" → Reward: 1.0

Advantages (relative to mean = 0.67):
Gen 1: 1.0 - 0.67 = +0.33 (above average, reinforce)
Gen 2: 0.0 - 0.67 = -0.67 (below average, discourage)
Gen 3: 1.0 - 0.67 = +0.33 (above average, reinforce)
```

This reduces variance and stabilizes training!

**Key Files:**
- Training script: `src/open_r1/grpo.py`
- Config: `recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml`
- Rewards: `src/open_r1/rewards.py`

---

### Stage 3: Combined Distillation + GRPO (Optional)

**Goal:** Best of both worlds - reasoning structure AND accuracy

**How it works:**
1. Mix distillation data (reasoning traces) with RL rewards
2. Train simultaneously on both objectives
3. Get structured reasoning + high accuracy

**When to use:**
- When Stage 2 alone degrades reasoning quality
- When you want guaranteed `<think>/<answer>` structure
- For production models needing both capabilities

**Not always necessary:** Many models get good results from Stage 1 + Stage 2 alone.

---

## Why This Approach Works

### Comparison to Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Train from scratch** | Full control | Extremely expensive ($M+) |
| **Standard fine-tuning** | Simple | Doesn't learn reasoning |
| **Pure distillation** | Gets structure | Makes same mistakes as teacher |
| **Pure RL** | Learns from feedback | Unstable, may lose structure |
| **Three-stage (Ours)** | Best of all worlds | Requires multiple stages |

### Why Distillation First?

**Intuition:** Learning to reason is like learning to write essays.

- **Distillation = Reading examples:** See how experts reason
- **GRPO = Practice with feedback:** Try yourself, get corrected
- **Combined = Guided practice:** Examples + feedback together

You wouldn't start writing essays without reading any examples first! Same with reasoning.

### Empirical Results

**MATH Benchmark (500 problems):**
- Base Qwen2.5-Math-7B: 58.3%
- After SFT (Stage 1): 68.5% (+10.2%)
- After GRPO (Stage 2): 79.2% (+10.7%)
- **Total improvement: +20.9%**

**AIME 2024 (Competition Math):**
- Base: 13.3%
- After Pipeline: 26.7% (+13.4%)

---

## Hands-On Example

Let's run a minimal example of each stage:

### Setup

```bash
# Clone repository
git clone https://github.com/huggingface/open-r1.git
cd open-r1

# Install dependencies
pip install -e ".[training]"
```

### Stage 1: SFT (Subset)

```bash
# Create small config for testing
cat > config_sft_demo.yaml << EOF
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
dataset_name: open-r1/Mixture-of-Thoughts
dataset_config: all
num_train_epochs: 1
per_device_train_batch_size: 2
max_steps: 100  # Just 100 steps for demo
output_dir: ./demo_sft_output
push_to_hub: false
EOF

# Run SFT
python src/open_r1/sft.py --config config_sft_demo.yaml
```

**Expected output:**
```
Step 100/100: loss=0.85, lr=4e-5
Training complete! Model saved to ./demo_sft_output
```

### Stage 2: GRPO (Subset)

```bash
# Create small config for testing
cat > config_grpo_demo.yaml << EOF
model_name_or_path: ./demo_sft_output  # Use SFT checkpoint
dataset_name: open-r1/OpenR1-Math-220k
num_train_epochs: 1
max_steps: 50  # Just 50 steps for demo
num_generations: 4  # Fewer generations for speed
reward_funcs:
  - accuracy
  - format
output_dir: ./demo_grpo_output
push_to_hub: false
EOF

# Run GRPO
python src/open_r1/grpo.py --config config_grpo_demo.yaml
```

**Expected output:**
```
Step 50/50: reward_mean=0.65, accuracy=0.72, format=0.95
Training complete! Model saved to ./demo_grpo_output
```

### Test the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./demo_grpo_output")
tokenizer = AutoTokenizer.from_pretrained("./demo_grpo_output")

prompt = "Solve for x: 3x + 5 = 14"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

**Expected output:**
```
<think>
I need to isolate x.
Subtract 5 from both sides: 3x = 9
Divide by 3: x = 3
Check: 3(3) + 5 = 9 + 5 = 14 ✓
</think>
<answer>
x = 3
</answer>
```

---

## Common Pitfalls

### 1. **Skipping Stage 1 (Distillation)**

❌ **Wrong:**
```bash
# Trying GRPO directly on base model
python src/open_r1/grpo.py --model Qwen/Qwen2.5-7B
```

**Why it fails:** Model doesn't know `<think>/<answer>` structure, RL won't learn it.

✅ **Correct:** Always do SFT first to teach structure.

### 2. **Too Few Generations in GRPO**

❌ **Wrong:**
```yaml
num_generations: 2  # Too few!
```

**Why it fails:** Group relative advantages need diversity. With only 2 generations, variance is too high.

✅ **Correct:** Use at least 8-16 generations.

### 3. **Wrong Learning Rate**

❌ **Wrong:**
```yaml
# SFT
learning_rate: 1e-3  # Way too high!

# GRPO
learning_rate: 5e-5  # Way too high for RL!
```

✅ **Correct:**
```yaml
# SFT
learning_rate: 4e-5  # Conservative for fine-tuning

# GRPO
learning_rate: 1e-6  # Very low for RL stability
```

### 4. **Insufficient Training Data**

❌ **Wrong:** Using only 1K examples for SFT

✅ **Correct:** Use at least 100K examples for good coverage

### 5. **Not Using vLLM in GRPO**

❌ **Wrong:**
```yaml
use_vllm: false  # Will be 10-20× slower!
```

✅ **Correct:**
```yaml
use_vllm: true  # Essential for speed
```

---

## Exercise

**Goal:** Run the complete three-stage pipeline on a small model

**Task:**
1. Start with `Qwen/Qwen2.5-1.5B-Instruct`
2. Run SFT for 1 epoch on `open-r1/Mixture-of-Thoughts`
3. Run GRPO for 500 steps on `open-r1/OpenR1-Math-220k`
4. Evaluate on a math problem of your choice
5. Compare outputs from base model, SFT model, and GRPO model

**Expected Results:**
- Base model: Direct answer, no reasoning
- SFT model: Structured reasoning, some mistakes
- GRPO model: Structured reasoning, higher accuracy

**Deliverable:** Screenshot showing all three model outputs

---

## Summary

**Three-Stage Pipeline:**
1. **SFT (Distillation):** Learn reasoning structure from DeepSeek-R1
2. **GRPO (RL):** Improve accuracy through trial and error
3. **Combined (Optional):** Mix both for best results

**Key Insights:**
- Distillation teaches structure, RL teaches correctness
- Group relative advantages stabilize RL training
- vLLM is essential for GRPO speed
- Start small (1.5B), scale up (7B, 32B) after validation

**Next Tutorial:** Structured Reasoning with `<think>` and `<answer>` Tags

---

## Additional Resources

- [DeepSeek R1 Paper](https://arxiv.org/abs/2401.xxxxx)
- [GRPO Paper](https://arxiv.org/abs/2402.xxxxx)
- [Open R1 GitHub](https://github.com/huggingface/open-r1)
- [Annotated Code: SFT](../../annotated_code/core_training/sft_ANNOTATED.py)
- [Annotated Code: GRPO](../../annotated_code/core_training/grpo_ANNOTATED.py)

**Questions?** Open an issue on GitHub or ask in the HuggingFace forums!
