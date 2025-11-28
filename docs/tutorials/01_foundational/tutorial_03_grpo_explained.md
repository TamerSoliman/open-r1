# Tutorial 3: Group Relative Policy Optimization (GRPO) Explained

**Target Audience:** Intermediate
**Duration:** 60 minutes
**Prerequisites:** Tutorial 1 (Three-Stage Pipeline), Tutorial 2 (Structured Reasoning)

## Table of Contents
1. [Overview](#overview)
2. [Why Reinforcement Learning?](#why-reinforcement-learning)
3. [From PPO to GRPO](#from-ppo-to-grpo)
4. [Group Relative Advantages](#group-relative-advantages)
5. [The GRPO Algorithm](#the-grpo-algorithm)
6. [Multi-Objective Rewards](#multi-objective-rewards)
7. [Hands-On Example](#hands-on-example)
8. [Common Pitfalls](#common-pitfalls)
9. [Exercise](#exercise)

---

## Overview

**Group Relative Policy Optimization (GRPO)** is the core RL algorithm in DeepSeek R1's Stage 2 training. It's a simplified, more efficient alternative to PPO (Proximal Policy Optimization) that works exceptionally well for language model reasoning.

**What you'll learn:**
- Why RL is needed after supervised learning
- How GRPO simplifies PPO by removing the value network
- Computing group relative advantages
- Multi-objective reward design
- Running GRPO training in practice

**Key Innovation:**
GRPO uses **relative rewards within each prompt's generations** instead of a separate value network, making it:
- ✅ Simpler (no critic network to train)
- ✅ More stable (no value function collapse)
- ✅ More memory-efficient (one model instead of two)
- ✅ Faster to train (fewer parameters to update)

---

## Why Reinforcement Learning?

### Limitations of Supervised Learning (SFT)

After Stage 1 (SFT), your model can produce reasoning traces in the correct format, but:

**Problem 1: Imitates without understanding correctness**

```
SFT Model Output:
<think>
Let me solve x + 2 = 5
Subtracting 2 from both sides: x = 3
Wait, let me check: 3 + 2 = 5 ✓
Actually, I made a mistake. Let me recalculate.
Subtracting 2: x = 7  ← WRONG, but formatted correctly!
</think>
<answer>
x = 7
</answer>
```

**Problem 2: Can't explore beyond training data**

SFT learns to mimic the teacher (DeepSeek-R1), but:
- Can't discover new reasoning paths
- Can't learn from its own mistakes
- Limited to teacher's capabilities

**Solution: Reinforcement Learning**

RL rewards **correct outcomes**, not just correct format:
- Reward = 1.0 if answer is correct
- Reward = 0.0 if answer is wrong
- Model learns to maximize reward

---

## From PPO to GRPO

### Standard PPO (Proximal Policy Optimization)

**PPO Architecture (2 models):**

```
┌─────────────────┐
│  Policy Model   │  Generates text, gets updated
│  (Actor)        │
└─────────────────┘

┌─────────────────┐
│  Value Model    │  Estimates expected reward, provides baseline
│  (Critic)       │
└─────────────────┘
```

**PPO Advantage Calculation:**

```python
# Standard PPO advantage
advantage = reward - value_estimate

# Example:
reward = 0.8  # Got 80% correct
value_estimate = 0.6  # Expected 60% correct
advantage = 0.2  # Better than expected! (positive reinforcement)
```

**PPO Challenges:**

1. **Two models to train:** Policy + Value network
2. **Value function instability:** Critic can diverge
3. **Memory overhead:** Store both models
4. **Complexity:** More hyperparameters to tune

### GRPO: Simplifying PPO

**Key Insight:** For language models, we can estimate baseline from **group statistics** instead of a separate value network.

**GRPO Architecture (1 model):**

```
┌─────────────────┐
│  Policy Model   │  Generates text, gets updated
│  (Actor only)   │
└─────────────────┘

No critic network! Baseline comes from group mean.
```

**GRPO Advantage Calculation:**

```python
# GRPO advantage (no value network needed)
advantage = reward - group_mean_reward

# Example (4 generations for same prompt):
rewards = [0.9, 0.7, 0.8, 0.4]  # Individual rewards
group_mean = 0.7  # Average across group

advantages = [
    0.9 - 0.7 = +0.2,  # Above average → reinforce
    0.7 - 0.7 =  0.0,  # Average → neutral
    0.8 - 0.7 = +0.1,  # Slightly above → reinforce
    0.4 - 0.7 = -0.3,  # Below average → discourage
]
```

**Why This Works:**

- Each prompt gets N generations (typically 16)
- Some generations are better, some worse
- **Relative comparison** within group provides learning signal
- No need for separate value estimator

---

## Group Relative Advantages

### Mathematical Formulation

**Setup:**
- Prompt: $p$
- Generations: $\{o_1, o_2, ..., o_N\}$ sampled from policy $\pi_\theta$
- Rewards: $\{r_1, r_2, ..., r_N\}$

**Group Mean (Baseline):**

$$\bar{r}_p = \frac{1}{N} \sum_{i=1}^{N} r_i$$

**Advantage for generation $i$:**

$$A_i = r_i - \bar{r}_p$$

**GRPO Objective:**

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}_{p, o_i \sim \pi_\theta} \left[ A_i \cdot \log \pi_\theta(o_i | p) \right]$$

**Interpretation:**
- Positive advantage ($A_i > 0$): Increase probability of this generation
- Negative advantage ($A_i < 0$): Decrease probability of this generation
- Zero advantage ($A_i = 0$): No change

### Concrete Example

**Prompt:** "Solve: 2x + 3 = 7"

**Generate 4 completions:**

| Generation | Answer | Reward | Advantage | Effect |
|------------|--------|--------|-----------|--------|
| Gen 1 | x = 2 ✓ | 1.0 | +0.375 | **Reinforce strongly** |
| Gen 2 | x = 5 ✗ | 0.0 | -0.625 | **Discourage strongly** |
| Gen 3 | x = 2 ✓ | 1.0 | +0.375 | **Reinforce strongly** |
| Gen 4 | x = 3 ✗ | 0.0 | -0.625 | **Discourage strongly** |

**Calculations:**

```python
rewards = [1.0, 0.0, 1.0, 0.0]
group_mean = 0.625

advantages = [
    1.0 - 0.625 = +0.375,
    0.0 - 0.625 = -0.625,
    1.0 - 0.625 = +0.375,
    0.0 - 0.625 = -0.625,
]
```

**Training Effect:**

- Generations with x = 2 get reinforced (probability ↑)
- Generations with x ≠ 2 get discouraged (probability ↓)
- Over many examples, model learns correct reasoning

### Why Relative Instead of Absolute?

**Absolute rewards (naive approach):**

```python
# Problem: Different prompts have different difficulty
prompt_easy = "What is 1 + 1?"  # reward = 1.0 (always)
prompt_hard = "Prove Fermat's Last Theorem"  # reward = 0.0 (always)

# Result: Model only learns easy problems!
```

**Relative advantages (GRPO):**

```python
# Easy problem:
rewards_easy = [1.0, 1.0, 1.0, 1.0]
advantages_easy = [0.0, 0.0, 0.0, 0.0]  # All equal → no learning signal

# Hard problem:
rewards_hard = [0.0, 0.0, 0.1, 0.0]  # One partially correct
advantages_hard = [-0.025, -0.025, +0.075, -0.025]  # Reinforce best attempt!

# Result: Model learns from both easy and hard problems
```

**Key Insight:** Relative advantages provide learning signal across all difficulty levels.

---

## The GRPO Algorithm

### High-Level Workflow

```
For each batch of prompts:
    1. Generate N completions per prompt (typically N=16)
    2. Compute rewards for all completions
    3. Calculate group mean reward per prompt
    4. Compute advantages (reward - group_mean)
    5. Update policy to increase probability of high-advantage completions
    6. Clip updates to prevent large policy changes (stability)
```

### Detailed Algorithm (from `grpo.py`)

**From `src/open_r1/grpo.py` - Annotated Training Loop:**

```python
# Step 1: Generate completions
prompts = batch["prompt"]  # List of N prompts
completions = policy_model.generate(
    prompts,
    num_generations=16,  # Generate 16 per prompt
    temperature=0.6,
    max_tokens=2048,
)

# Step 2: Compute rewards
rewards_list = []
for reward_func in reward_funcs:
    rewards = reward_func(
        prompts=repeated_prompts,  # Repeat each prompt 16 times
        completions=completions,
        **batch_metadata,
    )
    rewards_list.append(rewards)

# Combine multiple reward functions
# Shape: (N_prompts * 16, num_reward_funcs)
rewards_per_func = torch.stack(rewards_list, dim=1)

# Weighted sum of rewards
# Example: 1.0 * accuracy + 0.1 * format
final_rewards = rewards_per_func @ reward_weights
# Shape: (N_prompts * 16,)

# Step 3: Reshape to groups
# Shape: (N_prompts, 16)
grouped_rewards = final_rewards.view(num_prompts, num_generations)

# Step 4: Compute group mean (baseline)
# Shape: (N_prompts, 1)
group_means = grouped_rewards.mean(dim=1, keepdim=True)

# Step 5: Compute advantages
# Shape: (N_prompts, 16)
advantages = grouped_rewards - group_means

# Step 6: Policy gradient loss
# For each completion: advantage * log_prob
log_probs = policy_model.get_log_probs(prompts, completions)
policy_loss = -(advantages.flatten() * log_probs).mean()

# Step 7: Clip for stability (PPO-style clipping)
ratio = torch.exp(log_probs - old_log_probs)
clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
clipped_loss = -(advantages.flatten() * clipped_ratio).mean()
final_loss = torch.max(policy_loss, clipped_loss)

# Step 8: Backprop and update
optimizer.zero_grad()
final_loss.backward()
optimizer.step()
```

**Key Parameters:**

- `num_generations`: 16 (default) - More = better advantage estimates, but slower
- `clip_range`: 0.2 - Prevents drastic policy changes
- `learning_rate`: 1e-6 (lower than SFT) - RL is sensitive to LR
- `kl_penalty`: 0.01 - Keeps policy close to reference model

---

## Multi-Objective Rewards

### Why Multiple Rewards?

We want models to:
1. ✅ Get correct answers (accuracy)
2. ✅ Use proper format (`<think>/<answer>`)
3. ✅ Show detailed reasoning (tag count)

**Solution:** Combine multiple reward functions with weights

### Reward Configuration (from `config_demo.yaml`)

```yaml
reward_funcs:
  - accuracy        # Verifies mathematical correctness
  - format          # Checks <think>/<answer> structure
  - tag_count       # Counts reasoning steps

reward_weights:
  - 1.0             # Accuracy is primary objective
  - 0.1             # Format is important but secondary
  - 0.05            # Encourage reasoning, but don't overweight

# Final reward = 1.0 * accuracy + 0.1 * format + 0.05 * tag_count
```

### Reward Calculation Example

**Completion:**

```
<think>
Let me solve 2x + 3 = 7
Subtracting 3: 2x = 4
Dividing by 2: x = 2
Verification: 2(2) + 3 = 7 ✓
</think>
<answer>
x = 2
</answer>
```

**Individual Rewards:**

```python
accuracy_reward = 1.0  # Answer is correct
format_reward = 1.0    # Has <think> and <answer> tags
tag_count_reward = 1.0  # Has reasoning (4 steps)

# Weighted sum
final_reward = 1.0 * 1.0 + 0.1 * 1.0 + 0.05 * 1.0
             = 1.0 + 0.1 + 0.05
             = 1.15
```

**Wrong Answer Example:**

```python
accuracy_reward = 0.0  # Answer is wrong
format_reward = 1.0    # Still has correct format
tag_count_reward = 0.5  # Some reasoning present

final_reward = 1.0 * 0.0 + 0.1 * 1.0 + 0.05 * 0.5
             = 0.0 + 0.1 + 0.025
             = 0.125  # Low reward, but not zero (format bonus)
```

### Balancing Reward Weights

**Rule of Thumb:**

- **Primary objective (accuracy):** Weight = 1.0
- **Structure/format:** Weight = 0.05-0.2
- **Auxiliary signals (length, fluency):** Weight ≤ 0.1

**Example Mistake (too high format weight):**

```yaml
reward_weights:
  - 1.0   # Accuracy
  - 2.0   # Format TOO HIGH!

# Problem: Model learns to prioritize format over correctness
# Result: Perfect format, wrong answers
```

---

## Hands-On Example

### Step 1: Prepare Configuration

```yaml
# config_grpo_tutorial.yaml
model_name_or_path: ./sft_checkpoint  # From Stage 1
dataset_name: open-r1/OpenR1-Math-220k
num_train_epochs: 1
max_steps: 500

# GRPO-specific settings
num_generations: 16  # Generate 16 per prompt
temperature: 0.6
top_p: 0.95
max_completion_length: 2048

# Reward configuration
reward_funcs:
  - accuracy
  - format

reward_weights:
  - 1.0
  - 0.1

# Stability settings
learning_rate: 1e-6
clip_range: 0.2
kl_penalty: 0.01

output_dir: ./grpo_tutorial_output
```

### Step 2: Run GRPO Training

```bash
# Single GPU
accelerate launch src/open_r1/grpo.py \
  --config config_grpo_tutorial.yaml

# Multi-GPU (8 GPUs with DeepSpeed ZeRO-2)
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
  src/open_r1/grpo.py \
  --config config_grpo_tutorial.yaml
```

### Step 3: Monitor Training

**Console Output:**

```
Step 10/500:
  reward/accuracy: 0.45
  reward/format: 0.98
  reward/total: 0.548
  advantage/mean: 0.0
  advantage/std: 0.32
  policy/kl_divergence: 0.008

Step 100/500:
  reward/accuracy: 0.68  ← Improving!
  reward/format: 1.00
  reward/total: 0.78
  advantage/mean: 0.0
  advantage/std: 0.28
  policy/kl_divergence: 0.015

Step 500/500:
  reward/accuracy: 0.82  ← Much better!
  reward/format: 1.00
  reward/total: 0.92
  advantage/mean: 0.0
  advantage/std: 0.22
  policy/kl_divergence: 0.012
```

**What to watch:**

- ✅ `reward/accuracy` should increase over time
- ✅ `reward/format` should reach 1.0 quickly
- ✅ `advantage/std` should stabilize (0.2-0.4)
- ⚠️ `policy/kl_divergence` should stay low (<0.05)
  - If too high: Reduce learning rate or increase KL penalty

### Step 4: Test the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./grpo_tutorial_output")
tokenizer = AutoTokenizer.from_pretrained("./grpo_tutorial_output")

# Test problem
prompt = """Solve: 3x - 5 = 10"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.6,
)

print(tokenizer.decode(outputs[0]))
```

**Expected Output:**

```
<think>
I need to solve 3x - 5 = 10 for x.

Step 1: Add 5 to both sides
3x - 5 + 5 = 10 + 5
3x = 15

Step 2: Divide both sides by 3
x = 15 / 3
x = 5

Verification: 3(5) - 5 = 15 - 5 = 10 ✓
</think>
<answer>
x = 5
</answer>
```

---

## Common Pitfalls

### 1. **Too Few Generations Per Prompt**

❌ **Wrong:**

```yaml
num_generations: 4  # Too few!
```

**Problem:** Group mean is noisy with small N
**Effect:** Unstable advantages, poor learning
**Solution:** Use ≥8, ideally 16

✅ **Correct:**

```yaml
num_generations: 16  # Recommended
```

### 2. **Learning Rate Too High**

❌ **Wrong:**

```yaml
learning_rate: 5e-6  # Same as SFT
```

**Problem:** RL is more sensitive than SFT
**Effect:** Policy collapse, NaN losses
**Solution:** Use 1e-6 or lower

✅ **Correct:**

```yaml
learning_rate: 1e-6  # 5× lower than SFT
```

### 3. **Ignoring KL Divergence**

❌ **Wrong:**

```yaml
kl_penalty: 0.0  # No constraint!
```

**Problem:** Policy drifts too far from reference
**Effect:** Model forgets SFT knowledge, generates gibberish
**Solution:** Always use KL penalty

✅ **Correct:**

```yaml
kl_penalty: 0.01  # Keep policy anchored
```

### 4. **Imbalanced Reward Weights**

❌ **Wrong:**

```yaml
reward_weights:
  - 1.0   # Accuracy
  - 5.0   # Format way too high!
```

**Problem:** Model optimizes format over correctness
**Result:** Perfect `<think>/<answer>` tags, wrong math

✅ **Correct:**

```yaml
reward_weights:
  - 1.0   # Accuracy dominant
  - 0.1   # Format secondary
```

### 5. **Not Monitoring Advantage Statistics**

❌ **Wrong:** Only watching loss

**Problem:** Don't know if advantages are informative
**Solution:** Track `advantage/mean` and `advantage/std`

✅ **Correct Indicators:**

- `advantage/mean ≈ 0.0` (by construction)
- `advantage/std = 0.2-0.4` (healthy diversity)
- If `advantage/std < 0.1`: Rewards too similar, no signal
- If `advantage/std > 0.6`: Rewards too noisy, unstable

---

## Exercise

**Goal:** Train a GRPO model and understand advantage dynamics

### Task 1: Setup

```bash
# 1. Prepare small dataset
python -c "
from datasets import load_dataset
ds = load_dataset('open-r1/OpenR1-Math-220k', split='train[:100]')
ds.save_to_disk('./math_100_samples')
"

# 2. Create config
cat > config_grpo_exercise.yaml << EOF
model_name_or_path: ./sft_checkpoint
dataset_name: ./math_100_samples
num_train_epochs: 1
max_steps: 50

num_generations: 8
temperature: 0.6

reward_funcs:
  - accuracy
  - format

reward_weights:
  - 1.0
  - 0.1

learning_rate: 1e-6
output_dir: ./grpo_exercise
EOF
```

### Task 2: Run Training

```bash
python src/open_r1/grpo.py --config config_grpo_exercise.yaml
```

### Task 3: Analysis

**Answer these questions:**

1. What was the accuracy at step 10 vs step 50?
2. What is the average `advantage/std` across training?
3. Did `policy/kl_divergence` stay below 0.05?
4. Plot reward vs step (use WandB or logs)

### Task 4: Experiment with Advantages

**Modify `num_generations` and observe effects:**

```yaml
# Experiment 1: Very few generations
num_generations: 2  # Expect: Noisy advantages

# Experiment 2: Many generations
num_generations: 32  # Expect: Stable advantages, slower training
```

**Expected Observations:**

| num_generations | advantage/std | Training Speed | Stability |
|-----------------|---------------|----------------|-----------|
| 2 | High (0.5+) | Fast | Unstable |
| 8 | Medium (0.3) | Medium | Good |
| 16 | Low (0.25) | Slow | Very stable |
| 32 | Very low (0.2) | Very slow | Extremely stable |

### Task 5: Reward Weight Sensitivity

**Test different format weights:**

```yaml
# Experiment 1: No format reward
reward_weights: [1.0, 0.0]

# Experiment 2: High format reward
reward_weights: [1.0, 1.0]

# Question: Does high format weight hurt accuracy?
```

**Deliverable:**

- Table comparing accuracy with different format weights
- Explanation of which configuration works best and why

---

## Summary

**Key Takeaways:**

1. **GRPO simplifies PPO** by using group mean as baseline (no critic network)
2. **Group relative advantages** = reward - mean(group rewards)
3. **N generations per prompt** (typically 16) provide stable advantage estimates
4. **Multi-objective rewards** combine accuracy, format, and other signals
5. **Stability matters:** Use low LR (1e-6), KL penalty (0.01), and clipping (0.2)

**GRPO vs PPO:**

| Aspect | PPO | GRPO |
|--------|-----|------|
| Models | Policy + Value | Policy only |
| Baseline | Value network | Group mean |
| Memory | High (2 models) | Low (1 model) |
| Stability | Can diverge | More stable |
| Complexity | High | Low |

**When to Use GRPO:**

- ✅ Language model RL (reasoning, dialogue, code)
- ✅ Multi-generation sampling available
- ✅ Want simplicity and stability

**When to Use PPO:**

- Sequential decision-making (robotics, games)
- Single trajectory per episode
- Need value estimates for long-horizon credit assignment

**Next Tutorial:** Dataset Preparation and Chat Templates

---

## Additional Resources

- [GRPO Paper](https://arxiv.org/abs/2402.03300) (DeepSeek Team)
- [PPO Paper](https://arxiv.org/abs/1707.06347) (Schulman et al.)
- [Annotated Code: GRPO Training Loop](../../annotated_code/core_training/grpo_ANNOTATED.py#L200-L400)
- [Annotated Code: Reward Functions](../../annotated_code/core_training/rewards_ANNOTATED.py)

**Questions?** Open an issue on GitHub!
