# Tutorial 14: Reward Function Design and Debugging

**Target Audience:** Advanced
**Duration:** 50 minutes
**Prerequisites:** Tutorial 3 (GRPO), Annotated rewards.py

## Table of Contents
1. [Overview](#overview)
2. [Reward Design Principles](#reward-design-principles)
3. [Common Reward Functions](#common-reward-functions)
4. [Debugging Reward Issues](#debugging-reward-issues)
5. [Multi-Objective Balancing](#multi-objective-balancing)
6. [Hands-On Example](#hands-on-example)
7. [Summary](#summary)

---

## Overview

**Reward functions** are the most critical component of GRPO. Poor rewards lead to:
- ❌ Model optimizing wrong behavior
- ❌ Reward hacking
- ❌ Training instability

**What you'll learn:**
- Principles of good reward design
- Common reward functions for reasoning
- Debugging reward issues
- Balancing multiple objectives

---

## Reward Design Principles

### Principle 1: Alignment with Goal

```python
# ❌ BAD: Reward length instead of correctness
def bad_reward(completion):
    return len(completion) / 1000  # Longer = better?

# ✅ GOOD: Reward correctness
def good_reward(completion, answer):
    predicted = extract_answer(completion)
    return 1.0 if predicted == answer else 0.0
```

### Principle 2: Clarity

```python
# ❌ BAD: Ambiguous partial credit
def bad_reward(completion):
    # What does 0.5 mean? Half right? Close?
    return random.uniform(0, 1)

# ✅ GOOD: Clear criteria
def good_reward(completion):
    has_think_tag = "<think>" in completion  # +0.1
    has_answer_tag = "<answer>" in completion  # +0.1
    is_correct = verify(completion)  # +1.0
    return has_think_tag * 0.1 + has_answer_tag * 0.1 + is_correct
```

### Principle 3: No Reward Hacking

```python
# ❌ BAD: Model can game this
def bad_reward(completion):
    # Counts reasoning steps, but model can fake it!
    return completion.count("Step ")

# Example hack:
# "Step Step Step Step" → High reward, no actual reasoning!

# ✅ GOOD: Verify actual correctness
def good_reward(completion, answer):
    return symbolic_verify(completion, answer)
```

---

## Common Reward Functions

### 1. Accuracy Reward (Binary)

```python
def accuracy_reward(completions, answers):
    """Binary: 1.0 if correct, 0.0 otherwise"""
    rewards = []
    for completion, answer in zip(completions, answers):
        predicted = extract_answer(completion)
        reward = 1.0 if predicted == answer else 0.0
        rewards.append(reward)
    return rewards
```

**Use:** Math, code execution (clear correct/wrong)

### 2. Format Reward

```python
def format_reward(completions):
    """Enforce <think>/<answer> structure"""
    rewards = []
    for completion in completions:
        has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
        proper_order = completion.find("<think>") < completion.find("<answer>")

        if has_think and has_answer and proper_order:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards
```

**Use:** Enforcing structured output

### 3. Partial Credit Reward

```python
def partial_credit_reward(completions, test_cases):
    """Gradual reward based on test cases passed"""
    rewards = []
    for completion in completions:
        passed = sum(1 for tc in test_cases if execute(completion, tc))
        reward = passed / len(test_cases)  # 0.0 to 1.0
        rewards.append(reward)
    return rewards
```

**Use:** Code problems with multiple test cases

### 4. Length Penalty

```python
def length_penalty_reward(completions, target_length=500):
    """Penalize extremely long/short outputs"""
    rewards = []
    for completion in completions:
        length = len(completion.split())
        if 50 < length < target_length:
            rewards.append(1.0)
        elif length <= 50:
            rewards.append(0.5)  # Too short
        else:
            rewards.append(max(0.0, 1.0 - (length - target_length) / 1000))
    return rewards
```

**Use:** Prevent degenerate solutions (too short/verbose)

---

## Debugging Reward Issues

### Issue 1: Reward Hacking

**Symptom:** Training loss decreases, but actual performance doesn't improve

```python
# Debug: Log completions with high rewards
def debug_reward_hacking(completions, rewards):
    top_indices = np.argsort(rewards)[-10:]  # Top 10 rewards

    print("=== HIGH REWARD COMPLETIONS ===")
    for idx in top_indices:
        print(f"Reward: {rewards[idx]:.3f}")
        print(f"Completion: {completions[idx][:200]}...")
        print()

# If high-reward completions are nonsense → reward hacking!
```

**Fix:** Add verification step

```python
def fixed_reward(completion, answer):
    # Don't just check format, verify answer!
    predicted = extract_answer(completion)
    is_correct = verify(predicted, answer)
    has_format = check_format(completion)

    # Format alone gives small reward
    # Correctness is primary objective
    return 1.0 * is_correct + 0.1 * has_format
```

### Issue 2: Reward Saturation

**Symptom:** All rewards are 0.0 or 1.0, no learning signal

```python
# Debug: Plot reward distribution
import matplotlib.pyplot as plt

rewards = compute_rewards(completions)
plt.hist(rewards, bins=20)
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.title("Reward Distribution")

# If all rewards at 0 or 1 → no gradient signal!
```

**Fix:** Add intermediate rewards

```python
def graded_reward(completion, answer):
    """Multiple levels of partial credit"""
    predicted = extract_answer(completion)

    # Level 1: Has answer tag (0.1)
    if "<answer>" not in completion:
        return 0.0

    # Level 2: Answer is numeric (0.3)
    if not is_numeric(predicted):
        return 0.1

    # Level 3: Answer close to correct (0.6)
    if abs(float(predicted) - float(answer)) < 1.0:
        return 0.3

    # Level 4: Exactly correct (1.0)
    if predicted == answer:
        return 1.0

    return 0.0
```

### Issue 3: Conflicting Rewards

**Symptom:** Model can't satisfy both objectives

```python
# Example: Format vs Brevity conflict
format_reward = 1.0 if has_think_tags else 0.0
brevity_reward = 1.0 if len(text) < 50 else 0.0

# Problem: Can't have detailed thinking AND be brief!
```

**Fix:** Adjust weights or relax constraints

```python
# Allow longer text if using proper format
format_reward = 1.0 if has_think_tags else 0.0
brevity_reward = 1.0 if len(text) < 500 else 0.5  # Relax constraint

# Or: Reduce brevity weight
total_reward = 1.0 * correctness + 0.1 * format + 0.01 * brevity
```

---

## Multi-Objective Balancing

### Strategy 1: Weighted Sum

```python
rewards = {
    "accuracy": accuracy_reward(completions),
    "format": format_reward(completions),
    "length": length_reward(completions),
}

weights = {
    "accuracy": 1.0,   # Primary objective
    "format": 0.1,     # Secondary
    "length": 0.01,    # Tertiary
}

final_reward = sum(weights[k] * rewards[k] for k in rewards)
```

### Strategy 2: Hierarchical

```python
def hierarchical_reward(completion, answer):
    """Must satisfy lower levels before higher ones count"""

    # Level 0: Format (required)
    if not has_proper_format(completion):
        return 0.0

    # Level 1: Plausible answer (0.3)
    predicted = extract_answer(completion)
    if not is_valid_number(predicted):
        return 0.0

    # Level 2: Close (0.6)
    if abs(float(predicted) - float(answer)) < 1.0:
        return 0.6

    # Level 3: Exact (1.0)
    if predicted == answer:
        return 1.0

    return 0.3
```

---

## Hands-On Example

### Example: Debug Reward Function

```python
from src.open_r1.rewards import accuracy_reward, format_reward

# Generate completions
completions = model.generate(prompts, n=16)

# Compute individual rewards
acc_rewards = accuracy_reward(completions, answers)
fmt_rewards = format_reward(completions)

# Debug: Check distributions
print(f"Accuracy mean: {np.mean(acc_rewards):.3f} (want 0.3-0.7)")
print(f"Format mean: {np.mean(fmt_rewards):.3f} (want 0.8+)")

# Debug: Check correlation
correlation = np.corrcoef(acc_rewards, fmt_rewards)[0, 1]
print(f"Correlation: {correlation:.3f} (want low)")

# If correlation is high → rewards are redundant!
# If accuracy mean is 0.0 → problem too hard
# If format mean is 0.0 → model not learning format
```

### Example: A/B Test Reward Functions

```python
# Variant A: Binary accuracy only
reward_a = accuracy_reward(completions)

# Variant B: Accuracy + format
reward_b = accuracy_reward(completions) + 0.1 * format_reward(completions)

# Train two models
model_a = train_grpo(dataset, reward_fn=lambda x: reward_a)
model_b = train_grpo(dataset, reward_fn=lambda x: reward_b)

# Evaluate
eval_a = evaluate(model_a, test_set)
eval_b = evaluate(model_b, test_set)

print(f"Model A (accuracy only): {eval_a['accuracy']:.1%}")
print(f"Model B (acc + format): {eval_b['accuracy']:.1%}")
print(f"Model B format compliance: {eval_b['format_rate']:.1%}")
```

---

## Summary

**Key Takeaways:**

1. **Align rewards with goals** (correctness > format > style)
2. **Prevent reward hacking** (verify, don't just count)
3. **Balance multiple objectives** (1.0 primary, 0.1 secondary)
4. **Debug reward issues** (plot distributions, check hacking)
5. **Iterate on reward design** (A/B test, refine)

**Recommended Weights (Math):**
```python
rewards = {
    "accuracy": 1.0,    # Must be correct
    "format": 0.1,      # Proper structure
    "reasoning": 0.05,  # Show work
}
```

**Red Flags:**
- ❌ All rewards near 0 or 1 (saturation)
- ❌ Training improves, eval doesn't (hacking)
- ❌ High correlation between rewards (redundant)

**Next Tutorial:** KL Divergence and Policy Regularization

---

## Resources
- [Annotated: rewards.py](../../annotated_code/core_training/rewards_ANNOTATED.py)
- [RLHF Best Practices](https://arxiv.org/abs/2203.02155)
- [Reward Hacking Examples](https://openai.com/research/faulty-reward-functions)
