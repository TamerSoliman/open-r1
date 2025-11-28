# Tutorial 16: Multi-Objective Optimization Strategies

**Target Audience:** Advanced
**Duration:** 40 minutes
**Prerequisites:** Tutorial 3 (GRPO), Tutorial 14 (Reward Design)

## Table of Contents
1. [Overview](#overview)
2. [Why Multiple Objectives?](#why-multiple-objectives)
3. [Combination Strategies](#combination-strategies)
4. [Pareto Optimization](#pareto-optimization)
5. [Dynamic Weighting](#dynamic-weighting)
6. [Hands-On Example](#hands-on-example)
7. [Summary](#summary)

---

## Overview

**Multi-objective optimization** balances competing goals:
- Accuracy (get correct answer)
- Format (use proper structure)
- Brevity (don't be too verbose)
- Reasoning quality (show clear steps)

**Challenge:** These objectives sometimes conflict!

---

## Why Multiple Objectives?

### Single Objective Problems

```python
# Only optimize accuracy
reward = 1.0 if correct else 0.0

# Problems:
# ❌ No format enforcement → inconsistent outputs
# ❌ No length control → extremely verbose
# ❌ No reasoning quality → lucky guesses rewarded
```

**Example:**
```
Model output (high reward, poor quality):
"The answer is x=2 because I tried x=1,x=2,x=3,x=4,x=5,x=6... and x=2 worked."

Better output (same reward, better quality):
"<think>2x+3=7, so 2x=4, thus x=2</think><answer>x=2</answer>"
```

### Multi-Objective Approach

```python
rewards = {
    "accuracy": 1.0 if correct else 0.0,
    "format": 1.0 if has_tags else 0.0,
    "brevity": 1.0 - len(text)/1000,
    "reasoning": count_steps / 5,
}

# Combine into single reward
total_reward = weighted_sum(rewards)
```

---

## Combination Strategies

### Strategy 1: Weighted Sum (Most Common)

```python
def weighted_sum_reward(completion, answer):
    r_accuracy = check_correctness(completion, answer)
    r_format = check_format(completion)
    r_brevity = check_length(completion)

    # Weights sum to > 1.0 (accuracy is primary)
    weights = {"accuracy": 1.0, "format": 0.1, "brevity": 0.01}

    return (
        weights["accuracy"] * r_accuracy +
        weights["format"] * r_format +
        weights["brevity"] * r_brevity
    )
```

**Pros:** Simple, interpretable
**Cons:** Weights are hyperparameters to tune

### Strategy 2: Lexicographic (Hierarchical)

```python
def lexicographic_reward(completion, answer):
    """Satisfy objectives in order of priority"""

    # Priority 1: Must have format (required)
    if not has_format(completion):
        return 0.0  # Fail immediately

    # Priority 2: Must be correct
    if not is_correct(completion, answer):
        return 0.5  # Partial credit for format

    # Priority 3: Bonus for brevity (if correct)
    brevity_bonus = max(0, 1.0 - len(completion)/500) * 0.2

    return 1.0 + brevity_bonus  # Max 1.2
```

**Pros:** Clear priorities, no weight tuning
**Cons:** Lower priorities rarely optimized

### Strategy 3: Product (Multiplicative)

```python
def product_reward(completion, answer):
    """All objectives must be satisfied"""

    r_accuracy = 1.0 if correct else 0.0
    r_format = 1.0 if has_format else 0.5  # Partial if no format
    r_quality = reasoning_score / 10  # 0.0 to 1.0

    # Product: If any is 0, total is 0!
    return r_accuracy * r_format * r_quality
```

**Pros:** Forces satisfying all objectives
**Cons:** One failure → zero reward (harsh)

---

## Pareto Optimization

### Concept: Pareto Front

**Pareto optimal:** Can't improve one objective without hurting another

```
Example: Accuracy vs Brevity

Accuracy ↑
  1.0 │     C ← Pareto optimal (high acc, medium length)
      │
  0.8 │  B    ← Pareto optimal (medium acc, short)
      │
  0.6 │A      ← Not optimal (low acc, short)
      │         Can improve to B without cost!
  0.0 └────────────────→ Brevity (shorter is better)
      0   100  200  300
```

### Multi-Objective Evolution

```python
# Instead of single best model, maintain population
population = [model_1, model_2, ..., model_n]

# Evaluate on multiple objectives
for model in population:
    model.accuracy = evaluate_accuracy(model)
    model.brevity = evaluate_brevity(model)
    model.format_compliance = evaluate_format(model)

# Keep Pareto-optimal models
pareto_front = [
    m for m in population
    if not exists(m2 in population where m2 dominates m)
]

# User picks from Pareto front based on preference
```

**Use case:** Want multiple model variants for different use cases

---

## Dynamic Weighting

### Adaptive Weights Based on Performance

```python
class DynamicWeights:
    def __init__(self):
        self.weights = {"accuracy": 1.0, "format": 0.1, "brevity": 0.01}

    def update(self, metrics, step):
        """Adjust weights based on current performance"""

        # If accuracy is high, increase format weight
        if metrics["accuracy"] > 0.8:
            self.weights["format"] = min(0.3, self.weights["format"] * 1.1)

        # If format compliance is low, increase its weight
        if metrics["format_compliance"] < 0.7:
            self.weights["format"] *= 1.2

        # If outputs are too long, increase brevity weight
        if metrics["avg_length"] > 500:
            self.weights["brevity"] *= 1.5

        return self.weights
```

### Curriculum-Based Weighting

```python
def curriculum_weights(step, total_steps):
    """Change objective importance over training"""
    progress = step / total_steps

    # Early: Focus on format and basic correctness
    if progress < 0.3:
        return {"accuracy": 1.0, "format": 0.5, "reasoning": 0.0}

    # Mid: Add reasoning quality
    elif progress < 0.7:
        return {"accuracy": 1.0, "format": 0.3, "reasoning": 0.2}

    # Late: Fine-tune all aspects
    else:
        return {"accuracy": 1.0, "format": 0.1, "reasoning": 0.1}
```

---

## Hands-On Example

### Example 1: Compare Weighting Schemes

```python
# Test different weight combinations
weight_configs = [
    {"name": "accuracy_only", "accuracy": 1.0, "format": 0.0, "brevity": 0.0},
    {"name": "balanced", "accuracy": 1.0, "format": 0.1, "brevity": 0.01},
    {"name": "format_heavy", "accuracy": 1.0, "format": 0.5, "brevity": 0.05},
]

results = {}
for config in weight_configs:
    model = train_grpo(
        reward_weights=config,
        max_steps=5000
    )
    results[config["name"]] = evaluate(model)

# Compare
print("Configuration | Accuracy | Format | Avg Length")
for name, metrics in results.items():
    print(f"{name:15s} | {metrics['acc']:6.1%} | {metrics['fmt']:6.1%} | {metrics['len']:8.0f}")
```

**Expected Output:**
```
Configuration   | Accuracy | Format | Avg Length
accuracy_only   |   85.2% |   45.3% |      823
balanced        |   84.1% |   92.7% |      412
format_heavy    |   82.3% |   98.1% |      387
```

### Example 2: Pareto Front Visualization

```python
import matplotlib.pyplot as plt

# Train multiple models with different weight combinations
models = []
for w_format in [0.0, 0.05, 0.1, 0.2, 0.5]:
    model = train(weights={"accuracy": 1.0, "format": w_format})
    accuracy = evaluate_accuracy(model)
    format_rate = evaluate_format(model)
    models.append((accuracy, format_rate, w_format))

# Plot Pareto front
accuracies = [m[0] for m in models]
format_rates = [m[1] for m in models]

plt.scatter(format_rates, accuracies)
plt.xlabel("Format Compliance")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Format Trade-off")
plt.savefig("pareto_front.png")
```

---

## Summary

**Key Takeaways:**

1. **Multiple objectives** prevent degenerate solutions
2. **Weighted sum** is simplest and most common
3. **Recommended weights:** 1.0 (primary), 0.1 (secondary), 0.01 (tertiary)
4. **Pareto optimization** for multiple model variants
5. **Dynamic weighting** adapts to training progress

**Recommended Configuration (DeepSeek R1):**

```yaml
reward_funcs:
  - accuracy      # Primary objective
  - format        # Structural correctness
  - tag_count     # Reasoning quality

reward_weights:
  - 1.0           # Accuracy is most important
  - 0.1           # Format is secondary
  - 0.05          # Reasoning is tertiary
```

**Debugging:**
```python
# Check individual reward distributions
for name in ["accuracy", "format", "reasoning"]:
    rewards = compute_reward(name, completions)
    print(f"{name}: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}")

# Look for conflicts
correlation = np.corrcoef(accuracy_rewards, brevity_rewards)[0,1]
if correlation < -0.5:
    print("WARNING: Accuracy and brevity are conflicting!")
```

**Part 3 COMPLETE!** Moving to Part 4: Competitive Programming...

---

## Resources
- [Multi-Objective RL Survey](https://arxiv.org/abs/2103.02631)
- [Pareto Optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization)
- [Annotated: Reward Configuration](../../annotated_code/core_training/configs_ANNOTATED.py#L150-L200)
