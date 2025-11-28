# Tutorial 25: Error Analysis and Model Diagnostics

**Target Audience:** Advanced
**Duration:** 30 minutes

## Overview

**Error analysis** identifies patterns in model failures to guide improvements.

## Categorizing Errors

```python
errors = {
    "format": [],  # Wrong structure
    "arithmetic": [],  # Basic math errors
    "logic": [],  # Wrong reasoning
    "edge_case": [],  # Missed corner cases
}

for example in failed_examples:
    error_type = classify_error(example)
    errors[error_type].append(example)

# Print summary
for error_type, examples in errors.items():
    print(f"{error_type}: {len(examples)} ({len(examples)/len(failed_examples):.1%})")
```

**Example Output:**
```
format: 12 (8%)
arithmetic: 45 (30%)
logic: 78 (52%)
edge_case: 15 (10%)
```

## Difficulty Analysis

```python
# Group by difficulty
results_by_difficulty = {
    "easy": {"correct": 0, "total": 0},
    "medium": {"correct": 0, "total": 0},
    "hard": {"correct": 0, "total": 0},
}

for example in eval_set:
    difficulty = example["difficulty"]
    results_by_difficulty[difficulty]["total"] += 1
    if model_correct(example):
        results_by_difficulty[difficulty]["correct"] += 1

# Print accuracy by difficulty
for diff, stats in results_by_difficulty.items():
    acc = stats["correct"] / stats["total"]
    print(f"{diff}: {acc:.1%} ({stats['correct']}/{stats['total']})")
```

## Length Analysis

```python
import matplotlib.pyplot as plt

# Analyze solution lengths
lengths_correct = [len(s) for s in correct_solutions]
lengths_wrong = [len(s) for s in wrong_solutions]

plt.hist([lengths_correct, lengths_wrong], label=["Correct", "Wrong"])
plt.xlabel("Solution Length (tokens)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("length_analysis.png")
```

## Summary

- **Categorize errors** to find patterns
- **Difficulty analysis** shows where model struggles
- **Length analysis** reveals over/under-generation
- **Use insights** to improve training data/rewards

**Part 5 COMPLETE!** Moving to Part 6...

## Resources
- [Error Analysis Best Practices](https://arxiv.org/abs/2202.03286)
