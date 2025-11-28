# Tutorial 22: Pass@K Metrics and Sampling Strategies

**Target Audience:** Advanced
**Duration:** 30 minutes

## Overview

**Pass@K** measures: "Generate K solutions, what's the probability at least one is correct?"

## Definition

```
Pass@K = Probability(at least 1 correct in K samples)

Example:
  Generate 10 solutions
  3 are correct
  Pass@10 ≈ 1.0 (very likely to get ≥1 correct)

  Generate 1 solution
  Correct 30% of time
  Pass@1 = 0.3
```

## Implementation

```python
def compute_pass_at_k(n, c, k):
    """
    n: total samples generated
    c: number correct
    k: number we're evaluating

    Pass@k = 1 - (n-c choose k) / (n choose k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod([1.0 - k/(n-i) for i in range(c)])
```

## Sampling Strategies

```python
# Strategy 1: High temperature
samples = model.generate(prompt, n=10, temperature=0.8)

# Strategy 2: Nucleus sampling
samples = model.generate(prompt, n=10, top_p=0.95)

# Strategy 3: Diverse beam search
samples = model.generate(prompt, num_beams=10, num_return_sequences=10)
```

## Summary

- **Pass@K** more robust than Pass@1
- **K=10** typical for code generation
- **Higher temperature** → more diversity

**Next Tutorial:** Math Verification

## Resources
- [Pass@K Paper](https://arxiv.org/abs/2107.03374) (Codex)
