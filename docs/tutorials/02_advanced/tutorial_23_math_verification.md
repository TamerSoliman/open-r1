# Tutorial 23: Math Verification with Symbolic Solvers

**Target Audience:** Advanced
**Duration:** 30 minutes

## Overview

**Symbolic verification** checks math answers using computer algebra systems (SymPy).

## Why Symbolic Verification?

```python
# String matching (unreliable)
answer = "x = 2"
if answer == "x = 2":  # What about "x=2" or "2" or "x = 2.0"?
    correct = True

# Symbolic verification (robust)
from sympy import sympify, Eq
answer_sym = sympify("x - 2")
if answer_sym == 0:  # Symbolically equivalent
    correct = True
```

## SymPy Integration

```python
from sympy import sympify, simplify, Eq

def verify_equation(student_answer, correct_answer):
    """Verify algebraic equivalence"""
    try:
        student = sympify(student_answer)
        correct = sympify(correct_answer)

        # Check if symbolically equivalent
        diff = simplify(student - correct)
        return diff == 0
    except:
        return False

# Examples
verify_equation("2*x", "x + x")  # True
verify_equation("x^2 - 1", "(x-1)*(x+1)")  # True
verify_equation("sin(x)^2 + cos(x)^2", "1")  # True
```

## Implementation in Rewards

```python
def accuracy_reward_symbolic(completions, answers):
    """Use symbolic verification for math problems"""
    rewards = []
    for completion, answer in zip(completions, answers):
        predicted = extract_answer(completion)

        # Try symbolic verification first
        if verify_equation(predicted, answer):
            rewards.append(1.0)
        # Fallback to string matching
        elif predicted.strip() == answer.strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
```

## Summary

- **Symbolic verification** handles equivalent forms
- **SymPy** for symbolic math
- **Fallback** to string matching if symbolic fails

**Next Tutorial:** Leaderboard Submission

## Resources
- [SymPy Documentation](https://docs.sympy.org/)
- [Annotated: rewards.py](../../annotated_code/core_training/rewards_ANNOTATED.py#L400-L500)
