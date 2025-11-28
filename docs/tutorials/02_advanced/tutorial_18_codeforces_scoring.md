# Tutorial 18: Codeforces Scoring and Test Generation

**Target Audience:** Advanced
**Duration:** 40 minutes
**Prerequisites:** Tutorial 17 (IOI Scoring), Annotated cf_scoring.py

## Table of Contents
1. [Overview](#overview)
2. [Scoring Modes](#scoring-modes)
3. [Generated Test Cases](#generated-test-cases)
4. [Partial Credit System](#partial-credit-system)
5. [Hands-On Example](#hands-on-example)
6. [Summary](#summary)

---

## Overview

**Codeforces** is a competitive programming platform with different scoring than IOI. Key differences:
- **Multiple scoring modes** (pass_fail, partial, weighted_sum)
- **Generated test cases** from generators
- **Gradual rewards** (0.0 to 1.1)

---

## Scoring Modes

### Mode 1: Pass/Fail (Binary)

```python
def pass_fail_scoring(problem_data, solution):
    """1.0 if all tests pass, 0.0 otherwise"""
    results = run_tests(solution, problem_data["tests"])

    if all(r == "AC" for r in results):
        return 1.0
    else:
        return 0.0
```

**Use:** Simple problems where partial credit doesn't make sense

### Mode 2: Partial Credit

```python
def partial_scoring(problem_data, solution):
    """Gradual reward based on fraction passed"""
    results = run_tests(solution, problem_data["tests"])

    passed = sum(1 for r in results if r == "AC")
    total = len(results)

    return passed / total  # 0.0 to 1.0
```

**Use:** Most Codeforces problems

### Mode 3: Weighted Sum

```python
def weighted_sum_scoring(problem_data, solution):
    """Combine sample and generated tests with weights"""
    sample_tests = problem_data["sample_tests"]
    generated_tests = problem_data["generated_tests"]

    # Score sample tests (basic)
    sample_results = run_tests(solution, sample_tests)
    sample_score = sum(1 for r in sample_results if r == "AC") / len(sample_tests)

    # Score generated tests (harder)
    gen_results = run_tests(solution, generated_tests)
    gen_score = sum(1 for r in gen_results if r == "AC") / len(generated_tests)

    # Weighted combination
    return 0.1 * sample_score + 1.0 * gen_score
```

**Maximum score:** 1.1 (sample tests give bonus)

---

## Generated Test Cases

### Why Generate Tests?

**Static tests:**
- Limited coverage
- Can be memorized
- Don't adapt to solution

**Generated tests:**
- Thousands of test cases
- Cover edge cases
- Harder to game

### Test Generator

```python
def generate_test_case(seed, difficulty):
    """Generate Codeforces test case"""
    random.seed(seed)

    if difficulty == "easy":
        n = random.randint(1, 10)
    elif difficulty == "medium":
        n = random.randint(10, 100)
    else:  # hard
        n = random.randint(100, 10000)

    arr = [random.randint(-1000, 1000) for _ in range(n)]

    return {"input": arr, "expected_output": correct_answer(arr)}
```

### Loading Generated Tests

```python
from datasets import load_dataset

# Load from parquet file
test_df = pd.read_parquet(problem_data["generated_tests_path"])

tests = []
for _, row in test_df.iterrows():
    tests.append({
        "input": row["input"],
        "output": row["output"],
        "difficulty": row["difficulty"],
    })
```

---

## Partial Credit System

### Graduated Rewards

```python
def graduated_scoring(solution, test_cases):
    """Different test groups worth different amounts"""

    easy_tests = [t for t in test_cases if t["difficulty"] == "easy"]
    medium_tests = [t for t in test_cases if t["difficulty"] == "medium"]
    hard_tests = [t for t in test_cases if t["difficulty"] == "hard"]

    easy_score = score_tests(solution, easy_tests) * 0.3
    medium_score = score_tests(solution, medium_tests) * 0.5
    hard_score = score_tests(solution, hard_tests) * 0.2

    return easy_score + medium_score + hard_score  # Max 1.0
```

---

## Hands-On Example

### Example: Score Codeforces Problem

```python
from src.open_r1.competitive_programming.cf_scoring import score_codeforces

problem_data = {
    "problem_id": "1234A",
    "sample_tests": [...],  # 3 sample tests
    "generated_tests_path": "data/cf_1234A_tests.parquet",
    "scoring_mode": "weighted_sum",
}

solution = """
def solve(arr):
    return max(arr) - min(arr)
"""

score = score_codeforces(problem_data, solution)
print(f"Score: {score:.2f}/1.10")  # e.g., 0.95/1.10
```

### Example: Compare Solutions

```python
# Correct solution
def solution_correct(arr):
    return max(arr) - min(arr)

# Buggy solution (doesn't handle negatives)
def solution_buggy(arr):
    return max(arr)  # Wrong!

scores = {
    "correct": score_codeforces(problem, solution_correct),
    "buggy": score_codeforces(problem, solution_buggy),
}

print(f"Correct: {scores['correct']:.2f}")  # 1.10
print(f"Buggy: {scores['buggy']:.2f}")     # 0.15 (only passes some tests)
```

---

## Summary

**Key Takeaways:**

1. **Three scoring modes:** pass_fail, partial, weighted_sum
2. **Generated tests** provide broader coverage
3. **Gradual rewards** (0.0 to 1.1) better for RL
4. **Sample tests** give small bonus (0.1)
5. **Main tests** determine most of score (1.0)

**Recommended Mode:**
```yaml
scoring_mode: weighted_sum  # Best for RL training
# Rewards both passing samples AND generated tests
```

**Configuration:**
```python
cf_config = {
    "sample_weight": 0.1,
    "generated_weight": 1.0,
    "max_tests": 100,  # Don't run all 10k tests
}
```

**Next Tutorial:** Code Execution Sandboxing

---

## Resources
- [Codeforces Platform](https://codeforces.com/)
- [Annotated: cf_scoring.py](../../annotated_code/competitive_programming/cf_scoring_ANNOTATED.py)
