# Tutorial 17: IOI Scoring System Deep Dive

**Target Audience:** Advanced
**Duration:** 45 minutes
**Prerequisites:** Annotated ioi_scoring.py

## Table of Contents
1. [Overview](#overview)
2. [IOI Competition Format](#ioi-competition-format)
3. [Subtask-Based Scoring](#subtask-based-scoring)
4. [Test Result Types](#test-result-types)
5. [Early Stopping Optimization](#early-stopping-optimization)
6. [Hands-On Example](#hands-on-example)
7. [Summary](#summary)

---

## Overview

**IOI (International Olympiad in Informatics)** uses subtask-based scoring where problems are divided into subtasks, each worth a portion of the total points.

**Key Concepts:**
- Subtasks group test cases by difficulty
- Score = fraction of subtasks fully passed
- Early stopping when subtask fails
- More complex than simple pass/fail

---

## IOI Competition Format

### Problem Structure

```
Problem: "Find Maximum Subarray Sum"

Subtask 1 (20 points): N ≤ 100, all positive numbers
  - Test 1.1: [1,2,3,4,5]
  - Test 1.2: [10,20,5,15]

Subtask 2 (30 points): N ≤ 1000, mixed numbers
  - Test 2.1: [1,-2,3,4,-1]
  - Test 2.2: [-5,8,-3,10]
  - Test 2.3: [0,0,0,5,0]

Subtask 3 (50 points): N ≤ 10^6, any numbers
  - Test 3.1: Large random array
  - Test 3.2: Worst case (all negative)
  - Test 3.3: Alternating signs
```

### Scoring Rules

**To get subtask points:**
- Must pass ALL tests in that subtask
- Passing some tests in a subtask = 0 points
- Total score = sum of passed subtasks

**Example:**
```
Submission results:
  Subtask 1: All tests AC → 20 points
  Subtask 2: Test 2.2 WA → 0 points (not all tests passed)
  Subtask 3: All tests AC → 50 points

Total: 20 + 0 + 50 = 70 points
```

---

## Subtask-Based Scoring

### Implementation

```python
async def score_ioi_problem(problem_data, solution_code):
    """Score IOI problem with subtasks"""
    subtasks = problem_data["subtasks"]
    total_score = 0.0

    for subtask in subtasks:
        # Score this subtask
        subtask_passed = await score_subtask(subtask, solution_code)

        if subtask_passed:
            # Get all test results
            all_ac = all(result == "AC" for result in subtask_results)

            if all_ac:
                total_score += subtask["points"]
                # Example: 20/100 points for this subtask

    # Normalize to 0-1
    return total_score / 100.0
```

### Subtask Dependencies

**Some subtasks depend on others:**

```python
Subtask 1 (20 pts): N ≤ 100
Subtask 2 (30 pts): N ≤ 1000, requires Subtask 1
Subtask 3 (50 pts): N ≤ 10^6, requires Subtask 2

# If you fail Subtask 1, you automatically fail 2 and 3!
```

**Implementation:**
```python
def handle_dependencies(subtasks, results):
    """Mark dependent subtasks as failed if parent fails"""
    for subtask in subtasks:
        if subtask.get("depends_on"):
            parent_id = subtask["depends_on"]
            if not results[parent_id]["passed"]:
                results[subtask["id"]]["passed"] = False
                results[subtask["id"]]["reason"] = "Parent subtask failed"
```

---

## Test Result Types

### Standard Results

```python
class TestResult:
    AC = "Accepted"           # Correct answer
    WA = "Wrong Answer"        # Incorrect output
    TLE = "Time Limit Exceeded"  # Too slow
    MLE = "Memory Limit Exceeded"  # Too much memory
    RE = "Runtime Error"       # Crashed
    CE = "Compilation Error"   # Didn't compile
```

### Scoring Logic

```python
def subtask_score(test_results, subtask_points):
    """All tests must be AC for any points"""

    for result in test_results:
        if result != "AC":
            return 0.0  # Any non-AC → 0 points

    return subtask_points  # All AC → full points
```

**Example:**
```
Subtask 1 (20 points):
  Test 1.1: AC
  Test 1.2: AC
  Test 1.3: TLE  ← One failure!

Score: 0 points (not 13.3 points for 2/3)
```

---

## Early Stopping Optimization

### Problem: Wasted Computation

```python
# Naive approach
for test in subtask.tests:
    result = run_test(solution, test)  # 1 second per test

# If first test fails, why run the other 9?
# Subtask score is 0 anyway!
```

### Solution: Early Stopping

```python
async def score_subtask_optimized(subtask, solution):
    """Stop as soon as one test fails"""
    for test in subtask.tests:
        result = await run_test(solution, test)

        if result != "AC":
            # Subtask failed, no need to run more tests
            return 0.0

    # All tests passed!
    return subtask.points
```

**Speedup:**
```
Without early stopping:
  10 tests × 1 second = 10 seconds (even if first test fails)

With early stopping:
  First test fails → stop immediately
  Speedup: 10× faster!
```

### Batch Early Stopping

```python
async def score_subtask_batched(subtask, solution, batch_size=3):
    """Run tests in small batches, stop if any batch fails"""
    tests = subtask.tests

    for i in range(0, len(tests), batch_size):
        batch = tests[i:i+batch_size]

        # Run batch in parallel
        results = await asyncio.gather(*[
            run_test(solution, test) for test in batch
        ])

        # Check if any failed
        if any(r != "AC" for r in results):
            return 0.0  # Subtask failed

    return subtask.points  # All passed
```

---

## Hands-On Example

### Example 1: Score IOI Problem

```python
from src.open_r1.competitive_programming.ioi_scoring import score_ioi_problem

problem_data = {
    "subtasks": [
        {
            "id": 1,
            "points": 20,
            "tests": ["test_1_1.in", "test_1_2.in"],
        },
        {
            "id": 2,
            "points": 30,
            "tests": ["test_2_1.in", "test_2_2.in", "test_2_3.in"],
            "depends_on": 1,
        },
        {
            "id": 3,
            "points": 50,
            "tests": ["test_3_1.in", "test_3_2.in"],
            "depends_on": 2,
        },
    ]
}

solution_code = """
def max_subarray_sum(arr):
    max_sum = float('-inf')
    current_sum = 0
    for num in arr:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum
"""

# Score
score = await score_ioi_problem(problem_data, solution_code)
print(f"Score: {score:.1%}")  # e.g., 70.0%
```

### Example 2: Compare Algorithms

```python
# Brute force (works for small N only)
def brute_force_solution(arr):
    max_sum = float('-inf')
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            max_sum = max(max_sum, sum(arr[i:j+1]))
    return max_sum

# Kadane's algorithm (works for all N)
def kadane_solution(arr):
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

# Test both
results_brute = score_ioi_problem(problem, brute_force_solution)
results_kadane = score_ioi_problem(problem, kadane_solution)

print(f"Brute force: {results_brute:.0f} points (passes small cases)")
print(f"Kadane: {results_kadane:.0f} points (passes all cases)")
```

**Expected Output:**
```
Brute force: 20 points (Subtask 1 only, TLE on larger)
Kadane: 100 points (All subtasks)
```

---

## Summary

**Key Takeaways:**

1. **IOI uses subtask-based scoring** (not binary pass/fail)
2. **Must pass ALL tests** in subtask for points
3. **Partial credit per problem**, not per test
4. **Early stopping** critical for efficiency
5. **Dependencies** between subtasks reduce redundant testing

**Scoring Formula:**
```
Total score = Σ (subtask_points if all_tests_AC else 0)
Normalized = total_score / max_possible_points
```

**Implementation:**
```python
# From ioi_scoring.py
for subtask in problem.subtasks:
    all_ac = all(run_test(solution, test) == "AC" for test in subtask.tests)
    if all_ac:
        score += subtask.points
return score / 100.0  # Normalize
```

**Optimization:**
- Use early stopping (10× faster)
- Batch tests (3-5 per batch)
- Cache test results
- Skip dependent subtasks if parent fails

**Next Tutorial:** Codeforces Scoring

---

## Resources
- [IOI Regulations](https://ioinformatics.org/page/regulations/3)
- [Annotated: ioi_scoring.py](../../annotated_code/competitive_programming/ioi_scoring_ANNOTATED.py)
- [Example IOI Problems](https://oj.uz/)
