# Tutorial 20: Debugging Code Generation Failures

**Target Audience:** Advanced
**Duration:** 30 minutes

## Table of Contents
1. [Overview](#overview)
2. [Common Failure Modes](#common-failure-modes)
3. [Debugging Strategies](#debugging-strategies)
4. [Fix Patterns](#fix-patterns)
5. [Summary](#summary)

---

## Overview

**Code generation failures** fall into categories:
- Syntax errors (CE)
- Wrong logic (WA)
- Timeout (TLE)
- Memory issues (MLE)

---

## Common Failure Modes

### 1. Syntax Errors

```python
# Model output (broken)
def solve(arr)
    return max arr)  # Missing colon, parens

# Fix: Add syntax validation
if not is_valid_python(code):
    reward = 0.0
```

### 2. Wrong Algorithm

```python
# Inefficient algorithm (TLE for N>1000)
def sort_array(arr):
    # Bubble sort O(n²)
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

# Should use: arr.sort()  # O(n log n)
```

### 3. Edge Cases

```python
# Fails on empty input
def max_element(arr):
    max_val = arr[0]  # IndexError if arr is empty!
    for x in arr:
        if x > max_val:
            max_val = x
    return max_val

# Fix: Handle edge case
def max_element(arr):
    if not arr:
        return None
    return max(arr)
```

---

## Debugging Strategies

### Strategy 1: Test on Failed Cases

```python
# Save failed test cases
failed_tests = []
for test in tests:
    result = execute(solution, test)
    if result != "AC":
        failed_tests.append({
            "input": test["input"],
            "expected": test["output"],
            "actual": result["output"],
        })

# Analyze patterns
print(f"Failed {len(failed_tests)}/{len(tests)} tests")
print(f"First failure: {failed_tests[0]}")
```

### Strategy 2: Error Analysis

```python
error_types = {"CE": 0, "WA": 0, "TLE": 0, "RE": 0}
for result in results:
    error_types[result["status"]] += 1

print(error_types)
# {"CE": 2, "WA": 15, "TLE": 3, "RE": 0}
# Conclusion: Mostly wrong logic (WA)
```

---

## Fix Patterns

### Pattern 1: Add Input Validation

```python
# Before
def solve(n, arr):
    return arr[n-1]  # Can crash!

# After
def solve(n, arr):
    if n < 1 or n > len(arr):
        return None
    return arr[n-1]
```

### Pattern 2: Optimize Algorithm

```python
# Before: O(n²) → TLE
def has_duplicate(arr):
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j]:
                return True
    return False

# After: O(n) using set
def has_duplicate(arr):
    return len(arr) != len(set(arr))
```

---

## Summary

**Common Issues:**
1. Syntax errors (5-10% of failures)
2. Wrong logic (60-70%)
3. Timeout (15-20%)
4. Edge cases (5-10%)

**Debugging Workflow:**
1. Categorize errors (CE/WA/TLE/RE)
2. Analyze failed test cases
3. Identify patterns
4. Apply fixes

**Next Tutorial:** LightEval Integration (Part 5)

---

## Resources
- [Python Common Errors](https://realpython.com/python-traceback/)
- [Algorithm Optimization](https://www.bigocheatsheet.com/)
