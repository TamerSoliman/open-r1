# ==============================================================================
# FILE: src/open_r1/utils/competitive_programming/ioi_utils.py
# CATEGORY: Utilities - IOI Problem Utilities
# PRIORITY: MEDIUM
# LINES: 42
# DEPENDENCIES: datasets, functools
# ==============================================================================
#
# OVERVIEW:
# Utilities for working with International Olympiad in Informatics (IOI) problems.
# Handles C++ header includes and test case loading from HuggingFace datasets.
#
# KEY FUNCTIONALITY:
# - add_includes(): Add problem-specific headers to C++ code
# - load_ioi_tests_for_year(): Load all tests for IOI year (cached)
# - load_ioi_tests(): Load tests for specific problem
#
# IOI problems require special headers and have structured test cases organized
# by problem and subtask.
# ==============================================================================

from collections import defaultdict
from functools import lru_cache

from datasets import load_dataset


def add_includes(code: str, problem_id: str) -> str:
    """
    WHAT: Add required headers and namespace declarations to C++ code for IOI problems

    WHY:
    - IOI problems require problem-specific header files (e.g., "prize.h")
    - Models forget to include these headers
    - bits/stdc++.h provides standard library functionality
    - using namespace std simplifies code

    HOW:
    1. Start with #include <bits/stdc++.h>
    2. Add problem-specific header: #include "problem_id.h"
    3. Add using namespace std if appropriate
    4. Prepend to code

    PROXIMAL CONTEXT:
    - INPUT: C++ code and problem identifier
    - OUTPUT: C++ code with headers

    DISTAL CONTEXT:
    - ORIGIN: Called before compiling IOI submissions
    - DESTINATION: Compilation with grader files

    EXAMPLE:
    ```python
    code = '''
    int find_best(int n) {
        return n;
    }
    '''
    patched = add_includes(code, "prize")
    # Result:
    # #include <bits/stdc++.h>
    # #include "prize.h"
    #
    # using namespace std;
    #
    # int find_best(int n) {
    #     return n;
    # }
    ```

    IOI PROBLEM STRUCTURE:
    - Each problem has a header file: problem_id.h
    - Header declares function signatures
    - Grader provides main() and calls student's functions
    - Student implements functions declared in header

    Fix common compilation errors for IOI problems.
    """
    if not code:
        return code

    # ADD STANDARD LIBRARY HEADER
    # bits/stdc++.h includes most C++ standard library headers
    # Non-standard but widely used in competitive programming
    code_header = "#include <bits/stdc++.h>\n"

    # ADD PROBLEM-SPECIFIC HEADER
    # Each IOI problem has a header file with function declarations
    # Example: prize.h declares int find_best(int n)
    problem_header_include = f'#include "{problem_id}.h"'
    if problem_header_include not in code:
        code_header += problem_header_include + "\n"

    # ADD using namespace std IF NEEDED
    # Skip if already present or if std:: is used explicitly
    # WHY: Models often forget std:: prefix
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"

    return code_header + code


@lru_cache
def load_ioi_tests_for_year(year: int) -> dict[str, dict[str, tuple[str, str]]]:
    """
    WHAT: Load all IOI test cases for a given year (with caching)

    WHY:
    - IOI test cases stored in HuggingFace dataset
    - Loading dataset is expensive (network + parsing)
    - Results are reused across multiple problems
    - @lru_cache prevents redundant loads

    HOW:
    1. Load dataset from HuggingFace Hub
    2. Organize by problem_id → test_name → (input, output)
    3. Cache results in memory
    4. Return nested dictionary

    PROXIMAL CONTEXT:
    - INPUT: Year (e.g., 2024)
    - OUTPUT: Nested dict of test cases

    DISTAL CONTEXT:
    - ORIGIN: Called by load_ioi_tests()
    - DESTINATION: Used by scoring functions

    EXAMPLE:
    ```python
    tests = load_ioi_tests_for_year(2024)
    # Result: {
    #   "prize": {
    #     "01-01": ("10\n1 2 3...", "5"),
    #     "01-02": ("20\n4 5 6...", "7"),
    #     ...
    #   },
    #   "vision": {...},
    #   ...
    # }
    ```

    DATASET STRUCTURE:
    - repo: "open-r1/ioi-test-cases"
    - name: "{year}" (e.g., "2024")
    - fields: problem_id, test_name, test_input, test_output

    CACHING BEHAVIOR:
    - @lru_cache caches by year
    - First call: Loads from HuggingFace Hub
    - Subsequent calls: Returns cached dict
    - Cache persists for Python session

    Load IOI tests for a given year.
    """
    # LOAD DATASET FROM HUGGINGFACE HUB
    # Format: open-r1/ioi-test-cases, config={year}
    tests_dataset = load_dataset("open-r1/ioi-test-cases", name=f"{year}", split="train")

    # ORGANIZE BY PROBLEM → TEST → (INPUT, OUTPUT)
    test_cases = defaultdict(dict)
    for test_case in tests_dataset:
        # Build nested dictionary:
        # test_cases[problem_id][test_name] = (input, output)
        test_cases[test_case["problem_id"]][test_case["test_name"]] = test_case["test_input"], test_case["test_output"]

    return test_cases


def load_ioi_tests(year: int, problem_id: str) -> dict[str, tuple[str, str]]:
    """
    WHAT: Load test cases for specific IOI problem

    WHY:
    - Convenient interface for getting single problem's tests
    - Leverages cached year data from load_ioi_tests_for_year()
    - Returns only relevant tests

    HOW:
    1. Call load_ioi_tests_for_year() (uses cache if available)
    2. Extract tests for specific problem_id
    3. Return test dictionary

    PROXIMAL CONTEXT:
    - INPUT: Year and problem identifier
    - OUTPUT: Dict mapping test name to (input, output) tuple

    DISTAL CONTEXT:
    - ORIGIN: Called by IOI scoring functions
    - DESTINATION: Used to run submissions against test cases

    EXAMPLE:
    ```python
    tests = load_ioi_tests(2024, "prize")
    # Result: {
    #   "01-01": ("10\n1 2 3...", "5"),
    #   "01-02": ("20\n4 5 6...", "7"),
    #   "02-01": ("15\n7 8 9...", "3"),
    #   ...
    # }

    for test_name, (input_data, expected_output) in tests.items():
        result = run_submission(code, input_data)
        if result == expected_output:
            print(f"Test {test_name}: PASS")
    ```

    TEST NAMING CONVENTION:
    - Format: "{subtask}-{testnum}"
    - Example: "01-01" = subtask 1, test 1
    - Example: "03-05" = subtask 3, test 5

    PERFORMANCE:
    - First call for year: Loads from HuggingFace Hub
    - Subsequent calls: Uses cached data (very fast)

    Load IOI tests for a given year and problem id.
    """
    # GET ALL TESTS FOR YEAR (cached)
    # Then extract tests for specific problem
    return load_ioi_tests_for_year(year)[problem_id]

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================
# 1. **add_includes()**:
#    - IOI problems require problem-specific headers
#    - Adds bits/stdc++.h and problem_id.h
#    - Adds using namespace std if appropriate
#
# 2. **Test Case Loading**:
#    - Tests stored in HuggingFace dataset
#    - Organized by year → problem → test
#    - @lru_cache prevents redundant loads
#
# 3. **IOI Structure**:
#    - Each problem has header file (problem_id.h)
#    - Tests organized by subtasks
#    - Test naming: "{subtask}-{testnum}"
#
# 4. **Caching Strategy**:
#    - load_ioi_tests_for_year() cached by year
#    - All problems for year loaded together
#    - Amortizes dataset loading cost
#
# 5. **Dataset Format**:
#    ```python
#    {
#        "problem_id": "prize",
#        "test_name": "01-01",
#        "test_input": "10\n1 2 3...",
#        "test_output": "5"
#    }
#    ```
#
# 6. **Usage Pattern**:
#    ```python
#    # Patch code with headers
#    code = add_includes(student_code, "prize")
#
#    # Load tests
#    tests = load_ioi_tests(2024, "prize")
#
#    # Run and score
#    for test_name, (input_data, output_data) in tests.items():
#        result = run_with_grader(code, input_data)
#        score += check_result(result, output_data)
#    ```
#
# 7. **Why Separate Functions**:
#    - add_includes: Code preprocessing
#    - load_ioi_tests_for_year: Bulk loading with cache
#    - load_ioi_tests: Convenient single-problem interface
# ==============================================================================
