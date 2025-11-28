"""
==============================================================================
FILE: src/open_r1/utils/competitive_programming/cf_scoring.py
CATEGORY: Competitive Programming - Codeforces Evaluation
PRIORITY: HIGH
LINES: 147
DEPENDENCIES:
    - Piston: Code execution backend
    - async_lru: Async caching for test cases
    - pandas: Parquet file reading
==============================================================================

OVERVIEW:
This module handles evaluation for Codeforces programming problems. Codeforces
is one of the world's most popular competitive programming platforms with
millions of users and thousands of problems.

ROLE IN DEEPSEEK R1:
- Evaluates models on Codeforces problems (real contest problems)
- Provides cf_code reward function for GRPO training
- Enables training on diverse, real-world coding challenges
- Supports both C++ and Python submissions

KEY FEATURES:
1. Multiple Scoring Modes: pass_fail, partial, weighted_sum
2. Generated Tests: Large test suites from parquet files
3. Custom Checkers: Problem-specific output validation
4. Batch Execution: Parallel test running with early stopping
5. Test Caching: LRU cache for test case loading

CODEFORCES SCORING:
- Problems have official tests (from contest) + generated tests
- All tests must pass for AC (Accepted)
- Partial credit available in some scoring modes
- Custom checkers for non-deterministic outputs

DATA FLOW:
Submission code → Piston compiler → Execute tests → Check outputs
    → Test results → Score (0.0-1.0) → Reward function

TYPICAL WORKFLOW:
1. Load Codeforces problem with official + generated tests
2. Compile submission code (C++ or Python)
3. Run code against all test inputs
4. Compare outputs using custom checker or exact match
5. Return score based on scoring mode
==============================================================================
"""

import asyncio
import os
from io import BytesIO
from typing import Literal

from async_lru import alru_cache

from .piston_client import PistonClient
from .utils import batched


"""
==============================================================================
SINGLE TEST CASE SCORING
==============================================================================
"""


async def score_single_test_case(
    client: PistonClient,
    problem_data: dict,
    test_input: str,
    test_output: str,
    submission: str,
    submission_language: str = "cpp",
) -> tuple[str, str]:
    """
    WHAT: Scores a single Codeforces test case by executing submission

    WHY: Codeforces problems require running code against test inputs
         and comparing outputs with expected results

    HOW:
        1. Prepare files (submission, input, expected output, checker)
        2. Configure grader (time/memory limits, input mode)
        3. Execute via Piston using Codeforces language runtime
        4. Return execution result

    PROXIMAL CONTEXT:
        - Input: Submission code, test input/output
        - Output: Piston execution result

    DISTAL CONTEXT:
        - Originates from: score_submission batch execution
        - Flows to: Test result aggregation → Score calculation

    Args:
        client: PistonClient instance for executing code
        problem_data: Dictionary containing problem configuration
                     Fields: time_limit, memory_limit, input_mode,
                            generated_checker (optional custom checker)
        test_input: Input data for the test case
        test_output: Expected output for the test case
        submission: Source code of the submission
        submission_language: Language of submission ("cpp" or "python")

    Returns:
        tuple[str, str] or False: Execution result or False on error

    FILE STRUCTURE:
        main.cpp or main.python: Submission code
        input.txt: Test input (read by submission)
        correct_output.txt: Expected output
        checker.py: Custom output checker (if provided)
        grader_config: Configuration (time/memory limits, input mode)

    GRADER_CONFIG FORMAT:
        TIME_LIMIT=2
        MEMORY_LIMIT=256
        INPUT_MODE=stdin

    INPUT MODES:
        - stdin: Read from standard input
        - file: Read from file (e.g., input.txt)

    CUSTOM CHECKER:
        Some problems have non-deterministic outputs
        Example: "Print any valid solution"
        Checker validates output correctness
        Returns "1" (pass) or "0" (fail)

    LANGUAGE RUNTIMES:
        - "cpp" → "c++17" (Piston language)
        - "python" → "cf_python3" (Piston language with Codeforces config)

    TIME LIMITS:
        run_timeout = (time_limit + 10) × 1000 ms
        +10 seconds hard limit for safety
        Actual time limit enforced by Codeforces grader script

    EXAMPLE:
        result = await score_single_test_case(
            client,
            {
                "time_limit": 2,
                "memory_limit": 256,
                "input_mode": "stdin",
                "generated_checker": None
            },
            "5\n1 2 3 4 5\n",
            "15\n",
            cpp_code,
            "cpp"
        )
        # result["run"]["stdout"] = "1" (pass) or "0" (fail)
    """
    # STEP 1: Validate submission language
    # WHY: Only C++ and Python are supported
    if submission_language not in ["python", "cpp"]:
        raise ValueError(f"Invalid submission language: {submission_language}")

    # STEP 2: Prepare execution request
    try:
        result = await client.send_execute(
            {
                "files": [
                    # WHAT: Submission code file
                    # WHY: Extension determines compilation/execution
                    {"name": f"main.{submission_language}", "content": submission},
                    # WHAT: Custom output checker (optional)
                    # WHY: Some problems need non-exact output validation
                    # EXAMPLE: "Print any permutation" → checker validates it's a valid permutation
                    *(
                        [{"name": "checker.py", "content": problem_data["generated_checker"]}]
                        if problem_data["generated_checker"]
                        else []
                    ),
                    # WHAT: Test input file
                    # WHY: Codeforces grader redirects this to stdin or provides as file
                    {"name": "input.txt", "content": test_input},
                    # WHAT: Expected output file
                    # WHY: Used by checker to validate correctness
                    {"name": "correct_output.txt", "content": test_output},
                    # WHAT: Grader configuration file
                    # WHY: Tells Codeforces grader the problem constraints
                    {
                        "name": "grader_config",
                        "content": "\n".join(
                            f"{key}={value}"
                            for key, value in {
                                "TIME_LIMIT": problem_data["time_limit"],
                                "MEMORY_LIMIT": problem_data["memory_limit"],
                                "INPUT_MODE": problem_data["input_mode"],
                            }.items()
                        ),
                    },
                ],
                # WHAT: Hard timeout for Piston execution
                # WHY: +10 seconds safety margin beyond problem time limit
                #      Prevents infinite loops from hanging
                "run_timeout": (problem_data["time_limit"] + 10) * 1000,
                # +10 seconds hard limit. time limits are handled by the codeforces script
            },
            # WHAT: Piston language runtime
            # WHY: Different runtimes for Python (cf_python3) vs C++ (c++17)
            #      cf_python3 includes Codeforces-specific setup
            language="cf_python3" if submission_language == "python" else "c++17",
        )
    except Exception as e:
        # WHAT: Catch execution errors
        # WHY: Piston errors shouldn't crash evaluation
        print(f"Error scoring submission: {e}")
        return False

    # STEP 3: Return Piston result
    # WHY: Caller will parse compile/run status
    return result


"""
==============================================================================
GENERATED TEST LOADING
==============================================================================
"""


@alru_cache(maxsize=32)  # TODO make this configurable
async def get_generated_contest_tests(contest_id: str) -> list[dict]:
    """
    WHAT: Loads generated test cases for a Codeforces contest from parquet file

    WHY: Official Codeforces tests are limited (5-20 tests)
         Generated tests provide comprehensive coverage (100-1000 tests)
         Parquet format provides efficient storage and fast loading

    HOW:
        1. Read parquet file from CF_TESTS_FOLDER
        2. Group tests by problem_id
        3. Cache results for future use

    PROXIMAL CONTEXT:
        - Input: Contest ID (e.g., "1234")
        - Output: Dictionary mapping problem_id → test list

    DISTAL CONTEXT:
        - Originates from: CF_TESTS_FOLDER environment variable
        - Flows to: get_generated_tests → score_submission

    Args:
        contest_id: Codeforces contest ID (e.g., "1234")

    Returns:
        dict: Dictionary mapping problem_id to list of test cases
              Example: {"1234/A": [{input: "...", output: "..."}], ...}

    CACHING:
        @alru_cache(maxsize=32) provides LRU caching
        WHY: Avoid re-reading parquet files
             Typical workflow: evaluate many submissions on same contest
             Cache hit rate: 90%+ in practice

    ENVIRONMENT VARIABLE:
        CF_TESTS_FOLDER must point to folder with parquet files
        Example: /data/codeforces_tests/
        Contains: test_cases_0001.parquet, test_cases_0002.parquet, ...

    PARQUET FILE FORMAT:
        Columns: problem_id, input, output
        Example row:
            problem_id: "1234/A"
            input: "5\n1 2 3 4 5\n"
            output: "15\n"

    DATA SOURCE:
        Generated tests from HuggingFace:
        https://huggingface.co/datasets/open-r1/codeforces

    EXAMPLE:
        tests = await get_generated_contest_tests("1234")
        # tests = {
        #     "1234/A": [{"input": "...", "output": "..."}],
        #     "1234/B": [{"input": "...", "output": "..."}],
        #     ...
        # }
    """
    import pandas as pd

    import aiofiles
    import aiofiles.os

    # STEP 1: Check CF_TESTS_FOLDER environment variable
    # WHY: Generated tests are large, stored separately
    tests_folder = os.environ.get("CF_TESTS_FOLDER", None)
    if not tests_folder:
        raise ValueError(
            "CF_TESTS_FOLDER environment variable not set! Please download the codeforces generated tests and set CF_TESTS_FOLDER to the folder path. See https://huggingface.co/datasets/open-r1/codeforces for more information."
        )

    # STEP 2: Verify folder exists
    if not await aiofiles.os.path.exists(tests_folder):
        raise ValueError(
            f"CF_TESTS_FOLDER path '{tests_folder}' does not exist! Please download the codeforces generated tests and set CF_TESTS_FOLDER to the folder path. See https://huggingface.co/datasets/open-r1/codeforces for more information."
        )

    # STEP 3: Construct parquet file path
    # WHY: Files named test_cases_0001.parquet, test_cases_0002.parquet, ...
    #      Contest ID padded to 4 digits
    parquet_path = os.path.join(tests_folder, f"test_cases_{int(contest_id):04d}.parquet")

    # STEP 4: Check if parquet file exists
    # WHY: Not all contests have generated tests
    if not await aiofiles.os.path.exists(parquet_path):
        return {}

    # STEP 5: Read parquet file asynchronously
    # WHY: Async I/O avoids blocking event loop
    #      Important for concurrent evaluation
    async with aiofiles.open(parquet_path, "rb") as f:
        content = await f.read()
        df = pd.read_parquet(BytesIO(content))

    # STEP 6: Group by problem_id and convert to dictionary
    # WHY: Need mapping from problem_id to test list
    # EXAMPLE:
    #   Input DataFrame:
    #     problem_id  input  output
    #     1234/A      "1"    "2"
    #     1234/A      "2"    "4"
    #     1234/B      "3"    "6"
    #   Output:
    #     {"1234/A": [{"input": "1", "output": "2"}, {"input": "2", "output": "4"}],
    #      "1234/B": [{"input": "3", "output": "6"}]}
    grouped_tests = df.groupby("problem_id").apply(lambda x: x[["input", "output"]].to_dict("records")).to_dict()

    return grouped_tests


async def get_generated_tests(problem_id: str) -> list[dict]:
    """
    WHAT: Loads generated test cases for a specific Codeforces problem

    WHY: Convenience function to get tests for one problem
         Wrapper around get_generated_contest_tests

    HOW:
        1. Extract contest_id from problem_id
        2. Load all contest tests
        3. Return tests for specific problem

    Args:
        problem_id: Codeforces problem ID (format: "contest_id/problem_letter")
                   Example: "1234/A"

    Returns:
        list[dict]: List of test cases
                   Example: [{"input": "...", "output": "..."}, ...]

    PROBLEM_ID FORMAT:
        Format: "contest_id/problem_letter"
        Examples:
            "1234/A" → Contest 1234, Problem A
            "1789/C" → Contest 1789, Problem C

    CACHING:
        Leverages get_generated_contest_tests cache
        Loading same contest problems shares cache

    EXAMPLE:
        tests = await get_generated_tests("1234/A")
        # tests = [{"input": "1\n", "output": "2\n"}, ...]
    """
    # STEP 1: Extract contest ID from problem ID
    # WHY: problem_id format is "contest_id/problem_letter"
    # EXAMPLE: "1234/A" → contest_id = "1234"
    contest_id = problem_id.split("/")[0]

    # STEP 2: Load contest tests and extract problem tests
    # WHY: get_generated_contest_tests is cached, efficient
    # FALLBACK: Return empty list if problem not found
    return (await get_generated_contest_tests(contest_id)).get(problem_id, [])


"""
==============================================================================
SUBMISSION SCORING
==============================================================================
"""


async def score_submission(
    client: PistonClient,
    problem_data: dict,
    submission: str,
    test_batch_size: int = 1,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    no_compile_reward: float = -0.1,
    no_submission_reward: float = -1.0,
    submission_language: str = "cpp",
) -> float:
    """
    WHAT: Scores a submission against all official and generated tests

    WHY: Need single score value for GRPO reward function
         Different scoring modes provide flexibility

    HOW:
        1. Load official + generated tests
        2. Execute submission on all tests in batches
        3. Aggregate results based on scoring mode
        4. Return score (0.0 to 1.0 or negative for errors)

    PROXIMAL CONTEXT:
        - Input: Submission code, problem data
        - Output: Score value

    DISTAL CONTEXT:
        - Originates from: cf_code reward function
        - Flows to: GRPO advantage calculation

    Args:
        client: PistonClient instance for executing code
        problem_data: Dictionary containing:
                     - id: Problem ID (e.g., "1234/A")
                     - official_tests: Tests from contest
                     - time_limit, memory_limit, input_mode
                     - generated_checker (optional)
        submission: Source code of the submission
        test_batch_size: Number of tests to run in parallel
                        1 = Sequential with early stopping
                        -1 = All tests in parallel
        scoring_mode: How to calculate final score
        no_compile_reward: Penalty for compilation errors
        no_submission_reward: Penalty for empty submission
        submission_language: "cpp" or "python"

    Returns:
        float: Score value
              None if invalid problem (no tests)
              Negative for errors
              0.0-1.0+ for valid submissions

    SCORING MODES:
        1. "pass_fail":
           - 1.0 if all tests pass
           - 0.0 if any test fails
           - Binary reward, no partial credit

        2. "partial":
           - passed_tests / total_tests
           - Example: 8/10 → 0.8
           - Encourages progress even on hard problems

        3. "weighted_sum":
           - pass_fail_score + 0.1 × partial_score
           - Example: Pass all → 1.0 + 0.1 = 1.1
           - Example: Pass 8/10 → 0.0 + 0.1 × 0.8 = 0.08
           - Rewards full solution highly, partial progress weakly

    NEGATIVE REWARDS:
        no_compile_reward = -0.1:
            - Penalty for compilation errors
            - Discourages syntactically invalid code
            - Less severe than no submission

        no_submission_reward = -1.0:
            - Penalty for empty/missing code
            - Strong signal to generate code
            - Worst possible outcome

    EARLY STOPPING:
        In "pass_fail" mode, stop on first failure
        WHY: If any test fails, score = 0.0
             No need to run remaining tests
        SPEEDUP: 50-90% reduction in execution time

    BATCH PARALLELISM:
        test_batch_size = 1:
            - Run tests sequentially
            - Stop immediately on failure
            - Best for "pass_fail" mode

        test_batch_size = 10:
            - Run 10 tests concurrently
            - Check batch results before continuing
            - Good balance

        test_batch_size = -1:
            - Run all tests concurrently
            - No early stopping
            - Best for "partial" and "weighted_sum" modes

    TEST COMPOSITION:
        total_tests = official_tests + generated_tests
        Official: 5-20 tests (from contest)
        Generated: 100-1000 tests (comprehensive coverage)
        Total: 105-1020 tests

    EXAMPLE USAGE:
        # pass_fail mode (binary)
        score = await score_submission(
            client, problem_data, code,
            test_batch_size=1,
            scoring_mode="pass_fail"
        )
        # score = 1.0 (all pass) or 0.0 (any fail)

        # partial mode (gradual progress)
        score = await score_submission(
            client, problem_data, code,
            test_batch_size=-1,
            scoring_mode="partial"
        )
        # score = 0.85 (850/1000 tests passed)

        # weighted_sum mode (GRPO default)
        score = await score_submission(
            client, problem_data, code,
            test_batch_size=10,
            scoring_mode="weighted_sum"
        )
        # score = 1.1 (perfect) or 0.085 (85% partial)
    """
    # STEP 1: Validate submission language
    if submission_language not in ["python", "cpp"]:
        raise ValueError(f"Invalid submission language: {submission_language}")

    # STEP 2: Load test cases (official + generated)
    # WHY: Need comprehensive test coverage
    test_cases = problem_data["official_tests"] + (await get_generated_tests(problem_data["id"]))

    # STEP 3: Handle invalid problems
    # WHY: Some problems are not coding problems (e.g., interactive, special judge)
    if test_cases is None or len(test_cases) == 0:
        return None

    # STEP 4: Handle empty submissions
    # WHY: Model should be penalized for not generating code
    if not submission:
        return no_submission_reward

    # STEP 5: Initialize test counters
    passed_test_cases = 0

    # STEP 6: Run tests in batches with early stopping
    # WHY: Balance parallelism with early stopping
    for test_batch_to_run in batched(test_cases, test_batch_size) if test_batch_size >= 1 else [test_cases]:
        # STEP 6a: Execute all tests in this batch concurrently
        results = await asyncio.gather(
            *[
                asyncio.create_task(
                    score_single_test_case(
                        client, problem_data, test_case["input"], test_case["output"], submission, submission_language
                    )
                )
                for test_case in test_batch_to_run
            ]
        )

        # STEP 6b: Check for compilation errors
        # WHY: If code doesn't compile, stop immediately
        if any(result and result["compile"]["code"] != 0 for result in results):
            return no_compile_reward

        # STEP 6c: Check test results
        # WHY: Codeforces grader outputs "1" to stdout for pass, "0" for fail
        tests_passed_results = [
            result and result["run"]["code"] == 0 and result["run"]["stdout"].strip() == "1" for result in results
        ]

        # STEP 6d: Early stopping for pass_fail mode
        # WHY: If any test fails, final score = 0.0
        if scoring_mode == "pass_fail" and any(not test_passed for test_passed in tests_passed_results):
            break

        # STEP 6e: Count passed tests
        passed_test_cases += sum(1 for test_passed in tests_passed_results if test_passed)

    # STEP 7: Calculate pass_fail score
    # WHY: Used by all scoring modes
    pass_fail_score = 1.0 if passed_test_cases == len(test_cases) else 0.0

    # STEP 8: Return score based on mode
    if scoring_mode == "pass_fail":
        # Binary: 1.0 (all pass) or 0.0 (any fail)
        return pass_fail_score
    elif scoring_mode == "partial":
        # Ratio: passed / total
        # Example: 850/1000 = 0.85
        return passed_test_cases / len(test_cases)
    elif scoring_mode == "weighted_sum":
        # Weighted: pass_fail + 0.1 × partial
        # Example: 1.0 + 0.1×1.0 = 1.1 (perfect)
        # Example: 0.0 + 0.1×0.85 = 0.085 (partial)
        return pass_fail_score + 0.1 * (passed_test_cases / len(test_cases))
    else:
        raise ValueError(f"Invalid scoring mode: {scoring_mode}")


"""
==============================================================================
KEY TAKEAWAYS - CODEFORCES SCORING
==============================================================================

1. **Codeforces Platform**:
   - World's most popular competitive programming site
   - Real contest problems (not synthetic)
   - Millions of submissions for training data

2. **Test Composition**:
   - Official tests: 5-20 (from contest)
   - Generated tests: 100-1000 (comprehensive)
   - Total: 105-1020 tests per problem

3. **Scoring Modes**:
   - pass_fail: Binary (1.0 or 0.0)
   - partial: Gradual (0.0 to 1.0)
   - weighted_sum: Hybrid (0.0 to 1.1)

4. **Negative Rewards**:
   - No submission: -1.0 (worst)
   - Compilation error: -0.1 (syntax issue)
   - Runtime error: 0.0 (logic issue)

5. **Performance Optimization**:
   - Early stopping: Skip remaining tests on failure
   - Batch parallelism: Run multiple tests concurrently
   - Test caching: LRU cache for parquet files
   - Typical speedup: 50-90% vs sequential

6. **Language Support**:
   - C++ (c++17 runtime)
   - Python (cf_python3 runtime)
   - Extensible to other languages

7. **Custom Checkers**:
   - Non-deterministic outputs supported
   - Example: "Print any valid permutation"
   - Checker validates correctness, not exact match

8. **Data Storage**:
   - Parquet files for generated tests
   - Efficient storage and fast loading
   - Grouped by contest for caching

9. **Integration with GRPO**:
   - cf_code reward function uses this module
   - weighted_sum mode provides best signal
   - Encourages both correctness and progress

10. **Error Handling**:
    - Graceful degradation on missing files
    - Clear error messages for setup issues
    - Exception handling for Piston errors

11. **Typical Problem**:
    - Time limit: 1-3 seconds
    - Memory limit: 256-512 MB
    - Input: stdin or file
    - Output: stdout

12. **GRPO Training**:
    - Use weighted_sum mode
    - Batch size: 10-20 for balance
    - Reward range: -1.0 to 1.1
    - Encourages full solutions over partial

==============================================================================
"""
