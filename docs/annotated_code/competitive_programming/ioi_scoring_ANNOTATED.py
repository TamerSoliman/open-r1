"""
==============================================================================
FILE: src/open_r1/utils/competitive_programming/ioi_scoring.py
CATEGORY: Competitive Programming - IOI Evaluation
PRIORITY: HIGH
LINES: 336
DEPENDENCIES:
    - Piston: Code execution backend
    - asyncio: Asynchronous execution
    - ioi_utils: IOI test loading
==============================================================================

OVERVIEW:
This module handles evaluation for International Olympiad in Informatics (IOI)
problems, implementing the subtask-based scoring system used in IOI competitions.
IOI is one of the world's most prestigious programming competitions for high
school students.

ROLE IN DEEPSEEK R1:
- Evaluates models on Olympic-level programming problems
- Implements IOI-specific scoring rules (subtasks, partial credit)
- Provides feedback for GRPO training on code problems
- Enables the OlympicCoder model variant

KEY FEATURES:
1. Subtask-Based Scoring: IOI problems divided into subtasks
2. Test Result Types: AC, WA, TLE, RE, MLE, CE, PA
3. Batch Execution: Parallel test case execution with early stopping
4. Test Caching: Avoid re-running tests across subtasks
5. Piston Integration: Code execution via Piston workers

IOI SCORING SYSTEM:
- Problems have multiple subtasks, each worth some points
- Each subtask has multiple test cases
- Subtask score = minimum score across all its test cases
- Total score = sum of all subtask scores
- Partial credit possible (PA status)

DATA FLOW:
Submission code → Piston compiler → Execute tests → Compare outputs
    → Test results → Subtask scores → Total score → Reward function

TYPICAL WORKFLOW:
1. Load IOI problem with subtasks and test cases
2. Compile submission code
3. Run code against test inputs
4. Compare output with expected output
5. Aggregate scores by subtask (min score rule)
6. Return total score for reward function
==============================================================================
"""

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Union

from .ioi_utils import load_ioi_tests
from .piston_client import PistonClient, PistonError
from .utils import batched


"""
==============================================================================
DATA STRUCTURES
==============================================================================
"""


@dataclass
class TestResult:
    """
    WHAT: Represents the result of a single test case execution

    WHY: IOI problems have many test cases per subtask
         Need to track individual test results before aggregating

    HOW: Stores test name, score, status code, and detailed feedback

    PROXIMAL CONTEXT:
        - Input: Code execution result from Piston
        - Output: Used in SubtaskResult aggregation

    DISTAL CONTEXT:
        - Originates from: Piston code execution
        - Flows to: SubtaskResult → IOI reward function

    Attributes:
        test_name: Name of the test case (e.g., "01-easy-1.in")
        score: Score achieved for this test (0.0 to 1.0)
               1.0 = Accepted (AC), 0.0 = Failed, intermediate = Partial (PA)
        status: Status code of the test result
        feedback: Detailed feedback message from the judge or an error message

    STATUS CODES:
        - AC (Accepted): Perfect output, score = 1.0
        - WA (Wrong Answer): Incorrect output, score = 0.0
        - TLE (Time Limit Exceeded): Exceeded time limit, score = 0.0
        - MLE (Memory Limit Exceeded): Exceeded memory limit, score = 0.0
        - RE (Runtime Error): Crashed during execution, score = 0.0
        - CE (Compilation Error): Failed to compile, score = 0.0
        - PA (Partial Answer): Partially correct, 0.0 < score < 1.0
        - SKIPPED: Not yet executed (initial state)

    EXAMPLE:
        TestResult(
            test_name="01-easy-1.in",
            score=1.0,
            status="AC",
            feedback="Correct output"
        )
    """

    test_name: str
    score: float = 0.0
    status: str = "SKIPPED"
    feedback: str = None


@dataclass
class SubtaskResult:
    """
    WHAT: Represents the result of a subtask containing multiple test cases

    WHY: IOI uses subtask-based scoring where each subtask is worth points
         Subtask score = minimum score across all its test cases

    HOW: Aggregates multiple TestResult objects using min score rule

    PROXIMAL CONTEXT:
        - Input: List of TestResult objects
        - Output: Aggregated subtask score and status

    DISTAL CONTEXT:
        - Originates from: Multiple Piston executions
        - Flows to: Total IOI score → Reward function

    Attributes:
        problem: Problem identifier (e.g., "sorting", "maze")
        subtask: Subtask identifier (e.g., "1", "2a")
        points: Maximum points available for this subtask (e.g., 10, 20)
        score_precision: Number of decimal places for score rounding
        test_results: List of individual test case results

    SUBTASK SCORING RULES:
        1. All test cases must pass for full subtask score
        2. If any test fails, subtask score = 0
        3. Partial credit possible if test returns 0 < score < 1
        4. Subtask score = min(test_scores) × points

    EXAMPLE:
        SubtaskResult(
            problem="sorting",
            subtask="1",
            points=20.0,
            test_results=[
                TestResult("01.in", 1.0, "AC"),
                TestResult("02.in", 1.0, "AC"),
                TestResult("03.in", 0.5, "PA")  # Partial credit
            ]
        )
        # score = min(1.0, 1.0, 0.5) = 0.5
        # weighted_score = 0.5 × 20 = 10.0 points
    """

    problem: str = None
    subtask: str = None

    points: float = 0.0
    score_precision: int = 2

    test_results: list[TestResult] = field(default_factory=list)

    @property
    def status(self):
        """
        WHAT: Determines the overall status of the subtask

        WHY: Need single status code to represent subtask health
             Worst status across all tests determines subtask status

        HOW: Use priority ordering where CE (compilation error) is worst,
             AC (accepted) is best

        Returns:
            str: The status with the highest priority (lowest value)

        STATUS PRIORITY (worst to best):
            CE (-1): Compilation error → can't run any tests
            RE (0): Runtime error → code crashes
            WA (1): Wrong answer → logic error
            MLE (2): Memory limit exceeded → memory issue
            TLE (3): Time limit exceeded → performance issue
            PA (4): Partial answer → some correctness
            AC (5): Accepted → perfect
            SKIPPED (999): Not yet run

        EXAMPLE:
            test_results = [
                TestResult(..., status="AC"),
                TestResult(..., status="WA"),  # ← This determines status
                TestResult(..., status="AC")
            ]
            # status = "WA" (worst status wins)
        """
        status_prios = {"CE": -1, "RE": 0, "WA": 1, "MLE": 2, "TLE": 3, "PA": 4, "AC": 5, "SKIPPED": 999}
        return min([x.status for x in self.test_results], key=lambda x: status_prios[x])

    @property
    def score(self):
        """
        WHAT: Calculates the raw score (0.0 to 1.0) for the subtask

        WHY: IOI subtask scoring uses minimum score across all test cases
             All tests must pass for full score

        HOW: Take minimum score across all test results

        Returns:
            float: The rounded minimum score (0.0 to 1.0)

        SCORING RULE:
            score = min(score_1, score_2, ..., score_n)

        EXAMPLES:
            test_scores = [1.0, 1.0, 1.0] → score = 1.0 (all pass)
            test_scores = [1.0, 0.0, 1.0] → score = 0.0 (one fails)
            test_scores = [1.0, 0.5, 1.0] → score = 0.5 (partial credit)
        """
        return (
            0
            if not self.test_results
            else round(min([test_result.score for test_result in self.test_results]), self.score_precision)
        )

    @property
    def weighted_score(self):
        """
        WHAT: Calculates the weighted score in points for the subtask

        WHY: Subtasks are worth different amounts of points
             Need to multiply raw score (0-1) by point value

        HOW: weighted_score = raw_score × points

        Returns:
            float: The rounded weighted score

        EXAMPLES:
            score = 1.0, points = 20 → weighted_score = 20.0
            score = 0.5, points = 20 → weighted_score = 10.0
            score = 0.0, points = 20 → weighted_score = 0.0

        TOTAL PROBLEM SCORE:
            Sum weighted_score across all subtasks
            Max score = sum of all points
        """
        return (
            0
            if not self.test_results
            else round(
                min([test_result.score for test_result in self.test_results]) * self.points, self.score_precision
            )
        )

    def to_dict(self):
        """
        WHAT: Converts the SubtaskResult to a dictionary representation

        WHY: Need JSON-serializable format for logging, debugging, caching

        HOW: Convert all fields to dict, recursively convert test_results

        Returns:
            dict: Dictionary containing all subtask result data

        USAGE:
            Used for:
            - WandB logging
            - Test result caching
            - Debug output
            - Reward function input
        """
        return {
            "problem": self.problem,
            "subtask": self.subtask,
            "score": self.score,
            "weighted_score": self.weighted_score,
            "points": self.points,
            "score_precision": self.score_precision,
            "status": self.status,
            "test_results": [asdict(test_result) for test_result in self.test_results],
        }


"""
==============================================================================
STATUS EXTRACTION
==============================================================================
"""


def _extract_single_status(score: float, feedback: str) -> str:
    """
    WHAT: Determines the status code based on the score and feedback message

    WHY: Piston returns score (0.0-1.0) and feedback string
         Need to map these to IOI status codes

    HOW: Check score and parse feedback for error keywords

    Args:
        score: The numeric score (0.0 to 1.0)
        feedback: The feedback message from the execution

    Returns:
        str: Status code ('CE', 'MLE', 'TLE', 'WA', 'RE', 'AC', or 'PA')

    LOGIC:
        if score == 0.0:
            Check feedback for error type
        elif score == 1.0:
            Return "AC" (Accepted)
        else:
            Return "PA" (Partial Answer)

    FEEDBACK KEYWORDS:
        - "Compilation error" → CE
        - "Memory limit exceeded" → MLE
        - "Time limit exceeded" → TLE
        - "Output isn't correct" → WA
        - Other errors → RE (Runtime Error)

    EXAMPLES:
        _extract_single_status(0.0, "Time limit exceeded") → "TLE"
        _extract_single_status(1.0, "Correct output") → "AC"
        _extract_single_status(0.5, "Partially correct") → "PA"
    """
    if score == 0.0:
        if "Compilation error" in feedback:
            return "CE"
        elif "Memory limit exceeded" in feedback:
            return "MLE"
        elif "Time limit exceeded" in feedback:
            return "TLE"
        elif "Output isn't correct" in feedback:
            return "WA"
        else:
            return "RE"
    elif score == 1.0:
        return "AC"
    else:
        return "PA"


"""
==============================================================================
SINGLE TEST CASE SCORING
==============================================================================
"""


async def score_single_test_case(
    client: PistonClient, subtask: dict, test_name: str, test_input: str, test_output: str, submission: str
) -> TestResult:
    """
    WHAT: Scores a single test case by running the submission against test input/output

    WHY: IOI evaluation requires running code against many test cases
         Each test case is scored independently

    HOW:
        1. Execute submission with test_input via Piston
        2. Compare output with test_output
        3. Extract score and status from result

    PROXIMAL CONTEXT:
        - Input: Submission code, test input/output
        - Output: TestResult with score and status

    DISTAL CONTEXT:
        - Originates from: score_subtask batch execution
        - Flows to: SubtaskResult aggregation → Total score

    Args:
        client: PistonClient instance for executing code
        subtask: Dictionary containing subtask configuration
                 (time_limit, memory_limit, grader_files, etc.)
        test_name: Name of the test case (e.g., "01.in")
        test_input: Input data for the test case
        test_output: Expected output for the test case
        submission: Source code of the submission

    Returns:
        TestResult: Result of the test case execution

    EXECUTION FLOW:
        submission + test_input → Piston → actual_output
        actual_output vs test_output → score (0.0, 0.5, or 1.0)
        score + feedback → TestResult

    EXAMPLE:
        result = await score_single_test_case(
            client,
            {"id": "sorting", "time_limit": 1.0, "memory_limit": 256},
            "01-easy.in",
            "3\n1 3 2\n",
            "1 2 3\n",
            submission_code
        )
        # result.score = 1.0, result.status = "AC"
    """
    # Run submission for this test case
    # WHY: Need to execute code and get score + feedback
    score, feedback = await run_submission(client, subtask, test_input, submission, test_output)
    score = float(score)

    return TestResult(
        test_name=test_name, score=score, status=_extract_single_status(score, feedback), feedback=feedback
    )


"""
==============================================================================
SUBTASK SCORING
==============================================================================
"""


async def score_subtask(
    client: PistonClient,
    subtask: dict,
    submission: str,
    test_case_run_cache: Union[dict, None] = None,
    test_batch_size: int = 1,
) -> SubtaskResult:
    """
    WHAT: Scores all test cases in a subtask with batching and early stopping

    WHY: IOI subtasks can have many test cases (10-50+)
         Optimize by:
         1. Caching results (tests may appear in multiple subtasks)
         2. Early stopping (if any test fails, subtask score = 0)
         3. Batch parallelism (run multiple tests concurrently)

    HOW:
        1. Initialize with cached results
        2. Run non-cached tests in batches
        3. Stop early if any test fails (score = 0.0)
        4. Aggregate results into SubtaskResult

    PROXIMAL CONTEXT:
        - Input: Subtask config, submission code, cache
        - Output: SubtaskResult with all test results

    DISTAL CONTEXT:
        - Originates from: score_subtasks (all subtasks)
        - Flows to: IOI total score → Reward function

    Args:
        client: PistonClient instance for executing code
        subtask: Dictionary containing subtask configuration
                 Fields: id, subtask, score (points), score_precision,
                        test_names, test_cases (or year for loading)
        submission: Source code of the submission
        test_case_run_cache: Optional cache of previously run test cases
                            Avoids re-running tests in multiple subtasks
        test_batch_size: Number of test cases to run in parallel
                        1 = Run tests sequentially with early stopping
                        -1 = Run all tests in parallel (no early stopping)

    Returns:
        SubtaskResult: Result of the subtask evaluation

    BATCHING STRATEGY:
        test_batch_size = 1:
            - Run one test at a time
            - Stop immediately on first failure
            - Fastest for likely failures

        test_batch_size = 10:
            - Run 10 tests concurrently
            - Stop after batch if any failed
            - Good balance

        test_batch_size = -1:
            - Run all tests concurrently
            - No early stopping
            - Fastest for likely successes

    CACHING:
        Multiple subtasks may share test cases
        Cache avoids re-execution
        Cache key = test_name

    EARLY STOPPING:
        Since subtask score = min(test_scores),
        if any test scores 0.0, final score = 0.0
        No need to run remaining tests

    EXAMPLE:
        subtask = {
            "id": "sorting",
            "subtask": "1",
            "score": 20.0,
            "score_precision": 2,
            "test_names": ["01.in", "02.in", "03.in"],
            "test_cases": [(...), (...), (...)]
        }
        result = await score_subtask(client, subtask, code, cache={}, batch_size=1)
        # result.weighted_score = min(scores) × 20.0
    """
    # STEP 1: Initialize SubtaskResult
    subtask_result = SubtaskResult(
        problem=subtask["id"],
        subtask=subtask["subtask"],
        points=subtask["score"],
        score_precision=subtask["score_precision"],
        test_results=[],
    )

    # STEP 2: Identify tests that are not cached
    # WHY: Avoid re-running tests that were executed in previous subtasks
    tests_to_run = [
        (ti, test_name)
        for ti, test_name in enumerate(subtask["test_names"])
        if test_case_run_cache is None or test_name not in test_case_run_cache
    ]

    # STEP 3: Initialize test results with cached results or SKIPPED status
    # WHY: Pre-populate results array to preserve test order
    subtask_result.test_results = [
        test_case_run_cache[test_name]
        if test_case_run_cache is not None and test_name in test_case_run_cache
        else TestResult(test_name=test_name)
        for test_name in subtask["test_names"]
    ]

    # STEP 4: Early exit checks
    # WHY: Don't waste time if we already know the outcome
    # CASE 1: No submission code extracted
    # CASE 2: Cached result already shows failure
    if not submission or any(
        test_result.status != "SKIPPED" and test_result.score == 0.0 for test_result in subtask_result.test_results
    ):
        return subtask_result

    # STEP 5: Load test cases
    # WHY: Test cases can be inline (in subtask dict) or loaded from disk
    if "test_cases" in subtask:
        test_cases = subtask["test_cases"]
        # Convert list to dict for easier lookup
        if isinstance(subtask["test_cases"], list):
            test_cases = {test_name: test for test_name, test in zip(subtask["test_names"], subtask["test_cases"])}
    else:
        # Load from IOI test directory
        # WHY: Large test cases stored on disk, not in config
        test_cases = load_ioi_tests(subtask["year"], subtask["id"])

    # STEP 6: Run tests in batches with early stopping
    # WHY: Balance parallelism with early stopping optimization
    for test_batch_to_run in batched(tests_to_run, test_batch_size):
        # Run all tests in this batch concurrently
        results = await asyncio.gather(
            *[
                asyncio.create_task(
                    score_single_test_case(
                        client, subtask, test_name, test_cases[test_name][0], test_cases[test_name][1], submission
                    )
                )
                for _, test_name in test_batch_to_run
            ]
        )

        # Update results and cache
        for (ti, test_name), test_result in zip(test_batch_to_run, results):
            if test_case_run_cache is not None:
                test_case_run_cache[test_name] = test_result
            subtask_result.test_results[ti] = test_result

        # STEP 7: Early stopping check
        # WHY: If any test in batch failed (score = 0.0), subtask score = 0.0
        #      No need to run remaining tests
        if any(test_result.score == 0.0 for test_result in results):
            break

    return subtask_result


async def score_subtasks(
    client: PistonClient, subtasks: list[dict], submission: str, skip_mode: bool = True
) -> list[SubtaskResult]:
    """
    WHAT: Scores multiple subtasks for a submission

    WHY: IOI problems have multiple subtasks (typically 3-8)
         Need to evaluate all subtasks and sum weighted scores

    HOW:
        1. Create shared test case cache
        2. Score each subtask sequentially (to benefit from cache)
        3. Return list of SubtaskResult objects

    PROXIMAL CONTEXT:
        - Input: List of subtask configs, submission code
        - Output: List of SubtaskResult objects

    DISTAL CONTEXT:
        - Originates from: IOI reward function
        - Flows to: Total score = sum(subtask.weighted_score)

    Args:
        client: PistonClient instance for executing code
        subtasks: List of dictionaries containing subtask configurations
        submission: Source code of the submission
        skip_mode: If True, use batch_size=1 for early stopping
                   If False, use batch_size=-1 to run all tests in parallel
                   Should be True when evaluating many submissions

    Returns:
        list[SubtaskResult]: Results for all subtasks

    CACHING STRATEGY:
        Some test cases appear in multiple subtasks
        Example: subtask 1 (easy), subtask 2 (easy + medium)
        Cache avoids re-execution

    SKIP MODE:
        True (default): Optimize for likely failures
            - Run tests sequentially
            - Stop immediately on failure
            - Faster for weak submissions

        False: Optimize for likely successes
            - Run all tests in parallel
            - No early stopping
            - Faster for strong submissions

    EXAMPLE:
        subtasks = [
            {"id": "sorting", "subtask": "1", "score": 20, ...},
            {"id": "sorting", "subtask": "2", "score": 30, ...},
            {"id": "sorting", "subtask": "3", "score": 50, ...}
        ]
        results = await score_subtasks(client, subtasks, code)
        total_score = sum(r.weighted_score for r in results)
        # Max score = 20 + 30 + 50 = 100 points
    """
    # Avoid rerunning tests present in multiple subtasks
    # WHY: Test cache shared across all subtasks
    test_case_run_cache = {}

    # Score subtasks sequentially (not in parallel)
    # WHY: Sequential allows cache to build up and benefit later subtasks
    return [await score_subtask(client, subtask, submission, test_case_run_cache, skip_mode) for subtask in subtasks]


"""
==============================================================================
PISTON EXECUTION
==============================================================================
"""


async def run_submission(
    client: PistonClient, problem: dict, test_input: str, submission: str, test_output: str | None = None
) -> tuple[str, str]:
    """
    WHAT: Executes a submission against a test case using Piston

    WHY: Need to compile and run C++ code with specific test input
         Compare output with expected output

    HOW:
        1. Prepare files (submission, input, expected output, grader files)
        2. Set time and memory limits
        3. Execute via Piston
        4. Parse result into (score, feedback)

    PROXIMAL CONTEXT:
        - Input: Submission code, test input/output
        - Output: (score, feedback) tuple

    DISTAL CONTEXT:
        - Originates from: score_single_test_case
        - Flows to: TestResult → SubtaskResult

    Args:
        client: PistonClient instance for executing code
        problem: Dictionary containing problem configuration
                 Fields: id, time_limit, memory_limit, grader_files
        test_input: Input data for the test case
        submission: Source code of the submission
        test_output: Optional expected output for the test case

    Returns:
        tuple[str, str]: A tuple containing (score, feedback)
                        score: "0" (fail), "1" (pass), or "0.5" (partial)
                        feedback: Error message or explanation

    FILE STRUCTURE:
        graders/<problem_id>.cpp: Submission code
        input.txt: Test input
        correct_output.txt: Expected output (if provided)
        graders/<grader>.h: Grader header files (IOI-specific)

    TIME LIMITS:
        run_timeout = (time_limit + 3) × 1000 ms
        +3 seconds hard limit for safety
        Actual time limit enforced by IOI grader script

    MEMORY LIMITS:
        run_memory_limit in MB
        Enforced by Piston sandbox

    EXAMPLE:
        score, feedback = await run_submission(
            client,
            {
                "id": "sorting",
                "time_limit": 1.0,  # 1 second
                "memory_limit": 256,  # 256 MB
                "grader_files": [("grader.h", "#include ...")]
            },
            "3\n1 3 2\n",
            submission_code,
            "1 2 3\n"
        )
        # score = "1", feedback = "Correct output"
    """
    # Prepare Piston execution request
    data = {
        "files": [
            # WHAT: The actual submission code
            # WHY: IOI convention: code in graders/<problem>.cpp
            {"name": f"graders/{problem['id'].lower()}.cpp", "content": submission},
            # WHAT: Test input data
            # WHY: Read by IOI grader script via stdin redirection
            {"name": "input.txt", "content": test_input},
            # WHAT: Expected output for comparison
            # WHY: IOI grader compares actual vs expected output
            *([{"name": "correct_output.txt", "content": test_output}] if test_output else []),
            # WHAT: Grader header files (IOI-specific includes)
            # WHY: IOI problems often provide grader.h with helper functions
            *({"name": name, "content": content} for name, content in problem["grader_files"] if content),
        ],
        # WHAT: Timeout in milliseconds
        # WHY: +3 seconds hard limit prevents infinite loops
        #      Actual time limit enforced by IOI script
        "run_timeout": round((problem["time_limit"] + 3) * 1000),
        # WHAT: Memory limit in MB
        # WHY: Prevents memory bombs, enforces problem constraints
        "run_memory_limit": problem["memory_limit"],
    }
    return await execute_ioi(client, data)


async def execute_ioi(client, data) -> tuple[str, str]:
    """
    WHAT: Requests to the IOI package return the score as a float in stdout,
          as well as optional feedback/errors in stderr

    WHY: IOI grader script:
         1. Compiles C++ code
         2. Runs with test input
         3. Compares output with expected
         4. Prints score (0.0-1.0) to stdout
         5. Prints feedback to stderr

    HOW:
        1. Send execute request to Piston
        2. Parse response for compile/run results
        3. Extract score from stdout, feedback from stderr
        4. Map errors to (score, feedback)

    Returns:
        tuple[str, str]: (score, feedback)

    PISTON RESPONSE FORMAT:
        {
            "compile": {"code": 0, "stdout": "...", "stderr": "..."},
            "run": {"code": 0, "stdout": "1.0", "stderr": "Correct", "signal": None},
            "language": "cpp",
            "version": "10.2.0"
        }

    ERROR HANDLING:
        1. Compilation error: return ("0", "Compilation error...")
        2. Runtime error: check exit code and signal
        3. Memory limit: check for MemoryError in stderr
        4. Time limit: check for SIGKILL signal
        5. Successful: parse score from stdout

    EXAMPLES:
        Success: ("1.0", "Correct output")
        Wrong answer: ("0", "Output isn't correct")
        TLE: ("0", "Time limit exceeded")
        MLE: ("0", "Memory limit exceeded")
        CE: ("0", "Compilation error exit code 1\\n...")
    """
    # STEP 1: Send execution request to Piston
    response = await client.send_execute(data)

    # STEP 2: Check for Piston-level errors
    if "message" in response:
        raise PistonError(response["message"])

    # STEP 3: Check for compilation errors
    # WHY: If compilation fails, can't run any tests
    if "compile" in response and response["compile"]["code"] != 0:
        return "0", "Compilation error exit code " + str(response["compile"]["code"]) + "\n" + response["compile"][
            "stderr"
        ]

    # STEP 4: Check for missing run result
    if "run" not in response:
        raise PistonError(response)

    # STEP 5: Check for memory limit exceeded
    # WHY: Python MemoryError indicates OOM
    if response["run"]["code"] == 1 and "MemoryError" in response["run"]["stderr"]:
        return "0", "Memory limit exceeded"

    # STEP 6: Successful result
    # WHY: IOI grader prints score to stdout
    # EXAMPLE: stdout="1.0", stderr="Correct output"
    if response["run"]["stdout"]:
        return response["run"]["stdout"], response["run"]["stderr"]

    # STEP 7: Check for time limit exceeded
    # WHY: SIGKILL signal indicates timeout
    if response["run"]["signal"] == "SIGKILL":
        return "0", "Time limit exceeded"

    # STEP 8: Other runtime errors
    # WHY: Non-zero exit code indicates runtime error
    if response["run"]["code"] != 0:
        raise PistonError(
            f"language={response['language']}, version={response['version']}, exit code={response['run']['code']}, stderr={response['run']['stderr']}, signal={response['run']['signal']}"
        )

    # STEP 9: Unknown error (shouldn't reach here)
    return "0", "Unknown error"


"""
==============================================================================
KEY TAKEAWAYS - IOI SCORING
==============================================================================

1. **IOI Competition Format**:
   - International Olympiad in Informatics (high school programming)
   - World's most challenging programming competition
   - OlympicCoder model trained on IOI problems

2. **Subtask-Based Scoring**:
   - Problems divided into subtasks (e.g., easy, medium, hard)
   - Each subtask worth points (e.g., 20, 30, 50)
   - Subtask score = min(test_scores) × points
   - Total score = sum(subtask_scores)

3. **Test Result Types**:
   - AC (Accepted): Perfect, score = 1.0
   - WA (Wrong Answer): Incorrect, score = 0.0
   - TLE (Time Limit Exceeded): Too slow, score = 0.0
   - MLE (Memory Limit Exceeded): Too much memory, score = 0.0
   - RE (Runtime Error): Crashed, score = 0.0
   - CE (Compilation Error): Won't compile, score = 0.0
   - PA (Partial Answer): Partially correct, 0 < score < 1

4. **Optimization Strategies**:
   - Test caching: Avoid re-running shared tests
   - Early stopping: Stop on first failure (since min score)
   - Batch parallelism: Run multiple tests concurrently
   - Skip mode: Sequential for likely failures, parallel for successes

5. **Piston Integration**:
   - Piston provides sandboxed C++ execution
   - Time and memory limits enforced
   - IOI grader script compares outputs
   - Returns score (0-1) and feedback

6. **Typical IOI Problem**:
   - 3-5 subtasks
   - 5-15 tests per subtask
   - Total: 15-75 test cases
   - Max score: 100 points

7. **Cache Benefits**:
   - Subtasks often share test cases
   - Example: subtask 1 (easy), subtask 2 (easy + medium)
   - Cache avoids re-execution
   - Significant speedup for multi-subtask problems

8. **Status Priority**:
   - CE > RE > WA > MLE > TLE > PA > AC
   - Worst status determines subtask status
   - Helps identify root cause of failure

9. **Integration with GRPO**:
   - IOI reward = total_score / max_score
   - Used in ioi_code reward function
   - Trains OlympicCoder model variant
   - Enables learning on competition-level problems

10. **Error Handling**:
    - Compilation errors caught early
    - Memory/time limits enforced by Piston
    - Graceful degradation on unknown errors
    - Detailed feedback for debugging

==============================================================================
"""
