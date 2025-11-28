"""
==============================================================================
FILE: src/open_r1/rewards.py
CATEGORY: Core Training - Reward Functions for GRPO
PRIORITY: CRITICAL
LINES: 706
FUNCTIONS: 20+ reward functions
DEPENDENCIES:
    - latex2sympy2_extended: LaTeX math parsing
    - math_verify: Math answer verification
    - open_r1.utils.code_providers: Code execution (E2B, MorphCloud, Piston)
    - open_r1.utils.competitive_programming: IOI and Codeforces evaluation
==============================================================================

==============================================================================
OVERVIEW
==============================================================================

This module is the **heart of GRPO training** - it defines the reward functions
that guide what the model learns to optimize for. Reward functions are the
"objective" in reinforcement learning.

ROLE IN DEEPSEEK R1:
-------------------
Reward functions are critical to Stage 2 (GRPO training):

1. **Define Success**: What makes a completion "good"?
   - Math accuracy: Is the answer mathematically correct?
   - Format compliance: Does it follow <think>/<answer> structure?
   - Code correctness: Does the code pass test cases?
   - Quality metrics: Is it concise? Does it avoid repetition?

2. **Multi-Objective Optimization**:
   - DeepSeek R1 combines multiple rewards (accuracy + format + quality)
   - Prevents "reward hacking" (exploiting a single metric)
   - Balances correctness with structure and efficiency

3. **Verifiable vs Heuristic Rewards**:
   - VERIFIABLE: accuracy_reward, code_reward (objective ground truth)
   - HEURISTIC: format_reward, repetition_penalty (pattern matching)
   - Best results combine both types

KEY INNOVATIONS:
----------------
1. **Math Verification with LaTeX Parsing**:
   - Parses LaTeX expressions to SymPy
   - Symbolic comparison (not string matching)
   - Handles multiple equivalent forms: 1/2 = 0.5 = 50%

2. **Code Execution in Sandboxes**:
   - Secure execution via E2B, MorphCloud, or Piston
   - Supports multiple languages (Python, C++, Java, etc.)
   - Timeout and error handling
   - Test case-based evaluation

3. **Competitive Programming Evaluation**:
   - IOI (International Olympiad in Informatics)
   - Codeforces contest-style problems
   - Subtask-based scoring with partial credit

4. **Length-Based Rewards**:
   - Cosine-scaled rewards (from Kimi 1.5 tech report)
   - Encourages concise reasoning
   - Prevents excessive verbosity

5. **Repetition Detection**:
   - N-gram-based repetition penalty
   - Prevents mode collapse during RL
   - Exponential penalty for repeated patterns

REWARD FUNCTION CATEGORIES:
---------------------------
1. **Math Rewards** (3 functions):
   - accuracy_reward: Verifies math correctness
   - len_reward: Length-based for math (Kimi 1.5 style)
   - get_cosine_scaled_reward: Smooth length scaling

2. **Format Rewards** (3 functions):
   - format_reward: Checks <think>/<answer> structure
   - tag_count_reward: Counts correct tag usage
   - reasoning_steps_reward: Detects step-by-step reasoning

3. **Code Rewards** (4 functions):
   - code_reward: General code execution with test cases
   - binary_code_reward: Binary 0/1 code correctness
   - ioi_code_reward: IOI problem evaluation
   - cf_code_reward: Codeforces problem evaluation

4. **Quality Rewards** (3 functions):
   - get_repetition_penalty_reward: Penalizes n-gram repetition
   - get_soft_overlong_punishment: Penalizes excessive length
   - get_code_format_reward: Code-specific format checking

5. **Helper Functions** (7 functions):
   - extract_code: Extracts code from markdown blocks
   - _init_event_loop: Async event loop management
   - get_reward_funcs: Reward function registry/factory

DATA FLOW IN GRPO TRAINING:
---------------------------
    Model Completions (vLLM generation)
            ↓
    Reward Functions (this file)
            ↓
    Scalar Rewards (one per completion)
            ↓
    Group Relative Advantages (rewards - mean)
            ↓
    Policy Gradients (update model to maximize advantages)
            ↓
    Improved Model (better at maximizing rewards)

TYPICAL REWARD COMPOSITION:
---------------------------
For math problems:
    total_reward = 1.0 * accuracy_reward
                 + 1.0 * format_reward
                 + 1.0 * tag_count_reward
                 - 0.5 * repetition_penalty
                 + 0.3 * cosine_scaled_reward

For code problems:
    total_reward = 1.0 * code_reward (pass rate)
                 + 0.5 * code_format_reward
                 - 0.3 * repetition_penalty

==============================================================================
IMPORTS
==============================================================================
"""

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
# [License text omitted]

"""Reward functions for GRPO training."""

import asyncio  # For async code execution
import json  # For code evaluation script formatting
import math  # For cosine scaling
import re  # For pattern matching (format, tags, repetition)
from functools import partial, update_wrapper  # For parameterized reward functions
from typing import Callable, Dict, Literal, Optional

# WHAT: LaTeX and math verification libraries
# WHY: Enable symbolic comparison of math expressions
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# WHAT: Code execution providers
# WHY: Execute code in sandboxes for correctness checking
from .utils.code_providers import get_provider

# WHAT: Competitive programming evaluation
# WHY: IOI and Codeforces problem evaluation with subtasks
from .utils.competitive_programming import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
)
from .utils.competitive_programming import patch_code as cf_patch_code
from .utils.competitive_programming import score_submission as cf_score_submission
from .utils.competitive_programming import score_subtask


"""
==============================================================================
MATH REWARD FUNCTIONS
==============================================================================
"""


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """
    WHAT: Verifies if model's math answer matches ground truth symbolically

    WHY: Core objective for math reasoning - model must produce correct answers
         Symbolic verification handles equivalent forms (1/2 = 0.5 = 50%)

    HOW:
        1. Extract <answer>...</answer> content from completion
        2. Parse LaTeX in both completion and ground truth using latex2sympy2
        3. Convert to SymPy expressions (symbolic math)
        4. Use math_verify.verify() to check equivalence
        5. Return 1.0 if correct, 0.0 if wrong, None if unparseable

    PROXIMAL CONTEXT:
        - Input: Model completions + ground truth solutions
        - Output: List of rewards (1.0/0.0/None per completion)

    DISTAL CONTEXT:
        - Originates from: vLLM generation → Completions
        - Flows to: Advantage calculation → Policy gradients

    Args:
        completions: List of completions (each is list of dicts with role/content)
        solution: List of ground truth solutions (LaTeX format)
        **kwargs: Additional dataset fields (ignored)

    Returns:
        List of rewards: 1.0 (correct), 0.0 (wrong), None (skip this example)
    """
    # WHAT: Extract text content from completion messages
    # NOTE: completion[0] is the assistant message content
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # STEP 1: Parse ground truth LaTeX
        # WHY: Some solutions may not be parseable (ambiguous/malformed)
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",  # Take first valid LaTeX expression
        )

        if len(gold_parsed) != 0:
            # STEP 2: Parse model's answer with strict normalization
            # WHY: We require properly formatted LaTeX (no malformed operators)
            # HOW: LatexExtractionConfig specifies parsing rules
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,  # Don't ignore small formatting issues
                            malformed_operators=False,  # Reject malformed operators
                            basic_latex=True,  # Allow basic LaTeX
                            equations=True,  # Extract from equations
                            boxed="all",  # Try \boxed{} first (common in math)
                            units=True,  # Handle units (5m, $10, etc.)
                        ),
                        boxed_match_priority=0,  # Prioritize \boxed{} extraction
                        try_extract_without_anchor=False,  # Require anchor (answer tag)
                    )
                ],
                extraction_mode="first_match",
            )

            # STEP 3: Verify symbolic equivalence
            # WHY: Handles equivalent forms: 1/2 = 0.5 = 0.50 = 50%
            try:
                # WHAT: verify() uses SymPy to check mathematical equivalence
                # RETURNS: True/False
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                # WHAT: If verification fails (parsing issues), skip example
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # WHAT: If ground truth is unparseable, skip example
            # WHY: Can't verify correctness without valid ground truth
            reward = None
            print("Failed to parse gold solution: ", sol)

        rewards.append(reward)

    return rewards


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """
    WHAT: Computes length-based rewards to discourage verbosity (Kimi 1.5 style)

    WHY: Encourages concise correct answers, penalizes lengthy wrong answers
         Prevents model from "hiding" wrong answers in verbose reasoning

    HOW:
        1. Check correctness of each answer (using accuracy_reward logic)
        2. Compute lengths of all completions
        3. For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
           → Shorter correct answers get higher rewards (up to 0.5)
        4. For wrong answers: reward = min(0, lambda_val)
           → Wrong answers never get positive rewards
           → Shorter wrong answers get less penalty

    MOTIVATION (from Kimi 1.5 tech report):
        - Long completions waste compute
        - Verbosity can mask incorrect reasoning
        - Prefer concise correct solutions

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of length-based rewards (-0.5 to +0.5 range)
    """
    contents = [completion[0]["content"] for completion in completions]

    # STEP 1: Check correctness (same logic as accuracy_reward)
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            correctness.append(True)  # Skip unparseable (treat as correct)
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # STEP 2: Calculate lengths and normalize
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # EDGE CASE: All same length → no length-based reward
    if max_len == min_len:
        return [0.0] * len(completions)

    # STEP 3: Compute length-based rewards
    rewards = []
    for length, is_correct in zip(lengths, correctness):
        # WHAT: Linear interpolation from 0 (min_len) to 1 (max_len)
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)
        # Range: 0.5 (min_len) to -0.5 (max_len)

        if is_correct:
            # CORRECT: Shorter is better (reward up to 0.5)
            reward = lambda_val
        else:
            # WRONG: Never positive, but shorter is less penalized
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    WHAT: Factory function that creates a cosine-scaled length reward function

    WHY: Provides smooth (non-linear) length scaling using cosine schedule
         Smoother than linear scaling, helps with training stability

    HOW:
        1. Returns a closure (inner function) with parameters baked in
        2. Inner function computes cosine-scaled rewards based on length
        3. Cosine function: cos(progress * π) where progress ∈ [0, 1]
        4. Maps to reward range based on correctness

    COSINE SCALING INTUITION:
        - Progress 0 (short): cos(0) = 1 → max reward
        - Progress 0.5 (medium): cos(π/2) = 0 → mid reward
        - Progress 1 (long): cos(π) = -1 → min reward
        - Smooth transition vs linear steps

    Args:
        min_value_wrong: Min reward for wrong answers (most penalty)
        max_value_wrong: Max reward for wrong answers (least penalty)
        min_value_correct: Min reward for correct answers (longest)
        max_value_correct: Max reward for correct answers (shortest)
        max_len: Maximum length for scaling (chars)

    Returns:
        Reward function with baked-in parameters
    """

    def cosine_scaled_reward(completions, solution, **kwargs):
        """Inner function that computes cosine-scaled rewards."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            # Parse ground truth
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable
                print("Failed to parse gold solution: ", sol)
                continue

            # Parse model answer
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # WHAT: Compute cosine scaling
            # HOW: progress ∈ [0, 1], cosine ∈ [-1, 1]
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            # WHAT: Select reward range based on correctness
            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # SWAP for wrong answers: longer is worse
                min_value = max_value_wrong
                max_value = min_value_wrong

            # WHAT: Map cosine to reward range
            # HOW: reward = min + 0.5 * (max - min) * (1 + cos)
            # WHY: (1 + cos) ∈ [0, 2], so result ∈ [min, max]
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


"""
==============================================================================
FORMAT REWARD FUNCTIONS
==============================================================================
"""


def format_reward(completions, **kwargs):
    """
    WHAT: Checks if completion follows <think>...</think><answer>...</answer> format

    WHY: Enforces structured reasoning format - core to DeepSeek R1 approach
         Separates internal reasoning (<think>) from final answer (<answer>)

    HOW: Regex pattern matching for exact tag structure

    PROXIMAL CONTEXT:
        - Input: List of completions
        - Output: List of binary rewards (1.0 if formatted correctly, else 0.0)

    DISTAL CONTEXT:
        - Combined with accuracy_reward in GRPO
        - Model learns: correct answer AND correct format

    Returns:
        List of 1.0 (correct format) or 0.0 (incorrect format)
    """
    # WHAT: Regex pattern for exact format
    # ^ and $ ensure entire completion matches
    # .*? is non-greedy match (shortest match)
    # re.DOTALL makes . match newlines
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"

    completion_contents = [completion[0]["content"] for completion in completions]

    # WHAT: Check each completion against pattern
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]

    # WHAT: Convert matches to binary rewards
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """
    WHAT: Soft reward for correct tag usage (not just presence, but also format)

    WHY: More lenient than format_reward - provides partial credit
         Encourages learning tags even if full format isn't perfect

    HOW: Counts each of 4 tag components (0.25 points each):
         - <think>\n (opening tag with newline)
         - \n</think>\n (closing tag with newlines)
         - \n<answer>\n (opening tag with newlines)
         - \n</answer> (closing tag)

    ADAPTED FROM: Kimi 1.5 GRPO demo (willccbb GitHub gist)

    Args:
        completions: List of model completions

    Returns:
        List of rewards (0.0 to 1.0, in 0.25 increments)
    """

    def count_tags(text: str) -> float:
        """Count each tag component (max 1.0 = 4 * 0.25)."""
        count = 0.0
        # Each tag component is worth 0.25 points
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    """
    WHAT: Detects structured step-by-step reasoning patterns

    WHY: Encourages explicit reasoning steps, not just final answer
         Rewards clear logical progression

    HOW: Pattern matching for:
         - "Step 1:", "Step 2:", etc.
         - Numbered lists "1.", "2.", etc.
         - Bullet points "- ...", "* ..."
         - Transition words "First,", "Second,", "Next,", "Finally,"

    REWARD SCALING:
        - Count all step indicators
        - Normalize: min(1.0, count / 3)
        - Magic number 3: encourages at least 3 steps

    Args:
        completions: List of model completions

    Returns:
        List of rewards (0.0 to 1.0)
    """
    # WHAT: Regex for step indicators
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    completion_contents = [completion[0]["content"] for completion in completions]

    # WHAT: Count matches in each completion
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # WHAT: Normalize to [0, 1] with 3 steps = 1.0
    # WHY: Encourages at least 3 reasoning steps
    return [min(1.0, count / 3) for count in matches]


"""
==============================================================================
CODE REWARD FUNCTIONS
(Detailed annotations for key functions, summaries for others due to length)
==============================================================================
"""


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """
    WHAT: Evaluates code by executing in sandbox and checking test cases

    WHY: Objective verification of code correctness
         Essential for code-focused GRPO training

    HOW:
        1. Extract code from ```python...``` markdown blocks
        2. Create evaluation script that:
           - Runs code with each test case input
           - Compares output with expected output
           - Counts passing tests
        3. Execute in sandbox (E2B/MorphCloud)
        4. Return pass rate (0.0 to 1.0)

    PROXIMAL CONTEXT:
        - Input: Completions + verification_info (test cases)
        - Output: Pass rates (fraction of tests passed)

    DISTAL CONTEXT:
        - Used in GRPO for code tasks (Codeforces, competitive programming)
        - Enables learning from verifiable feedback

    Args:
        completions: List of model completions
        num_parallel: Concurrent executions (default 2, suitable for E2B free tier)
        provider_type: "e2b", "morph", or "local"
        enforce_same_language: Validate all problems use same language
        **kwargs: Must include "verification_info" with test cases

    Returns:
        List of pass rates (0.0 to 1.0, or None if execution fails)
    """
    # Template for evaluation script (executed in sandbox)
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            # Run code with test input
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # Compare output line-by-line
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    # Extract code from completions
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]

    # Format evaluation scripts for each code snippet
    template = evaluation_script_template
    scripts = [
        template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    # Validate language consistency if required
    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    # Get execution provider and run scripts
    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))


# [Additional code reward functions with summary annotations]

def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """
    WHAT: Binary (0/1) version of code_reward
    WHY: Sometimes partial credit isn't desired - either code works or it doesn't
    HOW: Calls code_reward, then applies threshold (0.99) to convert to 0 or 1
    """
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """
    WHAT: Evaluates code on IOI (International Olympiad in Informatics) problems
    WHY: IOI represents Olympic-level programming challenges
    HOW:
        1. Extract C++ code from completions
        2. Add problem-specific include files
        3. Execute on subtask test cases (batched for efficiency)
        4. Compute weighted score based on subtask points
        5. Stop early if any test fails (optimization)

    DATASET FORMAT: hf.co/datasets/open-r1/ioi
    EXECUTION: Piston workers (see slurm/piston/README.md)
    """
    # Get execution client (Piston or MorphCloud)
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        execution_client = get_piston_client_from_env()

    # Extract code and add includes
    code_snippets = [
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        """Wrapper to catch exceptions from workers."""
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    # Prepare problem data
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    # Create async tasks for evaluation
    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def cf_code_reward(
    completions,
    test_batch_size: int = 1,
    patch_code: bool = False,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    **kwargs,
) -> list[float]:
    """
    WHAT: Evaluates code on Codeforces-style competitive programming problems
    WHY: Codeforces is a major competitive programming platform
    HOW:
        1. Extract code from completions (optionally patch for completeness)
        2. Execute on generated test cases (stored in parquet format)
        3. Score based on mode:
           - pass_fail: 1.0 if all tests pass, else 0.0
           - partial: Fraction of tests passed
           - weighted_sum: Weighted by test case importance
        4. Batch test execution for efficiency

    DATASET FORMAT: hf.co/datasets/open-r1/codeforces (verifiable-prompts subset)
    EXECUTION: Piston workers with CF package
    """
    piston_client = get_piston_client_from_env()

    languages = kwargs["language"] if "language" in kwargs else [None] * len(completions)
    code_snippets = [
        cf_patch_code(extract_code(completion[-1]["content"], language), language)
        if patch_code
        else extract_code(completion[-1]["content"], language)
        for completion, language in zip(completions, languages)
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return None

    # Prepare problem data
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    # Create async evaluation tasks
    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                cf_score_submission(
                    piston_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                    scoring_mode=scoring_mode,
                    submission_language=problem_data.get("language", None),
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return results


"""
==============================================================================
QUALITY CONTROL REWARDS
==============================================================================
"""


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    WHAT: Creates a reward function that penalizes n-gram repetition
    WHY: Prevents mode collapse during RL training (repetitive text)
    HOW:
        1. Extract n-grams from text
        2. Count unique vs total n-grams
        3. Compute repetition rate = 1 - (unique / total)
        4. Apply exponential penalty: scaling * max_penalty

    REFERENCE: Appendix C.2 of https://huggingface.co/papers/2502.03373
    IMPLEMENTATION: From demystify-long-cot repository

    Args:
        ngram_size: Size of n-grams (typically 3-5)
        max_penalty: Maximum negative penalty (e.g., -1.0)
        language: "en" or "zh" (for word splitting)

    Returns:
        Reward function with baked-in parameters
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    # Define n-gram extraction based on language
    if language == "en":

        def zipngram(text: str, ngram_size: int):
            """Extract n-grams from English text (space-delimited)."""
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            """Extract n-grams from Chinese text (using jieba)."""
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """Compute repetition penalty for each completion."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            # Extract unique and total n-grams
            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            # Compute penalty based on repetition rate
            scaling = 1 - len(ngrams) / total  # 0 = no repetition, 1 = all repeated
            reward = scaling * max_penalty  # Negative reward
            rewards.append(reward)

        return rewards

    return repetition_penalty_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    WHAT: Penalizes completions that exceed length limits (soft threshold)
    WHY: Prevents excessively long completions that waste compute
    HOW:
        - No penalty if length <= max_len - cache
        - Linear penalty in cache range
        - Full penalty (-1.0) if length > max_len

    REFERENCE: Eq. (13) from DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum allowed length
        soft_punish_cache: Soft threshold range (gradual penalty)

    Returns:
        Reward function with baked-in parameters
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Compute soft overlong penalty for each completion."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)

            if completion_length <= max_completion_len - soft_punish_cache:
                # Within acceptable range: no penalty
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                # In soft range: linear penalty
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                # Exceeded limit: full penalty
                rewards.append(-1.0)

        return rewards

    return soft_overlong_punishment_reward


def get_code_format_reward(language: str = "python"):
    """
    WHAT: Format reward specifically for code responses
    WHY: Ensures code is properly formatted in markdown blocks
    HOW: Checks for <think>...</think><answer>```language...```</answer> structure

    Args:
        language: Programming language (python, cpp, java, etc.)

    Returns:
        Code format reward function
    """

    def code_format_reward(completions, **kwargs):
        """Check code formatting."""
        # Allow per-example language override
        languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)

        completion_contents = [completion[0]["content"] for completion in completions]

        # Check for proper structure with code block
        matches = [
            re.match(
                rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content, sample_language in zip(completion_contents, languages)
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


"""
==============================================================================
HELPER FUNCTIONS
==============================================================================
"""


def extract_code(completion: str, language: str | None = "python") -> str:
    """
    WHAT: Extracts code from markdown code blocks

    WHY: Models output code in ```language...``` format
         Need to extract just the code for execution

    HOW: Regex match for ```{language}...```
         Returns last match (in case multiple blocks)

    Args:
        completion: Model completion with markdown
        language: Programming language to extract

    Returns:
        Extracted code string (empty if not found)
    """
    if language is None:
        return ""

    # Regex for markdown code blocks
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)

    # Return last match (model might show examples then final code)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def _init_event_loop():
    """
    WHAT: Initializes or gets current asyncio event loop

    WHY: Code execution uses async operations
         Need event loop for await/async syntax

    HOW: Try to get existing loop, create new if none exists
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


"""
==============================================================================
REWARD FUNCTION REGISTRY
==============================================================================
"""


def get_reward_funcs(script_args) -> list[Callable]:
    """
    WHAT: Factory function that builds list of reward functions from config

    WHY: Enables flexible reward composition via configuration
         Different tasks need different reward combinations

    HOW:
        1. Define registry mapping names to functions
        2. For parameterized functions, use partial() to bake in params
        3. Return list of callables based on script_args.reward_funcs

    PROXIMAL CONTEXT:
        - Input: script_args with reward_funcs list and parameters
        - Output: List of callable reward functions

    DISTAL CONTEXT:
        - Called from grpo.py during trainer initialization
        - Reward functions used throughout GRPO training

    Args:
        script_args: GRPOScriptArguments with reward configuration

    Returns:
        List of reward functions (callables)
    """
    REWARD_FUNCS_REGISTRY = {
        # Math rewards
        "accuracy": accuracy_reward,
        "length": len_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        # Format rewards
        "format": format_reward,
        "tag_count": tag_count_reward,
        "reasoning_steps": reasoning_steps_reward,
        # Quality rewards
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
        # Code rewards
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "cf_code": update_wrapper(
            partial(
                cf_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                scoring_mode=script_args.code_eval_scoring_mode,
            ),
            cf_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
    }

    # Build list from config
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs


"""
==============================================================================
KEY TAKEAWAYS - REWARD FUNCTION DESIGN
==============================================================================

1. **Multi-Objective is Critical**:
   - Single rewards are easily gamed (reward hacking)
   - Combine verifiable (accuracy, code) + heuristic (format, quality)
   - Weighted sum allows balancing objectives

2. **Math Verification Requires Symbolic Parsing**:
   - String matching fails: "0.5" ≠ "1/2" ≠ "50%"
   - LaTeX → SymPy → symbolic comparison
   - Handles equivalent mathematical forms

3. **Code Execution Must Be Sandboxed**:
   - Never execute untrusted code directly
   - E2B/MorphCloud provide isolated environments
   - Timeout and error handling essential

4. **Length-Based Rewards Need Care**:
   - Encourage conciseness but not at cost of correctness
   - Correct answers can be long (complex problems)
   - Cosine scaling provides smooth gradients

5. **Repetition Penalty Prevents Collapse**:
   - RL training can lead to mode collapse (repeated text)
   - N-gram detection catches various repetition types
   - Exponential penalty scales with severity

6. **Format Rewards Should Be Soft**:
   - Binary format reward (0/1) is harsh
   - tag_count_reward provides partial credit
   - Helps model learn structure gradually

7. **Reward Function Testing is Essential**:
   - Test on diverse examples (edge cases, errors)
   - Verify rewards match intuition
   - Check for unintended reward hacking

8. **Scaling and Normalization Matter**:
   - Rewards should be roughly same magnitude
   - Use weights to balance importance
   - Avoid one reward dominating others

==============================================================================
"""
