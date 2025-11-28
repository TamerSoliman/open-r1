"""
==============================================================================
FILE: scripts/benchmark_e2b.py
CATEGORY: Scripts - E2B Code Execution Benchmarking
PRIORITY: MEDIUM
LINES: 86
DEPENDENCIES:
    - datasets: load_dataset (HuggingFace datasets library)
    - open_r1.rewards: code_reward (code execution reward function)
    - E2B: External code execution sandbox service
    - dotenv: Environment variable management
==============================================================================

==============================================================================
OVERVIEW
==============================================================================

This script benchmarks the performance of the code_reward function that uses E2B
(a cloud-based code execution sandbox) to verify code correctness. It measures
execution time and throughput across different sample sizes and parallelization
levels.

ROLE IN DEEPSEEK R1:
-------------------
The code_reward function is critical for GRPO training on coding tasks. During
reinforcement learning, the model generates multiple code solutions per problem,
and these solutions must be executed against test cases to compute rewards.
Understanding the performance characteristics of code execution is essential for:

1. **Training Pipeline Optimization**: Knowing execution times helps configure
   batch sizes and parallelization settings for efficient GRPO training
2. **Cost Estimation**: E2B is a paid service with rate limits; benchmarking
   helps estimate costs and identify optimal parallelization levels
3. **Bottleneck Identification**: If code execution is slow, it becomes the
   bottleneck in GRPO training (generation → execution → gradient update)

WHAT IS BEING BENCHMARKED:
--------------------------
The script tests code_reward across three dimensions:

1. **Sample Size**: Number of code solutions to execute (16, 64, 256)
2. **Parallelization**: Concurrent executions (varies per sample size)
3. **Accuracy**: Verifies that gold standard solutions pass test cases

Each sample is a CodeForces problem with:
- A gold standard (known correct) Python solution
- A set of public test cases (inputs/expected outputs)
- Verification metadata for execution

TYPICAL BENCHMARK RESULTS:
--------------------------
Expected patterns:
- Linear scaling with sample size (2x samples ≈ 2x time)
- Sub-linear improvement with parallelization (diminishing returns)
- Near-perfect accuracy (gold solutions should pass all tests)
- Execution time dominated by sandbox creation overhead

Example output:
| Sample Size | Parallelization | Execution Time (s) | Mean Reward |
|-------------|-----------------|-------------------|-------------|
|     16      |        1        |      45.23        |   0.9875    |
|     16      |        4        |      15.67        |   0.9875    |
|     16      |       16        |       8.34        |   0.9875    |
|     64      |        4        |      62.11        |   0.9844    |

DATA FLOW:
----------
    DISTAL ORIGIN:
    └─> HuggingFace Hub → verifiable-coding-problems-python_decontaminated
        └─> Contains: CodeForces problems with gold solutions and test cases

    PROXIMAL PROCESSING (this script):
    1. Load dataset from Hub (shuffled with fixed seed for reproducibility)
    2. Select subset based on sample size (16/64/256)
    3. Extract gold_standard_solution and verification_info
    4. Call code_reward with varying parallelization levels
    5. Measure execution time and collect reward statistics
    6. Format results as markdown table

    DISTAL DESTINATION:
    └─> Console output: Performance table for analysis
    └─> Informally: Insights inform GRPO training configuration

==============================================================================
IMPORTS AND SETUP
==============================================================================
"""

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Benchmark script for the code_reward function with E2B.

This script measures the performance of the code_reward function with varying numbers
of samples and parallelization levels.

Each sample is a CodeForces problem with a gold standard solution that is executed against a set of public test cases.
"""

from datasets import load_dataset
import time
from tqdm.auto import tqdm

from dotenv import load_dotenv
load_dotenv()  # WHY: Loads E2B_API_KEY from .env file for authentication

from open_r1.rewards import code_reward

"""
==============================================================================
WHAT: benchmark_code_reward()
WHY:  Test a single example to verify code_reward functionality
HOW:  Wraps solution in expected format and measures execution time
==============================================================================

This function is currently defined but unused in the main benchmarking loop.
It was likely used during development to test individual examples.

PROXIMAL CONTEXT: Called per-example during development/debugging
DISTAL CONTEXT: Never called in production; main() uses batch processing

The function demonstrates the expected input format:
- test_completions: [[{"content": code_string}]] (nested list structure)
- reward_kwargs: {"verification_info": [test_case_metadata]}

Returns the example dict with added fields:
- test_reward: Scalar reward (0.0 to 1.0, typically binary for correctness)
- reward_time: Execution time in seconds
"""
def benchmark_code_reward(example):
    start_time = time.time()
    test_completions = [[{"content": example["gold_standard_solution"]}]]
    reward_kwargs = {"verification_info": [example["verification_info"]]}
    rewards = code_reward(test_completions, **reward_kwargs)
    end_time = time.time()
    example["test_reward"] = rewards[0]
    example["reward_time"] = end_time - start_time
    return example

"""
==============================================================================
WHAT: Main benchmarking loop
WHY:  Measure code_reward performance across different configurations
HOW:  Nested loops over sample sizes and parallelization levels
==============================================================================

ARCHITECTURE:
------------
The script uses a two-level nested loop structure:

Outer loop: Sample sizes (16, 64, 256)
  └─> Inner loop: Parallelization levels (varies per sample size)
      └─> For each configuration:
          1. Load dataset fresh (clean state)
          2. Shuffle with seed=42 (reproducibility)
          3. Select first N samples
          4. Batch execute with code_reward
          5. Measure total execution time
          6. Compute reward statistics
          7. Store results for table

PARALLELIZATION STRATEGY:
------------------------
The parallel_dict maps sample size to parallelization levels to test:

    16 samples → [1, 4, 16] parallel executions
    64 samples → [4, 16, 64] parallel executions
    256 samples → [16, 64, 96] parallel executions (capped at 96 for PRO account)

WHY THESE LEVELS?
- Start with 1 to establish baseline (no parallelization)
- Increase to match sample size (maximum possible parallelization)
- Cap at 96 due to E2B PRO account limit (100 sandboxes max)

PROXIMAL PROCESSING DETAILS:
----------------------------
For each configuration, the script:

1. LOADING: Fetches dataset from HuggingFace Hub
   - Dataset: open-r1/verifiable-coding-problems-python_decontaminated
   - Why decontaminated? Removes problems overlapping with eval benchmarks
   - Shuffle ensures variety across runs (but seed=42 ensures reproducibility)

2. PREPARATION: Extracts gold solutions and test cases
   - test_completions: Nested list of message dicts (matches GRPO format)
   - reward_kwargs: Contains verification_info for test case execution

3. EXECUTION: Calls code_reward with timing
   - num_parallel: Controls concurrent sandbox executions
   - Higher parallelization → faster total time but more API load

4. ANALYSIS: Computes statistics
   - mean_reward: Should be ~1.0 (gold solutions are correct)
   - min/max_reward: Identifies any failures (bugs in gold solutions or tests)
   - execution_time: Total wall-clock time for batch

5. STORAGE: Accumulates results for final table

DISTAL CONTEXT:
--------------
Results inform production decisions:
- If 16 parallel is 3x faster than 4 parallel → use high parallelization
- If mean_reward < 1.0 → investigate dataset quality issues
- If execution_time scales poorly → consider alternative execution backends

KEY INSIGHT:
-----------
Benchmarking gold standard solutions provides a performance upper bound.
Actual GRPO training will be slower because:
1. Model-generated solutions may be incorrect (execution still required)
2. Each prompt generates multiple candidates (16x more executions)
3. Generation time adds overhead before execution

This benchmark establishes the code execution bottleneck in isolation.
"""
if __name__ == "__main__":
    # WHY: Map each sample size to appropriate parallelization levels
    # HOW: Smaller samples test lower parallelization; larger samples need higher
    parallel_dict = {
        16:[1,4,16],      # Test from serial to fully parallel
        64:[4,16, 64],    # Skip serial (too slow); test medium to high parallelism
        256:[16, 64, 96], # cap at 96 as PRO account is limited to 100 sandboxes
    }
    # Store results for table formatting
    results = []

    # WHAT: Main benchmarking loop - test all combinations of samples and parallelization
    # WHY: Understand performance characteristics across realistic training scenarios
    for num_samples in tqdm([16, 64,256], desc="Benchmarking samples"):
        for num_parallel in parallel_dict[num_samples]:
            # WHAT: Load dataset fresh for each configuration
            # WHY: Ensures clean state and avoids any caching effects
            code_dataset = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated")
            code_dataset = code_dataset["train"].shuffle(seed=42).select(range(num_samples))

            # WHAT: Prepare batch data for code_reward
            # WHY: code_reward expects nested list structure matching GRPO conversation format
            # HOW: Each completion is wrapped as [{"content": code}] to match message format
            test_completions = [[{"content": example["gold_standard_solution"]}] for example in code_dataset]
            reward_kwargs = {"verification_info": [example["verification_info"] for example in code_dataset]}

            # WHAT: Execute batch and measure time
            # WHY: Wall-clock time is most relevant metric for training pipeline throughput
            start_time = time.time()
            rewards = code_reward(test_completions, num_parallel=num_parallel, **reward_kwargs)
            execution_time = time.time() - start_time

            # WHAT: Calculate reward statistics
            # WHY: Verify correctness (mean should be ~1.0) and identify outliers
            # HOW: Gold standard solutions should pass all test cases (reward = 1.0)
            mean_reward = sum(rewards) / len(rewards)
            min_reward = min(rewards)
            max_reward = max(rewards)

            # Store results for final table
            results.append({
                "num_samples": num_samples,
                "num_parallel": num_parallel,
                "execution_time": execution_time,
                "mean_reward": mean_reward,
                "min_reward": min_reward,
                "max_reward": max_reward
            })

    # WHAT: Format results as markdown table
    # WHY: Easy to copy into documentation or reports
    # HOW: Fixed-width columns with center/right alignment for readability
    print("\n## Benchmark Results\n")
    print("| Sample Size | Parallelization | Execution Time (s) | Mean Reward | Min Reward | Max Reward |")
    print("|:-----------:|:---------------:|------------------:|:-----------:|:-----------:|:-----------:|")

    for result in results:
        print(f"| {result['num_samples']:^11} | {result['num_parallel']:^15} | {result['execution_time']:17.2f} | {result['mean_reward']:^11.4f} | {result['min_reward']:^11.4f} | {result['max_reward']:^11.4f} |")


"""
==============================================================================
KEY TAKEAWAYS
==============================================================================

1. **PURPOSE**: This script benchmarks E2B code execution performance to inform
   GRPO training configuration decisions (batch size, parallelization, cost).

2. **ARCHITECTURE**: Nested loops test all combinations of sample sizes (16/64/256)
   and parallelization levels (1 to 96), measuring execution time and accuracy.

3. **CRITICAL INSIGHT**: Code execution is often the bottleneck in GRPO training
   for coding tasks. Understanding performance characteristics enables:
   - Optimal parallelization settings (balance speed vs API limits)
   - Accurate cost estimation (E2B charges per execution)
   - Identification of dataset quality issues (mean_reward < 1.0)

4. **DATA FORMAT**: The script demonstrates the expected input format for code_reward:
   - test_completions: [[{"content": code}]] (nested list of message dicts)
   - reward_kwargs: {"verification_info": [test_metadata]}

   This matches the GRPO conversation format where completions are message lists.

5. **GOLD STANDARD TESTING**: Using gold (known correct) solutions provides a
   performance upper bound. Actual training is slower due to:
   - Multiple candidates per prompt (typically 16x executions)
   - Model-generated code may fail (still requires execution)
   - Generation time adds overhead

6. **PARALLELIZATION STRATEGY**: Tests exponentially increasing parallelism:
   - 1 (serial baseline)
   - 4 (modest parallelism)
   - 16-96 (high parallelism, API limit constrained)

   Results show diminishing returns at high parallelism due to sandbox overhead.

7. **REPRODUCIBILITY**: Fixed seed (42) ensures consistent benchmark results
   across runs, enabling meaningful performance comparisons.

8. **PRO ACCOUNT LIMITATION**: The 96 parallel cap reflects E2B PRO account limits
   (100 concurrent sandboxes). Higher parallelism requires enterprise tier.

==============================================================================
USAGE EXAMPLE
==============================================================================

To run the benchmark:

    $ export E2B_API_KEY="your_api_key_here"
    $ python scripts/benchmark_e2b.py

Expected output:

    ## Benchmark Results

    | Sample Size | Parallelization | Execution Time (s) | Mean Reward | Min Reward | Max Reward |
    |:-----------:|:---------------:|------------------:|:-----------:|:-----------:|:-----------:|
    |     16      |        1        |      45.23        |   0.9875    |   0.0000    |   1.0000    |
    |     16      |        4        |      15.67        |   0.9875    |   0.0000    |   1.0000    |
    |     16      |       16        |       8.34        |   0.9875    |   0.0000    |   1.0000    |

Interpreting results:
- Execution time decreases with higher parallelization (expected)
- Mean reward ~0.98-1.0 indicates high gold solution correctness
- Min reward 0.0 suggests some test cases fail (investigate dataset)
- Max reward 1.0 confirms at least some solutions are fully correct

==============================================================================
"""
