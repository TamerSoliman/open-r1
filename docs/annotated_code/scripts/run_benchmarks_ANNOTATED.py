"""
==============================================================================
FILE: scripts/run_benchmarks.py
CATEGORY: Scripts - Benchmark Runner
PRIORITY: HIGH
LINES: 62
DEPENDENCIES:
    - open_r1.utils.evaluation: SUPPORTED_BENCHMARKS, run_benchmark_jobs
    - open_r1.configs: SFTConfig
    - trl: ModelConfig, TrlParser
==============================================================================

==============================================================================
OVERVIEW
==============================================================================

This script provides a standalone command-line interface for running evaluation
benchmarks on DeepSeek R1 models. It allows users to evaluate models on standard
benchmarks (AIME, MATH-500, GPQA, LiveCodeBench, etc.) without running the full
training pipeline.

ROLE IN DEEPSEEK R1:
-------------------
Evaluation is critical for measuring model performance and validating training
effectiveness. This script enables:

1. **Post-Training Evaluation**: Run benchmarks after training completes to
   measure final model quality and compare against baselines.

2. **Model Comparison**: Evaluate multiple models on the same benchmarks to
   identify the best checkpoint or compare different training approaches.

3. **Checkpoint Selection**: Run benchmarks on intermediate checkpoints to
   identify when the model peaked (before overfitting).

4. **Leaderboard Submission**: Generate standardized benchmark results for
   sharing with the community or submitting to leaderboards.

5. **Debugging**: Test specific benchmarks in isolation to debug evaluation
   issues without re-running full training.

WHY SEPARATE SCRIPT?
--------------------
While benchmarks can be run during training via callbacks, a standalone script
provides several advantages:

1. **Post-Hoc Analysis**: Evaluate already-trained models without re-training
2. **Resource Efficiency**: Run on CPU or smaller GPU when training is done
3. **Batch Processing**: Evaluate multiple models in sequence
4. **Custom Benchmarks**: Test subsets of benchmarks or custom configurations
5. **Debugging**: Isolate evaluation logic from training complexity

SUPPORTED BENCHMARKS:
---------------------
The script can run any benchmark defined in SUPPORTED_BENCHMARKS:

- AIME 2024/2025: American Invitational Mathematics Examination
- MATH-500: Curated subset of MATH dataset
- GPQA: Graduate-level science questions
- LiveCodeBench: Recent coding problems
- (and others defined in open_r1.utils.evaluation)

Use --list_benchmarks to see all available options.

TYPICAL WORKFLOW:
-----------------

1. Train a model (sft.py or grpo.py)
2. Model is pushed to HuggingFace Hub with hub_model_id
3. Run benchmarks on the pushed model:
   ```
   python scripts/run_benchmarks.py \
       --model_id "myusername/my-r1-model" \
       --benchmarks aime_2024 math_500 gpqa
   ```
4. Results are logged to console and optionally pushed to Hub

DATA FLOW:
----------
    DISTAL ORIGIN:
    ├─> HuggingFace Hub → Trained model checkpoint
    └─> HuggingFace Hub → Benchmark datasets (AIME, MATH-500, etc.)

    PROXIMAL PROCESSING (this script):
    1. Parse command-line arguments (model_id, benchmarks, system_prompt)
    2. Create SFTConfig with benchmark configuration
    3. Create ModelConfig with model location
    4. Call run_benchmark_jobs() to execute evaluations
    5. run_benchmark_jobs internally:
       a. Downloads model from Hub
       b. Downloads benchmark datasets
       c. Generates model responses for each problem
       d. Evaluates responses (accuracy, pass@k, etc.)
       e. Aggregates results across benchmarks
       f. Optionally pushes results to Hub

    DISTAL DESTINATION:
    ├─> Console → Printed results summary
    ├─> HuggingFace Hub → Results dataset (if configured)
    └─> Weights & Biases → Logged metrics (if configured)

SYSTEM PROMPT:
--------------
The --system_prompt argument allows customizing the instruction given to the
model before each problem. This is crucial for controlling reasoning format:

Example system prompts:
- Default (from config): "Think step by step and provide your final answer."
- Structured: "Use <think> for reasoning and <answer> for final answer."
- Chain-of-thought: "Let's think through this step by step."

Different prompts can significantly affect benchmark performance, so the script
makes this configurable.

==============================================================================
IMPORTS AND SETUP
==============================================================================
"""

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
from dataclasses import dataclass, field
from typing import List, Optional

from open_r1.utils.evaluation import SUPPORTED_BENCHMARKS, run_benchmark_jobs
from open_r1.configs import SFTConfig
from trl import ModelConfig, TrlParser


"""
==============================================================================
WHAT: ScriptArguments
WHY:  Define command-line interface for benchmark runner
HOW:  Dataclass with field metadata for argument parsing
==============================================================================

This dataclass defines the command-line arguments for the benchmark script:

FIELDS:
------

1. model_id (str):
   - Default: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
   - HuggingFace Hub model ID to evaluate
   - Can be official model or user's fine-tuned checkpoint
   - Example: "myusername/qwen-grpo-math"

2. model_revision (str):
   - Default: "main"
   - Git branch/tag/commit to use
   - Allows evaluating specific checkpoints
   - Example: "checkpoint-1000" or "v1.0"

3. trust_remote_code (bool):
   - Default: False
   - Whether to trust custom code in model repository
   - Required for models with custom modeling files
   - Security note: Only enable for trusted sources

4. benchmarks (List[str]):
   - Default: [] (empty list)
   - List of benchmark names to run
   - Must match entries in SUPPORTED_BENCHMARKS
   - Example: ["aime_2024", "math_500", "gpqa"]

5. list_benchmarks (bool):
   - Default: False
   - If True, print available benchmarks and exit
   - Useful for discovering what benchmarks are supported
   - Doesn't require model_id when used

6. system_prompt (Optional[str]):
   - Default: None (use config default)
   - Custom instruction prepended to each problem
   - Affects model's reasoning format and behavior
   - Example: "Use <think> tags for your reasoning."

PROXIMAL CONTEXT: Parsed by TrlParser in main()
DISTAL CONTEXT: Values propagate to SFTConfig → run_benchmark_jobs

The dataclass approach (vs raw argparse) provides:
- Type safety (List[str] not str)
- Default values in one place
- Integration with TRL's config system
- Help text via metadata
"""
@dataclass
class ScriptArguments:
    model_id: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        metadata={"help": "The Hub model id to push the model to."},
    )
    model_revision: str = field(default="main", metadata={"help": "The Hub model branch to push the model to."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust the remote code."})
    benchmarks: List[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    list_benchmarks: bool = field(default=False, metadata={"help": "List all supported benchmarks."})
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The system prompt to use for the benchmark."}
    )


"""
==============================================================================
WHAT: main()
WHY:  Execute benchmark evaluation on specified model
HOW:  Parse args, configure, delegate to run_benchmark_jobs
==============================================================================

ARCHITECTURE:
------------
The main function has two paths:

Path 1: List benchmarks (--list_benchmarks)
    1. Parse arguments
    2. Print SUPPORTED_BENCHMARKS
    3. Exit

Path 2: Run benchmarks (default)
    1. Parse arguments
    2. Create SFTConfig with benchmark settings
    3. Create ModelConfig with model location
    4. Call run_benchmark_jobs() to execute evaluations
    5. Results are handled by run_benchmark_jobs (logging, Hub upload, etc.)

CONFIGURATION OBJECT CREATION:
------------------------------

SFTConfig is created with minimal fields:
- output_dir: "" (not needed for eval-only)
- hub_model_id: args.model_id (where to download model)
- hub_model_revision: args.model_revision (which version)
- benchmarks: args.benchmarks (what to evaluate)
- system_prompt: args.system_prompt (how to prompt model)

ModelConfig is created with:
- model_name_or_path: "" (will use hub_model_id from SFTConfig)
- model_revision: "" (will use hub_model_revision from SFTConfig)
- trust_remote_code: args.trust_remote_code

WHY EMPTY STRINGS?
-----------------
The run_benchmark_jobs function prioritizes hub_model_id over model_name_or_path.
By setting model_name_or_path="", we force it to use the Hub ID instead of a
local path. This ensures we always evaluate the Hub-uploaded model, not a
local checkpoint.

PROXIMAL PROCESSING:
-------------------
The script itself is thin - it just:
1. Parses CLI arguments
2. Wraps them in config objects
3. Delegates to run_benchmark_jobs

All actual work (model loading, generation, evaluation) happens in
run_benchmark_jobs (see open_r1/utils/evaluation.py for details).

DISTAL IMPACT:
-------------
After run_benchmark_jobs completes:
- Benchmark results are logged to console
- Results may be pushed to Hub (if configured)
- Metrics may be logged to W&B (if configured)
- User can compare performance across models/checkpoints
"""
def main():
    # WHAT: Parse command-line arguments using TRL's parser
    # WHY: TRL parser supports both CLI args and config files
    # HOW: Returns tuple of (script_args, ...), we only need first element
    parser = TrlParser(ScriptArguments)
    args = parser.parse_args_and_config()[0]

    # WHAT: List available benchmarks if requested
    # WHY: Help users discover what benchmarks they can run
    # HOW: Print SUPPORTED_BENCHMARKS and exit early
    if args.list_benchmarks:
        print("Supported benchmarks:")
        for benchmark in SUPPORTED_BENCHMARKS:
            print(f"  - {benchmark}")
        return

    # WHAT: Create benchmark configuration from parsed arguments
    # WHY: run_benchmark_jobs expects SFTConfig object, not raw args
    # HOW: Wrap args in SFTConfig with benchmark-relevant fields
    benchmark_args = SFTConfig(
        output_dir="",  # WHY: Not training, so no output directory needed
        hub_model_id=args.model_id,  # WHY: Where to download model from
        hub_model_revision=args.model_revision,  # WHY: Which version to evaluate
        benchmarks=args.benchmarks,  # WHY: What benchmarks to run
        system_prompt=args.system_prompt,  # WHY: How to prompt the model
    )

    # WHAT: Run benchmark evaluation on specified model
    # WHY: Delegate to evaluation module for actual benchmark execution
    # HOW: Pass benchmark config and model config to run_benchmark_jobs
    run_benchmark_jobs(
        benchmark_args,
        ModelConfig(
            model_name_or_path="",  # WHY: Empty forces use of hub_model_id
            model_revision="",  # WHY: Empty forces use of hub_model_revision
            trust_remote_code=args.trust_remote_code  # WHY: Security setting for custom code
        ),
    )


"""
==============================================================================
WHAT: Main entry point
WHY:  Execute main() when script is run directly
==============================================================================
"""
if __name__ == "__main__":
    main()


"""
==============================================================================
KEY TAKEAWAYS
==============================================================================

1. **PURPOSE**: This script provides a standalone CLI for evaluating DeepSeek R1
   models on standard benchmarks without running the full training pipeline.

2. **ARCHITECTURE**: Thin wrapper around run_benchmark_jobs():
   - Parse CLI arguments
   - Wrap in config objects (SFTConfig, ModelConfig)
   - Delegate to evaluation module

3. **CRITICAL FEATURES**:
   - --list_benchmarks: Discover available benchmarks
   - --benchmarks: Select which benchmarks to run
   - --system_prompt: Control model prompting strategy
   - --model_revision: Evaluate specific checkpoints

4. **CONFIGURATION STRATEGY**: Uses empty strings for model_name_or_path to
   force evaluation of Hub model:
   - model_name_or_path="" → uses hub_model_id
   - model_revision="" → uses hub_model_revision

5. **TYPICAL WORKFLOW**:
   Step 1: Train model and push to Hub
   Step 2: Run benchmarks on Hub model
   Step 3: Compare results across checkpoints
   Step 4: Select best model for deployment

6. **SYSTEM PROMPT IMPORTANCE**: The --system_prompt argument is crucial:
   - Affects reasoning format (<think>/<answer> vs free-form)
   - Different prompts yield different performance
   - Should match training-time prompt for fair evaluation

7. **BENCHMARK DISCOVERY**: --list_benchmarks enables exploration:
   ```
   $ python scripts/run_benchmarks.py --list_benchmarks
   Supported benchmarks:
     - aime_2024
     - aime_2025
     - math_500
     - gpqa
     - lcb
   ```

8. **INTEGRATION**: Works seamlessly with training scripts:
   - sft.py/grpo.py push model to Hub with hub_model_id
   - run_benchmarks.py evaluates using same hub_model_id
   - Consistent evaluation across training runs

==============================================================================
USAGE EXAMPLES
==============================================================================

List available benchmarks:

    $ python scripts/run_benchmarks.py --list_benchmarks

    Supported benchmarks:
      - aime_2024
      - aime_2025
      - math_500
      - gpqa
      - lcb

Evaluate official DeepSeek R1 model:

    $ python scripts/run_benchmarks.py \
        --model_id "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
        --benchmarks aime_2024 math_500

    Loading model from deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B...
    Running AIME 2024 benchmark...
    AIME 2024: 12/30 correct (40.0%)
    Running MATH-500 benchmark...
    MATH-500: 385/500 correct (77.0%)

Evaluate custom fine-tuned model:

    $ python scripts/run_benchmarks.py \
        --model_id "myusername/qwen-grpo-math" \
        --model_revision "checkpoint-5000" \
        --benchmarks aime_2024 math_500 gpqa \
        --system_prompt "Use <think> for reasoning and <answer> for final answer."

    Loading model from myusername/qwen-grpo-math@checkpoint-5000...
    Running AIME 2024 benchmark...
    AIME 2024: 15/30 correct (50.0%)
    Running MATH-500 benchmark...
    MATH-500: 420/500 correct (84.0%)
    Running GPQA benchmark...
    GPQA: 45/100 correct (45.0%)

Compare multiple checkpoints:

    #!/bin/bash
    MODEL="myusername/qwen-grpo-math"
    BENCHMARKS="aime_2024 math_500"

    for CHECKPOINT in checkpoint-1000 checkpoint-3000 checkpoint-5000; do
        echo "Evaluating $CHECKPOINT..."
        python scripts/run_benchmarks.py \
            --model_id "$MODEL" \
            --model_revision "$CHECKPOINT" \
            --benchmarks $BENCHMARKS
    done

    # Compare results to identify best checkpoint

Evaluate with custom system prompt:

    $ python scripts/run_benchmarks.py \
        --model_id "myusername/my-model" \
        --benchmarks math_500 \
        --system_prompt "Let's solve this problem step by step. Show your work and provide the final answer."

    Running MATH-500 with custom prompt...
    MATH-500: 410/500 correct (82.0%)

==============================================================================
"""
