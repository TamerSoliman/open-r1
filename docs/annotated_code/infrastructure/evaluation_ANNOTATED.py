"""
==============================================================================
FILE: src/open_r1/utils/evaluation.py
CATEGORY: Infrastructure - Model Evaluation
PRIORITY: HIGH
LINES: 119
DEPENDENCIES:
    - lighteval: HuggingFace evaluation framework
    - vLLM: Fast inference for evaluation
    - Slurm: Job scheduling for distributed evaluation
==============================================================================

OVERVIEW:
This module provides integration with LightEval for benchmarking trained models
on standard reasoning and coding tasks. It handles Slurm job submission for
distributed evaluation using vLLM.

ROLE IN DEEPSEEK R1:
- Validates training progress on held-out benchmarks
- Compares model performance across iterations
- Enables automatic evaluation during training
- Supports both math and coding benchmarks

KEY FEATURES:
1. LightEval Integration: Standard benchmark framework
2. Slurm Job Submission: Distributed evaluation on clusters
3. vLLM Backend: Fast inference for evaluation
4. Tensor Parallelism: Shard large models across GPUs
5. Multiple Benchmarks: MATH, AIME, GPQA, LiveCodeBench

DATA FLOW:
Training checkpoint → Slurm job → vLLM server → LightEval
    → Benchmark results → Metrics logging → HuggingFace Hub

TYPICAL WORKFLOW:
1. During training, checkpoint is saved to Hub
2. run_benchmark_jobs() is called at eval intervals
3. Slurm jobs are submitted for each benchmark
4. vLLM server loads model and runs evaluation
5. Results logged to Hub for comparison
==============================================================================
"""

import subprocess
from typing import TYPE_CHECKING, Dict, Union

from .hub import get_gpu_count_for_vllm, get_param_count_from_repo_id


if TYPE_CHECKING:
    from trl import GRPOConfig, SFTConfig, ModelConfig

import base64
import os


"""
==============================================================================
SLURM ENVIRONMENT SETUP
==============================================================================
"""

# WHAT: Special environment setup for launching vLLM from within Slurm training jobs
# WHY: vLLM needs clean environment without training job's env vars
# HOW: Use env -i to start fresh bash, source profile, set HOME
#
# CONTEXT:
# - Training job runs in Slurm with DeepSpeed/FSDP env vars
# - vLLM server needs different env (no distributed training vars)
# - Solution: Launch new Slurm job with clean environment
#
# REFERENCE:
# - https://github.com/huggingface/brrr/blob/main/brrr/lighteval/one_job_runner.py
user_home_directory = os.path.expanduser("~")
VLLM_SLURM_PREFIX = [
    "env",  # WHAT: env command for environment manipulation
    "-i",  # WHAT: Start with empty environment (ignore current env vars)
    "bash",  # WHAT: Launch bash shell
    "-c",  # WHAT: Execute command string
    # WHAT: Command string to execute
    # WHY: Need to source /etc/profile.d/*.sh for PATH, modules, etc.
    # WHY: Set HOME so vLLM can find caches, configs
    f"for f in /etc/profile.d/*.sh; do source $f; done; export HOME={user_home_directory}; sbatch ",
]


"""
==============================================================================
BENCHMARK REGISTRATION
==============================================================================
"""


def register_lighteval_task(
    configs: Dict[str, str],
    eval_suite: str,
    task_name: str,
    task_list: str,
    num_fewshot: int = 0,
):
    """
    WHAT: Registers a LightEval task configuration

    WHY: LightEval requires specific task format: "suite|task|fewshot|0"
         This helper simplifies registering tasks

    HOW:
        1. Take task names (comma-separated)
        2. Format each as "eval_suite|task_name|num_fewshot|0"
        3. Store in configs dict

    PROXIMAL CONTEXT:
        - Input: Task name and parameters
        - Output: Formatted task string in configs dict

    DISTAL CONTEXT:
        - Originates from: LIGHTEVAL_TASKS dict initialization
        - Flows to: run_lighteval_job() → Slurm script → LightEval

    Args:
        configs: Dictionary to store task configurations
        eval_suite: Evaluation suite ("lighteval" for built-in, "extended" for custom)
        task_name: Short name for this task (e.g., "math_500")
        task_list: Comma-separated list of task identifiers
        num_fewshot: Number of few-shot examples (0 for zero-shot)

    LIGHTEVAL FORMAT:
        - "lighteval|task_name|num_fewshot|0"
        - First field: suite (lighteval or extended)
        - Second field: task name
        - Third field: number of few-shot examples
        - Fourth field: always 0 (reserved)

    TASK SUITES:
        - lighteval: Built-in tasks from lighteval/tasks/tasks_table.jsonl
        - extended: Custom tasks in scripts/evaluation/extended_lighteval_tasks

    EXAMPLE:
        register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "math_500", "math_500", 0)
        # Result: LIGHTEVAL_TASKS["math_500"] = "lighteval|math_500|0|0"

        register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb", "lcb:codegeneration", 0)
        # Result: LIGHTEVAL_TASKS["lcb"] = "extended|lcb:codegeneration|0|0"
    """
    # Format task list in lighteval format
    # WHY: LightEval expects specific string format
    # HOW: Join all tasks with suite, fewshot, and trailing 0
    task_list = ",".join(f"{eval_suite}|{task}|{num_fewshot}|0" for task in task_list.split(","))
    configs[task_name] = task_list


# WHAT: Global registry of all supported benchmarks
# WHY: Centralize benchmark definitions for easy access
LIGHTEVAL_TASKS = {}

# MATH BENCHMARKS
# WHY: Math reasoning is core capability for DeepSeek R1
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "math_500", "math_500", 0)
# WHAT: MATH-500 benchmark (subset of MATH dataset)
# WHY: Fast evaluation of math reasoning (500 problems vs full 5,000)

register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "aime24", "aime24", 0)
# WHAT: American Invitational Mathematics Examination 2024
# WHY: Very challenging competition math (top high school level)

register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "aime25", "aime25", 0)
# WHAT: AIME 2025
# WHY: Latest year's competition for up-to-date evaluation

# SCIENCE BENCHMARKS
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "gpqa", "gpqa:diamond", 0)
# WHAT: Google-Proof Q&A (GPQA) Diamond subset
# WHY: PhD-level science questions (physics, chemistry, biology)
# WHY: Tests advanced reasoning beyond memorization

# CODING BENCHMARKS
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb", "lcb:codegeneration", 0)
# WHAT: LiveCodeBench code generation
# WHY: Real-world coding problems, constantly updated
# WHY: Tests code generation capability

register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb_v4", "lcb:codegeneration_v4", 0)
# WHAT: LiveCodeBench v4 (newer version)
# WHY: Updated problem set


def get_lighteval_tasks():
    """
    WHAT: Returns list of all registered benchmark names

    WHY: Used for --benchmarks=all option
         Enables listing available benchmarks

    Returns:
        List of benchmark names (e.g., ["math_500", "aime24", ...])
    """
    return list(LIGHTEVAL_TASKS.keys())


SUPPORTED_BENCHMARKS = get_lighteval_tasks()


"""
==============================================================================
EVALUATION JOB EXECUTION
==============================================================================
"""


def run_lighteval_job(
    benchmark: str,
    training_args: Union["SFTConfig", "GRPOConfig"],
    model_args: "ModelConfig",
) -> None:
    """
    WHAT: Submits a Slurm job to run LightEval on a specific benchmark

    WHY: Evaluation is compute-intensive and should run on separate GPUs
         from training to avoid interfering with training throughput

    HOW:
        1. Get task configuration for benchmark
        2. Determine GPU requirements (tensor parallelism for large models)
        3. Build sbatch command with arguments
        4. Submit Slurm job

    PROXIMAL CONTEXT:
        - Input: Benchmark name, training config, model config
        - Output: Slurm job submitted (asynchronous)

    DISTAL CONTEXT:
        - Originates from: run_benchmark_jobs() during training
        - Flows to: Slurm scheduler → evaluate.slurm script → vLLM + LightEval
        - Results flow to: HuggingFace Hub metrics

    Args:
        benchmark: Benchmark name (e.g., "math_500")
        training_args: Training configuration with hub_model_id, hub_model_revision
        model_args: Model configuration with trust_remote_code flag

    EVALUATION WORKFLOW:
        1. Slurm job allocated GPUs
        2. evaluate.slurm script runs
        3. vLLM server launched with model
        4. LightEval queries vLLM for predictions
        5. Metrics computed and logged to Hub

    EXAMPLE:
        run_lighteval_job(
            benchmark="math_500",
            training_args=SFTConfig(
                hub_model_id="open-r1/OpenR1-7B-sft",
                hub_model_revision="step_1000"
            ),
            model_args=ModelConfig(trust_remote_code=False)
        )
    """
    # STEP 1: Get task configuration
    # WHY: Need to pass task list to LightEval
    task_list = LIGHTEVAL_TASKS[benchmark]

    # STEP 2: Get model identifiers
    # WHY: Load specific checkpoint from Hub for evaluation
    model_name = training_args.hub_model_id
    model_revision = training_args.hub_model_revision

    # STEP 3: Determine GPU requirements
    # WHY: Large models need tensor parallelism to fit in memory
    # HOW: Check parameter count, allocate GPUs accordingly
    #
    # TENSOR PARALLELISM:
    # - Models >= 30B params: Shard across GPUs with tensor parallelism
    # - Smaller models: Can fit on 1-2 GPUs
    num_gpus = get_gpu_count_for_vllm(model_name, model_revision)

    if get_param_count_from_repo_id(model_name) >= 30_000_000_000:
        # WHAT: Large model (>= 30B parameters)
        # WHY: Need tensor parallelism to shard model across GPUs
        # HOW: vLLM --tensor-parallel-size flag
        tensor_parallel = True
    else:
        # WHAT: Smaller model (< 30B parameters)
        # WHY: Can fit on fewer GPUs without tensor parallelism
        num_gpus = 2  # Hack while cluster is full
        tensor_parallel = False

    # STEP 4: Build sbatch command
    # WHY: Need to submit Slurm job with proper resource allocation
    # HOW: Start with clean environment prefix, add sbatch args

    cmd = VLLM_SLURM_PREFIX.copy()
    cmd_args = [
        # GPU ALLOCATION
        # WHAT: Request GPUs from Slurm
        # WHY: vLLM needs GPUs for inference
        f"--gres=gpu:{num_gpus}",

        # JOB NAME
        # WHAT: Descriptive name for Slurm job
        # WHY: Easy to identify in squeue/sacct
        # FORMAT: or1_{benchmark}_{model}_{revision}
        f"--job-name=or1_{benchmark}_{model_name.split('/')[-1]}_{model_revision}",

        # SLURM SCRIPT
        # WHAT: Path to evaluation script
        # WHY: Contains logic to launch vLLM + LightEval
        "slurm/evaluate.slurm",

        # SCRIPT ARGUMENTS (passed to evaluate.slurm)
        benchmark,  # Benchmark name (e.g., "math_500")
        f'"{task_list}"',  # Task list in LightEval format (quoted for shell)
        model_name,  # HuggingFace model ID
        model_revision,  # Git revision/tag
        f"{tensor_parallel}",  # Whether to use tensor parallelism
        f"{model_args.trust_remote_code}",  # Whether to trust remote code
    ]

    # STEP 5: Add system prompt (if provided)
    # WHY: Some models need system prompt for proper evaluation
    # HOW: Base64 encode to avoid shell escaping issues
    if training_args.system_prompt is not None:
        # WHAT: Encode system prompt to base64
        # WHY: Avoid issues with special characters in shell command
        # HOW: Decode in evaluate.slurm script
        prompt_encoded = base64.b64encode(training_args.system_prompt.encode()).decode()
        cmd_args.append(prompt_encoded)

    # STEP 6: Finalize command and submit
    # WHY: Slurm sbatch expects single command string
    # HOW: Append all args to the sbatch command
    cmd[-1] += " " + " ".join(cmd_args)

    # STEP 7: Run sbatch command
    # WHY: Submit job to Slurm scheduler
    # HOW: subprocess.run() with check=True (raise on error)
    subprocess.run(cmd, check=True)


def run_benchmark_jobs(training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig") -> None:
    """
    WHAT: Runs evaluation jobs for all specified benchmarks

    WHY: Training configs can specify multiple benchmarks to evaluate
         "all" option runs every supported benchmark

    HOW:
        1. Get benchmark list from training_args
        2. Expand "all" to full benchmark list
        3. Submit Slurm job for each benchmark

    PROXIMAL CONTEXT:
        - Input: Training config with benchmarks list
        - Output: Multiple Slurm jobs submitted

    DISTAL CONTEXT:
        - Originates from: Training script at evaluation intervals
        - Flows to: Slurm jobs → vLLM + LightEval → Hub metrics

    INTEGRATION POINTS:
        - Called from SFT/GRPO training scripts
        - Usually at end of training or at regular checkpoints
        - Can be called manually for ad-hoc evaluation

    Args:
        training_args: Training configuration with benchmarks list
        model_args: Model configuration

    EXAMPLE:
        # In training script:
        if training_args.benchmarks:
            run_benchmark_jobs(training_args, model_args)

    TYPICAL USAGE:
        # Evaluate on specific benchmarks
        training_args.benchmarks = ["math_500", "aime24"]
        run_benchmark_jobs(training_args, model_args)

        # Evaluate on all benchmarks
        training_args.benchmarks = ["all"]
        run_benchmark_jobs(training_args, model_args)
    """
    # STEP 1: Get benchmark list
    benchmarks = training_args.benchmarks

    # STEP 2: Expand "all" to full list
    # WHY: User convenience - don't need to list every benchmark
    # HOW: Replace ["all"] with get_lighteval_tasks()
    if len(benchmarks) == 1 and benchmarks[0] == "all":
        benchmarks = get_lighteval_tasks()
        # Evaluate on all supported benchmarks. Later we may want to include a `chat` option
        # that just evaluates on `ifeval` and `mt_bench` etc.

    # STEP 3: Submit job for each benchmark
    # WHY: Each benchmark runs as separate Slurm job
    # WHY: Allows parallel evaluation, easier to debug failures
    for benchmark in benchmarks:
        print(f"Launching benchmark `{benchmark}`")
        if benchmark in get_lighteval_tasks():
            run_lighteval_job(benchmark, training_args, model_args)
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")


"""
==============================================================================
KEY TAKEAWAYS - MODEL EVALUATION
==============================================================================

1. **LightEval Integration**:
   - Standard framework for evaluating LLMs
   - Consistent metrics across research community
   - Supports both built-in and custom tasks

2. **Supported Benchmarks**:
   - MATH-500: Fast math reasoning evaluation
   - AIME 24/25: Competition math (very challenging)
   - GPQA: PhD-level science questions
   - LiveCodeBench: Real-world coding problems

3. **Slurm Job Submission**:
   - Evaluation runs on separate GPUs from training
   - Clean environment needed for vLLM
   - Asynchronous job submission (don't block training)

4. **Tensor Parallelism**:
   - Models >= 30B params sharded across GPUs
   - Smaller models can use 1-2 GPUs
   - vLLM handles sharding automatically

5. **System Prompt Handling**:
   - Base64 encoding for safe shell escaping
   - Decoded in Slurm script
   - Important for chat-tuned models

6. **Evaluation Workflow**:
   1. Training script checkpoints model to Hub
   2. run_benchmark_jobs() called
   3. Slurm jobs submitted (one per benchmark)
   4. vLLM server loads model
   5. LightEval queries vLLM for predictions
   6. Metrics logged to Hub

7. **Integration with Training**:
   - Called at end of training or at intervals
   - training_args.benchmarks = ["all"] or specific list
   - Results tracked on HuggingFace Hub

8. **Typical Evaluation Schedule**:
   - SFT: Evaluate at end of training
   - GRPO: Evaluate every N steps (e.g., every 500 steps)
   - Final model: Evaluate on all benchmarks

9. **Performance Considerations**:
   - vLLM provides fast inference (PagedAttention)
   - Tensor parallelism for large models
   - Separate GPUs avoid slowing down training

10. **Custom Tasks**:
    - Extended tasks in scripts/evaluation/extended_lighteval_tasks
    - Follow LightEval task format
    - Register with register_lighteval_task()

==============================================================================
"""
