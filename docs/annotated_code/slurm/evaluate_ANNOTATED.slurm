#!/bin/bash
# ==============================================================================
# FILE: slurm/evaluate.slurm
# CATEGORY: Slurm Scripts - Model Evaluation
# PRIORITY: HIGH
# LINES: 84
# DEPENDENCIES: Slurm, vLLM, LightEval, HuggingFace Hub
# ==============================================================================
#
# OVERVIEW:
# Slurm script for evaluating trained models on benchmarks using LightEval.
# Runs vLLM for fast inference and uploads results to HuggingFace leaderboard.
#
# ROLE IN DEEPSEEK R1:
# - Evaluates checkpoints on MATH, AIME, GPQA, LiveCodeBench
# - Runs automatically after training via callbacks
# - Uploads results to open-r1-eval-leaderboard
# - Tracks model performance across training
#
# WORKFLOW:
# 1. Allocate Slurm node with 8 GPUs
# 2. Load model with vLLM (tensor or data parallelism)
# 3. Run LightEval on specified benchmarks
# 4. Upload results to HuggingFace Spaces leaderboard
# 5. Upload detailed results to separate repo
# 6. Cleanup and finish
#
# TYPICAL USAGE (called from evaluation.py):
# sbatch slurm/evaluate.slurm \
#   math_500 \
#   "lighteval|math_500|0|0" \
#   open-r1/OpenR1-7B-GRPO \
#   step_1000 \
#   True \
#   False
# ==============================================================================

# ==============================================================================
# SLURM JOB CONFIGURATION
# ==============================================================================

#SBATCH --ntasks-per-node=1
# One task per node (single-node eval)

#SBATCH --gres=gpu:8
# Request 8 GPUs per node

#SBATCH --partition=hopper-prod
# Hopper GPU partition (H100 GPUs)

#SBATCH --output=./logs/%x-%j.out
# Stdout log file

#SBATCH --error=./logs/%x-%j.err
# Stderr log file

#SBATCH --requeue
# WHAT: Automatically requeue if preempted
# WHY: Cluster may preempt low-priority jobs
#      Requeue ensures eval completes eventually

#SBATCH --time=1-00:00:00
# WHAT: Maximum runtime (1 day)
# WHY: Evaluation usually takes hours, not days
#      Much shorter than training or generation

# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.4

# Refresh Weka on h4 cache
echo "Refreshing Weka filesystem..."
find -L /fsx/h4/ -type f | xargs -d '\n' -r -n512 -P64 weka fs tier fetch
# WHAT: Prefetch files from Weka filesystem
# WHY: Weka is distributed filesystem with tiering
#      Prefetch ensures fast access during eval

# Needed for vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# WHAT: Use spawn method for vLLM worker processes
# WHY: Fork can cause issues with CUDA contexts
#      Spawn creates fresh processes

set -x -e  # Print commands, exit on error

source ~/.bashrc
source openr1/bin/activate  # Activate Python virtual environment

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

TASK_NAME=$1
# WHAT: Benchmark name (e.g., "math_500", "aime24")
# WHY: Used for organizing results

TASKS=$2
# WHAT: LightEval task list (e.g., "lighteval|math_500|0|0")
# WHY: Defines which tasks to run in LightEval format

MODEL_ID=$3
# WHAT: HuggingFace model ID (e.g., "open-r1/OpenR1-7B-GRPO")
# WHY: Model to evaluate

MODEL_REVISION=$4
# WHAT: Git revision (e.g., "step_1000", "main")
# WHY: Specific checkpoint to evaluate

# Optional args
[ -z "$5"] && TENSOR_PARALLEL=False || TENSOR_PARALLEL=$5
# WHAT: Whether to use tensor parallelism
# WHY: Large models (30B+) need sharding across GPUs

[ -z "$6"] && TRUST_REMOTE_CODE=False || TRUST_REMOTE_CODE=$6
# WHAT: Whether to trust remote code
# WHY: Some models have custom code in repo

# $7 is reserved for system_prompt, see line 51
# WHAT: Optional system prompt (base64 encoded)
# WHY: Some models need specific system prompts

NUM_GPUS=$(nvidia-smi -L | wc -l)
# WHAT: Count available GPUs
# WHY: Determines parallelism strategy

# ==============================================================================
# VLLM CONFIGURATION
# ==============================================================================

# Use TP to shard model across GPUs
if [ "$TENSOR_PARALLEL" = "True" ]; then
    # WHAT: Tensor parallelism (shard model across GPUs)
    # WHY: Large models don't fit on single GPU
    # EXAMPLE: 30B model split across 8 GPUs
    MODEL_ARGS="model_name=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
else
    # WHAT: Data parallelism (replicate model on each GPU)
    # WHY: Smaller models fit on single GPU
    # BENEFIT: Faster than tensor parallelism for small models
    MODEL_ARGS="model_name=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
fi

LM_EVAL_REPO_ID="open-r1/open-r1-eval-leaderboard"
# WHAT: HuggingFace Spaces repo for leaderboard
# WHY: Central location for tracking all model evaluations

MODEL_NAME=$(echo $MODEL_ID | sed 's/\//_/g') # replaces / with _
# WHAT: Convert model ID to filesystem-safe name
# EXAMPLE: "open-r1/OpenR1-7B" → "open-r1_OpenR1-7B"

DETAILS_REPO_ID="open-r1/details-$MODEL_NAME"
# WHAT: Separate repo for detailed results
# WHY: Leaderboard shows summary, details repo has full outputs

OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME"
# WHAT: Local directory for evaluation results
# EXAMPLE: eval_results/open-r1/OpenR1-7B/step_1000/math_500

# We need this flag since we run this script from training jobs that use DeepSpeed and the env vars get progated which causes errors during evaluation
ACCELERATE_USE_DEEPSPEED=false
# WHAT: Disable DeepSpeed for evaluation
# WHY: Training uses DeepSpeed, but eval uses vLLM
#      Env vars from training can cause conflicts

# ==============================================================================
# RUN LIGHTEVAL
# ==============================================================================

echo "Running lighteval script ..."
echo "Eval results will be saved to $OUTPUT_DIR"
lighteval vllm "$MODEL_ARGS" $TASKS \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details \
    ${7:+--system-prompt "$(echo "$7" | base64 --decode)"}
# WHAT: Run LightEval with vLLM backend
# WHY: LightEval standardizes evaluation across benchmarks
#      vLLM provides fast inference (10-20× faster than HF)
# FLAGS:
#   --use-chat-template: Apply chat template to prompts
#   --save-details: Save detailed outputs (not just metrics)
#   --system-prompt: Optional system prompt (base64 decoded)

# ==============================================================================
# UPLOAD RESULTS TO HUB
# ==============================================================================

OUTPUT_FILEPATHS=$(find $OUTPUT_DIR/results/ -type f \( -name "*.json" \))
for filepath in $OUTPUT_FILEPATHS; do
    echo "Uploading $filepath to Hugging Face Hub..."
    filename=$(basename -- "$filepath")
    # Retry upload up to 20 times
    # WHY: Network can be flaky, ensure upload succeeds
    for attempt in {1..20}; do
        if huggingface-cli upload --repo-type space --private $LM_EVAL_REPO_ID $filepath $OUTPUT_DIR/$filename; then
            echo "Upload succeeded for $filepath"
            break
        else
            echo "Upload failed for $filepath. Attempt $attempt of 20. Retrying in 5 seconds..."
            sleep 5
        fi
    done
done

echo "Uploading details to Hugging Face Hub..."
DETAILS_FILEPATHS=$(find $OUTPUT_DIR/details/ -type f \( -name "*.parquet" \))
echo "DETAILS_FILEPATHS: $DETAILS_FILEPATHS"
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
# WHAT: Upload detailed results (full outputs) to separate repo
# WHY: Detailed results are large (parquet files with all outputs)
#      Separate repo keeps leaderboard repo clean
python scripts/upload_details.py --data_files $DETAILS_FILEPATHS --hub_repo_id $DETAILS_REPO_ID --config_name $MODEL_REVISION.$TASK_NAME.$TIMESTAMP

echo "Cleaning up ..."
rm -rf $OUTPUT_DIR
# WHAT: Remove local evaluation results
# WHY: Save disk space (results uploaded to Hub)

echo "Done!"

# ==============================================================================
# KEY TAKEAWAYS - EVALUATION SLURM SCRIPT
# ==============================================================================
#
# 1. **LightEval Integration**:
#    - Standard framework for LLM evaluation
#    - Consistent metrics across models
#    - Supports vLLM backend for speed
#
# 2. **vLLM Parallelism**:
#    - Tensor parallelism for large models (>30B)
#    - Data parallelism for small models (<30B)
#    - 8 GPUs per node (H100)
#
# 3. **Leaderboard Upload**:
#    - Results → open-r1-eval-leaderboard (summary)
#    - Details → open-r1/details-{model} (full outputs)
#    - Retry logic (20 attempts) for reliability
#
# 4. **Configuration**:
#    - max_model_length: 32768 (long context)
#    - gpu_memory_utilization: 0.8 (80% GPU memory)
#    - temperature: 0.6, top_p: 0.95 (sampling params)
#
# 5. **Integration with Training**:
#    - Called from evaluation.py via callbacks
#    - Runs at checkpoints (e.g., every 500 steps)
#    - Tracks performance across training
#
# 6. **Benchmarks**:
#    - MATH-500: Fast math evaluation
#    - AIME: Competition math
#    - GPQA: PhD-level science
#    - LiveCodeBench: Real-world coding
#
# 7. **HuggingFace Cluster Specific**:
#    - Weka filesystem refresh
#    - May not work on other clusters
#    - Optimized for H4 cluster
#
# 8. **Job Management**:
#    - Auto-requeue on preemption
#    - 1 day timeout (usually takes hours)
#    - Logs saved to ./logs/
#
# 9. **System Prompt**:
#    - Optional base64-encoded prompt
#    - Decoded before passing to LightEval
#    - Important for chat-tuned models
#
# 10. **Cleanup**:
#     - Delete local results after upload
#     - Saves disk space on cluster
#     - All results preserved on Hub
#
# ==============================================================================
