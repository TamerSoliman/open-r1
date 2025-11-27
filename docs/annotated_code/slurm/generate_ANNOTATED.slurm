#!/bin/bash
# ==============================================================================
# FILE: slurm/generate.slurm
# CATEGORY: Slurm Scripts - Distributed Data Generation
# PRIORITY: HIGH
# LINES: 244
# DEPENDENCIES: Slurm, Ray, vLLM, HuggingFace datasets
# ==============================================================================
#
# OVERVIEW:
# Slurm script for large-scale synthetic data generation using distributed
# vLLM inference. Sets up Ray cluster across multiple nodes, launches vLLM
# server with pipeline parallelism, and runs Distilabel generation pipeline.
#
# ROLE IN DEEPSEEK R1:
# - Generates synthetic reasoning traces for SFT training
# - Scales to multi-node generation (2+ nodes × 8 GPUs each)
# - Uses vLLM for fast inference (10-20× faster than HF transformers)
# - Produces Mixture-of-Thoughts dataset
#
# WORKFLOW:
# 1. Allocate Slurm nodes (2 nodes × 8 GPUs = 16 GPUs total)
# 2. Start Ray cluster (head node + workers)
# 3. Launch vLLM server with tensor + pipeline parallelism
# 4. Wait for vLLM server to be ready
# 5. Submit generation job to Ray cluster
# 6. Generate completions and save to HuggingFace dataset
#
# PARALLELISM STRATEGY:
# - Tensor Parallelism: Shard model across 8 GPUs per node
# - Pipeline Parallelism: Split model layers across nodes
# - Example: 2 nodes × 8 GPUs = 16-way parallelism for large models
#
# TYPICAL USAGE:
# sbatch slurm/generate.slurm \
#   --model deepseek-ai/DeepSeek-R1 \
#   --hf-dataset open-r1/math-prompts \
#   --num-generations 16 \
#   --hf-output-dataset open-r1/Mixture-of-Thoughts
# ==============================================================================

# ==============================================================================
# SLURM JOB CONFIGURATION
# ==============================================================================

#SBATCH --job-name=deepseek-r1-generation
# Job name for squeue/sacct

#SBATCH --partition=hopper-prod
# WHAT: Hopper GPU partition (H100 GPUs)
# WHY: Need latest GPUs for fast generation

#SBATCH --qos=normal
# Quality of service (priority level)

#SBATCH --nodes=2
# WHAT: Request 2 nodes
# WHY: Large model needs pipeline parallelism
#      Example: DeepSeek-R1 (70B+) split across 2 nodes

#SBATCH --exclusive
# WHAT: Exclusive node access (no sharing)
# WHY: Maximum performance, no interference

#SBATCH --gpus-per-node=8
# WHAT: 8 GPUs per node (standard H100 node config)
# WHY: Tensor parallelism across 8 GPUs per node

#SBATCH --output=./logs/%x-%j.out
# WHAT: Stdout log file (%x = job name, %j = job ID)
# EXAMPLE: ./logs/deepseek-r1-generation-12345.out

#SBATCH --error=./logs/%x-%j.err
# WHAT: Stderr log file
# EXAMPLE: ./logs/deepseek-r1-generation-12345.err

#SBATCH --time=04-00:00:00
# WHAT: Maximum runtime (4 days)
# WHY: Large-scale generation can take days
#      Generating millions of samples takes time

# ==============================================================================
# COMMAND LINE ARGUMENT PARSING
# ==============================================================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-dataset)
            HF_DATASET="$2"
            shift 2
            ;;
        --hf-dataset-config)
            HF_DATASET_CONFIG="$2"
            shift 2
            ;;
        --hf-dataset-split)
            HF_DATASET_SPLIT="$2"
            shift 2
            ;;
        --prompt-column)
            PROMPT_COLUMN="$2"
            shift 2
            ;;
        --prompt-template)
            PROMPT_TEMPLATE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --num-generations)
            NUM_GENERATIONS="$2"
            shift 2
            ;;
        --input-batch-size)
            INPUT_BATCH_SIZE="$2"
            shift 2
            ;;
        --client-replicas)
            CLIENT_REPLICAS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --retries)
            RETRIES="$2"
            shift 2
            ;;
        --hf-output-dataset)
            HF_OUTPUT_DATASET="$2"
            shift 2
            ;;
        --private)
            PRIVATE="true"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$MODEL" ] || [ -z "$HF_DATASET" ]; then
    echo "Error: --model and --hf-dataset are required parameters"
    exit 1
fi

# Set default values for optional parameters
HF_DATASET_SPLIT=${HF_DATASET_SPLIT:-"train"}
PROMPT_COLUMN=${PROMPT_COLUMN:-"prompt"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"{{ instruction }}"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}  # Long for reasoning chains
NUM_GENERATIONS=${NUM_GENERATIONS:-1}
INPUT_BATCH_SIZE=${INPUT_BATCH_SIZE:-64}
CLIENT_REPLICAS=${CLIENT_REPLICAS:-1}
TIMEOUT=${TIMEOUT:-900}  # 15 minutes per request
RETRIES=${RETRIES:-0}
PRIVATE=${PRIVATE:-"false"}

# Print all input arguments for debugging
echo "Input arguments:"
echo "MODEL: $MODEL"
echo "HF_DATASET: $HF_DATASET"
echo "HF_DATASET_CONFIG: $HF_DATASET_CONFIG"
echo "HF_DATASET_SPLIT: $HF_DATASET_SPLIT"
echo "PROMPT_COLUMN: $PROMPT_COLUMN"
echo "PROMPT_TEMPLATE: $PROMPT_TEMPLATE"
echo "TEMPERATURE: $TEMPERATURE"
echo "TOP_P: $TOP_P"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "NUM_GENERATIONS: $NUM_GENERATIONS"
echo "INPUT_BATCH_SIZE: $INPUT_BATCH_SIZE"
echo "CLIENT_REPLICAS: $CLIENT_REPLICAS"
echo "TIMEOUT: $TIMEOUT"
echo "RETRIES: $RETRIES"
echo "HF_OUTPUT_DATASET: $HF_OUTPUT_DATASET"
echo "PRIVATE: $PRIVATE"
echo "-------------------"

set -ex  # Exit on error, print commands

module load cuda/12.4  # Load CUDA module

export LD_LIBRARY_PATH=.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

source openr1/bin/activate  # Activate Python virtual environment

# ==============================================================================
# RAY CLUSTER SETUP
# ==============================================================================

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Start Ray head node
port=6379  # Ray default port
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --block &

# Give some time to head node to start...
sleep 10

# Start Ray worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))

# Start from 1 (0 is head node)
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --block &
    sleep 5
done

# Give some time to the Ray cluster to gather info
echo "Waiting a bit for Ray cluster to gather node info..."
sleep 60

# ==============================================================================
# VLLM SERVER SETUP
# ==============================================================================

# Run vllm
RAY_ADDRESS="http://$head_node_ip:8265" ray job submit \
    --working-dir src/open_r1 \
    --no-wait \
    --job-id vllm-server \
    -- vllm serve $MODEL \
    --tensor-parallel-size $SLURM_GPUS_PER_NODE \
    --pipeline-parallel-size $SLURM_JOB_NUM_NODES \
    --gpu-memory-utilization=0.85 \
    --max-model-len 16384 \
    --enable-chunked-prefill \
    --trust-remote-code \
    --distributed-executor-backend ray

# wait for vllm to load the model
echo "Waiting for vLLM (http://$head_node_ip:8000) server to be up..."

# wait for vllm to load and serve the model
while true; do
    if curl -s -o /dev/null -w "%{http_code}" http://$head_node_ip:8000 >/dev/null 2>&1; then
        echo "Received response from http://$head_node_ip:8000"
        break
    else
        echo "Still waiting... (Press Ctrl+C to cancel)"
        sleep 60
    fi
done

echo "Checking available models..."
curl http://$head_node_ip:8000/v1/models

echo "Executing sanity check..."
curl http://$head_node_ip:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"<｜begin▁of▁sentence｜><｜User｜>hi, how are you?<｜Assistant｜>\",
        \"max_tokens\": 2048,
        \"temperature\": 0.6
    }"

# ==============================================================================
# DATA GENERATION JOB
# ==============================================================================

# Finally submit the job to the cluster
echo "Submitting job to ray cluster..."
RAY_ADDRESS="http://$head_node_ip:8265" ray job submit \
    --working-dir src/open_r1 \
    --job-id generate \
    -- python -u generate.py \
    --model "$MODEL" \
    --hf-dataset "$HF_DATASET" \
    ${HF_DATASET_CONFIG:+--hf-dataset-config "$HF_DATASET_CONFIG"} \
    --hf-dataset-split "$HF_DATASET_SPLIT" \
    --prompt-column "$PROMPT_COLUMN" \
    --prompt-template "$PROMPT_TEMPLATE" \
    ${TEMPERATURE:+--temperature "$TEMPERATURE"} \
    ${TOP_P:+--top-p "$TOP_P"} \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --num-generations "$NUM_GENERATIONS" \
    --input-batch-size "$INPUT_BATCH_SIZE" \
    --client-replicas "$CLIENT_REPLICAS" \
    --timeout "$TIMEOUT" \
    --retries "$RETRIES" \
    ${HF_OUTPUT_DATASET:+--hf-output-dataset "$HF_OUTPUT_DATASET"} \
    ${PRIVATE:+--private} \
    --vllm-server-url "http://$head_node_ip:8000/v1"

mkdir -p ray_logs

echo "Downloading Ray job logs..."
RAY_ADDRESS="http://$head_node_ip:8265" ray job logs --job-id vllm-server > ray_logs/vllm-server-${SLURM_JOB_ID}.log
RAY_ADDRESS="http://$head_node_ip:8265" ray job logs --job-id generate > ray_logs/generate-${SLURM_JOB_ID}.log

# ==============================================================================
# KEY TAKEAWAYS - DATA GENERATION SLURM SCRIPT
# ==============================================================================
#
# 1. **Multi-Node Ray Cluster**:
#    - Sets up Ray across 2+ nodes
#    - Head node coordinates, workers execute
#    - Dashboard on port 8265 for monitoring
#
# 2. **vLLM Parallelism**:
#    - Tensor parallelism: 8 GPUs per node
#    - Pipeline parallelism: Split across nodes
#    - Total: 16 GPUs (2 nodes × 8 GPUs)
#
# 3. **Generation Parameters**:
#    - max_new_tokens: 8192 (long reasoning chains)
#    - num_generations: 1 (can set higher for diversity)
#    - input_batch_size: 64 (prompts processed in parallel)
#
# 4. **Typical Use Case**:
#    - Generate Mixture-of-Thoughts from DeepSeek-R1
#    - Input: Math/code prompts
#    - Output: Reasoning traces for SFT training
#
# 5. **Performance**:
#    - vLLM: 10-20× faster than HF transformers
#    - Multi-node: Scale to very large models (70B+)
#    - Chunked prefill: Handle long contexts efficiently
#
# 6. **Job Management**:
#    - Ray jobs: vllm-server + generate
#    - Logs saved to ray_logs/
#    - Can monitor via Ray dashboard
#
# 7. **Configuration**:
#    - gpu-memory-utilization: 0.85 (leave 15% buffer)
#    - max-model-len: 16384 (context window)
#    - timeout: 900 seconds (15 min per request)
#
# 8. **Output**:
#    - Saves to HuggingFace dataset
#    - Can be private or public
#    - Ready for SFT training
#
# ==============================================================================
