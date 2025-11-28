"""
==============================================================================
FILE: scripts/get_tensor_parallel_size.py
CATEGORY: Scripts - Tensor Parallelism Configuration
PRIORITY: MEDIUM
LINES: 29
DEPENDENCIES:
    - transformers: AutoConfig (model configuration loading)
    - math: gcd (greatest common divisor calculation)
==============================================================================

==============================================================================
OVERVIEW
==============================================================================

This script calculates the optimal tensor parallelism (TP) size for a given
model based on its architecture, specifically the number of attention heads.
Tensor parallelism is a distributed training technique that splits model layers
across multiple GPUs to enable training models larger than single-GPU memory.

ROLE IN DEEPSEEK R1:
-------------------
During GRPO training with vLLM for generation, models are often distributed
across multiple GPUs using tensor parallelism. The TP size must be chosen
carefully to ensure:

1. **Hardware Compatibility**: TP size must divide evenly into available GPUs
   - 8 GPUs → TP can be 1, 2, 4, or 8
   - Using TP=3 on 8 GPUs would fail (not evenly divisible)

2. **Model Compatibility**: TP size must divide evenly into attention heads
   - Model with 32 heads → TP can be 1, 2, 4, 8, 16, or 32
   - Using TP=3 would split 32 heads unevenly (10.67 heads per GPU)

3. **Optimal Utilization**: Choose largest TP size that satisfies both constraints
   - Available GPUs: 8
   - Attention heads: 32
   - Optimal TP: gcd(8, 32) = 8 (use all GPUs)

WHY ATTENTION HEADS MATTER:
---------------------------
In transformer models, attention is the most memory-intensive operation.
Tensor parallelism splits attention heads across GPUs:

- 32 heads, TP=8 → 4 heads per GPU
- Each GPU computes attention for its subset of heads
- Results are gathered and concatenated

If TP doesn't divide heads evenly, some GPUs get unequal workloads or
the implementation fails entirely.

WHAT IS GCD (GREATEST COMMON DIVISOR)?
---------------------------------------
The GCD is the largest number that divides both inputs evenly.

Examples:
- gcd(8, 32) = 8 (both divisible by 8)
- gcd(8, 40) = 8 (both divisible by 8)
- gcd(8, 30) = 2 (both divisible by 2, but not 4 or 8)
- gcd(7, 13) = 1 (prime numbers, only divisible by 1)

For TP calculation:
- gcd(num_gpus, num_heads) gives the largest TP that satisfies both constraints

TYPICAL SCENARIOS:
------------------

Scenario 1: Qwen2.5-1.5B on 8 GPUs
    - num_heads = 12
    - default_tp = 8 (GPUs per node)
    - gcd(12, 8) = 4
    - Optimal TP: 4 (uses 4 GPUs, leaves 4 idle or for data parallelism)

Scenario 2: Qwen2.5-7B on 8 GPUs
    - num_heads = 32
    - default_tp = 8
    - gcd(32, 8) = 8
    - Optimal TP: 8 (uses all GPUs)

Scenario 3: Custom model with 30 heads on 8 GPUs
    - num_heads = 30
    - default_tp = 8
    - gcd(30, 8) = 2
    - Optimal TP: 2 (uses 2 GPUs, suboptimal but safe)

DATA FLOW:
----------
    DISTAL ORIGIN:
    └─> HuggingFace Hub → Model config.json (contains num_attention_heads)

    PROXIMAL PROCESSING (this script):
    1. Fetch model config from Hub (or local cache)
    2. Extract num_attention_heads attribute
    3. If heads don't divide evenly by default_tp:
       a. Compute gcd(num_heads, default_tp)
       b. Return gcd (safe TP size)
    4. Otherwise, return default_tp
    5. Print result to stdout

    DISTAL DESTINATION:
    └─> Shell script captures stdout as TP size
        └─> Passed to vLLM server via --tensor-parallel-size flag

==============================================================================
IMPORTS AND SETUP
==============================================================================
"""

import argparse
from transformers import AutoConfig
from math import gcd

"""
==============================================================================
WHAT: get_tensor_parallel_size()
WHY:  Calculate safe TP size based on model architecture
HOW:  Fetch config, check head divisibility, compute GCD if needed
==============================================================================

PROXIMAL CONTEXT: Called from __main__ with CLI arguments
DISTAL CONTEXT: Result used to configure vLLM server startup

This function determines the tensor parallelism size using this logic:

1. Load model config from HuggingFace Hub (or local cache)
2. Extract num_attention_heads (core architectural parameter)
3. Check if num_heads is evenly divisible by default_tp
   - If yes: Safe to use default_tp (all GPUs utilized)
   - If no: Compute gcd(num_heads, default_tp) for safe alternative
4. Return max(gcd_result, 1) to ensure TP >= 1

EDGE CASES:
----------

Case 1: num_heads not in config
    - Some architectures use different parameter names
    - Function returns default_tp (conservative fallback)
    - Example: Encoder-only models might use num_attention_heads_encoder

Case 2: Config fetch fails (network error, private model, invalid name)
    - Exception caught, warning printed
    - Returns default_tp (safe fallback)
    - Example: Typo in model name, rate limit exceeded

Case 3: gcd results in TP=1
    - Happens with prime number heads (e.g., 7, 11, 13)
    - Falls back to TP=1 (no tensor parallelism, slower but works)
    - Example: Custom model with 13 heads on 8 GPUs

PARAMETERS:
----------
- model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
- revision: Git branch/tag/commit (default: None = "main")
- default_tp: Number of GPUs available (default: 8)

RETURN VALUE:
------------
Integer TP size that is:
1. Compatible with model architecture (divides num_heads)
2. Compatible with hardware (divides available GPUs)
3. As large as possible (maximizes parallelism)

WHY trust_remote_code=True?
---------------------------
Some models on HuggingFace Hub include custom code in modeling files.
Setting trust_remote_code=True allows AutoConfig to load this code.

Security note: Only use with trusted models (official releases, vetted sources).
For untrusted models, set trust_remote_code=False and handle exceptions.
"""
def get_tensor_parallel_size(model_name: str, revision: str = None, default_tp: int = 8) -> int:
    try:
        # WHAT: Load model configuration from HuggingFace Hub
        # WHY: Need num_attention_heads to validate TP compatibility
        # HOW: AutoConfig fetches config.json and parses into Python object
        config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)

        # WHAT: Extract number of attention heads from config
        # WHY: TP size must divide num_heads evenly for tensor parallelism to work
        # HOW: getattr with default=None to handle missing attribute gracefully
        num_heads = getattr(config, 'num_attention_heads', None)

        # WHAT: Check if default_tp is compatible with num_heads
        # WHY: If num_heads % default_tp != 0, some GPUs would get unequal heads
        if num_heads is not None and num_heads % default_tp != 0:
            # WHAT: Compute largest TP size compatible with both heads and GPUs
            # WHY: gcd ensures even division of heads across GPUs
            # HOW: gcd(a, b) returns largest number dividing both a and b
            tp = gcd(num_heads, default_tp)

            # WHAT: Ensure TP is at least 1
            # WHY: TP=0 would be invalid (no GPUs used)
            # HOW: max(tp, 1) returns larger of tp or 1
            return max(tp, 1)
        else:
            # WHAT: Use default_tp if compatible or num_heads not found
            # WHY: No adjustment needed when heads divide evenly or config incomplete
            return default_tp

    except Exception as e:
        # WHAT: Handle any errors in config loading
        # WHY: Network failures, invalid model names, permission issues can occur
        # HOW: Print warning and return safe fallback (default_tp)
        print(f"Warning: Failed to fetch config for {model_name}@{revision}: {e}")
        return default_tp

"""
==============================================================================
WHAT: Main entry point
WHY:  Parse CLI arguments and compute TP size
HOW:  argparse + function call + stdout print
==============================================================================

Command-line interface:
    python scripts/get_tensor_parallel_size.py \
        --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
        --revision "main" \
        --default_tp 8

Output: Single integer printed to stdout (e.g., "4")

WHY PRINT TO STDOUT?
- Shell scripts can capture output: TP=$(python script.py ...)
- Clean interface: No logging noise, just the result
- Composable: Can pipe to other tools

TYPICAL USAGE IN SHELL SCRIPT:
------------------------------
    #!/bin/bash
    MODEL="Qwen/Qwen2.5-1.5B-Instruct"
    NUM_GPUS=8

    # Calculate optimal TP size
    TP_SIZE=$(python scripts/get_tensor_parallel_size.py \
        --model_name "$MODEL" \
        --default_tp "$NUM_GPUS")

    echo "Using TP size: $TP_SIZE"

    # Start vLLM server with computed TP size
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size "$TP_SIZE" \
        ...

This ensures vLLM is configured correctly for the model architecture.
"""
if __name__ == "__main__":
    # WHAT: Parse command-line arguments for TP calculation
    # WHY: Make script configurable for different models and hardware
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--revision", type=str, default=None, help="Model revision if applicable")
    parser.add_argument("--default_tp", type=int, default=8, help="Default TP size (usually GPUs per node)")

    args = parser.parse_args()

    # WHAT: Compute optimal TP size for model and hardware
    # WHY: Ensures tensor parallelism is compatible with model architecture
    tp = get_tensor_parallel_size(args.model_name, args.revision, args.default_tp)

    # WHAT: Print TP size to stdout for shell script capture
    # WHY: Clean output format for programmatic consumption
    print(tp)


"""
==============================================================================
KEY TAKEAWAYS
==============================================================================

1. **PURPOSE**: This script calculates the optimal tensor parallelism size for
   vLLM serving based on model architecture and available GPU count.

2. **CORE ALGORITHM**: Uses GCD (greatest common divisor) to find the largest
   TP size that satisfies both constraints:
   - Divides num_attention_heads evenly (model compatibility)
   - Divides available GPUs evenly (hardware compatibility)

3. **CRITICAL INSIGHT**: Tensor parallelism requires attention heads to be
   split evenly across GPUs. If TP doesn't divide num_heads, the model cannot
   be loaded or will fail during inference.

4. **FALLBACK STRATEGY**: Multiple layers of fault tolerance:
   - If num_heads not in config → use default_tp
   - If config fetch fails → use default_tp
   - If gcd results in 0 → use max(gcd, 1) = 1

5. **TYPICAL SCENARIOS**:
   - Qwen2.5-1.5B (12 heads) + 8 GPUs → TP=4 (gcd(12,8)=4)
   - Qwen2.5-7B (32 heads) + 8 GPUs → TP=8 (32%8==0, use 8)
   - Custom (7 heads) + 8 GPUs → TP=1 (gcd(7,8)=1, no parallelism)

6. **INTEGRATION**: Output is captured by shell scripts and passed to vLLM:
   - Compute TP: TP=$(python get_tensor_parallel_size.py ...)
   - Configure vLLM: --tensor-parallel-size $TP

7. **WHY GCD?**: The GCD is the mathematically optimal solution:
   - Maximizes parallelism (largest possible TP)
   - Guarantees compatibility (divides both inputs evenly)
   - Computationally efficient (Euclidean algorithm)

8. **ARCHITECTURAL DEPENDENCY**: The script assumes:
   - Model uses standard num_attention_heads config key
   - TP splits attention heads (not other tensors)
   - All GPUs have equal memory (homogeneous cluster)

==============================================================================
USAGE EXAMPLE
==============================================================================

Check TP for Qwen2.5-1.5B:

    $ python scripts/get_tensor_parallel_size.py \
        --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
        --default_tp 8

    4

Interpretation: Model has 12 attention heads, gcd(12, 8) = 4, so use TP=4.

Check TP for Qwen2.5-7B:

    $ python scripts/get_tensor_parallel_size.py \
        --model_name "Qwen/Qwen2.5-7B-Instruct" \
        --default_tp 8

    8

Interpretation: Model has 32 attention heads, 32 % 8 == 0, so use TP=8.

Use in shell script:

    #!/bin/bash
    MODEL="Qwen/Qwen2.5-1.5B-Instruct"
    TP=$(python scripts/get_tensor_parallel_size.py --model_name "$MODEL" --default_tp 8)
    echo "Optimal TP size: $TP"

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.9

Output:
    Optimal TP size: 4
    INFO: Started vLLM server with TP=4 on 4 GPUs...

==============================================================================
"""
