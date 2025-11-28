# ==============================================================================
# FILE: src/open_r1/utils/hub.py
# CATEGORY: Utilities - HuggingFace Hub Integration
# PRIORITY: MEDIUM
# LINES: 133
# DEPENDENCIES: huggingface_hub, transformers, trl
# ==============================================================================
#
# OVERVIEW:
# Utilities for interacting with HuggingFace Hub:
# 1. Pushing checkpoints to Hub with branch/revision management
# 2. Checking if revisions exist (avoid overwriting)
# 3. Extracting model parameter counts from repo IDs or metadata
# 4. Computing optimal GPU count for vLLM inference
#
# KEY FUNCTIONS:
# - push_to_hub_revision(): Push checkpoint to specific Hub branch
# - check_hub_revision_exists(): Validate revision doesn't exist
# - get_param_count_from_repo_id(): Extract model size (7b, 70b, etc.)
# - get_gpu_count_for_vllm(): Compute GPU count based on attention heads
#
# ==============================================================================

#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import re
from concurrent.futures import Future

from transformers import AutoConfig

from huggingface_hub import (
    create_branch,
    create_repo,
    get_safetensors_metadata,
    list_repo_commits,
    list_repo_files,
    list_repo_refs,
    repo_exists,
    upload_folder,
)
from trl import GRPOConfig, SFTConfig


logger = logging.getLogger(__name__)


def push_to_hub_revision(training_args: SFTConfig | GRPOConfig, extra_ignore_patterns=[]) -> Future:
    """
    WHAT: Push checkpoint to HuggingFace Hub on a specific branch/revision

    WHY:
    - Track multiple checkpoints separately (not just main branch)
    - Enable versioned checkpoint management
    - Allow rollback to any checkpoint

    HOW:
    1. Create repo if doesn't exist (private by default)
    2. Get initial commit to branch from
    3. Create target branch (or use existing)
    4. Upload checkpoint folder to branch
    5. Return Future for async operation

    PROXIMAL CONTEXT:
    - INPUT: training_args (with hub_model_id, hub_model_revision, output_dir)
    - OUTPUT: Future object (upload operation)

    DISTAL CONTEXT:
    - ORIGIN: Called by PushToHubRevisionCallback during training
    - DESTINATION: HuggingFace Hub (specific branch)

    EXAMPLE:
    ```python
    push_to_hub_revision(
        training_args,  # hub_model_id="user/model", hub_model_revision="exp1-step-5000"
        extra_ignore_patterns=["*.pt"]  # Don't upload optimizer states
    )
    # Result: Checkpoint uploaded to user/model @ exp1-step-5000 branch
    ```

    IGNORE PATTERNS:
    - checkpoint-*: Don't upload sub-checkpoints
    - *.pth: Legacy PyTorch checkpoints
    - extra_ignore_patterns: Additional patterns (e.g., *.pt for optimizer)

    RETURNS: Future object (can add callbacks, check status)
    """
    """Pushes the model to branch on a Hub repo."""

    # CREATE REPO IF NEEDED
    # private=True: Keep checkpoints private by default
    # exist_ok=True: Don't error if repo already exists
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=True, exist_ok=True)

    # GET INITIAL COMMIT TO BRANCH FROM
    # Uses last commit (oldest) as base
    # WHY: Ensures branch starts from repo root
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]

    # CREATE TARGET BRANCH
    # exist_ok=True: Reuse branch if already exists
    # revision=initial_commit.commit_id: Branch from first commit
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info(f"Created target repo at {repo_url}")
    logger.info(f"Pushing to the Hub revision {training_args.hub_model_revision}...")

    # CONFIGURE IGNORE PATTERNS
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    # Example with extra_ignore_patterns=["*.pt"]:
    # Final: ["checkpoint-*", "*.pth", "*.pt"]

    # UPLOAD CHECKPOINT FOLDER
    # run_as_future=True: Async upload (training continues)
    future = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        commit_message=f"Add {training_args.hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    logger.info(f"Pushed to {repo_url} revision {training_args.hub_model_revision} successfully!")

    return future


def check_hub_revision_exists(training_args: SFTConfig | GRPOConfig):
    """
    WHAT: Validate that Hub revision doesn't already exist (avoid overwriting)

    WHY:
    - Prevent accidental overwrite of existing checkpoints
    - Force explicit --overwrite_hub_revision flag for overwrites
    - Safety mechanism for expensive training runs

    HOW:
    1. Check if repo exists
    2. If push_to_hub_revision enabled:
       a. List all branches
       b. If target branch exists:
          - Check if README.md exists (indicates complete checkpoint)
          - If complete and no overwrite flag → raise error

    PROXIMAL CONTEXT:
    - INPUT: training_args (hub_model_id, hub_model_revision, overwrite flag)
    - OUTPUT: None (raises ValueError if conflict)

    DISTAL CONTEXT:
    - ORIGIN: Called at training script start
    - DESTINATION: Prevents training start if revision exists

    RAISES:
    ValueError if revision exists with README and overwrite not enabled

    EXAMPLE ERROR:
    "Revision main-step-005000 already exists. Use --overwrite_hub_revision to overwrite it."
    """
    """Checks if a given Hub revision exists."""
    if repo_exists(training_args.hub_model_id):
        if training_args.push_to_hub_revision is True:
            # FIRST CHECK IF REVISION/BRANCH EXISTS
            revisions = [rev.name for rev in list_repo_refs(training_args.hub_model_id).branches]

            # IF REVISION EXISTS, CHECK IT HAS README
            # README.md indicates complete checkpoint (not partial upload)
            if training_args.hub_model_revision in revisions:
                repo_files = list_repo_files(
                    repo_id=training_args.hub_model_id,
                    revision=training_args.hub_model_revision,
                )
                if "README.md" in repo_files and training_args.overwrite_hub_revision is False:
                    raise ValueError(
                        f"Revision {training_args.hub_model_revision} already exists. "
                        "Use --overwrite_hub_revision to overwrite it."
                    )


def get_param_count_from_repo_id(repo_id: str) -> int:
    """
    WHAT: Extract model parameter count from repo ID or safetensors metadata

    WHY:
    - Determine model size for resource allocation
    - Enable size-based logic (e.g., GPU count, batch size)
    - Parse common naming patterns (7b, 70b, 8x7b, etc.)

    HOW:
    1. Try to get exact count from safetensors metadata (most accurate)
    2. If fails, parse repo ID with regex for patterns:
       - Single numbers: "7b", "1.5b", "42m"
       - Products: "8x7b" (Mixtral), "2x3b"
    3. Convert to actual count (b = billion, m = million)
    4. Return largest number found

    PROXIMAL CONTEXT:
    - INPUT: repo_id (e.g., "meta-llama/Llama-2-7b-hf")
    - OUTPUT: int (parameter count, or -1 if not found)

    PATTERN EXAMPLES:
    - "Llama-2-7b" → 7,000,000,000
    - "Qwen-1.5B" → 1,500,000,000
    - "Mistral-8x7B" → 56,000,000,000 (8 × 7)
    - "gpt2" → -1 (no pattern match, no metadata)

    RETURNS:
    - int: Parameter count
    - -1 if cannot determine
    """
    """Function to get model param counts from safetensors metadata or find patterns like 42m, 1.5b, 0.5m or products like 8x7b in a repo ID."""
    try:
        # TRY TO GET FROM METADATA (MOST ACCURATE)
        metadata = get_safetensors_metadata(repo_id)
        return list(metadata.parameter_count.values())[0]
    except Exception:
        # FALLBACK: PARSE REPO ID FOR PATTERNS

        # Pattern to match products (like 8x7b) and single values (like 42m)
        # Examples:
        #   7b → group: (7, b)
        #   1.5b → group: (1.5, b)
        #   8x7b → group: (8, x, 7, b)
        pattern = r"((\d+(\.\d+)?)(x(\d+(\.\d+)?))?)([bm])"
        matches = re.findall(pattern, repo_id.lower())

        param_counts = []
        for full_match, number1, _, _, number2, _, unit in matches:
            if number2:  # If there's a second number, it's a product
                number = float(number1) * float(number2)
                # Example: 8x7b → 8 * 7 = 56
            else:  # Otherwise, it's a single value
                number = float(number1)
                # Example: 7b → 7

            # CONVERT TO ACTUAL COUNT
            if unit == "b":
                number *= 1_000_000_000  # Convert to billion
            elif unit == "m":
                number *= 1_000_000  # Convert to million

            param_counts.append(number)

        if len(param_counts) > 0:
            # Return the largest number
            # WHY: If multiple matches, assume largest is model size
            # Example: "Llama-2-7b-chat-4k" → returns 7b (not 2 or 4k)
            return int(max(param_counts))
        else:
            # Return -1 if no match found
            return -1


def get_gpu_count_for_vllm(model_name: str, revision: str = "main", num_gpus: int = 8) -> int:
    """
    WHAT: Calculate optimal GPU count for vLLM inference based on model architecture

    WHY:
    - vLLM enforces constraints:
      1. num_attention_heads must be divisible by num_gpus
      2. 64 must be divisible by num_gpus (tensor parallelism requirement)
    - Using invalid GPU count causes vLLM to fail

    HOW:
    1. Load model config (get num_attention_heads)
    2. Starting from num_gpus, decrement until constraints satisfied
    3. Return valid GPU count

    PROXIMAL CONTEXT:
    - INPUT: model_name, revision, desired num_gpus
    - OUTPUT: Adjusted num_gpus (valid for vLLM)

    DISTAL CONTEXT:
    - ORIGIN: Called before starting vLLM server
    - DESTINATION: Used in vLLM launch command (--tensor-parallel-size)

    EXAMPLE 1:
    ```python
    get_gpu_count_for_vllm("Qwen/Qwen-7B", num_gpus=8)
    # Config has 32 attention heads
    # 32 % 8 = 0 ✓
    # 64 % 8 = 0 ✓
    # Returns: 8
    ```

    EXAMPLE 2:
    ```python
    get_gpu_count_for_vllm("some-model-with-40-heads", num_gpus=8)
    # Config has 40 attention heads
    # 40 % 8 = 0 ✓
    # 64 % 8 = 0 ✓
    # Returns: 8
    ```

    EXAMPLE 3:
    ```python
    get_gpu_count_for_vllm("model-with-36-heads", num_gpus=8)
    # Config has 36 attention heads
    # 36 % 8 = 4 ✗ (not divisible)
    # Try 7: 36 % 7 = 1 ✗
    # Try 6: 36 % 6 = 0 ✓, 64 % 6 = 4 ✗
    # Try 5: 36 % 5 = 1 ✗
    # Try 4: 36 % 4 = 0 ✓, 64 % 4 = 0 ✓
    # Returns: 4
    ```

    CONSTRAINTS:
    - num_heads % num_gpus == 0
    - 64 % num_gpus == 0
    - Both must be satisfied
    """
    """vLLM enforces a constraint that the number of attention heads must be divisible by the number of GPUs and 64 must be divisible by the number of GPUs.
    This function calculates the number of GPUs to use for decoding based on the number of attention heads in the model.
    """
    config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)

    # Get number of attention heads
    num_heads = config.num_attention_heads

    # Reduce num_gpus so that num_heads is divisible by num_gpus and 64 is divisible by num_gpus
    while num_heads % num_gpus != 0 or 64 % num_gpus != 0:
        logger.info(f"Reducing num_gpus from {num_gpus} to {num_gpus - 1} to make num_heads divisible by num_gpus")
        num_gpus -= 1

    return num_gpus

# ==============================================================================
# KEY TAKEAWAYS - HUB UTILITIES
# ==============================================================================
#
# 1. **push_to_hub_revision()**:
#    - Async upload (returns Future)
#    - Creates branches automatically
#    - Ignores optimizer states by default
#    - Private repos by default
#
# 2. **check_hub_revision_exists()**:
#    - Safety check before training
#    - Prevents accidental overwrites
#    - Requires --overwrite_hub_revision flag
#
# 3. **get_param_count_from_repo_id()**:
#    - Try metadata first (accurate)
#    - Fallback to regex parsing
#    - Handles products (8x7b)
#    - Returns -1 if unknown
#
# 4. **get_gpu_count_for_vllm()**:
#    - Enforces vLLM constraints
#    - Decrements from desired GPU count
#    - Returns valid count or smaller number
#    - Critical for vLLM to start successfully
#
# 5. **Common Patterns**:
#    - exist_ok=True: Idempotent operations
#    - run_as_future=True: Async for long operations
#    - Logging at all key steps
#    - Private by default (security)
#
# ==============================================================================
