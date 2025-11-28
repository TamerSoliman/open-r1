# ==============================================================================
# FILE: src/open_r1/utils/callbacks.py
# CATEGORY: Utilities - Training Callbacks
# PRIORITY: MEDIUM
# LINES: 93
# DEPENDENCIES: transformers, subprocess
# ==============================================================================
#
# OVERVIEW:
# Training callbacks for automating checkpoint management during training.
# Primary function is pushing checkpoints to HuggingFace Hub with revision
# tracking and optional benchmark evaluation on Slurm clusters.
#
# KEY FUNCTIONALITY:
# 1. Automatic checkpoint pushing to Hub with revision naming
# 2. Optional benchmark job submission when Slurm is available
# 3. Callback registration system for extensibility
#
# MAIN CALLBACK: PushToHubRevisionCallback
# - Triggers on every checkpoint save
# - Creates unique revision per checkpoint (step-000001234)
# - Excludes optimizer states (.pt files) to save space
# - Optionally launches benchmark jobs after push completes
#
# WHY THIS EXISTS:
# - Automates checkpoint versioning (manual versioning error-prone)
# - Enables checkpoint-specific benchmarks for tracking training progress
# - Provides async push (training continues while uploading)
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

import subprocess
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision


def is_slurm_available() -> bool:
    """
    WHAT: Check if Slurm workload manager is available

    WHY: Benchmarks are submitted as Slurm jobs, so we only enable
         benchmark callback if Slurm is present

    HOW: Run 'sinfo' command and check if it succeeds

    RETURNS:
    - True if Slurm is available
    - False otherwise
    """
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    """
    WHAT: Lightweight config object that doesn't break Accelerator state

    WHY: Using dataclasses.replace(args, ...) or instantiating new SFTConfig
         breaks the accelerator distributed state. This workaround creates
         a simple object with needed attributes without affecting accelerator.

    HOW: Takes kwargs and sets them as attributes

    WORKAROUND FOR: Accelerator state preservation during callback
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    """
    WHAT: Callback that pushes checkpoints to HuggingFace Hub with unique revisions

    WHY:
    - Track each checkpoint separately (not just "main" branch)
    - Enable checkpoint-specific benchmarks
    - Preserve training history on Hub

    HOW:
    1. Triggered on_save (every checkpoint)
    2. Creates revision name: "{base_revision}-step-{global_step:09d}"
    3. Pushes checkpoint to Hub asynchronously
    4. Optionally submits benchmark job when push completes (if Slurm available)

    EXAMPLE REVISION NAMING:
    - Base revision: "experiment-1"
    - Step 1234: "experiment-1-step-000001234"
    - Step 5000: "experiment-1-step-000005000"

    PROXIMAL CONTEXT:
    - INPUT: Checkpoint saved by Trainer
    - OUTPUT: Checkpoint on Hub + optional benchmark job

    DISTAL CONTEXT:
    - ORIGIN: Called by Trainer during training loop
    - DESTINATION: HuggingFace Hub (versioned checkpoints) + Slurm queue (benchmarks)
    """
    def __init__(self, model_config) -> None:
        self.model_config = model_config

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        WHAT: Triggered when Trainer saves a checkpoint

        WHY: Automate checkpoint pushing and benchmark launching

        HOW:
        1. Check if main process (avoid duplicate pushes)
        2. Create dummy config with checkpoint-specific revision
        3. Push to Hub (async, returns future)
        4. If Slurm available, add benchmark callback to future

        IMPORTANT: Only runs on rank 0 (is_world_process_zero)
        """
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken, so I do this workaround
            # Also if you instantiate a new SFTConfig, the accelerator dist state will be broken

            # CREATE CHECKPOINT-SPECIFIC CONFIG
            # Revision format: base-step-000001234
            # Example: "main-step-000005000" for checkpoint at step 5000
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",  # 9-digit zero-padded
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            # PUSH TO HUB (ASYNC)
            # extra_ignore_patterns=["*.pt"]: Don't push optimizer states
            # WHY: Optimizer states are large (2× model size) and not needed for inference
            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )  # don't push the optimizer states

            # OPTIONAL: LAUNCH BENCHMARK JOB AFTER PUSH COMPLETES
            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    """
                    WHAT: Callback that runs when push completes

                    WHY: Submit benchmarks only after checkpoint is on Hub
                         (benchmarks need to download checkpoint)

                    HOW: Submit Slurm job via run_benchmark_jobs()
                    """
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                # Add callback to future (runs when push finishes)
                future.add_done_callback(run_benchmark_callback)


# ==============================================================================
# CALLBACK REGISTRY
# ==============================================================================

CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
}
"""
WHAT: Registry of available callbacks

WHY: Enable configuration-based callback selection
     (e.g., callbacks: ["push_to_hub_revision"] in config)

HOW TO EXTEND:
Add new callback class to CALLBACKS dict:
    CALLBACKS["my_callback"] = MyCallbackClass
"""


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    """
    WHAT: Initialize callbacks from config

    WHY: Allow configuration-driven callback selection

    HOW:
    1. Iterate through callback names in config
    2. Look up callback class in CALLBACKS registry
    3. Instantiate with model_config
    4. Return list of callback instances

    PROXIMAL CONTEXT:
    - INPUT: train_config (with callbacks list), model_config
    - OUTPUT: List of instantiated TrainerCallback objects

    DISTAL CONTEXT:
    - ORIGIN: Called by training script setup
    - DESTINATION: Passed to Trainer(..., callbacks=callbacks)

    EXAMPLE CONFIG:
    ```yaml
    callbacks:
      - push_to_hub_revision
    ```

    RAISES:
    ValueError if callback_name not found in CALLBACKS registry
    """
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks

# ==============================================================================
# KEY TAKEAWAYS - CALLBACKS
# ==============================================================================
#
# 1. **PushToHubRevisionCallback**:
#    - Automates checkpoint pushing with unique revisions
#    - Format: "{base}-step-{step:09d}"
#    - Excludes optimizer states to save bandwidth
#
# 2. **Benchmark Integration**:
#    - Optionally launches benchmarks after push completes
#    - Only if Slurm is available (checked via sinfo command)
#    - Uses async callback on push future
#
# 3. **DummyConfig Workaround**:
#    - Needed to avoid breaking Accelerator distributed state
#    - Don't use dataclasses.replace() or instantiate new configs in callbacks
#
# 4. **World Process Zero**:
#    - Callbacks only run on rank 0 (is_world_process_zero)
#    - Prevents duplicate pushes in distributed training
#
# 5. **Extensibility**:
#    - Add new callbacks to CALLBACKS registry
#    - Reference by name in config.callbacks list
#
# ==============================================================================
