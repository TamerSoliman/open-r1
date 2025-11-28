# ==============================================================================
# FILE: src/open_r1/utils/wandb_logging.py
# CATEGORY: Utilities - Weights & Biases Logging
# PRIORITY: LOW
# LINES: 14
# DEPENDENCIES: os
# ==============================================================================
#
# OVERVIEW:
# Simple utility for configuring Weights & Biases (W&B) logging through environment
# variables. Sets up W&B entity, project, and run group for experiment tracking.
#
# KEY FUNCTIONALITY:
# - init_wandb_training(): Configure W&B environment variables from training args
#
# This module ensures consistent W&B configuration across training scripts without
# requiring direct W&B SDK calls.
# ==============================================================================

import os


def init_wandb_training(training_args):
    """
    WHAT: Configure Weights & Biases logging through environment variables

    WHY:
    - W&B SDK reads configuration from environment variables
    - Allows setting W&B config without direct SDK calls
    - Centralizes logging configuration from training arguments
    - Enables consistent experiment tracking across runs

    HOW:
    1. Check if wandb_entity is provided in training_args
    2. Set WANDB_ENTITY environment variable
    3. Repeat for wandb_project and wandb_run_group
    4. W&B SDK will automatically pick up these variables

    PROXIMAL CONTEXT:
    - INPUT: training_args with optional W&B configuration
    - OUTPUT: None (modifies os.environ)

    DISTAL CONTEXT:
    - ORIGIN: Called at training script start (before Trainer initialization)
    - DESTINATION: Used by W&B SDK when Trainer creates WandbCallback

    EXAMPLE 1: Full W&B configuration
    ```python
    training_args = SFTConfig(
        wandb_entity="huggingface",
        wandb_project="open-r1-sft",
        wandb_run_group="llama-7b-experiments"
    )
    init_wandb_training(training_args)
    # Result: All three W&B env vars set
    ```

    EXAMPLE 2: Partial configuration
    ```python
    training_args = SFTConfig(
        wandb_project="open-r1-sft"
    )
    init_wandb_training(training_args)
    # Result: Only WANDB_PROJECT set, entity uses W&B default
    ```

    ENVIRONMENT VARIABLES SET:
    - WANDB_ENTITY: Organization or username (e.g., "huggingface")
    - WANDB_PROJECT: Project name (e.g., "open-r1-sft")
    - WANDB_RUN_GROUP: Group name for organizing runs (e.g., "exp-1")

    W&B INTEGRATION:
    - Transformers Trainer automatically creates WandbCallback
    - WandbCallback reads these environment variables
    - No need to explicitly initialize wandb.init()

    CONFIGURATION HIERARCHY:
    1. Environment variables (set by this function)
    2. wandb.init() arguments (not used in this codebase)
    3. W&B defaults (entity from wandb login)

    RETURNS: None (side effect: modifies os.environ)
    """
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    # SET WANDB ENTITY (organization or username)
    # Only set if provided in training_args
    # WHY: Not all users have an organization, W&B can use default
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity

    # SET WANDB PROJECT (project name)
    # Groups related runs together in W&B UI
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    # SET WANDB RUN GROUP (experiment group)
    # Useful for comparing variants within same project
    # Example: "baseline", "experiment-1", "ablation-study"
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================
# 1. **Environment Variable Configuration**:
#    - W&B SDK reads config from environment variables
#    - No need to call wandb.init() explicitly
#    - Transformers Trainer handles W&B integration automatically
#
# 2. **Optional Configuration**:
#    - All three settings are optional
#    - W&B uses sensible defaults if not provided
#    - Entity defaults to logged-in user
#
# 3. **Run Organization**:
#    - Entity: Organization or personal account
#    - Project: Groups related experiments
#    - Run Group: Groups variants within project
#
# 4. **Integration Pattern**:
#    - Call before Trainer initialization
#    - Trainer automatically creates WandbCallback
#    - Logs metrics, hyperparameters, model checkpoints
#
# 5. **Why Not wandb.init()**:
#    - Trainer handles initialization automatically
#    - Environment variables allow Trainer to control lifecycle
#    - Cleaner separation of concerns
#
# 6. **Common Usage**:
#    ```python
#    init_wandb_training(training_args)
#    trainer = Trainer(...)  # WandbCallback auto-created
#    trainer.train()  # Metrics logged to W&B automatically
#    ```
# ==============================================================================
