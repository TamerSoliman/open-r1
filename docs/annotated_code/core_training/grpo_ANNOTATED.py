"""
==============================================================================
FILE: src/open_r1/grpo.py
CATEGORY: Core Training - GRPO Training Script
PRIORITY: CRITICAL
DEPENDENCIES:
    - open_r1.configs: GRPOConfig, GRPOScriptArguments (configuration dataclasses)
    - open_r1.rewards: get_reward_funcs (reward function registry)
    - open_r1.utils: get_dataset, get_model, get_tokenizer (data and model loading)
    - trl: GRPOTrainer (GRPO training implementation)
    - transformers: Standard HuggingFace utilities
==============================================================================

==============================================================================
OVERVIEW
==============================================================================

This script implements the Group Relative Policy Optimization (GRPO) training
pipeline for DeepSeek R1 reasoning models. GRPO is the **core innovation** that
enables reinforcement learning-based fine-tuning of language models for
reasoning tasks.

ROLE IN DEEPSEEK R1:
-------------------
GRPO represents **Stage 2** of the DeepSeek R1 three-stage training pipeline:

    Stage 1 (Distillation): sft.py - Learn reasoning from DeepSeek-R1
    Stage 2 (GRPO): grpo.py ← THIS FILE - Reinforce via RL on verifiable tasks
    Stage 3 (Combined): Multi-stage training combining distillation + GRPO

The GRPO approach is detailed in the DeepSeek R1 paper and represents a
significant improvement over standard RLHF/PPO methods for reasoning tasks.

KEY INNOVATIONS:
----------------
1. **Group Relative Advantages**:
   - Generates multiple completions per prompt (typically 16)
   - Computes advantages relative to group mean, not absolute rewards
   - Reduces variance in gradient estimates
   - More stable than PPO for long-form reasoning

2. **Multi-Objective Reward Functions**:
   - Combines accuracy (math correctness), format (structured reasoning),
     code execution, and quality metrics
   - Weighted reward composition prevents reward hacking
   - Enables learning across diverse objectives

3. **vLLM Integration for Generation**:
   - Uses vLLM server for efficient generation during training
   - Enables high-throughput sampling (16 completions per prompt)
   - Critical for GRPO's group-based advantage calculation

4. **Structured Reasoning Enforcement**:
   - System prompts and format rewards enforce <think>/<answer> structure
   - Separates internal reasoning from final response
   - Improves interpretability and performance

DATA FLOW:
----------
This script orchestrates the complete GRPO training pipeline:

    DISTAL ORIGIN (where data comes from):
    ├─> HuggingFace Hub → Dataset (e.g., OpenR1-Math-220k)
    ├─> HuggingFace Hub → Base Model (e.g., Qwen2.5-1.5B-Instruct)
    └─> Config YAML → Hyperparameters, reward functions, system prompts

    PROXIMAL PROCESSING (this script):
    1. Load dataset and format as conversations (user prompts)
    2. Load model and tokenizer with specified settings
    3. Initialize GRPOTrainer with:
       - Model (policy to optimize)
       - Reward functions (what to optimize for)
       - Dataset (prompts to generate from)
    4. Training loop:
       a. GRPOTrainer samples prompts from dataset
       b. vLLM generates multiple completions per prompt
       c. Reward functions compute scalar rewards for each completion
       d. Group relative advantages calculated (rewards - group mean)
       e. Policy gradients computed and applied
       f. Repeat until convergence
    5. Save final model and metrics

    DISTAL DESTINATION (where results go):
    ├─> Local filesystem: data/model_name/ (checkpoints)
    ├─> HuggingFace Hub: hub_model_id (for sharing)
    ├─> Weights & Biases: Training metrics, reward distributions
    └─> Evaluation benchmarks: Triggered via callbacks

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

# WHAT: Standard library imports for logging, file operations, and system interaction
import logging
import os
import sys

# WHAT: HuggingFace datasets and transformers imports
# WHY: Core infrastructure for dataset loading, model loading, and reproducibility
import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

# WHAT: Open R1 custom imports
# WHY: Configuration system, reward functions, and utility functions specific to this project
from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs  # Reward function registry
from open_r1.utils import get_dataset, get_model, get_tokenizer  # Helper functions
from open_r1.utils.callbacks import get_callbacks  # Training callbacks (push to hub, eval)
from open_r1.utils.wandb_logging import init_wandb_training  # W&B initialization

# WHAT: TRL (Transformer Reinforcement Learning) imports
# WHY: GRPOTrainer is the core training class that implements GRPO algorithm
#      TrlParser handles parsing of YAML configs into dataclass instances
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


"""
==============================================================================
MAIN TRAINING FUNCTION
==============================================================================
"""


def main(script_args, training_args, model_args):
    """
    WHAT: Main entry point for GRPO training that orchestrates the complete pipeline

    WHY: Encapsulates all training logic in a single function for clarity and
         enables easy testing/debugging by separating config parsing from execution

    HOW:
        1. Set up logging and reproducibility (seed)
        2. Load dataset, model, and tokenizer
        3. Format dataset as conversations with system prompts
        4. Initialize GRPOTrainer with reward functions
        5. Run training loop with checkpointing
        6. Save model, metrics, and push to hub
        7. Optionally run evaluation

    PROXIMAL CONTEXT:
        - Input: Parsed configuration objects (script_args, training_args, model_args)
        - Output: Trained model saved to disk/hub, training metrics logged

    DISTAL CONTEXT:
        - Originates from: CLI arguments → YAML config → TrlParser → dataclass instances
        - Flows to:
            * Saved model → HuggingFace Hub → Inference/evaluation
            * Metrics → W&B → Experiment tracking
            * Callbacks → SLURM jobs → Automated benchmarking

    Args:
        script_args (GRPOScriptArguments): Dataset and reward function configuration
        training_args (GRPOConfig): Training hyperparameters (learning rate, batch size, etc.)
        model_args (ModelConfig): Model loading configuration (quantization, attention, etc.)

    Returns:
        None (side effects: saves model, logs metrics)
    """

    # ==========================================================================
    # STEP 1: REPRODUCIBILITY AND LOGGING SETUP
    # ==========================================================================
    # WHAT: Set random seed for reproducibility across runs
    # WHY: Critical for scientific reproducibility and debugging
    # HOW: Sets seeds for Python random, NumPy, PyTorch, and CUDA
    set_seed(training_args.seed)

    # WHAT: Configure logging with consistent format across distributed processes
    # WHY: Essential for debugging distributed training and tracking training progress
    # HOW: Sets up StreamHandler to stdout with timestamp and log level formatting
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # WHAT: Set log levels for all relevant libraries
    # WHY: Ensures consistent verbosity across datasets, transformers, and our code
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # WHAT: Log training configuration on each process
    # WHY: Critical for debugging distributed training (shows which process is which)
    # DATA FLOW: Logs device assignment, GPU count, distributed training status
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # ==========================================================================
    # STEP 2: CHECKPOINT DETECTION AND RESUMPTION
    # ==========================================================================
    # WHAT: Check if a previous checkpoint exists for resuming training
    # WHY: Enables fault tolerance in long training runs (hours/days)
    # HOW: Scans output directory for checkpoint-{step} folders
    # DATA FLOW: output_dir → checkpoint detection → resume_from_checkpoint flag
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # ==========================================================================
    # STEP 3: EXPERIMENT TRACKING INITIALIZATION
    # ==========================================================================
    # WHAT: Initialize Weights & Biases for experiment tracking
    # WHY: Enables tracking of reward distributions, learning curves, hyperparameters
    # HOW: Sets environment variables for W&B entity/project/run group
    # DATA FLOW: training_args → W&B env vars → W&B dashboard
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # ==========================================================================
    # STEP 4: DATASET LOADING AND FORMATTING
    # ==========================================================================
    # WHAT: Load dataset from HuggingFace Hub or create dataset mixture
    # WHY: Provides prompts for GRPO training (questions to generate answers for)
    # HOW: get_dataset handles both single datasets and weighted mixtures
    # DATA FLOW: HF Hub / local files → DatasetDict with train/test splits
    dataset = get_dataset(script_args)

    # Example datasets for GRPO:
    # - open-r1/OpenR1-Math-220k: Math problems with verifiable answers
    # - open-r1/codeforces: Coding problems with test cases
    # - Custom mixtures combining multiple task types

    # ==========================================================================
    # STEP 5: TOKENIZER LOADING
    # ==========================================================================
    # WHAT: Load tokenizer matching the model
    # WHY: Required for text → token conversion and chat template formatting
    # HOW: AutoTokenizer.from_pretrained with custom chat template override
    # DATA FLOW: Model name → HF Hub → Tokenizer with chat template
    tokenizer = get_tokenizer(model_args, training_args)

    # NOTE: Chat template is critical for formatting system prompts and user messages
    # into the correct token format for the model (e.g., ChatML, Llama format, etc.)

    # ==========================================================================
    # STEP 6: MODEL LOADING
    # ==========================================================================
    # WHAT: Load the base model (policy) to be optimized
    # WHY: This is the model we'll improve via GRPO reinforcement learning
    # HOW: Handles quantization, attention implementation, device mapping
    # DATA FLOW: Model name → HF Hub → Model loaded to GPU(s)
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # KEY POINT: This model starts from a pre-trained checkpoint (often SFT'd)
    # and will be improved via GRPO to better solve reasoning/coding tasks

    # ==========================================================================
    # STEP 7: REWARD FUNCTIONS INITIALIZATION
    # ==========================================================================
    # WHAT: Load reward functions from the registry based on config
    # WHY: Reward functions define what behaviors we want to reinforce
    # HOW: Parses script_args.reward_funcs list and instantiates each function
    # DATA FLOW: Config → Reward function names → Instantiated callables
    reward_funcs = get_reward_funcs(script_args)

    # CRITICAL CONCEPT: Reward functions are the "objective" in RL
    # Examples:
    # - accuracy_reward: Checks if math answer matches ground truth
    # - format_reward: Ensures <think>...</think><answer>...</answer> structure
    # - code_reward: Executes code and checks test cases
    #
    # Multiple rewards are weighted and summed to create composite objective
    # This prevents "reward hacking" where model exploits a single metric

    # ==========================================================================
    # STEP 8: DATASET FORMATTING FOR GRPO
    # ==========================================================================
    # WHAT: Convert dataset rows into chat-formatted conversations
    # WHY: GRPOTrainer expects prompts in chat format (list of dicts with role/content)
    # HOW: Map function adds system prompt and user message to each example
    # DATA FLOW: Raw dataset → Formatted conversations → GRPO-ready prompts

    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        """
        WHAT: Formats a dataset example as a conversation with system and user messages

        WHY: GRPO operates on conversations, not raw text. This enables:
             - System prompts to guide reasoning format
             - Multi-turn conversations for complex tasks
             - Consistent formatting across different tasks

        HOW:
            1. Start with empty prompt list
            2. Add system message if specified (guides reasoning format)
            3. Add user message from dataset
            4. Return as {"prompt": [{"role": "system"/"user", "content": "..."}]}

        PROXIMAL CONTEXT:
            - Input: Dataset example (dict with various fields)
            - Output: {"prompt": conversation_list}

        DISTAL CONTEXT:
            - Originates from: Dataset row in HF Hub
            - Flows to: GRPOTrainer → vLLM generation → Reward computation

        Args:
            example: Single row from dataset (dict)
            prompt_column: Which field contains the user prompt (default: "prompt")

        Returns:
            Dict with "prompt" key containing conversation list
        """
        prompt = []

        # WHAT: Add system prompt if provided in config
        # WHY: System prompts are critical for enforcing reasoning format
        # EXAMPLE: "Respond in format: <think>...</think><answer>...</answer>"
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        # WHAT: Validate that prompt column exists in dataset
        # WHY: Fail fast with clear error message if misconfigured
        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        # WHAT: Add user message with the actual question/prompt
        # WHY: This is what the model will generate a response to
        prompt.append({"role": "user", "content": example[prompt_column]})

        return {"prompt": prompt}

    # WHAT: Apply conversation formatting to entire dataset
    # HOW: dataset.map applies function to each row
    # RESULT: Dataset now has "prompt" column with chat-formatted conversations
    dataset = dataset.map(make_conversation)

    # WHAT: Remove "messages" column if present (cleanup)
    # WHY: Some datasets have pre-formatted messages that conflict with our format
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # ==========================================================================
    # STEP 9: GRPO TRAINER INITIALIZATION
    # ==========================================================================
    # WHAT: Initialize the GRPOTrainer with all components
    # WHY: GRPOTrainer encapsulates the GRPO algorithm implementation
    # HOW: Combines model, rewards, dataset, and config into training loop
    #
    # KEY CONCEPT: GRPOTrainer handles:
    #   1. Prompt sampling from dataset
    #   2. Generation of multiple completions via vLLM
    #   3. Reward computation for each completion
    #   4. Group relative advantage calculation
    #   5. Policy gradient computation and application
    #   6. Logging, checkpointing, and callbacks

    trainer = GRPOTrainer(
        # WHAT: Model to optimize (the policy in RL terminology)
        # DATA FLOW: Pre-trained model → GRPO updates → Optimized model
        model=model,

        # WHAT: List of reward functions to evaluate completions
        # DATA FLOW: Completions → Rewards → Advantages → Policy gradients
        reward_funcs=reward_funcs,

        # WHAT: Training configuration (learning rate, batch size, etc.)
        args=training_args,

        # WHAT: Training dataset (prompts to generate from)
        # NOTE: GRPO doesn't use ground truth completions, only prompts + rewards
        train_dataset=dataset[script_args.dataset_train_split],

        # WHAT: Evaluation dataset (optional, for periodic evaluation)
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),

        # WHAT: PEFT config for LoRA/QLoRA (parameter-efficient fine-tuning)
        # WHY: Enables training large models by only updating small adapter layers
        peft_config=get_peft_config(model_args),

        # WHAT: Callbacks for push to hub, automated benchmarking, etc.
        # DATA FLOW: Training events → Callbacks → Hub pushes, eval jobs
        callbacks=get_callbacks(training_args, model_args),

        # WHAT: Processing class (tokenizer) for text ↔ tokens conversion
        processing_class=tokenizer,
    )

    # ==========================================================================
    # STEP 10: TRAINING LOOP
    # ==========================================================================
    # WHAT: Execute the GRPO training loop
    # WHY: This is where the actual learning happens
    # HOW: GRPOTrainer.train() implements the full GRPO algorithm:
    #
    # GRPO ALGORITHM (simplified):
    # For each training step:
    #   1. Sample batch of prompts from dataset
    #   2. For each prompt, generate N completions (typically 16)
    #      using vLLM server with specified temperature/top_p
    #   3. For each completion, compute rewards using reward_funcs
    #      rewards = weighted_sum([r(completion) for r in reward_funcs])
    #   4. For each prompt's completions, compute group relative advantages:
    #      advantages = rewards - mean(rewards_for_this_prompt)
    #      This is the key innovation: comparing within group, not global
    #   5. Compute policy gradient loss:
    #      loss = -mean(log_prob(completion) * advantage)
    #      Higher advantage → increase probability
    #      Lower advantage → decrease probability
    #   6. Backpropagate and update model parameters
    #   7. Log metrics (rewards, advantages, loss) to W&B
    #   8. Periodically save checkpoints and run callbacks
    #
    # DATA FLOW:
    #   Prompts → vLLM generation → Completions → Reward functions → Rewards
    #   → Advantages → Policy gradients → Parameter updates → Improved model

    logger.info("*** Train ***")

    # WHAT: Determine which checkpoint to resume from (if any)
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # WHAT: Run training loop
    # RETURNS: TrainOutput with metrics (loss, rewards, etc.)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # WHAT: Extract and log training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()  # Save optimizer state for resumption

    # ==========================================================================
    # STEP 11: MODEL SAVING AND HUB PUSH
    # ==========================================================================
    # WHAT: Save the trained model to disk and optionally push to HF Hub
    # WHY: Preserves trained model for inference, evaluation, and sharing
    # HOW: Saves model weights, config, tokenizer, and generation config
    # DATA FLOW: Trained model → Local filesystem → (optionally) HF Hub

    logger.info("*** Save model ***")

    # WHAT: Align model's generation config with tokenizer
    # WHY: Prevents unbounded generation in transformers pipeline
    # HOW: Ensures model knows when to stop generating (at EOS token)
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id

    # WHAT: Save model to output directory
    # SAVES: model.safetensors, config.json, generation_config.json
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # WHAT: Create model card and save additional artifacts (main process only)
    # WHY: Model card provides metadata for HF Hub (dataset, metrics, etc.)
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],  # Tags for discoverability on HF Hub
    }
    if trainer.accelerator.is_main_process:
        # WHAT: Create README.md with training info
        trainer.create_model_card(**kwargs)

        # WHAT: Re-enable KV cache for fast inference
        # WHY: Cache was disabled during training to save memory
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # ==========================================================================
    # STEP 12: EVALUATION (OPTIONAL)
    # ==========================================================================
    # WHAT: Run evaluation on test set if configured
    # WHY: Measures performance on held-out data
    # HOW: Generates completions and computes reward metrics
    # DATA FLOW: Test prompts → Generation → Rewards → Eval metrics → Logs

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ==========================================================================
    # STEP 13: HUB PUSH (OPTIONAL)
    # ==========================================================================
    # WHAT: Push final model to HuggingFace Hub
    # WHY: Enables sharing with community and versioning
    # HOW: Uploads all model files to hub_model_id repository
    # DATA FLOW: Local model → HF Hub → Public/private repository

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


"""
==============================================================================
SCRIPT ENTRY POINT
==============================================================================
"""

if __name__ == "__main__":
    """
    WHAT: Entry point when script is run directly from command line

    WHY: Separates config parsing from execution for clarity

    HOW:
        1. TrlParser parses CLI args and YAML config files
        2. Instantiates dataclass objects for each config type
        3. Calls main() with parsed configs

    USAGE EXAMPLE:
        accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml \
            src/open_r1/grpo.py \
            --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml

    DATA FLOW:
        Command line → YAML file → TrlParser → Dataclass instances → main()
    """

    # WHAT: Create parser for three config types
    # WHY: Separates concerns: dataset config, training config, model config
    # HOW: TrlParser handles merging of CLI args + YAML configs
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))

    # WHAT: Parse arguments from CLI and config files
    # RETURNS: Three dataclass instances with all configuration
    script_args, training_args, model_args = parser.parse_args_and_config()

    # WHAT: Execute main training function
    # DATA FLOW: Parsed configs → Training loop → Saved model
    main(script_args, training_args, model_args)

"""
==============================================================================
KEY TAKEAWAYS FOR UNDERSTANDING GRPO
==============================================================================

1. **Group Relative Advantages are the Core Innovation**:
   - Instead of absolute rewards, GRPO uses rewards relative to group mean
   - This dramatically reduces variance in policy gradients
   - Enables stable training on long-form reasoning tasks

2. **Multi-Objective Rewards Prevent Reward Hacking**:
   - Accuracy + Format + Code + Quality metrics combined
   - Model can't exploit single metric
   - Learns robust reasoning, not shortcuts

3. **vLLM is Critical for Throughput**:
   - GRPO needs many generations per prompt (16+)
   - vLLM's PagedAttention enables efficient batch generation
   - Without vLLM, training would be prohibitively slow

4. **System Prompts Enforce Reasoning Structure**:
   - <think>...</think><answer>...</answer> format
   - Separates internal reasoning from final response
   - Improves interpretability and debugging

5. **GRPO is Stage 2 of Multi-Stage Pipeline**:
   - Stage 1 (SFT): Learn reasoning format from strong model
   - Stage 2 (GRPO): Improve via RL on verifiable tasks
   - Stage 3 (Combined): Further improve with multi-stage approach

6. **Distributed Training via Accelerate**:
   - Supports DDP, DeepSpeed ZeRO-2/3, FSDP
   - Enables training on models up to 32B+ parameters
   - Critical for scaling GRPO to large models

7. **Production-Grade Infrastructure**:
   - Automatic checkpointing and resumption
   - W&B integration for experiment tracking
   - Callbacks for automated Hub pushes and benchmarking
   - SLURM integration for HF cluster deployment
==============================================================================
"""
