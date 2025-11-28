"""
==============================================================================
FILE: src/open_r1/sft.py
CATEGORY: Core Training - Supervised Fine-Tuning Script
PRIORITY: CRITICAL
DEPENDENCIES:
    - open_r1.configs: ScriptArguments, SFTConfig (configuration dataclasses)
    - open_r1.utils: get_dataset, get_model, get_tokenizer (utilities)
    - trl: SFTTrainer, setup_chat_format (supervised fine-tuning)
    - transformers: Standard HuggingFace utilities
==============================================================================

==============================================================================
OVERVIEW
==============================================================================

This script implements the Supervised Fine-Tuning (SFT) pipeline for DeepSeek R1
reasoning models. SFT is the **first stage** of the three-stage training process,
also known as the "distillation" stage.

ROLE IN DEEPSEEK R1:
-------------------
SFT represents **Stage 1** of the DeepSeek R1 three-stage training pipeline:

    Stage 1 (Distillation): sft.py ← THIS FILE - Learn reasoning from DeepSeek-R1
    Stage 2 (GRPO): grpo.py - Reinforce via RL on verifiable tasks
    Stage 3 (Combined): Multi-stage training combining distillation + GRPO

**Key Concept: Knowledge Distillation**
- Take a strong reasoning model (DeepSeek-R1 1.5T parameters)
- Generate reasoning traces on many problems (Mixture-of-Thoughts dataset)
- Train smaller model (Qwen 7B) to imitate these reasoning traces
- Result: Smaller model learns reasoning format and approach

WHY DISTILLATION WORKS:
- Strong model's reasoning traces act as "supervision signal"
- Teaches both the reasoning format (<think>/<answer>) and reasoning strategy
- Much more data-efficient than training from scratch
- Provides good initialization for Stage 2 (GRPO)

KEY INNOVATIONS:
----------------
1. **Long Context Support (32K+ tokens)**:
   - Uses Qwen2.5-Math-7B-RoPE-300k base (extended to 300K context)
   - Enables learning from long reasoning chains
   - Critical for complex multi-step problems

2. **Mixture-of-Thoughts Dataset**:
   - 350K problems with reasoning traces from DeepSeek-R1
   - Covers math, code, and reasoning tasks
   - Pre-formatted with <think>/<answer> structure

3. **Custom Chat Template**:
   - Enforces structured reasoning format
   - System prompt guides model behavior
   - Ensures consistency across training

4. **Memory-Efficient Training**:
   - Gradient checkpointing for large context windows
   - Liger kernel for optimized operations
   - DeepSpeed ZeRO-3 for model sharding across GPUs

5. **Chat Format Setup**:
   - Automatic ChatML format setup if no template exists
   - Handles special tokens (im_start, im_end)
   - Ensures model learns conversational format

DATA FLOW:
----------
This script orchestrates the complete SFT pipeline:

    DISTAL ORIGIN (where data comes from):
    ├─> HuggingFace Hub → Mixture-of-Thoughts dataset (reasoning traces)
    ├─> HuggingFace Hub → Base Model (Qwen2.5-Math-7B-RoPE-300k)
    └─> Config YAML → Hyperparameters, chat templates, system prompts

    PROXIMAL PROCESSING (this script):
    1. Load dataset with reasoning traces (prompt + completion pairs)
    2. Load model and tokenizer with extended context support
    3. Set up chat format (ChatML) if not present
    4. Initialize SFTTrainer with:
       - Model (to be fine-tuned)
       - Dataset (reasoning demonstrations)
       - Training config (batch size, learning rate, etc.)
    5. Training loop:
       a. Sample batch of (prompt, completion) pairs
       b. Tokenize conversations
       c. Compute cross-entropy loss (maximize likelihood of completions)
       d. Backpropagate and update parameters
       e. Repeat until convergence
    6. Save fine-tuned model and metrics

    DISTAL DESTINATION (where results go):
    ├─> Local filesystem: data/model_name/ (checkpoints)
    ├─> HuggingFace Hub: hub_model_id (for sharing/Stage 2)
    ├─> Weights & Biases: Training loss, perplexity metrics
    └─> Evaluation benchmarks: Triggered via callbacks

COMPARISON WITH GRPO (Stage 2):
-------------------------------
| Aspect               | SFT (Stage 1)              | GRPO (Stage 2)            |
|----------------------|----------------------------|---------------------------|
| Learning paradigm    | Supervised learning        | Reinforcement learning    |
| Training signal      | Ground truth completions   | Reward functions          |
| Data requirement     | Demonstration data         | Prompts + verification    |
| What it learns       | Reasoning format/strategy  | Task performance          |
| Output quality       | Matches demonstrations     | Optimized for rewards     |
| Typical duration     | 5 epochs (~1-2 days)       | 1-3 epochs (~hours)       |

==============================================================================
IMPORTS AND SETUP
==============================================================================
"""

# Copyright 2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# [License text omitted for brevity]

"""
MODULE DOCSTRING WITH USAGE EXAMPLE
This is prominently displayed to help users understand how to run this script.
"""

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \\
    --model_name_or_path open-r1/Qwen2.5-Math-7B-RoPE-300k \\
    --dataset_name open-r1/Mixture-of-Thoughts \\
    --dataset_config all \\
    --eos_token '<|im_end|>' \\
    --learning_rate 4.0e-5 \\
    --num_train_epochs 5 \\
    --max_seq_length 32768 \\
    --per_device_train_batch_size 2 \\
    --gradient_checkpointing \\
    --bf16 \\
    --use_liger_kernel \\
    --output_dir data/OpenR1-Distill-7B
"""

# WHAT: Standard library imports
import logging
import os
import sys

# WHAT: HuggingFace core libraries
import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

# WHAT: Open R1 custom modules
from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training

# WHAT: TRL for supervised fine-tuning
# WHY: SFTTrainer provides optimized SFT with packing, chat templates, etc.
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format


logger = logging.getLogger(__name__)


"""
==============================================================================
MAIN TRAINING FUNCTION
==============================================================================
"""


def main(script_args, training_args, model_args):
    """
    WHAT: Main entry point for supervised fine-tuning (distillation)

    WHY: Trains a smaller model to imitate reasoning traces from a larger model
         (DeepSeek-R1 → Qwen 7B). This provides a strong initialization for
         subsequent GRPO training.

    HOW:
        1. Set up logging and reproducibility
        2. Load distillation dataset (Mixture-of-Thoughts)
        3. Load base model and tokenizer
        4. Set up chat format (ChatML)
        5. Initialize SFTTrainer
        6. Run training loop (supervised learning on demonstrations)
        7. Save model and push to hub
        8. Optionally evaluate

    PROXIMAL CONTEXT:
        - Input: Configuration objects (dataset, model, training params)
        - Output: Fine-tuned model, training metrics

    DISTAL CONTEXT:
        - Originates from: CLI args → YAML config → Parsed configs
        - Flows to:
            * Saved model → Stage 2 (GRPO) initialization
            * Hub → Model sharing and versioning
            * Metrics → W&B → Training monitoring

    Args:
        script_args (ScriptArguments): Dataset configuration
        training_args (SFTConfig): Training hyperparameters
        model_args (ModelConfig): Model loading configuration

    Returns:
        None (side effects: saves model, logs metrics)
    """

    # ==========================================================================
    # STEP 1: REPRODUCIBILITY SETUP
    # ==========================================================================
    # WHAT: Set random seed for reproducibility
    # WHY: Ensures consistent results across runs
    set_seed(training_args.seed)

    # ==========================================================================
    # STEP 2: LOGGING CONFIGURATION
    # ==========================================================================
    # WHAT: Set up logging infrastructure
    # WHY: Essential for monitoring training progress and debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # WHAT: Log all configuration for debugging
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # ==========================================================================
    # STEP 3: CHECKPOINT DETECTION
    # ==========================================================================
    # WHAT: Check for existing checkpoints to resume training
    # WHY: Enables fault tolerance for long training runs
    # HOW: Scans output directory for checkpoint-{step} folders
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # ==========================================================================
    # STEP 4: WANDB INITIALIZATION
    # ==========================================================================
    # WHAT: Initialize Weights & Biases for experiment tracking
    # WHY: Track loss curves, learning rates, and other training metrics
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # ==========================================================================
    # STEP 5: DATA LOADING
    # ==========================================================================
    # WHAT: Load distillation dataset (demonstrations from strong model)
    # WHY: These are the "teacher" examples the model will learn from
    # HOW: get_dataset handles both single datasets and mixtures
    #
    # CRITICAL: For SFT, dataset must contain:
    # - "messages" column: List of dicts with role/content (ChatML format)
    #   OR
    # - Custom columns that can be formatted into conversations
    #
    # EXAMPLE (Mixture-of-Thoughts format):
    # {
    #   "messages": [
    #     {"role": "system", "content": "You are a helpful assistant..."},
    #     {"role": "user", "content": "Solve: 2x + 5 = 13"},
    #     {"role": "assistant", "content": "<think>...</think><answer>x=4</answer>"}
    #   ]
    # }
    dataset = get_dataset(script_args)

    # ==========================================================================
    # STEP 6: TOKENIZER LOADING
    # ==========================================================================
    # WHAT: Load tokenizer for text ↔ token conversion
    # WHY: Required for chat template and special token handling
    # DATA FLOW: Model name → HF Hub → Tokenizer
    tokenizer = get_tokenizer(model_args, training_args)

    # ==========================================================================
    # STEP 7: MODEL LOADING
    # ==========================================================================
    # WHAT: Load base model to be fine-tuned
    # WHY: This is the "student" that will learn from distillation data
    # HOW: Handles quantization, attention implementation, device mapping
    #
    # TYPICAL MODEL: Qwen2.5-Math-7B-RoPE-300k
    # - 7B parameters (tractable to train)
    # - Extended to 300K context via RoPE (for long reasoning chains)
    # - Math-focused pre-training (better base for reasoning)
    model = get_model(model_args, training_args)

    # ==========================================================================
    # STEP 8: CHAT FORMAT SETUP
    # ==========================================================================
    # WHAT: Set up chat template if not present in tokenizer
    # WHY: Chat template formats conversations into token sequences
    # HOW: Uses ChatML format by default (<|im_start|>, <|im_end|> tokens)
    #
    # CHAT TEMPLATE IMPORTANCE:
    # - Defines how messages are concatenated
    # - Adds special tokens (system, user, assistant markers)
    # - Ensures model learns conversational structure
    #
    # EXAMPLE ChatML format:
    # <|im_start|>system
    # You are a helpful assistant.
    # <|im_end|>
    # <|im_start|>user
    # What is 2+2?
    # <|im_end|>
    # <|im_start|>assistant
    # 4
    # <|im_end|>

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        # WHAT: setup_chat_format adds ChatML template and special tokens
        # RETURNS: Modified model and tokenizer
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    # ==========================================================================
    # STEP 9: SFT TRAINER INITIALIZATION
    # ==========================================================================
    # WHAT: Initialize the SFTTrainer with all components
    # WHY: SFTTrainer handles supervised fine-tuning with optimizations:
    #      - Efficient packing of sequences (optional)
    #      - Automatic chat template application
    #      - Loss masking (only compute loss on assistant responses)
    #      - Mixed precision training
    #      - Gradient checkpointing for memory efficiency
    #
    # KEY DIFFERENCE FROM GRPO:
    # - SFT learns from ground truth completions (supervised)
    # - GRPO learns from reward signals (reinforcement learning)
    # - SFT maximizes likelihood: P(completion | prompt)
    # - GRPO maximizes expected reward: E[R(completion)]

    trainer = SFTTrainer(
        # WHAT: Model to fine-tune
        model=model,

        # WHAT: Training configuration (learning rate, batch size, etc.)
        args=training_args,

        # WHAT: Training dataset with demonstrations
        # NOTE: SFTTrainer expects "messages" column in ChatML format
        train_dataset=dataset[script_args.dataset_train_split],

        # WHAT: Evaluation dataset (optional)
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),

        # WHAT: Processing class (tokenizer)
        processing_class=tokenizer,

        # WHAT: PEFT config for LoRA/QLoRA (optional)
        # WHY: Enables parameter-efficient fine-tuning
        peft_config=get_peft_config(model_args),

        # WHAT: Callbacks for Hub pushing, evaluation, etc.
        callbacks=get_callbacks(training_args, model_args),
    )

    # ==========================================================================
    # STEP 10: TRAINING LOOP
    # ==========================================================================
    # WHAT: Execute supervised fine-tuning
    # WHY: This is where the model learns reasoning from demonstrations
    # HOW: Standard supervised learning with cross-entropy loss
    #
    # SFT TRAINING LOOP (simplified):
    # For each training step:
    #   1. Sample batch of conversations from dataset
    #   2. Apply chat template to format as token sequences:
    #      [system tokens] [user tokens] [assistant tokens] [eos]
    #   3. Forward pass: compute model predictions
    #   4. Compute loss ONLY on assistant tokens (not system/user):
    #      loss = -log P(assistant_tokens | system + user tokens)
    #      This is called "causal language modeling" with masking
    #   5. Backpropagate: compute gradients
    #   6. Update parameters with optimizer (AdamW)
    #   7. Log metrics (loss, perplexity) to W&B
    #   8. Periodically save checkpoints
    #
    # KEY HYPERPARAMETERS:
    # - learning_rate: 4e-5 (higher than typical fine-tuning, lower than pre-training)
    # - max_seq_length: 32768 (long context for reasoning chains)
    # - num_train_epochs: 5 (multiple passes over distillation data)
    # - gradient_accumulation_steps: 8 (effective batch size = 2 * 8 * 8 = 128)
    # - gradient_checkpointing: True (saves memory for long sequences)
    #
    # DATA FLOW:
    #   Dataset → Batch sampling → Chat template → Tokenization
    #   → Forward pass → Loss computation → Backprop → Parameter update

    logger.info("*** Train ***")

    # WHAT: Determine checkpoint to resume from
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # WHAT: Run training loop
    # RETURNS: TrainOutput with metrics (loss, perplexity, etc.)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # WHAT: Extract and log metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ==========================================================================
    # STEP 11: MODEL SAVING
    # ==========================================================================
    # WHAT: Save fine-tuned model to disk and Hub
    # WHY: This model will be used as initialization for Stage 2 (GRPO)
    # DATA FLOW: Trained model → Filesystem → Hub → GRPO training

    logger.info("*** Save model ***")

    # WHAT: Align generation config with tokenizer
    # WHY: Ensures proper EOS token handling during inference
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id

    # WHAT: Save model files
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # WHAT: Create model card and save config (main process only)
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # WHAT: Re-enable KV cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # ==========================================================================
    # STEP 12: EVALUATION (OPTIONAL)
    # ==========================================================================
    # WHAT: Evaluate on held-out test set
    # WHY: Measures generalization to unseen demonstrations
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ==========================================================================
    # STEP 13: HUB PUSH (OPTIONAL)
    # ==========================================================================
    # WHAT: Upload model to HuggingFace Hub
    # WHY: Enables sharing and serves as checkpoint for Stage 2
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
    WHAT: Entry point for command-line execution

    HOW:
        1. Parse CLI arguments and YAML config
        2. Instantiate configuration dataclasses
        3. Call main() function

    USAGE:
        accelerate launch --config_file=zero3.yaml src/open_r1/sft.py \\
            --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml
    """

    # WHAT: Create parser for configuration types
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))

    # WHAT: Parse arguments from CLI and YAML files
    script_args, training_args, model_args = parser.parse_args_and_config()

    # WHAT: Execute training
    main(script_args, training_args, model_args)

"""
==============================================================================
KEY TAKEAWAYS FOR UNDERSTANDING SFT (DISTILLATION)
==============================================================================

1. **SFT is Knowledge Distillation**:
   - Strong model (DeepSeek-R1) generates reasoning traces
   - Weak model (Qwen 7B) learns to imitate these traces
   - Result: Smaller model captures reasoning format and strategy

2. **Why SFT Before GRPO**:
   - Provides strong initialization with reasoning format
   - Teaches <think>/<answer> structure
   - Makes GRPO training more sample-efficient and stable

3. **Long Context is Critical**:
   - 32K token sequences for multi-step reasoning
   - Requires gradient checkpointing to fit in memory
   - Extended RoPE for up to 300K context

4. **Chat Template Matters**:
   - Defines how conversations are formatted
   - ChatML adds special tokens for structure
   - Critical for model to learn conversational flow

5. **Memory Optimizations**:
   - Gradient checkpointing: trade compute for memory
   - DeepSpeed ZeRO-3: shard model across GPUs
   - Liger kernel: optimized operations
   - Mixed precision (BF16): 2x memory savings

6. **Typical Training Setup**:
   - 8x H100 80GB GPUs
   - DeepSpeed ZeRO-3 for model sharding
   - Batch size 2 per device, grad accum 8 (effective batch 128)
   - 5 epochs (~1-2 days)
   - Learning rate 4e-5 with cosine schedule

7. **Dataset Requirements**:
   - Must have "messages" column in ChatML format
   - Each example is a complete conversation
   - Assistant responses should include <think>/<answer> tags
   - Typical size: 100K-1M examples

8. **Output**: Distilled Model
   - Learns reasoning format from demonstrations
   - Ready for Stage 2 (GRPO) fine-tuning
   - Can be evaluated on math/code benchmarks
   - Typically achieves 60-70% of strong model performance

==============================================================================
DISTILLATION VS PRE-TRAINING VS GRPO
==============================================================================

| Aspect           | Pre-training         | Distillation (SFT)   | GRPO (RL)           |
|------------------|----------------------|----------------------|---------------------|
| Data source      | Web crawl            | Strong model traces  | Prompts + rewards   |
| Supervision      | Next token pred      | Ground truth comps   | Reward signals      |
| Scale            | Trillions of tokens  | 100K-1M examples     | 10K-100K prompts    |
| Cost             | $$$$$                | $$                   | $                   |
| What it learns   | Language patterns    | Reasoning format     | Task optimization   |
| Training time    | Weeks/months         | Days                 | Hours               |
| Key challenge    | Data quality         | Teacher quality      | Reward design       |

==============================================================================
"""
