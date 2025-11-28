# ==============================================================================
# FILE: src/open_r1/utils/model_utils.py
# CATEGORY: Utilities - Model Loading
# PRIORITY: MEDIUM
# LINES: 43
# DEPENDENCIES: transformers, trl, torch
# ==============================================================================
#
# OVERVIEW:
# Utility functions for loading models and tokenizers with proper configuration.
# Handles quantization, device mapping, attention implementations, and chat templates.
#
# KEY FUNCTIONALITY:
# - get_tokenizer(): Load tokenizer with optional chat template override
# - get_model(): Load model with quantization, device mapping, and training configs
#
# This module centralizes model/tokenizer loading to ensure consistent configuration
# across training scripts (SFT, GRPO, etc.).
# ==============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """
    WHAT: Load and configure tokenizer for the model

    WHY:
    - Centralize tokenizer loading with consistent configuration
    - Allow chat template override for instruction tuning
    - Ensure trust_remote_code is properly set

    HOW:
    1. Load tokenizer from HuggingFace Hub or local path
    2. Apply custom chat template if provided
    3. Return configured tokenizer

    PROXIMAL CONTEXT:
    - INPUT: model_args (model path, revision), training_args (chat template)
    - OUTPUT: PreTrainedTokenizer instance

    DISTAL CONTEXT:
    - ORIGIN: Called at training script initialization
    - DESTINATION: Used for tokenizing prompts and completions

    EXAMPLE:
    ```python
    model_args = ModelConfig(model_name_or_path="meta-llama/Llama-2-7b-hf")
    training_args = SFTConfig(chat_template="{% for message in messages %}...")
    tokenizer = get_tokenizer(model_args, training_args)
    # Result: Tokenizer with custom chat template applied
    ```

    PARAMETERS:
    - model_args.model_name_or_path: HuggingFace repo ID or local path
    - model_args.model_revision: Git revision (branch, tag, commit)
    - model_args.trust_remote_code: Allow custom model code
    - training_args.chat_template: Optional Jinja2 chat template

    RETURNS: Configured tokenizer instance
    """
    """Get the tokenizer for the model."""
    # LOAD TOKENIZER FROM HUB OR LOCAL PATH
    # Uses revision for versioned checkpoints
    # trust_remote_code enables custom tokenizer implementations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # APPLY CUSTOM CHAT TEMPLATE IF PROVIDED
    # Overrides default chat template from tokenizer
    # Useful for instruction tuning with specific formats
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
    """
    WHAT: Load and configure model for training

    WHY:
    - Handle quantization (4-bit, 8-bit) for memory efficiency
    - Configure attention implementation (flash, sdpa, eager)
    - Set device mapping for multi-GPU training
    - Enable gradient checkpointing for large models

    HOW:
    1. Determine torch dtype (auto, float16, bfloat16)
    2. Get quantization config if using QLoRA
    3. Configure attention implementation
    4. Set cache behavior based on gradient checkpointing
    5. Load model with all configurations

    PROXIMAL CONTEXT:
    - INPUT: model_args (model config), training_args (training config)
    - OUTPUT: AutoModelForCausalLM instance

    DISTAL CONTEXT:
    - ORIGIN: Called at training script initialization
    - DESTINATION: Used for training loop (SFT, GRPO)

    EXAMPLE 1: Full precision training
    ```python
    model_args = ModelConfig(
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2"
    )
    training_args = SFTConfig(gradient_checkpointing=True)
    model = get_model(model_args, training_args)
    # Result: Model in bfloat16 with flash attention and gradient checkpointing
    ```

    EXAMPLE 2: QLoRA training (4-bit quantization)
    ```python
    model_args = ModelConfig(
        model_name_or_path="meta-llama/Llama-2-70b-hf",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16"
    )
    training_args = SFTConfig(gradient_checkpointing=True)
    model = get_model(model_args, training_args)
    # Result: 4-bit quantized model with device map for multi-GPU
    ```

    CONFIGURATION OPTIONS:
    - torch_dtype: Data type (auto, float16, bfloat16, float32)
    - attn_implementation: Attention type (flash_attention_2, sdpa, eager)
    - quantization: 4-bit or 8-bit via bitsandbytes
    - use_cache: Disabled for gradient checkpointing, enabled otherwise
    - device_map: Auto-computed for quantization, None for DDP

    RETURNS: Configured model instance ready for training
    """
    """Get the model"""
    # DETERMINE TORCH DTYPE
    # "auto" or None: Let transformers decide based on model config
    # Otherwise: Convert string to torch dtype (e.g., "bfloat16" -> torch.bfloat16)
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    # GET QUANTIZATION CONFIG (if using QLoRA)
    # Returns None if no quantization requested
    # Returns BitsAndBytesConfig for 4-bit/8-bit quantization
    quantization_config = get_quantization_config(model_args)

    # BUILD MODEL KWARGS
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,  # flash_attention_2, sdpa, or eager
        torch_dtype=torch_dtype,
        # CACHE BEHAVIOR:
        # - Disabled (use_cache=False) when gradient checkpointing enabled
        #   WHY: Gradient checkpointing requires recomputation, cache is incompatible
        # - Enabled (use_cache=True) otherwise for faster inference
        use_cache=False if training_args.gradient_checkpointing else True,
        # DEVICE MAP:
        # - Auto device map when quantized (needed for multi-GPU QLoRA)
        # - None when not quantized (DDP/FSDP handles device placement)
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # LOAD MODEL FROM HUB OR LOCAL PATH
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================
# 1. **get_tokenizer()**:
#    - Simple wrapper around AutoTokenizer.from_pretrained
#    - Applies custom chat template if provided
#    - Ensures consistent tokenizer loading across scripts
#
# 2. **get_model()**:
#    - Handles complex model loading with many configurations
#    - Quantization-aware device mapping
#    - Gradient checkpointing disables cache (required)
#    - Supports QLoRA (4-bit/8-bit) and full precision
#
# 3. **Common Patterns**:
#    - trust_remote_code: Enable for models with custom code
#    - revision: Use for versioned checkpoints
#    - torch_dtype: "auto" for safety, explicit for control
#
# 4. **Training Configurations**:
#    - Full precision: torch_dtype="bfloat16", no quantization
#    - QLoRA: load_in_4bit=True, device_map auto-computed
#    - Gradient checkpointing: use_cache=False (required)
#
# 5. **Why Centralized Loading**:
#    - Consistent configuration across SFT, GRPO scripts
#    - Single place to update model loading logic
#    - Easier to debug and maintain
# ==============================================================================
