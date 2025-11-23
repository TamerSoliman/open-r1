"""
==============================================================================
FILE: src/open_r1/configs.py
CATEGORY: Core Training - Configuration System
PRIORITY: HIGH
LINES: 332
DEPENDENCIES: trl (TRL configuration base classes)
==============================================================================

OVERVIEW:
This module defines the configuration dataclasses for SFT and GRPO training.
Configurations are typically loaded from YAML files and validated at runtime.

ROLE IN DEEPSEEK R1:
- Centralizes all hyperparameters and settings
- Enables reproducible experiments via YAML configs
- Supports both single datasets and complex mixtures
- Extends TRL's base configs with custom options

KEY FEATURES:
1. Dataset Mixture Support: Weighted combinations of multiple datasets
2. Custom Training Options: Benchmarks, callbacks, Hub integration
3. Reward Function Configuration: Parameterized reward functions for GRPO
4. W&B Integration: Experiment tracking configuration
5. Validation Logic: Ensures configuration consistency

DATA FLOW:
YAML config file → TrlParser → Dataclass instances → Training scripts
==============================================================================
"""

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
# [License omitted]

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import trl


"""
==============================================================================
DATASET CONFIGURATION CLASSES
==============================================================================
"""


@dataclass
class DatasetConfig:
    """
    WHAT: Configuration for a single dataset in a mixture

    WHY: Enables flexible dataset combination with per-dataset settings

    FIELDS:
        id: HuggingFace dataset ID (e.g., "open-r1/OpenR1-Math-220k")
        config: Dataset configuration name (e.g., "default", "all")
        split: Dataset split ("train", "test", "validation")
        columns: Which columns to keep (e.g., ["prompt", "solution"])
        weight: Sampling weight for mixture (0.0 to 1.0)

    USAGE:
        DatasetConfig(
            id="open-r1/math-problems",
            config="algebra",
            split="train",
            columns=["problem", "solution"],
            weight=0.7
        )
    """

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """
    WHAT: Configuration for combining multiple datasets

    WHY: Multi-task training benefits from diverse data sources
         Enables balancing between dataset types

    FIELDS:
        datasets: List of DatasetConfig objects
        seed: Random seed for shuffling and sampling
        test_split_size: Fraction of data for test set (e.g., 0.1 for 10%)

    USAGE:
        DatasetMixtureConfig(
            datasets=[
                DatasetConfig(id="dataset1", weight=0.6),
                DatasetConfig(id="dataset2", weight=0.4),
            ],
            seed=42,
            test_split_size=0.1
        )
    """

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


"""
==============================================================================
SCRIPT ARGUMENTS (BASE)
==============================================================================
"""


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    WHAT: Extended version of TRL's ScriptArguments with mixture support

    WHY: TRL's base class only supports single datasets
         DeepSeek R1 training uses dataset mixtures for multi-task learning

    KEY INNOVATION:
        dataset_mixture: Enables weighted combination of multiple datasets
        Example: 60% math problems + 40% code problems

    VALIDATION:
        - Either dataset_name OR dataset_mixture must be provided
        - Dataset mixture must have consistent column names
        - Weights are optional (defaults to equal weighting)

    USAGE IN YAML:
        dataset_mixture:
          datasets:
            - id: open-r1/math-problems
              columns: [prompt, solution]
              weight: 0.6
            - id: open-r1/code-problems
              columns: [prompt, solution]
              weight: 0.4
          seed: 42
          test_split_size: 0.1
    """

    # WHAT: Override to make dataset_name optional
    # WHY: Can use dataset_mixture instead
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Dataset name. Can be omitted if using dataset_mixture."}
    )

    # WHAT: Advanced dataset mixture configuration
    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Configuration for creating dataset mixtures with advanced options like shuffling."},
    )

    def __post_init__(self):
        """
        WHAT: Validation logic run after initialization

        WHY: Ensures configuration is valid before training starts
             Fail fast with clear error messages

        VALIDATION STEPS:
            1. Check that either dataset_name or dataset_mixture is provided
            2. If mixture, validate structure and convert to DatasetMixtureConfig
            3. Validate column consistency across datasets in mixture

        DATA FLOW:
            YAML dict → __post_init__ → DatasetMixtureConfig object
        """
        # VALIDATION 1: Must have either dataset_name or mixture
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        # VALIDATION 2: If mixture, validate and convert to DatasetMixtureConfig
        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                # WHAT: Convert each dataset dict to DatasetConfig
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            # WHAT: Convert dict to DatasetMixtureConfig dataclass
            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # VALIDATION 3: Check column consistency
            # WHY: All datasets in mixture must have same columns after selection
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


"""
==============================================================================
GRPO CONFIGURATION
==============================================================================
"""


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    WHAT: Configuration for GRPO training (extends TRL's GRPOConfig)

    WHY: Adds custom options for benchmarks, callbacks, and Hub integration
         Specific to DeepSeek R1's training infrastructure

    KEY ADDITIONS:
        - benchmarks: List of tasks to evaluate on after training
        - callbacks: Custom callbacks (push to hub, eval jobs)
        - hub_model_revision: Branch name for Hub pushes
        - system_prompt: Prompt to guide reasoning format
        - wandb_*: W&B experiment tracking configuration

    USAGE:
        GRPOConfig(
            benchmarks=["math_500", "aime24"],
            callbacks=["push_to_hub_revision"],
            system_prompt="Respond in format: <think>...</think><answer>...</answer>",
            wandb_project="open-r1",
            ...
        )
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    num_completions_to_print: int = field(default=0, metadata={"help": "Number of completions to print."})
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log the unique prompts to wandb. This will create a new run for each unique prompt.")
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    WHAT: Configuration for SFT training (extends TRL's SFTConfig)

    WHY: Similar to GRPOConfig but for supervised fine-tuning
         Adds same custom options for infrastructure

    KEY ADDITIONS: Same as GRPOConfig
        - benchmarks, callbacks, hub options, wandb options
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )


"""
==============================================================================
GRPO SCRIPT ARGUMENTS
==============================================================================
"""


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    WHAT: Script arguments specific to GRPO training

    WHY: GRPO requires additional configuration for:
         - Reward functions and their parameters
         - Code execution providers
         - Completion length limits

    KEY FIELDS:
        reward_funcs: List of reward function names
        cosine_*: Parameters for cosine-scaled length reward
        repetition_*: Parameters for repetition penalty
        code_*: Code execution configuration
        parallel_code_exec_per_proc: Concurrency for code execution

    TYPICAL USAGE:
        reward_funcs: [accuracy, format, tag_count]
        reward_weights: [1.0, 1.0, 1.0]
        code_provider: e2b
        parallel_code_exec_per_proc: 2
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )

    # COSINE-SCALED LENGTH REWARD PARAMETERS
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )

    # REPETITION PENALTY PARAMETERS
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )

    # CODE EXECUTION PARAMETERS
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "for each generation, evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases. Useful to avoid overloading the eval server + save time on wrong solutions"
        },
    )
    code_eval_scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = field(
        default="weighted_sum",
        metadata={"help": "use fraction of passed test cases as reward. If false, use 0/1 scoring."},
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={
            "help": "Number of parallel E2B code executions per process. Default of 2 is suitable for the Free Hobby tier of E2B with 8 GPUs used for training."
        },
    )

    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column to use as prompts for training."},
    )

    # CODE EXECUTION PROVIDERS
    e2b_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the E2B router. See scripts/e2b_router.py"},
    )
    morph_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the MorphCloud router. See scripts/morph_router.py"},
    )
    code_provider: Optional[str] = field(
        default="e2b",
        metadata={
            "help": "Provider for code execution. Options: 'e2b', 'local', 'morph'.",
            "choices": ["e2b", "local", "morph"],
        },
    )
    ioi_provider: Optional[str] = field(
        default="piston",
        metadata={
            "help": "Provider for IOI code execution. Options: 'piston', 'morph'.",
            "choices": ["piston", "morph"],
        },
    )

    # COMPLETION LENGTH PARAMETERS
    max_completion_len: int = field(
        default=16384,
        metadata={"help": "Maximum number of characters in completion."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Minimum number of characters in completion."},
    )


"""
==============================================================================
KEY TAKEAWAYS - CONFIGURATION SYSTEM
==============================================================================

1. **Dataset Mixtures Enable Multi-Task Learning**:
   - Combine multiple datasets with weights
   - Enables training on math + code + reasoning simultaneously
   - Column consistency validation prevents errors

2. **Reward Function Configuration is Flexible**:
   - List of reward names: ["accuracy", "format", "tag_count"]
   - Parameterized rewards (cosine scaling, repetition penalty)
   - Easy to add new rewards via registry

3. **Code Execution Providers are Pluggable**:
   - E2B, MorphCloud, or Piston
   - Router URLs for batch processing
   - Concurrency control per process

4. **Hub Integration for Reproducibility**:
   - hub_model_revision: Push to specific branch
   - overwrite_hub_revision: Control versioning
   - push_to_hub_revision: Automatic uploads

5. **W&B Integration for Experiment Tracking**:
   - wandb_entity: Organization
   - wandb_project: Project name
   - wandb_run_group: Group related runs

6. **Validation Prevents Common Errors**:
   - Column consistency in mixtures
   - Either dataset_name or mixture required
   - Clear error messages for misconfigurations

7. **YAML as Configuration Format**:
   - Human-readable
   - Version controllable
   - TrlParser handles parsing + validation
   - Enables reproducible experiments

==============================================================================
"""
