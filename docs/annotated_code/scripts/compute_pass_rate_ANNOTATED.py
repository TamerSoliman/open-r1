# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
==============================================================================
FILE: scripts/pass_rate_filtering/compute_pass_rate.py
CATEGORY: Scripts - Dataset Quality Filtering
PRIORITY: HIGH
LINES: 206
DEPENDENCIES:
    - vLLM: Fast inference
    - HuggingFace datasets: Dataset processing
    - TRL: Training configs
    - open_r1: Reward functions
==============================================================================

OVERVIEW:
Script for filtering datasets by pass rate - the percentage of generations
that pass reward function checks. Generates multiple completions per prompt,
computes rewards, calculates pass rate, and filters to problems with intermediate
difficulty (not too easy, not too hard).

ROLE IN DEEPSEEK R1:
- Filters training data by difficulty (pass rate)
- Removes trivial problems (pass_rate > 0.9)
- Removes impossible problems (pass_rate < 0.1)
- Keeps challenging but learnable problems (0.1 < pass_rate < 0.9)
- Improves training efficiency on quality data

KEY CONCEPTS:
1. Pass Rate: % of generations that achieve reward > threshold
2. Sweet Spot: 0.1-0.9 pass rate (not too easy/hard)
3. Batch Generation: Generate N completions per prompt
4. Reward-Based Filtering: Use same rewards as GRPO

DATA FLOW:
Dataset → Generate N completions → Compute rewards → Calculate pass rate
    → Filter by pass rate → Filtered dataset → GRPO training

TYPICAL USAGE:
python scripts/pass_rate_filtering/compute_pass_rate.py \
  --config recipes/dataset_filtering/config_demo.yaml \
  --dataset-name open-r1/math-prompts \
  --num-generations 16 \
  --pass-rate-min 0.1 \
  --pass-rate-max 0.9
==============================================================================
"""

import logging
from dataclasses import dataclass
from git import Optional
import torch
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from trl import ModelConfig, TrlParser
from trl.data_utils import apply_chat_template
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

@dataclass
class PassRateScriptArguments(GRPOScriptArguments):
    """
    WHAT: Configuration for pass rate computation and filtering

    WHY: Extends GRPOScriptArguments to reuse GRPO configs
         (reward functions, dataset settings, etc.)

    Fields:
        output_dataset_name: Name for filtered dataset on Hub
        pass_rate_min: Minimum pass rate threshold (default: 0.1 = 10%)
        pass_rate_max: Maximum pass rate threshold (default: 0.9 = 90%)
        dataset_start_index: Start index for dataset slice
        dataset_end_index: End index for dataset slice
        dataset_split: Dataset split to use (default: "train")

    PASS RATE RATIONALE:
        - pass_rate < 0.1: Too hard, model can't learn
        - 0.1 < pass_rate < 0.9: Sweet spot for learning
        - pass_rate > 0.9: Too easy, waste of compute

    EXAMPLE:
        pass_rate_min = 0.1  # Keep problems with >=10% success
        pass_rate_max = 0.9  # Keep problems with <=90% success
    """
    # we can be lazy and just use the same script args as GRPO
    output_dataset_name: Optional[str] = None
    pass_rate_min: float = 0.1
    pass_rate_max: float = 0.9
    dataset_start_index: Optional[int] = None
    dataset_end_index: Optional[int] = None
    dataset_split: str = "train"


def main(script_args, training_args, model_args):
    """
    WHAT: Main function for pass rate computation and filtering

    WORKFLOW:
        1. Load dataset (with optional slicing)
        2. Format prompts with chat template
        3. Load vLLM for fast generation
        4. Generate N completions per prompt
        5. Compute rewards for all completions
        6. Calculate pass rate per problem
        7. Filter to problems in target pass rate range
        8. Push results to HuggingFace Hub

    WHY:
        - Focus training on learnable problems
        - Remove trivial and impossible problems
        - Improve training efficiency
        - Better use of compute budget

    METRICS:
        - pass_rate = mean(rewards > threshold)
        - Usually threshold = 0.5 (50% correct)
    """
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
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

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_split)

    # Optional slicing for large datasets
    # WHY: Can process dataset in chunks (parallelizable)
    if script_args.dataset_start_index is not None and script_args.dataset_end_index is not None:
        dataset = dataset.select(range(script_args.dataset_start_index, script_args.dataset_end_index))

    # Get reward functions from the registry
    # WHY: Use same rewards as GRPO training
    #      Ensures filtered data matches training objectives
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        """
        WHAT: Converts dataset example to conversation format

        WHY: Chat models expect conversation structure
             (system, user, assistant messages)

        HOW:
            1. Backup original prompt
            2. Create conversation list
            3. Add system prompt (if provided)
            4. Add user message with prompt
        """
        example["prompt_backup"] = example[prompt_column]

        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    tokenizer = get_tokenizer(model_args, training_args)

    # Remove messages column if exists (avoid conflicts)
    if "messages" in dataset.column_names:
        dataset = dataset.remove_columns("messages")

    # Apply chat template to format prompts
    # WHY: Converts conversation to model-specific format
    #      Example: ChatML, Llama-2 chat format, etc.
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    # Load vLLM for fast inference
    # WHY: vLLM 10-20× faster than HuggingFace generate()
    #      Essential for generating many completions
    llm = LLM(
        model=model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Configure sampling parameters
    # WHY: Generate diverse completions for pass rate estimation
    sampling_params=SamplingParams(
        temperature=training_args.temperature,
        top_p=training_args.top_p,
        top_k=training_args.top_k,
        n=training_args.num_generations,  # Generate N completions per prompt
        max_tokens=training_args.max_completion_length,
    )

    def batch_score(examples):
        """
        WHAT: Generates completions and computes rewards for batch of examples

        WORKFLOW:
            1. Extract prompts from batch
            2. Generate N completions per prompt using vLLM
            3. Format completions for reward functions
            4. Compute rewards using all reward functions
            5. Reshape rewards to (num_prompts, num_generations)
            6. Store generations and rewards in dataset

        WHY:
            - Batch processing for efficiency
            - Use vLLM for speed
            - Compute all rewards in one pass

        Returns:
            dict: Updated examples with pass_rate_generations and pass_rate_rewards
        """
        prompts = examples["prompt"]

        # Generate completions using vLLM
        # WHY: Fast batch generation
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Prepare data structures
        repeated_prompts = []
        reward_completions = []
        grouped_completions = []

        # Process vLLM outputs
        # WHY: Need to flatten for reward computation, then reshape
        for output in outputs:
            prompt = output.prompt
            group = []
            for completion in output.outputs:
                text = completion.text
                group.append(text)
                message = [{"role": "assistant", "content": text}]
                repeated_prompts.append(prompt)
                reward_completions.append(message)
            grouped_completions.append(group)

        def repeat_each_element_k_times(list_to_repeat: list, k: int) -> list:
            """
            WHAT: Repeats each element k times

            WHY: Reward functions expect repeated metadata for each generation
                 Example: [a, b] → [a, a, a, b, b, b] (k=3)
            """
            return [element for item in list_to_repeat for element in [item] * k]

        # Initialize reward tensor
        # Shape: (num_prompts * num_generations, num_reward_funcs)
        rewards_per_func = torch.zeros(len(repeated_prompts), len(reward_funcs))

        # Compute rewards for each reward function
        for i, reward_func in enumerate(reward_funcs):
            # Prepare kwargs for reward function
            # WHY: Rewards need metadata (e.g., ground truth answers)
            keys = [key for key in examples.data.keys() if key not in ["prompt", "completion"]]
            reward_kwargs = {key: repeat_each_element_k_times(examples[key], training_args.num_generations) for key in keys}

            # Compute rewards
            output_reward_func = reward_func(prompts=repeated_prompts, completions=reward_completions, **reward_kwargs)

            # Convert None values to NaN
            # WHY: Some problems may not have valid rewards
            #      NaN allows graceful handling
            output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32)

        # Reshape rewards to (num_prompts, num_generations)
        # WHY: Need rewards grouped by prompt for pass rate computation
        reshaped_rewards = rewards_per_func.view(-1, training_args.num_generations)

        # Store results in dataset
        examples["pass_rate_generations"] = grouped_completions
        examples["pass_rate_rewards"] = reshaped_rewards.tolist()


        return examples

    # Apply batch scoring to dataset
    # WHY: Process in batches for efficiency
    dataset = dataset.map(batch_score, batched=True, batch_size=64)

    # we need to restore the prompt for the final dataset
    def restore_prompt(example):
        """
        WHAT: Restores original prompt column

        WHY: Converted to conversation format for generation
             Need to restore original for downstream use
        """
        example["prompt"] = example["prompt_backup"]
        return example

    dataset = dataset.map(restore_prompt)
    dataset = dataset.remove_columns("prompt_backup")

    # Determine output dataset name
    if script_args.output_dataset_name is not None:
        output_dataset_name = script_args.output_dataset_name
    else:
        # Auto-generate name from model and dataset
        model_name = model_args.model_name_or_path
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        model_revision = model_args.model_revision

        output_dataset_name = f"{script_args.dataset_name}-{model_name}-{model_revision}-gen"

    # Configure dataset names
    config_name="default"
    filtered_config_name = f"filt-{script_args.pass_rate_min}-{script_args.pass_rate_max}"

    if script_args.dataset_start_index is not None and script_args.dataset_end_index is not None:
        config_name = f"gen-{script_args.dataset_start_index}-{script_args.dataset_end_index}"
        filtered_config_name = f"{filtered_config_name}-{script_args.dataset_start_index}-{script_args.dataset_end_index}"

    # Push unfiltered dataset with generations
    # WHY: Preserve raw data for analysis
    dataset.push_to_hub(output_dataset_name, config_name=config_name, revision="gen")

    def filter_func(example):
        """
        WHAT: Filters examples by pass rate

        WHY: Keep problems in target difficulty range

        HOW:
            1. Extract rewards for this example
            2. Compute mean reward (ignoring NaN)
            3. Check if in target range [min, max]

        PASS RATE:
            - mean_reward ≈ pass_rate (assuming binary rewards)
            - Example: 8/16 completions pass → mean_reward = 0.5
        """
        rewards = example["pass_rate_rewards"]
        # get the mean of the rewards that are not None
        mean_reward = torch.nanmean(torch.tensor(rewards, dtype=torch.float32))

        return script_args.pass_rate_min < mean_reward < script_args.pass_rate_max

    logger.info(f"Filtering dataset with low reward threshold {script_args.pass_rate_min} and high reward threshold {script_args.pass_rate_max}")
    logger.info(f"Dataset size before filtering: {dataset}")

    # Apply filter
    dataset = dataset.filter(filter_func)

    logger.info(f"Dataset size after filtering: {dataset}")

    # Push filtered dataset
    # WHY: This is the high-quality dataset for training
    dataset.push_to_hub(output_dataset_name, config_name=filtered_config_name, revision="pass_rate")



if __name__ == "__main__":
    parser = TrlParser((PassRateScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)


"""
==============================================================================
KEY TAKEAWAYS - PASS RATE FILTERING
==============================================================================

1. **Pass Rate Concept**:
   - % of generations that achieve reward > threshold
   - Proxy for problem difficulty
   - Sweet spot: 0.1-0.9 (10-90%)

2. **Filtering Rationale**:
   - pass_rate < 0.1: Too hard, model can't learn
   - 0.1 < pass_rate < 0.9: Learnable problems
   - pass_rate > 0.9: Too easy, waste of compute

3. **Workflow**:
   - Generate N completions per prompt (typically 16)
   - Compute rewards using same functions as GRPO
   - Calculate mean reward ≈ pass rate
   - Filter to target range

4. **vLLM Integration**:
   - Fast batch generation
   - SamplingParams for diversity
   - 10-20× faster than HF transformers

5. **Reward Functions**:
   - Use same rewards as GRPO training
   - Ensures filtered data matches objectives
   - Examples: accuracy, format, code execution

6. **Dataset Organization**:
   - "gen" revision: Raw generations + rewards
   - "pass_rate" revision: Filtered dataset
   - Multiple configs for different filtering ranges

7. **Chunking Support**:
   - dataset_start_index, dataset_end_index
   - Process large datasets in parallel
   - Combine chunks later

8. **Typical Parameters**:
   - num_generations: 16 (for reliable pass rate)
   - pass_rate_min: 0.1 (10%)
   - pass_rate_max: 0.9 (90%)
   - temperature: 0.6-0.7 (balanced diversity)

9. **Use Cases**:
   - Filter math problems by difficulty
   - Remove trivial code problems
   - Focus training on challenging data
   - Improve training efficiency

10. **Integration with GRPO**:
    - Filtered dataset → GRPO training
    - Better learning on quality data
    - Faster convergence
    - Higher final performance

11. **Quality Control**:
    - NaN handling for invalid rewards
    - Preserves original dataset fields
    - Metadata for analysis

12. **Efficiency**:
    - Batch processing (64 examples)
    - vLLM for speed
    - Parallel reward computation
    - Can process millions of examples

==============================================================================
"""
