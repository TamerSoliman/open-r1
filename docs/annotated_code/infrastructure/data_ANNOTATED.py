"""
==============================================================================
FILE: src/open_r1/utils/data.py
CATEGORY: Infrastructure - Dataset Loading
PRIORITY: HIGH
LINES: 66
DEPENDENCIES:
    - datasets: HuggingFace datasets library
    - open_r1.configs: ScriptArguments dataclass
==============================================================================

OVERVIEW:
This module provides dataset loading functionality with support for both
single datasets and weighted mixtures. It's used by SFT and GRPO training
scripts to load training data.

ROLE IN DEEPSEEK R1:
- Centralizes dataset loading logic
- Enables multi-task training via dataset mixtures
- Handles train/test splitting
- Column selection and standardization

KEY FEATURES:
1. Single Dataset Loading: Load from HuggingFace Hub
2. Dataset Mixtures: Weighted combination of multiple datasets
3. Column Selection: Keep only relevant columns
4. Train/Test Splitting: Create evaluation splits
5. Shuffling: Reproducible randomization with seed

DATA FLOW:
ScriptArguments (dataset_name or dataset_mixture)
    → get_dataset()
    → HuggingFace Hub
    → Dataset loading + mixing + splitting
    → DatasetDict (train/test splits)
    → Training script
==============================================================================
"""

import logging

import datasets
from datasets import DatasetDict, concatenate_datasets

from ..configs import ScriptArguments


logger = logging.getLogger(__name__)


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """
    WHAT: Load a dataset or a mixture of datasets based on the configuration

    WHY: Training on multiple datasets improves generalization
         Weighted mixtures enable balancing between task types
         (e.g., 60% math + 40% code)

    HOW:
        Single dataset mode:
            1. Load directly from HuggingFace Hub
            2. Return as-is

        Mixture mode:
            1. For each dataset in mixture:
               a. Load from Hub
               b. Select columns (if specified)
               c. Sample by weight (shuffle + select)
            2. Concatenate all datasets
            3. Shuffle combined dataset
            4. Optionally split into train/test
            5. Return DatasetDict

    PROXIMAL CONTEXT:
        - Input: ScriptArguments with dataset configuration
        - Output: DatasetDict with train (and optionally test) splits

    DISTAL CONTEXT:
        - Originates from: YAML config → TrlParser → ScriptArguments
        - Flows to: Training script (SFT/GRPO) → Batching → Training

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration

    Returns:
        DatasetDict: The loaded datasets with train (and optionally test) splits

    EXAMPLE (Single Dataset):
        args = ScriptArguments(
            dataset_name="open-r1/OpenR1-Math-220k",
            dataset_config="default"
        )
        dataset = get_dataset(args)
        # Returns: DatasetDict({'train': Dataset(...)})

    EXAMPLE (Mixture):
        args = ScriptArguments(
            dataset_mixture={
                'datasets': [
                    {'id': 'open-r1/math', 'weight': 0.6, 'columns': ['prompt', 'solution']},
                    {'id': 'open-r1/code', 'weight': 0.4, 'columns': ['prompt', 'solution']},
                ],
                'seed': 42,
                'test_split_size': 0.1
            }
        )
        dataset = get_dataset(args)
        # Returns: DatasetDict({'train': Dataset(...), 'test': Dataset(...)})
    """

    # =========================================================================
    # MODE 1: SINGLE DATASET
    # =========================================================================
    if args.dataset_name and not args.dataset_mixture:
        # WHAT: Load single dataset from HuggingFace Hub
        # WHY: Simplest case - just one dataset
        # HOW: datasets.load_dataset() handles download and caching
        logger.info(f"Loading dataset: {args.dataset_name}")
        return datasets.load_dataset(args.dataset_name, args.dataset_config)

    # =========================================================================
    # MODE 2: DATASET MIXTURE
    # =========================================================================
    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")

        # WHAT: Seed for reproducible shuffling
        seed = args.dataset_mixture.seed
        datasets_list = []

        # STEP 1: Load and process each dataset in mixture
        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")

            # SUBSTEP 1.1: Load dataset from Hub
            # WHY: Each dataset might be on different Hub repo
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,  # Usually "train"
            )

            # SUBSTEP 1.2: Select columns (if specified)
            # WHY: Standardize column names across datasets
            # EXAMPLE: Dataset1 has ["question", "answer"], Dataset2 has ["prompt", "response"]
            #          → Both selected to ["prompt", "solution"] for consistency
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)

            # SUBSTEP 1.3: Sample by weight
            # WHY: Weight controls proportion of each dataset in mixture
            # HOW: Shuffle → Select first (length * weight) examples
            # EXAMPLE: weight=0.6, length=1000 → select 600 examples
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) "
                    f"with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        # STEP 2: Combine all datasets
        if datasets_list:
            # SUBSTEP 2.1: Concatenate
            # WHY: Create single dataset from multiple sources
            # HOW: datasets.concatenate_datasets() stacks datasets vertically
            # REQUIREMENT: All datasets must have same columns (validated in configs.py)
            combined_dataset = concatenate_datasets(datasets_list)

            # SUBSTEP 2.2: Shuffle combined dataset
            # WHY: Ensure examples from different datasets are mixed
            # HOW: Use same seed for reproducibility
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            # STEP 3: Train/Test Split (Optional)
            # WHY: Need evaluation set to monitor overfitting
            # HOW: train_test_split() creates stratified or random split
            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size,
                    seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with "
                    f"test size: {args.dataset_mixture.test_split_size}"
                )
                return combined_dataset
            else:
                # WHAT: No test split requested, return as train-only DatasetDict
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")


"""
==============================================================================
KEY TAKEAWAYS - DATASET LOADING
==============================================================================

1. **Dataset Mixtures Enable Multi-Task Learning**:
   - Combine math + code + reasoning tasks
   - Weighted sampling controls proportions
   - Improves generalization vs single-task training

2. **Column Standardization is Critical**:
   - Different datasets use different column names
   - Standardize to ["prompt", "solution"] or similar
   - Validation in configs.py prevents mismatches

3. **Weighted Sampling Strategy**:
   - weight=0.6 → keep 60% of dataset
   - Applied AFTER shuffling (random subset)
   - Enables balancing dataset sizes

4. **Shuffling for Mixing**:
   - Each dataset shuffled before sampling
   - Combined dataset shuffled again
   - Ensures good mixing across sources

5. **Train/Test Splitting**:
   - Optional test_split_size parameter
   - Typical: 0.1 (10% for evaluation)
   - Uses same seed for reproducibility

6. **Seed for Reproducibility**:
   - Same seed → same shuffle order
   - Same seed → same train/test split
   - Critical for comparing experiments

7. **DatasetDict Format**:
   - Always returns DatasetDict
   - Keys: "train" (always), "test" (if split)
   - Consistent interface for training scripts

8. **Typical Mixture Example**:
   datasets:
     - id: open-r1/math-problems
       columns: [prompt, solution]
       weight: 0.5  # 50% math
     - id: open-r1/code-problems
       columns: [prompt, solution]
       weight: 0.3  # 30% code
     - id: open-r1/reasoning
       columns: [prompt, solution]
       weight: 0.2  # 20% reasoning
   seed: 42
   test_split_size: 0.1

9. **Common Pitfalls**:
   - Forgetting to standardize column names → errors
   - Weights don't need to sum to 1.0 (they're proportions)
   - Large datasets with high weight → memory issues
   - Different seeds → non-reproducible results

10. **Performance Considerations**:
    - load_dataset() caches on disk (fast on repeat)
    - Shuffling large datasets can be slow
    - Column selection reduces memory usage
    - Weighted sampling creates smaller datasets

==============================================================================
"""
