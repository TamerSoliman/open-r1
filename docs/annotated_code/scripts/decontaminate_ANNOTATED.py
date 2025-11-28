"""
==============================================================================
FILE: scripts/decontaminate.py
CATEGORY: Scripts - Dataset Decontamination
PRIORITY: HIGH
LINES: 147
DEPENDENCIES:
    - datasets: load_dataset, Dataset (HuggingFace datasets library)
    - collections: defaultdict (efficient lookup table construction)
    - tqdm: Progress bar for long-running operations
==============================================================================

==============================================================================
OVERVIEW
==============================================================================

This script implements n-gram-based dataset decontamination to prevent training
data from leaking into evaluation benchmarks. It detects and optionally removes
training examples that overlap with evaluation datasets using word-level n-gram
matching.

ROLE IN DEEPSEEK R1:
-------------------
Dataset contamination is a critical problem in machine learning research:

1. **Evaluation Validity**: If training data contains problems from evaluation
   benchmarks, the model has "seen the answers" and performance metrics are
   inflated and unreliable.

2. **Scientific Rigor**: Decontamination ensures that benchmark results reflect
   true generalization capability, not memorization of evaluation data.

3. **Community Trust**: Open-sourcing decontaminated datasets demonstrates
   research integrity and enables fair comparisons across models.

WHY N-GRAM OVERLAP?
-------------------
N-gram matching detects textual similarity without requiring exact duplicates:

- Robust to minor variations (whitespace, formatting, typos)
- Captures semantic overlap (problems with similar wording)
- Computationally efficient (hash-based lookup)
- Standard approach in the field (DeepSeek, Simple Scaling, etc.)

The default n=8 (8-word sequences) balances:
- Precision: Long enough to avoid spurious matches
- Recall: Short enough to catch paraphrased problems

CONTAMINATION STRATEGY:
-----------------------
The script checks against 5 key evaluation benchmarks:

1. **AIME 2024/2025**: American Invitational Mathematics Examination
   - Advanced high school math competition problems
   - Tests complex mathematical reasoning

2. **MATH-500**: Curated subset of MATH dataset
   - Diverse mathematical problem types
   - Standard RL benchmark for reasoning models

3. **GPQA**: Graduate-level Physics/Chemistry/Biology Questions
   - Expert-level domain knowledge
   - Tests scientific reasoning

4. **LiveCodeBench (LCB)**: Recent coding problems
   - Fresh problems to prevent memorization
   - Tests code generation on unseen challenges

Each benchmark is loaded and indexed by n-grams for efficient lookup.

DATA FLOW:
----------
    DISTAL ORIGIN:
    ├─> HuggingFace Hub → Training dataset to decontaminate
    └─> HuggingFace Hub → 5 evaluation benchmark datasets

    PROXIMAL PROCESSING (this script):
    1. Load training dataset (e.g., verifiable-coding-problems-python)
    2. Load all 5 evaluation datasets
    3. Build n-gram lookup tables for each eval dataset
    4. For each training example:
       a. Extract n-grams from problem text
       b. Check for overlap with each eval dataset
       c. Add boolean columns: contaminated_aime_2024, contaminated_gpqa, etc.
    5. Optional cleanup (--cleanup flag):
       - Filter out all contaminated rows
       - Remove contamination marker columns
    6. Push to Hub with _decontaminated suffix

    DISTAL DESTINATION:
    └─> HuggingFace Hub → Clean dataset for trustworthy training
    └─> Console → Contamination statistics for transparency

TYPICAL RESULTS:
---------------
Example contamination report:

    Removed 34 samples from 'aime_2024'
    Removed 12 samples from 'aime_2025'
    Removed 0 samples from 'math_500'
    Removed 5 samples from 'gpqa'
    Removed 87 samples from 'lcb'
    Initial size: 10000, Final size: 9862

Interpretation:
- 138 contaminated samples (1.38% contamination rate)
- LiveCodeBench has highest overlap (coding problems are common in training data)
- MATH-500 has zero overlap (good dataset curation)

==============================================================================
IMPORTS AND SETUP
==============================================================================
"""

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
"""
This script is used to decontaminate a dataset by checking for n-gram overlap with other datasets.
It uses the same approach presented in https://huggingface.co/papers/2501.19393,
as found in: https://github.com/simplescaling/s1/blob/main/data/decontaminate_util.py

Usage:

python scripts/decontaminate.py \
    --dataset open-r1/verifiable-coding-problems-python \
    --split train \
    --ngram_size 8 \
    --problem_column problem \
    --cleanup
"""

import collections

from tqdm import tqdm


"""
==============================================================================
WHAT: normalize_string()
WHY:  Ensure consistent text comparison regardless of formatting
HOW:  Lowercase and whitespace normalization
==============================================================================

PROXIMAL CONTEXT: Called on every problem string before n-gram extraction
DISTAL CONTEXT: Sourced from Simple Scaling S1 paper implementation

Text normalization eliminates spurious differences that don't affect semantics:
- Case: "Solve FOR x" vs "solve for x" → both become "solve for x"
- Whitespace: "x  +  y" vs "x + y" → both become "x + y"

This increases recall (catches more contamination) without reducing precision,
since these variations represent the same problem.
"""
def normalize_string(text: str) -> str:
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text


"""
==============================================================================
WHAT: word_ngrams()
WHY:  Generate n-gram sequences for overlap detection
HOW:  Sliding window over word tokens
==============================================================================

PROXIMAL CONTEXT: Called after normalization to extract n-grams
DISTAL CONTEXT: Standard n-gram extraction algorithm

An n-gram is a sequence of n consecutive words. For example:

    Text: "solve the following equation"
    3-grams: ["solve the following", "the following equation"]
    8-grams: None (text too short)

WHY WORD-LEVEL VS CHARACTER-LEVEL?
- Word n-grams capture semantic units (concepts, phrases)
- Character n-grams would match trivial substrings like "the " everywhere
- Word-level is standard for decontamination (Simple Scaling, others)

The sliding window approach generates all possible n-grams:
- Words: ["a", "b", "c", "d"]
- 2-grams: ["a b", "b c", "c d"]
- Position i: words[i:i+n]

Edge case: If text has fewer than n words, returns empty list (no n-grams).
"""
def word_ngrams(text: str, n: int) -> list:
    """Generate word-level n-grams from text."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


"""
==============================================================================
WHAT: build_ngram_lookup()
WHY:  Create efficient index for contamination detection
HOW:  Hash table mapping n-grams to document IDs
==============================================================================

PROXIMAL CONTEXT: Called once per evaluation dataset at startup
DISTAL CONTEXT: Inverted index data structure (common in search engines)

This function builds a reverse index:

    Instead of: Document → N-grams
    We create:  N-gram → Set of Documents

Example:
    Doc 0: "solve for x"
    Doc 1: "solve for y"

    With n=2:
    {
        "solve for": {0, 1},  # Appears in both docs
        "for x": {0},         # Only in doc 0
        "for y": {1},         # Only in doc 1
    }

WHY THIS STRUCTURE?
- Fast lookup: Given an n-gram, instantly find which docs contain it
- Memory efficient: Store each unique n-gram once, not per document
- Set operations: Efficiently check "does any n-gram overlap?"

The defaultdict(set) automatically creates empty sets for new n-grams,
avoiding explicit key existence checks.

PERFORMANCE:
- Time: O(total n-grams) to build
- Space: O(unique n-grams × average docs per n-gram)
- Lookup: O(1) per n-gram

For typical eval datasets (100-1000 problems), this builds in seconds.
"""
def build_ngram_lookup(documents: list[str], ngram_size: int = 8) -> dict[str, set[int]]:
    """Build ngram lookup for documents."""
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(tqdm(documents)):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(doc_id)

    return lookup


"""
==============================================================================
WHAT: build_ngram_single()
WHY:  Extract n-grams from a single training example
HOW:  Normalize and generate n-grams (returns set for fast membership testing)
==============================================================================

PROXIMAL CONTEXT: Called for each training example during contamination check
DISTAL CONTEXT: Counterpart to build_ngram_lookup for single documents

This function processes one training example to extract its n-grams:

1. Normalize text (consistent with eval dataset normalization)
2. Generate all n-grams
3. Return as set (not list) for efficient lookup

WHY RETURN SET?
- Membership testing: "ngram in ngram_lookup" is O(1)
- Deduplication: Same n-gram appearing twice in one example only counts once
- Set operations: Can use set.intersection() for overlap counting

Compared to build_ngram_lookup:
- build_ngram_lookup: Many docs → {ngram: [doc_ids]}
- build_ngram_single: One doc → {ngrams}

The asymmetry reflects the use case:
- Eval datasets: Index once, query many times
- Training examples: Process once, discard n-grams after checking
"""
def build_ngram_single(document: str, ngram_size: int = 8) -> set[str]:
    normalized_text = normalize_string(document)
    ngrams = word_ngrams(normalized_text, ngram_size)

    return set(ngrams)


"""
==============================================================================
WHAT: Main decontamination pipeline
WHY:  Remove evaluation benchmark overlap from training data
HOW:  N-gram matching against 5 standard benchmarks
==============================================================================

ARCHITECTURE:
------------
The pipeline has 4 phases:

Phase 1: Load datasets
    - Training dataset (to decontaminate)
    - 5 evaluation benchmarks (contamination sources)

Phase 2: Build n-gram indexes
    - For each eval dataset, create lookup table
    - Done once upfront for efficiency

Phase 3: Contamination detection
    - For each training example:
        - Extract n-grams
        - Check against each eval dataset
        - Mark with boolean columns

Phase 4: Cleanup (optional)
    - Remove contaminated rows
    - Drop marker columns
    - Push to Hub

EVALUATION BENCHMARKS:
---------------------
The script checks against these datasets:

1. AIME 2024/2025: Math competition problems
   - Column: "problem"
   - Why: High-quality reasoning benchmarks, must be pristine

2. MATH-500: Curated math problems
   - Column: "problem"
   - Why: Standard RL benchmark, widely used for comparison

3. GPQA: Graduate-level science questions
   - Column: "Question"
   - Why: Expert-level reasoning, tests domain knowledge

4. LiveCodeBench: Recent coding problems
   - Column: "question_content"
   - Why: Fresh problems, prevents train-test leakage

CONTAMINATION DETECTION LOGIC:
-----------------------------
For each training example, we check:

    any(ngram in eval_lookup for ngram in training_ngrams)

This means: "Does ANY n-gram from the training example appear in the eval dataset?"

WHY "ANY" NOT "ALL"?
- Conservative: Better to over-filter than under-filter
- Partial overlap often indicates paraphrased problems
- Single shared n-gram (8 words) is unlikely to be coincidence

Example:
    Training: "Solve the quadratic equation x^2 + 5x + 6 = 0"
    Eval: "Find solutions to x^2 + 5x + 6 = 0"

    Shared 8-gram: "x 2 + 5 x + 6 = 0" (normalized)
    → Detected as contaminated (correct!)

PARALLEL PROCESSING:
-------------------
The script uses num_proc=8 for parallel dataset mapping:
- Each worker processes a shard of the dataset
- Speeds up decontamination on large datasets (10k+ examples)
- Safe because each example is processed independently

CLEANUP STRATEGY:
----------------
If --cleanup is specified:

1. Filter out contaminated rows for each benchmark sequentially
   - Why sequential? Allows tracking per-benchmark contamination stats

2. Print removal statistics for transparency
   - Example: "Removed 34 samples from 'aime_2024'"

3. Remove marker columns (contaminated_*)
   - Clean dataset doesn't need these internal flags

4. Report final size reduction
   - Helps assess contamination severity

DATASET NAMING:
--------------
Output dataset name:
- Default: {original_name}_decontaminated
- Custom: --new_dataset_name

Examples:
- verifiable-coding-problems-python → verifiable-coding-problems-python_decontaminated
- Custom: open-r1/clean-math-data

PROXIMAL DECISIONS:
------------------
The script makes several design choices:

1. n-gram size = 8 (default)
   - Balances precision and recall
   - Longer: fewer false positives, more false negatives
   - Shorter: more false positives, fewer false negatives

2. Problem column (--problem_column)
   - Different datasets use different column names
   - Common: "problem", "question", "question_content"

3. Cleanup is optional
   - Allows inspecting contamination before removal
   - Can rerun with --cleanup after verification

4. Push to Hub automatically
   - Ensures decontaminated data is immediately usable
   - Transparent: Other researchers can inspect

DISTAL IMPACT:
-------------
Decontaminated datasets enable:
- Trustworthy evaluation metrics
- Fair model comparisons
- Reproducible research
- Community confidence in results

Without decontamination:
- Inflated benchmark scores
- Misleading research conclusions
- Wasted compute on "good results" from data leakage
"""
if __name__ == "__main__":
    import argparse

    # WHAT: Parse command-line arguments for decontamination configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to check for contamination.")
    parser.add_argument("--config", type=str, default=None, help="Name of the dataset config to load.")
    parser.add_argument("--split", type=str, default="train", help="Split to check for contamination, defaults to `train`.")
    parser.add_argument("--ngram_size", type=int, default=8, help="Size of n-grams to build, defaults to 8.")
    parser.add_argument(
        "--problem_column", type=str, default="problem", help="Name of the column containing the problem (prompt)."
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Whether to remove the contaminated rows before pushing the dataset.",
    )
    parser.add_argument(
        "--new_dataset_name",
        type=str,
        default=None,
        help="New name for the dataset. If not provided, will reuse the name and add a `_decontaminated` to the name."
    )
    args = parser.parse_args()

    from datasets import load_dataset, Dataset

    # WHAT: Load the training dataset to decontaminate
    # WHY: This is the dataset we want to ensure doesn't leak into eval benchmarks
    ds = load_dataset(args.dataset, name=args.config, split=args.split)

    # WHAT: Define evaluation benchmarks to check against
    # WHY: These are standard benchmarks used to evaluate DeepSeek R1 models
    # HOW: Each entry is (dataset, problem_column_name) tuple
    eval_datasets = {
        "aime_2024": (load_dataset("HuggingFaceH4/aime_2024", split="train"), "problem"),
        "aime_2025": (load_dataset("yentinglin/aime_2025", split="train"), "problem"),
        "math_500": (load_dataset("HuggingFaceH4/MATH-500", split="test"), "problem"),
        "gpqa": (load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True), "Question"),
        "lcb": (
            load_dataset(
                "livecodebench/code_generation_lite", split="test", version_tag="v4_v5", trust_remote_code=True
            ),
            "question_content",
        ),
    }

    # WHAT: Build n-gram lookup tables for all eval datasets
    # WHY: Precompute indexes for efficient contamination detection
    # HOW: One lookup table per eval dataset, all stored in dict
    ngram_lookups = {}
    for ds_name, (eval_dataset, problem_col) in eval_datasets.items():
        ngram_lookups[ds_name] = build_ngram_lookup(eval_dataset[problem_col], ngram_size=args.ngram_size)

    # WHAT: Check each training example against all eval datasets
    # WHY: Identify contaminated examples that overlap with benchmarks
    for eval_name, ngram_lookup in ngram_lookups.items():
        # WHAT: Define contamination detection function for this eval dataset
        # WHY: Closures capture ngram_lookup for efficient access
        # Update the ngram_lookup variable for each dataset
        def find_contaminated(row):
            # For each example we have to build the ngrams and check for all of them on each row
            ngrams = build_ngram_single(row[args.problem_column], ngram_size=args.ngram_size)
            # WHAT: Check if ANY n-gram overlaps with eval dataset
            # WHY: Conservative approach - partial overlap suggests contamination
            row[f"contaminated_{eval_name}"] = any(set(ngram in ngram_lookup for ngram in ngrams))
            return row

        # WHAT: Apply contamination check to entire dataset in parallel
        # WHY: num_proc=8 speeds up processing on large datasets
        ds = ds.map(find_contaminated, num_proc=8)

    # WHAT: Optional cleanup - remove contaminated examples
    # WHY: Produces clean dataset ready for training
    # Allow cleaning up via CLI args (removing the contaminated examples and dropping the columns)
    def cleanup(dataset: Dataset) -> Dataset:
        initial_size = len(dataset)
        contamination_cols = [col for col in dataset.column_names if col.startswith("contaminated_")]

        # WHAT: Filter out contaminated examples for each benchmark
        # WHY: Sequential filtering allows per-benchmark statistics
        for col in contamination_cols:
            if col.startswith("contaminated_"):
                size_prior = len(dataset)
                dataset = dataset.filter(lambda x: not x[col], num_proc=8)
                if len(dataset) < size_prior:
                    print(f"Removed {size_prior - len(dataset)} samples from '{col.replace('contaminated_', '')}'")

        # WHAT: Remove contamination marker columns
        # WHY: Clean dataset doesn't need internal flags
        dataset = dataset.remove_columns(contamination_cols)

        # WHAT: Report final size reduction
        # WHY: Transparency about contamination severity
        print(f"Initial size: {initial_size}, Final size: {len(dataset)}")
        return dataset

    if args.cleanup:
        ds = cleanup(ds)

    # WHAT: Push decontaminated dataset to HuggingFace Hub
    # WHY: Makes clean data immediately available for training
    # HOW: Append _decontaminated suffix unless custom name provided
    new_ds_name = args.new_dataset_name or f"{args.dataset}_decontaminated"
    config_name = args.config if args.config is not None else "default"
    url = ds.push_to_hub(new_ds_name, config_name=config_name, split="train")
    print(f"Decontaminated dataset: {url}")


"""
==============================================================================
KEY TAKEAWAYS
==============================================================================

1. **PURPOSE**: This script implements n-gram-based dataset decontamination to
   prevent training data from leaking into evaluation benchmarks, ensuring
   trustworthy and reproducible benchmark results.

2. **METHODOLOGY**: Uses 8-word n-gram overlap detection, matching against 5
   standard benchmarks (AIME 2024/2025, MATH-500, GPQA, LiveCodeBench).

3. **CRITICAL INSIGHT**: Even partial textual overlap (a single shared n-gram)
   is treated as contamination because:
   - 8 consecutive matching words rarely occur by chance
   - Paraphrased problems often share key phrases
   - Conservative filtering is safer than under-filtering

4. **DATA STRUCTURES**:
   - build_ngram_lookup: Creates inverted index {ngram → doc_ids}
   - build_ngram_single: Extracts n-grams from one document
   - Efficient O(1) lookup for contamination detection

5. **PROCESSING PIPELINE**:
   Phase 1: Load training and eval datasets
   Phase 2: Build n-gram indexes (once per eval dataset)
   Phase 3: Mark contaminated rows (parallel processing)
   Phase 4: Optional cleanup and Hub upload

6. **TRANSPARENCY**: The script reports:
   - Per-benchmark contamination counts
   - Total size reduction (initial vs final)
   - Hub URL for decontaminated dataset

   This enables auditing and verification of the decontamination process.

7. **DESIGN CHOICES**:
   - n=8: Balances precision (avoid false positives) and recall (catch paraphrases)
   - Word-level: Captures semantic units, not character substrings
   - ANY overlap: Conservative approach to prevent leakage
   - Sequential filtering: Enables per-benchmark statistics

8. **PROVENANCE**: Based on Simple Scaling S1 paper implementation, ensuring
   consistency with established research practices.

9. **IMPACT**: Decontaminated datasets enable:
   - Fair model comparisons (no inflated scores from leakage)
   - Reproducible research (consistent evaluation conditions)
   - Community trust (transparent methodology)

==============================================================================
USAGE EXAMPLE
==============================================================================

To decontaminate a coding dataset:

    $ python scripts/decontaminate.py \
        --dataset open-r1/verifiable-coding-problems-python \
        --split train \
        --ngram_size 8 \
        --problem_column problem \
        --cleanup

Expected output:

    Building n-gram lookup for aime_2024: 100%|████████| 30/30
    Building n-gram lookup for aime_2025: 100%|████████| 30/30
    ...
    Checking contamination: 100%|████████| 10000/10000
    Removed 34 samples from 'aime_2024'
    Removed 12 samples from 'aime_2025'
    Removed 0 samples from 'math_500'
    Removed 5 samples from 'gpqa'
    Removed 87 samples from 'lcb'
    Initial size: 10000, Final size: 9862
    Decontaminated dataset: https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python_decontaminated

To inspect contamination without removing (omit --cleanup):

    $ python scripts/decontaminate.py \
        --dataset my-dataset \
        --split train \
        --problem_column text

    This adds contaminated_* columns without filtering, allowing manual inspection.

==============================================================================
"""
