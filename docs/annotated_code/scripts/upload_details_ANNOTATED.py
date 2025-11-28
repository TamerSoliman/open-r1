"""
==============================================================================
FILE: scripts/upload_details.py
CATEGORY: Scripts - Hub Upload Utilities
PRIORITY: LOW
LINES: 56
DEPENDENCIES:
    - datasets: load_dataset (HuggingFace datasets library)
    - transformers: HfArgumentParser (argument parsing)
==============================================================================

==============================================================================
OVERVIEW
==============================================================================

This script uploads evaluation results from LightEval benchmark runs to the
HuggingFace Hub. It's a simple utility for sharing detailed evaluation results
(per-example predictions, scores, metadata) as datasets, enabling transparency
and reproducibility in model evaluation.

ROLE IN DEEPSEEK R1:
-------------------
After running evaluations with LightEval (a comprehensive evaluation framework),
the results are saved as local Parquet, JSON, or JSONL files. This script:

1. **Preserves Evaluation Details**: Upload per-example results, not just
   aggregate metrics (enables deeper analysis).

2. **Enables Reproducibility**: Share exact predictions and scores so others
   can verify results or analyze failure modes.

3. **Supports Error Analysis**: Detailed results allow identifying which types
   of problems the model struggles with.

4. **Community Transparency**: Public datasets of evaluation results build
   trust and enable meta-analysis across models.

WHAT IS LIGHTEVAL?
------------------
LightEval is HuggingFace's evaluation framework that:
- Runs models on standard benchmarks
- Saves detailed results (not just accuracy)
- Outputs Parquet/JSON files with per-example data

Example LightEval output:
    results_aime_2024.parquet:
    | problem_id | problem_text    | model_answer | correct | score |
    |------------|-----------------|--------------|---------|-------|
    | 1          | "Find x if..."  | "x = 5"      | True    | 1.0   |
    | 2          | "Solve for..." | "y = 3"      | False   | 0.0   |

This script uploads such files to Hub for sharing.

WHY SEPARATE FROM run_benchmarks.py?
-------------------------------------
run_benchmarks.py runs evaluations and logs aggregate metrics (accuracy, pass@k).
upload_details.py is for uploading detailed per-example results after the fact.

Separation allows:
- Running evaluations without automatic upload (review first)
- Uploading results from different evaluation frameworks
- Bulk upload of multiple evaluation runs

TYPICAL WORKFLOW:
-----------------

1. Run LightEval evaluation:
   ```
   lighteval accelerate \
       --model "myusername/my-model" \
       --tasks "aime_2024,math_500" \
       --output_dir "./eval_results"
   ```

2. LightEval saves results to Parquet files:
   ```
   eval_results/
   ├── results_aime_2024.parquet
   └── results_math_500.parquet
   ```

3. Upload results to Hub:
   ```
   python scripts/upload_details.py \
       --data_files eval_results/*.parquet \
       --hub_repo_id "myusername/my-model-eval-results" \
       --config_name "aime_math"
   ```

4. Results are accessible at:
   https://huggingface.co/datasets/myusername/my-model-eval-results

DATA FLOW:
----------
    DISTAL ORIGIN:
    └─> Local filesystem → LightEval output files (Parquet/JSON/JSONL)

    PROXIMAL PROCESSING (this script):
    1. Parse command-line arguments (data_files, hub_repo_id, config_name)
    2. Detect file format (Parquet, JSON, or JSONL)
    3. Load files as HuggingFace Dataset
    4. Upload to Hub with specified repo_id and config_name
    5. Print dataset URL

    DISTAL DESTINATION:
    └─> HuggingFace Hub → Public/private dataset for sharing

DATASET CONFIGS:
----------------
The --config_name parameter allows uploading multiple evaluation runs to the
same repository as different configs:

Example:
    myusername/my-model-eval-results
    ├─ Config: aime_2024 (AIME 2024 results)
    ├─ Config: math_500 (MATH-500 results)
    └─ Config: gpqa (GPQA results)

This keeps all evaluation results organized in one repository.

==============================================================================
IMPORTS AND SETUP
==============================================================================
"""

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
Push the details from a LightEval run to the Hub.

Usage:

python src/open_r1/utils/upload_details.py \
    --data_files {path_to_parquet_file} \
    --hub_repo_id {hub_repo_id} \
    --config_name {config_name}
"""

from dataclasses import dataclass, field
from typing import List

from datasets import load_dataset
from transformers import HfArgumentParser


"""
==============================================================================
WHAT: ScriptArguments
WHY:  Define command-line interface for upload script
HOW:  Dataclass with HfArgumentParser integration
==============================================================================

This dataclass defines three arguments:

1. data_files (List[str]):
   - Paths to local evaluation result files
   - Supports Parquet, JSON, or JSONL formats
   - Can be multiple files (e.g., different benchmarks)
   - Example: ["results_aime.parquet", "results_math.parquet"]

2. hub_repo_id (str):
   - HuggingFace Hub repository to upload to
   - Format: "username/repo-name"
   - Can be new (will be created) or existing
   - Example: "myusername/qwen-eval-results"

3. config_name (str):
   - Dataset configuration name (subset)
   - Allows multiple configs in one repository
   - Example: "aime_2024" or "math_500"

PROXIMAL CONTEXT: Parsed by HfArgumentParser in main()
DISTAL CONTEXT: Values passed to load_dataset and push_to_hub

The default_factory=list creates a new list for each instance (avoids
mutable default argument bug).
"""
@dataclass
class ScriptArguments:
    data_files: List[str] = field(default_factory=list)
    hub_repo_id: str = None
    config_name: str = None


"""
==============================================================================
WHAT: main()
WHY:  Upload evaluation results to HuggingFace Hub
HOW:  Load files, detect format, upload as dataset
==============================================================================

ARCHITECTURE:
------------
The main function has 4 steps:

1. Parse arguments
2. Detect file format (Parquet, JSON, or JSONL)
3. Load files as HuggingFace Dataset
4. Upload to Hub and print URL

PROXIMAL PROCESSING DETAILS:
----------------------------

Step 1: Parse Arguments
    Uses HfArgumentParser (transformers' argument parser):
    - Supports dataclasses (type-safe, documented)
    - Returns tuple, we extract first element (script args)

Step 2: Detect File Format
    Checks file extensions to determine format:
    - All .json → load as "json"
    - All .jsonl → load as "json" (same loader)
    - Otherwise → load as "parquet" (default)

    WHY ALL-OR-NOTHING?
    Mixed formats not supported by load_dataset. All files must be same type.

    WHY JSONL USES "json" LOADER?
    HuggingFace datasets library uses "json" for both JSON and JSONL.
    The loader auto-detects based on file structure.

Step 3: Load Dataset
    load_dataset(format, data_files=paths):
    - format: "json" or "parquet"
    - data_files: List of local file paths
    - Returns DatasetDict with "train" split by default

    WHY DatasetDict?
    load_dataset always returns dict of splits, even with one split.

Step 4: Upload to Hub
    ds.push_to_hub():
    - hub_repo_id: Where to upload
    - config_name: Subset name within repository
    - private=True: Create private dataset (not public)
    - Returns URL to uploaded dataset

    WHY private=True?
    Evaluation results may contain sensitive information or be incomplete.
    User can make public later via Hub UI if desired.

DISTAL IMPACT:
-------------
After upload:
- Dataset accessible at https://huggingface.co/datasets/{hub_repo_id}
- Can be loaded with: load_dataset(hub_repo_id, config_name)
- Can be shared, analyzed, or used for meta-studies
- Provides transparency for model evaluation claims
"""
def main():
    # WHAT: Parse command-line arguments into ScriptArguments object
    # WHY: Type-safe argument handling with automatic help text
    # HOW: HfArgumentParser converts CLI args to dataclass instance
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # WHAT: Detect file format based on extensions
    # WHY: load_dataset requires format specification
    # HOW: Check if all files end with .json or .jsonl, otherwise assume Parquet
    if all(file.endswith(".json") for file in args.data_files):
        # WHAT: Load JSON files
        # WHY: LightEval can output JSON format for evaluation results
        ds = load_dataset("json", data_files=args.data_files)
    elif all(file.endswith(".jsonl") for file in args.data_files):
        # WHAT: Load JSONL (JSON Lines) files
        # WHY: JSONL is common for large evaluation results (one result per line)
        # HOW: Same "json" loader handles both .json and .jsonl
        ds = load_dataset("json", data_files=args.data_files)
    else:
        # WHAT: Load Parquet files (default)
        # WHY: Parquet is efficient for large tabular data (typical for eval results)
        # HOW: load_dataset with "parquet" format
        ds = load_dataset("parquet", data_files=args.data_files)

    # WHAT: Upload dataset to HuggingFace Hub
    # WHY: Share evaluation results for transparency and reproducibility
    # HOW: push_to_hub creates/updates repository with dataset
    url = ds.push_to_hub(args.hub_repo_id, config_name=args.config_name, private=True)

    # WHAT: Print dataset URL to console
    # WHY: User needs URL to access and share uploaded dataset
    print(f"Dataset available at: {url}")


"""
==============================================================================
WHAT: Main entry point
WHY:  Execute main() when script is run directly
==============================================================================
"""
if __name__ == "__main__":
    main()


"""
==============================================================================
KEY TAKEAWAYS
==============================================================================

1. **PURPOSE**: This script uploads detailed LightEval benchmark results to
   HuggingFace Hub for sharing, transparency, and reproducibility.

2. **ARCHITECTURE**: Simple 4-step pipeline:
   - Parse arguments (data_files, hub_repo_id, config_name)
   - Detect file format (JSON/JSONL/Parquet)
   - Load as HuggingFace Dataset
   - Upload to Hub with private=True

3. **FORMAT DETECTION**: Automatically detects file format:
   - All .json → "json" loader
   - All .jsonl → "json" loader (same as .json)
   - Otherwise → "parquet" loader (default)

   Mixed formats not supported - all files must be same type.

4. **PRIVACY**: Uploads as private dataset by default (private=True):
   - Protects potentially sensitive evaluation data
   - User can make public later via Hub UI
   - Prevents accidental public exposure of incomplete results

5. **DATASET CONFIGS**: --config_name enables organizing multiple evaluations:
   ```
   myusername/my-model-evals
   ├─ aime_2024 (AIME 2024 results)
   ├─ math_500 (MATH-500 results)
   └─ gpqa (GPQA results)
   ```

   All in one repo, accessed via config parameter.

6. **LIGHTEVAL INTEGRATION**: Designed for LightEval output:
   - LightEval saves Parquet/JSON with per-example results
   - This script uploads those files to Hub
   - Enables sharing beyond aggregate metrics

7. **SIMPLICITY**: Minimal script (56 lines) focused on one task:
   - No complex logic or error handling
   - Trusts load_dataset and push_to_hub to handle edge cases
   - Easy to understand and modify

8. **REPRODUCIBILITY**: Uploaded datasets enable:
   - Verifying published benchmark claims
   - Analyzing model failure modes
   - Meta-studies across multiple models
   - Debugging evaluation pipelines

==============================================================================
USAGE EXAMPLES
==============================================================================

Upload single Parquet file:

    $ python scripts/upload_details.py \
        --data_files eval_results/results_aime_2024.parquet \
        --hub_repo_id "myusername/my-model-eval-results" \
        --config_name "aime_2024"

    Dataset available at: https://huggingface.co/datasets/myusername/my-model-eval-results

Upload multiple Parquet files (merged):

    $ python scripts/upload_details.py \
        --data_files eval_results/results_part1.parquet eval_results/results_part2.parquet \
        --hub_repo_id "myusername/my-model-eval-results" \
        --config_name "math_500"

    Dataset available at: https://huggingface.co/datasets/myusername/my-model-eval-results

Upload JSON files:

    $ python scripts/upload_details.py \
        --data_files eval_results/results_aime.json \
        --hub_repo_id "myusername/my-model-eval-results" \
        --config_name "aime_2025"

    Dataset available at: https://huggingface.co/datasets/myusername/my-model-eval-results

Upload JSONL files:

    $ python scripts/upload_details.py \
        --data_files eval_results/results_gpqa.jsonl \
        --hub_repo_id "myusername/my-model-eval-results" \
        --config_name "gpqa"

    Dataset available at: https://huggingface.co/datasets/myusername/my-model-eval-results

Upload with glob pattern (shell expands to file list):

    $ python scripts/upload_details.py \
        --data_files eval_results/*.parquet \
        --hub_repo_id "myusername/all-eval-results" \
        --config_name "combined"

    Dataset available at: https://huggingface.co/datasets/myusername/all-eval-results

Loading uploaded dataset:

    from datasets import load_dataset

    # Load specific config
    ds = load_dataset("myusername/my-model-eval-results", "aime_2024")

    # Analyze results
    correct = ds["train"].filter(lambda x: x["correct"])
    print(f"Accuracy: {len(correct) / len(ds['train']):.2%}")

Making dataset public:

    1. Upload as private (default)
    2. Review dataset on Hub
    3. Go to Settings → Visibility → Make Public
    4. Share URL with community

==============================================================================
"""
