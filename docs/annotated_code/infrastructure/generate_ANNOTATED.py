"""
==============================================================================
FILE: src/open_r1/generate.py
CATEGORY: Infrastructure - Data Generation Pipeline
PRIORITY: HIGH
LINES: 209
DEPENDENCIES:
    - distilabel: Pipeline framework for LLM data generation
    - datasets: HuggingFace datasets library
==============================================================================

OVERVIEW:
This module builds Distilabel pipelines for generating synthetic reasoning
traces using vLLM. It's used to create training datasets by generating
completions from a strong model (e.g., DeepSeek-R1).

ROLE IN DEEPSEEK R1:
- Creates synthetic training data for SFT (Stage 1)
- Generates diverse reasoning traces at scale
- Enables self-improvement via synthetic data
- Supports multi-generation for diversity

KEY FEATURES:
1. Distilabel Integration: Async generation pipeline
2. vLLM Backend: Fast generation via OpenAI-compatible API
3. Ray Parallelism: Scales across cores/machines
4. Batching: Efficient processing of large datasets
5. Retry Logic: Handles failures gracefully

DATA FLOW:
Input prompts → Distilabel Pipeline → vLLM Server → Generated completions
    → Dataset with generations → Push to Hub → Training data for SFT/GRPO
==============================================================================
"""

# Copyright 2025 The HuggingFace Team. All rights reserved.
# [License omitted for brevity]

from typing import Optional

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration


"""
==============================================================================
PIPELINE BUILDER
==============================================================================
"""


def build_distilabel_pipeline(
    model: str,
    base_url: str = "http://localhost:8000/v1",
    prompt_column: Optional[str] = None,
    prompt_template: str = "{{ instruction }}",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: int = 8192,
    num_generations: int = 1,
    input_batch_size: int = 64,
    client_replicas: int = 1,
    timeout: int = 900,
    retries: int = 0,
) -> Pipeline:
    """
    WHAT: Builds a Distilabel pipeline for generating responses using a vLLM server

    WHY: Distilabel provides efficient async generation at scale
         Ray backend enables parallelism across cores/machines
         vLLM server provides fast inference

    HOW:
        1. Create Pipeline with Ray backend
        2. Add TextGeneration step with OpenAILLM client
        3. Configure sampling parameters (temperature, top_p, etc.)
        4. Set batching and replica parameters
        5. Return configured pipeline

    TYPICAL USE CASE:
        Generate reasoning traces from DeepSeek-R1 to create training data
        for distillation (Mixture-of-Thoughts dataset)

    PROXIMAL CONTEXT:
        - Input: Configuration parameters
        - Output: Configured Distilabel Pipeline

    DISTAL CONTEXT:
        - Originates from: Command-line args or script config
        - Flows to: pipeline.run(dataset) → Generated dataset → Hub

    Args:
        model: Model name (for logging, not actual loading)
        base_url: vLLM server URL (OpenAI-compatible API)
        prompt_column: Dataset column to use for prompts (if not "instruction")
        prompt_template: Jinja2 template for formatting prompts
        temperature: Sampling temperature (None = model default)
        top_p: Nucleus sampling threshold (None = model default)
        max_new_tokens: Maximum tokens to generate per completion
        num_generations: Number of completions per prompt (for diversity)
        input_batch_size: Batch size for processing prompts
        client_replicas: Number of Ray replicas for parallelism
        timeout: Request timeout in seconds (900 = 15 minutes)
        retries: Number of retries for failed requests

    Returns:
        Configured Distilabel Pipeline ready to run

    EXAMPLE:
        pipeline = build_distilabel_pipeline(
            model="deepseek-ai/DeepSeek-R1",
            base_url="http://localhost:8000/v1",
            temperature=0.7,
            top_p=0.9,
            num_generations=16,
            input_batch_size=64,
        )
        dataset = load_dataset("open-r1/math-problems", split="train")
        distiset = pipeline.run(dataset=dataset)
    """

    # STEP 1: Build generation_kwargs dict
    # WHY: Only include parameters that are explicitly set (not None)
    generation_kwargs = {"max_new_tokens": max_new_tokens}

    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    # STEP 2: Create pipeline with Ray backend
    # WHY: Ray enables distributed processing across cores/machines
    # HOW: with Pipeline().ray() as pipeline: creates Ray-enabled pipeline
    with Pipeline().ray() as pipeline:
        TextGeneration(
            # LLM CONFIGURATION
            # WHAT: OpenAILLM client pointing to vLLM server
            # WHY: vLLM provides OpenAI-compatible API
            # HOW: base_url points to vLLM server (typically http://localhost:8000/v1)
            llm=OpenAILLM(
                base_url=base_url,
                api_key="something",  # WHAT: vLLM doesn't require auth, but OpenAI client needs a value
                model=model,  # WHAT: Model name (for logging only, vLLM server already loaded model)
                timeout=timeout,  # WHAT: Request timeout (15 minutes default for long reasoning)
                max_retries=retries,  # WHAT: Retry failed requests
                generation_kwargs=generation_kwargs,
            ),

            # PROMPT TEMPLATE
            # WHAT: Jinja2 template for formatting prompts
            # WHY: Enables flexible prompt formatting
            # DEFAULT: "{{ instruction }}" uses dataset's "instruction" column
            template=prompt_template,

            # INPUT MAPPING
            # WHAT: Maps dataset columns to template variables
            # WHY: Allows using different column names (e.g., "prompt" instead of "instruction")
            # HOW: If prompt_column provided, maps it to "instruction" template variable
            input_mappings=({"instruction": prompt_column} if prompt_column is not None else {}),

            # BATCHING
            # WHAT: Number of prompts to process in one batch
            # WHY: Larger batches improve throughput
            # TYPICAL: 64 prompts per batch
            input_batch_size=input_batch_size,

            # NUM_GENERATIONS
            # WHAT: Generate multiple completions per prompt
            # WHY: Enables diversity in generated data
            # TYPICAL: 16 generations for training data
            num_generations=num_generations,

            # GROUPING
            # WHAT: Keep all generations for same prompt together
            # WHY: Easier to analyze and filter generated data
            group_generations=True,

            # RAY RESOURCES
            # WHAT: Number of Ray replicas (parallel workers)
            # WHY: More replicas = faster generation (if you have CPU/GPU resources)
            # TYPICAL: 1 replica per vLLM server
            resources=StepResources(replicas=client_replicas),
        )

    return pipeline


"""
==============================================================================
COMMAND-LINE INTERFACE
==============================================================================
"""

if __name__ == "__main__":
    """
    WHAT: Command-line interface for running generation pipeline

    WHY: Enables easy generation from command line without writing code

    USAGE:
        python src/open_r1/generate.py \
            --hf-dataset open-r1/verifiable-coding-problems \
            --model deepseek-ai/DeepSeek-R1 \
            --vllm-server-url http://localhost:8000/v1 \
            --temperature 0.7 \
            --num-generations 16 \
            --hf-output-dataset open-r1/generated-data

    WORKFLOW:
        1. Parse command-line arguments
        2. Load input dataset from HuggingFace Hub
        3. Build Distilabel pipeline
        4. Run generation
        5. Push results to Hub
    """

    import argparse
    from datasets import load_dataset

    # ARGUMENT PARSER SETUP
    parser = argparse.ArgumentParser(description="Run distilabel pipeline for generating responses with DeepSeek R1")

    # DATASET ARGUMENTS
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="HuggingFace dataset to load",
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        required=False,
        help="Dataset config to use",
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{{ instruction }}",
        help="Template string for formatting prompts.",
    )

    # MODEL AND SERVER ARGUMENTS
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for generation",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the vLLM server",
    )

    # GENERATION PARAMETERS
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p value for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Number of generations per problem",
    )

    # INFRASTRUCTURE PARAMETERS
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=64,
        help="Batch size for input processing",
    )
    parser.add_argument(
        "--client-replicas",
        type=int,
        default=1,
        help="Number of client replicas for parallel processing",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries for failed requests (default: 0)",
    )

    # OUTPUT ARGUMENTS
    parser.add_argument(
        "--hf-output-dataset",
        type=str,
        required=False,
        help="HuggingFace repo to push results to",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the output dataset private when pushing to HF Hub",
    )

    args = parser.parse_args()

    # LOG CONFIGURATION
    print("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # STEP 1: LOAD INPUT DATASET
    # WHY: Need prompts to generate completions from
    # DATA FLOW: HF Hub → Input dataset
    print(f"Loading '{args.hf_dataset}' (config: {args.hf_dataset_config}, split: {args.hf_dataset_split}) dataset...")
    dataset = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
    print("Dataset loaded!")

    # STEP 2: BUILD PIPELINE
    # WHY: Configure generation with specified parameters
    pipeline = build_distilabel_pipeline(
        model=args.model,
        base_url=args.vllm_server_url,
        prompt_template=args.prompt_template,
        prompt_column=args.prompt_column,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )

    # STEP 3: RUN GENERATION
    # WHY: Generate completions for all prompts
    # HOW:
    #   - Ray distributes work across replicas
    #   - Each replica processes batches of prompts
    #   - vLLM server generates completions
    #   - Results collected into Distiset
    # DATA FLOW: Input dataset → vLLM → Distiset (dataset with generations)
    print("Running generation pipeline...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,  # WHAT: Large batches for Ray
        use_cache=False,  # WHAT: Don't cache results (for fresh generation)
    )
    print("Generation pipeline finished!")

    # STEP 4: PUSH TO HUB (OPTIONAL)
    # WHY: Share generated data, use for training
    # DATA FLOW: Distiset → HF Hub → Training dataset
    if args.hf_output_dataset:
        print(f"Pushing resulting dataset to '{args.hf_output_dataset}'...")
        distiset.push_to_hub(args.hf_output_dataset, private=args.private)
        print("Dataset pushed!")


"""
==============================================================================
KEY TAKEAWAYS - DATA GENERATION
==============================================================================

1. **vLLM is Critical for Speed**:
   - vLLM provides fast inference (PagedAttention, continuous batching)
   - OpenAI-compatible API makes integration easy
   - Must launch vLLM server before running generation

2. **Distilabel Scales Generation**:
   - Ray backend enables parallelism
   - Async processing improves throughput
   - Handles retries and errors gracefully

3. **num_generations for Diversity**:
   - Generate multiple completions per prompt (typically 16)
   - Enables filtering by quality (keep best N)
   - More diverse training data

4. **Batching Improves Throughput**:
   - input_batch_size=64 is typical
   - Larger batches better utilize GPU
   - Trade-off: memory vs throughput

5. **Temperature Controls Diversity**:
   - temperature=0.7: Good balance for reasoning
   - temperature=1.0: More diverse, less focused
   - temperature=0.0: Deterministic (greedy)

6. **Timeout is Important**:
   - Long reasoning chains can take 5+ minutes
   - 15-minute timeout (900s) is safe
   - Adjust based on max_new_tokens

7. **Typical Workflow**:
   1. Launch vLLM server with strong model
   2. Load seed dataset (problems/prompts)
   3. Generate completions (16 per prompt)
   4. Filter by quality (pass rate, accuracy)
   5. Push to Hub
   6. Use for SFT training

8. **Generated Dataset Format**:
   - Original columns preserved
   - Added columns: generation, model_name, etc.
   - group_generations=True keeps them together

==============================================================================
"""
