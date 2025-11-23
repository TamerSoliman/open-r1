# Comprehensive Annotation Plan for DeepSeek R1 Implementation

This document outlines the plan for creating comprehensively annotated versions of key files in the Open R1 (DeepSeek R1) repository.

## Annotation Guidelines

Each annotated file will include:

### The WHAT
- Purpose and role of each function/class/module
- Input/output specifications
- Data structures and their meanings
- Key variables and their purposes

### The WHY
- Design decisions and rationale
- Why specific algorithms or approaches were chosen
- Connection to DeepSeek R1 paper concepts
- Trade-offs and alternatives considered

### The HOW
- Step-by-step logic flow
- Algorithm implementations
- Integration with other components
- Performance considerations

### Data Flow Context
- **Proximal context**: How data flows in/out of immediate function
- **Distal context**: Where data originates and where it goes in the larger pipeline
- **State management**: How state changes throughout the pipeline

---

## Part 1: Core Training Infrastructure (10 files)

### 1. `src/open_r1/grpo.py` (181 lines)
**Priority: CRITICAL**
- **What**: GRPO (Group Relative Policy Optimization) training loop
- **Why annotate**: Heart of the RL training - implements the key innovation
- **Key concepts to explain**:
  - Group relative advantages vs absolute rewards
  - Policy gradient optimization
  - Integration with vLLM for generation
  - Multi-generation per prompt strategy
  - How GRPO differs from PPO/RLHF
- **Data flow**: Dataset → Prompt → vLLM generation → Reward computation → Advantage calculation → Policy update
- **Connection to paper**: DeepSeek R1 paper sections on RL training

### 2. `src/open_r1/sft.py` (169 lines)
**Priority: CRITICAL**
- **What**: Supervised Fine-Tuning trainer for distillation
- **Why annotate**: Step 1 of the 3-stage pipeline
- **Key concepts to explain**:
  - Chat template setup and formatting
  - Gradient checkpointing for memory efficiency
  - LoRA/QLoRA integration
  - Packing vs non-packing strategies
  - System prompt injection for reasoning format
- **Data flow**: Raw dataset → Tokenization → Batching → Training loop → Checkpointing
- **Connection to paper**: Distillation from strong to weak models

### 3. `src/open_r1/generate.py` (208 lines)
**Priority: HIGH**
- **What**: Data generation pipeline using Distilabel
- **Why annotate**: Creates synthetic reasoning traces for training
- **Key concepts to explain**:
  - Distilabel pipeline architecture
  - Async generation with replicas
  - Temperature/top_p sampling strategies
  - Batch processing and retry logic
  - Output formatting (JSONL)
- **Data flow**: Seed prompts → vLLM API → Generated completions → Filtering → Training dataset
- **Connection to paper**: Self-improvement through synthetic data generation

### 4. `src/open_r1/rewards.py` (706 lines)
**Priority: CRITICAL**
- **What**: 20+ reward functions for RL training
- **Why annotate**: Defines what behaviors are reinforced
- **Key concepts to explain**:
  - Math answer verification (LaTeX parsing, exact match)
  - Format rewards for structured reasoning
  - Code execution rewards with multiple backends
  - Length-based rewards and cosine scaling (from Kimi 1.5)
  - Repetition penalty implementation
  - Soft overlong punishment
  - Reward composition and weighting
- **Data flow**: Model completion → Reward function → Scalar reward → Advantage calculation
- **Connection to paper**: Multi-objective RL with math/code/reasoning rewards

### 5. `src/open_r1/configs.py` (331 lines)
**Priority: HIGH**
- **What**: Configuration dataclasses for all training modes
- **Why annotate**: Central configuration system, defines all hyperparameters
- **Key concepts to explain**:
  - ScriptArguments: Dataset mixtures and column selection
  - SFTConfig: SFT-specific params
  - GRPOConfig: RL-specific params (num_generations, reward weights)
  - Validation logic for dataset mixtures
  - Relationship between config and actual training behavior
- **Data flow**: YAML config → Dataclass validation → Training script parameters
- **Connection to paper**: Hyperparameters from DeepSeek experiments

### 6. `src/open_r1/utils/model_utils.py` (42 lines)
**Priority: MEDIUM**
- **What**: Model and tokenizer loading utilities
- **Why annotate**: Handles quantization, attention implementations, device mapping
- **Key concepts to explain**:
  - BitsAndBytes quantization (4-bit, 8-bit)
  - Flash Attention 2 vs SDPA vs Eager
  - Device mapping for quantized models
  - Trust remote code implications
  - Cache configuration for training
- **Data flow**: Model name → HF Hub download → Quantization → Device placement → Ready model
- **Connection to implementation**: Enables training large models on limited hardware

### 7. `src/open_r1/utils/data.py` (65 lines)
**Priority: HIGH**
- **What**: Dataset loading and mixture support
- **Why annotate**: Core data infrastructure for multi-dataset training
- **Key concepts to explain**:
  - Single dataset vs mixture loading
  - Weighted sampling from multiple datasets
  - Column selection and standardization
  - Train/test splitting strategies
  - Shuffling and interleaving
- **Data flow**: Dataset specs → HF Hub → Loading → Mixing → Sampling → Training batches
- **Connection to paper**: Multi-task training with dataset mixtures

### 8. `src/open_r1/utils/evaluation.py` (118 lines)
**Priority: HIGH**
- **What**: LightEval integration for benchmarking
- **Why annotate**: Standardized evaluation across math/code/reasoning tasks
- **Key concepts to explain**:
  - LightEval task specifications
  - vLLM backend for efficient inference
  - Tensor parallelism for large models
  - Results saving and Hub integration
  - Benchmark task definitions (MATH-500, AIME, GPQA, LCB)
- **Data flow**: Trained model → Task prompts → vLLM inference → Answer evaluation → Metrics
- **Connection to paper**: Reproducing DeepSeek evaluation results

### 9. `src/open_r1/utils/callbacks.py` (92 lines)
**Priority: MEDIUM**
- **What**: Training callbacks for Hub pushing and evaluation
- **Why annotate**: Automates model versioning and continuous evaluation
- **Key concepts to explain**:
  - PushToHubRevisionCallback: Branch management
  - Automatic evaluation triggering
  - Checkpoint tracking
  - SLURM job submission from callback
- **Data flow**: Training checkpoint → Hub push → Evaluation SLURM job → Results tracking
- **Connection to workflow**: Enables continuous integration/deployment

### 10. `src/open_r1/utils/code_providers.py` (366 lines)
**Priority: CRITICAL**
- **What**: Pluggable code execution backends (E2B, MorphCloud, Piston)
- **Why annotate**: Core infrastructure for code reward functions
- **Key concepts to explain**:
  - Abstract CodeExecutionProvider interface
  - E2B sandboxes: security, language support, timeout handling
  - MorphCloud API integration
  - Piston worker pools for competitive programming
  - Router pattern for batch execution
  - Error handling and timeout management
- **Data flow**: Code string → Sandbox creation → Execution → Output/error → Pass/fail determination
- **Connection to paper**: Enables code-based RL training

---

## Part 2: Competitive Programming Infrastructure (7 files)

### 11. `src/open_r1/utils/competitive_programming/ioi_scoring.py`
**Priority: HIGH**
- **What**: International Olympiad in Informatics (IOI) problem evaluation
- **Why annotate**: Evaluates on world's most challenging programming problems
- **Key concepts to explain**:
  - Subtask-based scoring system
  - Test result types (AC, WA, TLE, RE, MLE, CE)
  - Point allocation by subtask weight
  - Batch execution and aggregation
  - IOI-specific test format
- **Data flow**: Code submission → Subtask test cases → Execution → Result aggregation → Score
- **Connection to paper**: OlympicCoder model evaluation

### 12. `src/open_r1/utils/competitive_programming/cf_scoring.py`
**Priority: HIGH**
- **What**: Codeforces problem evaluation
- **Why annotate**: Contest-style programming evaluation
- **Key concepts to explain**:
  - Generated test case format (parquet)
  - Scoring modes: pass_fail, partial, weighted_sum
  - Language-specific compilation
  - Input/output matching
  - Batch processing across problems
- **Data flow**: Problem spec → Test generation → Code execution → Output comparison → Score
- **Connection to paper**: Codeforces benchmark in DeepSeek paper

### 13. `src/open_r1/utils/competitive_programming/code_patcher.py`
**Priority: MEDIUM**
- **What**: Auto-completion of code submissions
- **Why annotate**: Handles incomplete/partial code from models
- **Key concepts to explain**:
  - Pattern matching for code completion
  - Language-specific patching logic
  - Template insertion
  - Edge case handling
- **Data flow**: Raw model output → Pattern detection → Template insertion → Complete code
- **Connection to workflow**: Improves robustness of code evaluation

### 14. `src/open_r1/utils/competitive_programming/ioi_utils.py`
**Priority: MEDIUM**
- **What**: IOI test loading and include file injection
- **Why annotate**: Handles IOI-specific test format
- **Key concepts to explain**:
  - Test file structure parsing
  - Include file injection for C++
  - Subtask configuration
  - Error handling for malformed tests
- **Data flow**: IOI test directory → Parse structure → Inject includes → Test list
- **Connection to workflow**: Enables IOI evaluation

### 15. `src/open_r1/utils/competitive_programming/piston_client.py`
**Priority: MEDIUM**
- **What**: Piston code execution API client
- **Why annotate**: Low-level interface to Piston workers
- **Key concepts to explain**:
  - Piston API format (files, language, version)
  - Synchronous execution
  - Output parsing
  - Error handling
- **Data flow**: Code + inputs → Piston API → Execution → Stdout/stderr
- **Connection to infrastructure**: Enables local code execution

### 16. `src/open_r1/utils/competitive_programming/morph_client.py`
**Priority: MEDIUM**
- **What**: MorphCloud API client
- **Why annotate**: Alternative to E2B for code execution
- **Key concepts to explain**:
  - MorphCloud API authentication
  - Sandbox lifecycle (create, execute, delete)
  - Async operations
  - Rate limiting
- **Data flow**: Code → MorphCloud API → Sandbox → Execution → Results
- **Connection to infrastructure**: Provides scalable code execution

### 17. `src/open_r1/utils/competitive_programming/__init__.py`
**Priority: LOW**
- **What**: Module initialization and exports
- **Why annotate**: Shows module interface
- **Key concepts to explain**:
  - Exported functions and classes
  - Module organization
- **Data flow**: N/A (structural)
- **Connection to infrastructure**: Module interface definition

---

## Part 3: Utility and Infrastructure Files (8 files)

### 18. `src/open_r1/utils/hub.py` (132 lines)
**Priority: MEDIUM**
- **What**: HuggingFace Hub integration utilities
- **Why annotate**: Handles model versioning and parameter counting
- **Key concepts to explain**:
  - Branch-based checkpoint management
  - SafeTensors metadata parsing for param count
  - Revision existence checking
  - GPU count estimation for vLLM
  - Hub API patterns
- **Data flow**: Local checkpoint → Hub upload → Branch creation → Metadata indexing
- **Connection to workflow**: Enables model sharing and reproducibility

### 19. `src/open_r1/utils/wandb_logging.py` (13 lines)
**Priority: LOW**
- **What**: Weights & Biases initialization
- **Why annotate**: Sets up experiment tracking
- **Key concepts to explain**:
  - Environment variable configuration
  - Entity/project/run group organization
- **Data flow**: Training args → W&B env vars → W&B dashboard
- **Connection to workflow**: Enables experiment tracking

### 20. `src/open_r1/utils/routed_sandbox.py`
**Priority: MEDIUM**
- **What**: E2B router client for batch code execution
- **Why annotate**: Enables scalable batch execution
- **Key concepts to explain**:
  - Router pattern for load balancing
  - Batch request formatting
  - Async HTTP requests
  - Error handling and retry logic
- **Data flow**: Batch of code → Router API → Worker pool → Results aggregation
- **Connection to infrastructure**: Scales code execution

### 21. `src/open_r1/utils/routed_morph.py`
**Priority: MEDIUM**
- **What**: MorphCloud router client for batch execution
- **Why annotate**: Alternative to E2B router
- **Key concepts to explain**:
  - Similar router pattern
  - MorphCloud-specific API format
  - Batch processing
- **Data flow**: Batch of code → Router API → MorphCloud workers → Results
- **Connection to infrastructure**: Alternative scaling solution

### 22. `src/open_r1/__init__.py`
**Priority: LOW**
- **What**: LightEval task registry
- **Why annotate**: Registers custom evaluation tasks
- **Key concepts to explain**:
  - LightEval task registration
  - Custom task definitions
  - Module initialization
- **Data flow**: Task definitions → LightEval registry → Evaluation tasks
- **Connection to workflow**: Enables custom benchmarks

### 23. `setup.py`
**Priority: MEDIUM**
- **What**: Package installation and dependency management
- **Why annotate**: Defines all dependencies and installation options
- **Key concepts to explain**:
  - Dependency versions and constraints
  - Optional extras (code, eval, tests, dev)
  - CUDA requirements
  - Package structure
- **Data flow**: pip install → dependency resolution → environment setup
- **Connection to workflow**: Critical for reproducibility

### 24. `Makefile`
**Priority: LOW**
- **What**: Development command shortcuts
- **Why annotate**: Common workflows and commands
- **Key concepts to explain**:
  - Installation commands
  - Quality checks (ruff, isort)
  - Testing commands
  - Evaluation shortcuts
- **Data flow**: make command → script execution
- **Connection to workflow**: Developer convenience

### 25. `pyproject.toml`
**Priority: LOW**
- **What**: Modern Python project configuration
- **Why annotate**: Tool configurations (ruff, pytest, etc.)
- **Key concepts to explain**:
  - Tool configurations
  - Code quality standards
  - Test settings
- **Data flow**: N/A (configuration)
- **Connection to workflow**: Defines code standards

---

## Part 4: Training Recipes and Configurations (12 files)

### 26. `recipes/OpenR1-Distill-7B/sft/config_distill.yaml`
**Priority: CRITICAL**
- **What**: Step 1 distillation configuration (DeepSeek → Qwen)
- **Why annotate**: Reference configuration for distillation phase
- **Key concepts to explain**:
  - Qwen2.5-Math-7B base model choice
  - Mixture-of-Thoughts dataset
  - 32k context length for long reasoning
  - Cosine learning rate schedule
  - System prompt for <think>/<answer> format
  - Gradient accumulation strategy
- **Data flow**: Config → Training script → Model training
- **Connection to paper**: DeepSeek distillation methodology

### 27. `recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml`
**Priority: CRITICAL**
- **What**: Step 2 GRPO configuration (RL on math tasks)
- **Why annotate**: Reference configuration for RL phase
- **Key concepts to explain**:
  - 1.5B model for faster iteration
  - OpenR1-Math-220k dataset
  - num_generations=16 (multiple attempts per prompt)
  - Reward function selection (accuracy, format, tag_count)
  - Learning rate for RL (2e-5)
  - Generation parameters (temperature, top_p)
- **Data flow**: Config → GRPO script → RL training
- **Connection to paper**: RL fine-tuning phase

### 28. `recipes/Qwen2.5-Coder-7B-Instruct/grpo/config_codeforces.yaml`
**Priority: HIGH**
- **What**: Code-focused GRPO configuration (Codeforces)
- **Why annotate**: Shows code-specific RL setup
- **Key concepts to explain**:
  - Qwen2.5-Coder as base (pre-trained on code)
  - Codeforces verifiable-prompts dataset
  - cf_code reward function
  - High gradient accumulation for large batches
  - Code-specific system prompts
- **Data flow**: Config → GRPO script → Code RL training
- **Connection to paper**: Code capabilities in DeepSeek R1

### 29. `recipes/OlympicCoder-7B/grpo/config_ioi.yaml`
**Priority: HIGH**
- **What**: IOI-focused GRPO configuration
- **Why annotate**: Most challenging coding benchmark
- **Key concepts to explain**:
  - IOI dataset (Olympic-level problems)
  - ioi_code reward function
  - Subtask-based evaluation
  - Long context for complex problems
- **Data flow**: Config → GRPO script → IOI training
- **Connection to paper**: OlympicCoder model results

### 30. `recipes/OlympicCoder-32B/grpo/config_ioi_32b.yaml`
**Priority: MEDIUM**
- **What**: Large model IOI configuration
- **Why annotate**: Shows scaling to 32B parameters
- **Key concepts to explain**:
  - FSDP for model parallelism
  - Reduced batch size per device
  - Higher gradient accumulation
  - Memory optimization techniques
- **Data flow**: Config → GRPO script → Large model training
- **Connection to paper**: Scaling laws and large model training

### 31. `recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_r1_distill.yaml`
**Priority: HIGH**
- **What**: GRPO on DeepSeek-R1 distilled base
- **Why annotate**: Multi-stage pipeline (distill + RL)
- **Key concepts to explain**:
  - Starting from distilled checkpoint
  - Further RL optimization
  - Combined benefits of distillation and RL
- **Data flow**: Config → GRPO script → Multi-stage training
- **Connection to paper**: Complete 3-stage pipeline

### 32. `recipes/accelerate_configs/ddp.yaml`
**Priority: MEDIUM**
- **What**: Data Parallel (DDP) configuration
- **Why annotate**: Simplest multi-GPU setup
- **Key concepts to explain**:
  - DDP vs other parallelism strategies
  - When to use DDP (small models, few GPUs)
  - Communication overhead
- **Data flow**: Config → Accelerate → Training
- **Connection to infrastructure**: Distributed training

### 33. `recipes/accelerate_configs/zero2.yaml`
**Priority: HIGH**
- **What**: DeepSpeed ZeRO-2 configuration
- **Why annotate**: Medium model parallelism
- **Key concepts to explain**:
  - ZeRO-2 optimizer state sharding
  - Gradient sharding
  - Memory savings vs communication cost
  - When to use ZeRO-2 (7B-13B models)
- **Data flow**: Config → DeepSpeed → Training
- **Connection to infrastructure**: Memory-efficient training

### 34. `recipes/accelerate_configs/zero3.yaml`
**Priority**: CRITICAL**
- **What**: DeepSpeed ZeRO-3 configuration
- **Why annotate**: Default for large models, most important config
- **Key concepts to explain**:
  - Full model sharding (parameters + gradients + optimizer)
  - Stage 3 offloading options
  - Communication patterns
  - When to use ZeRO-3 (7B+ models)
  - Trade-offs: memory vs speed
- **Data flow**: Config → DeepSpeed → Training
- **Connection to paper**: Enables training large reasoning models

### 35. `recipes/accelerate_configs/fsdp.yaml`
**Priority: HIGH**
- **What**: Fully Sharded Data Parallel (FSDP) configuration
- **Why annotate**: PyTorch native alternative to ZeRO-3
- **Key concepts to explain**:
  - FSDP vs ZeRO-3
  - Sharding strategies
  - Mixed precision settings
  - When to use FSDP (32B+ models)
- **Data flow**: Config → PyTorch FSDP → Training
- **Connection to infrastructure**: Large model training

### 36. `recipes/dataset_filtering/config_filter_by_pass_rate.yaml`
**Priority: MEDIUM**
- **What**: Pass rate filtering configuration
- **Why annotate**: Data quality improvement
- **Key concepts to explain**:
  - Pass rate as quality metric
  - Threshold selection
  - Filtering strategy
  - Dataset refinement
- **Data flow**: Config → Filtering script → Refined dataset
- **Connection to workflow**: Improves training data quality

### 37. `recipes/README.md`
**Priority: LOW**
- **What**: Recipe documentation
- **Why annotate**: Training commands and examples
- **Key concepts to explain**:
  - How to use recipes
  - Command-line arguments
  - Resource requirements
- **Data flow**: Documentation → User commands
- **Connection to workflow**: User guide

---

## Part 5: SLURM Infrastructure (6 files)

### 38. `slurm/train.slurm`
**Priority: CRITICAL**
- **What**: Main SLURM training script
- **Why annotate**: Orchestrates entire training workflow on HF cluster
- **Key concepts to explain**:
  - Weka filesystem caching
  - vLLM server launch for GRPO
  - Checkpoint detection and resumption
  - DP/TP configuration for vLLM
  - Environment setup
  - Log management
- **Data flow**: sbatch → Resource allocation → vLLM server → Training → Checkpoints
- **Connection to infrastructure**: Production training on HF cluster

### 39. `slurm/generate.slurm`
**Priority: HIGH**
- **What**: Data generation SLURM script
- **Why annotate**: Scales synthetic data generation
- **Key concepts to explain**:
  - vLLM server for generation
  - Distilabel pipeline execution
  - Resource allocation for generation
  - Output dataset management
- **Data flow**: sbatch → vLLM server → Generation → Dataset
- **Connection to workflow**: Creates training datasets

### 40. `slurm/evaluate.slurm`
**Priority: HIGH**
- **What**: Evaluation SLURM script
- **Why annotate**: Automated benchmarking
- **Key concepts to explain**:
  - LightEval execution
  - Model loading and inference
  - Results aggregation
  - Hub upload
- **Data flow**: sbatch → Model loading → Evaluation → Results
- **Connection to workflow**: Continuous evaluation

### 41. `slurm/serve_r1.slurm`
**Priority: MEDIUM**
- **What**: Model serving SLURM script
- **Why annotate**: Production inference setup
- **Key concepts to explain**:
  - vLLM server configuration
  - Tensor/data parallelism
  - OpenAI-compatible API
  - Resource allocation
- **Data flow**: sbatch → vLLM server → API endpoint
- **Connection to workflow**: Model deployment

### 42. `slurm/piston/launch_piston_workers.sh`
**Priority: MEDIUM**
- **What**: Piston worker pool launcher
- **Why annotate**: Scales code execution for competitive programming
- **Key concepts to explain**:
  - Worker pool architecture
  - SLURM array jobs
  - Environment configuration
  - Package installation (IOI, Codeforces)
- **Data flow**: Script → SLURM array → Worker pool
- **Connection to infrastructure**: Code execution scaling

### 43. `slurm/piston/launch_single_piston.sh`
**Priority: LOW**
- **What**: Single Piston worker launcher
- **Why annotate**: Worker process setup
- **Key concepts to explain**:
  - Docker container setup
  - Port allocation
  - Package mounting
- **Data flow**: Script → Docker → Piston worker
- **Connection to infrastructure**: Worker implementation

---

## Part 6: Scripts and Utilities (7 files)

### 44. `scripts/generate_reasoning.py`
**Priority: HIGH**
- **What**: Async data generation with retry logic
- **Why annotate**: Production-grade generation pipeline
- **Key concepts to explain**:
  - Async HTTP requests to vLLM
  - Batch processing
  - Retry logic and error handling
  - JSONL output format
  - Streaming writes
- **Data flow**: Prompts → Async vLLM requests → Completions → JSONL
- **Connection to workflow**: Scalable data generation

### 45. `scripts/decontaminate.py`
**Priority: MEDIUM**
- **What**: Dataset decontamination
- **Why annotate**: Prevents benchmark leakage
- **Key concepts to explain**:
  - N-gram matching
  - Contamination detection
  - Dataset filtering
  - Benchmark overlap removal
- **Data flow**: Training dataset + benchmarks → Matching → Clean dataset
- **Connection to workflow**: Ensures fair evaluation

### 46. `scripts/e2b_router.py`
**Priority: MEDIUM**
- **What**: E2B router service
- **Why annotate**: Load balancing for code execution
- **Key concepts to explain**:
  - HTTP server setup
  - Worker pool management
  - Request routing
  - Error handling
- **Data flow**: HTTP requests → Router → E2B workers → Responses
- **Connection to infrastructure**: Scales E2B execution

### 47. `scripts/morph_router.py`
**Priority: MEDIUM**
- **What**: MorphCloud router service
- **Why annotate**: Load balancing for MorphCloud
- **Key concepts to explain**:
  - Similar to E2B router
  - MorphCloud-specific logic
  - API authentication
- **Data flow**: HTTP requests → Router → MorphCloud → Responses
- **Connection to infrastructure**: Scales MorphCloud execution

### 48. `scripts/run_benchmarks.py`
**Priority: MEDIUM**
- **What**: Benchmark runner script
- **Why annotate**: Automated benchmark execution
- **Key concepts to explain**:
  - Task selection
  - Model loading
  - Result aggregation
  - Comparison with baselines
- **Data flow**: Model + tasks → Evaluation → Results
- **Connection to workflow**: Standardized evaluation

### 49. `scripts/pass_rate_filtering/compute_pass_rate.py`
**Priority: HIGH**
- **What**: Pass rate computation for code problems
- **Why annotate**: Data quality metric calculation
- **Key concepts to explain**:
  - Pass@k metric
  - Multiple generations per problem
  - Aggregation across problems
  - Threshold-based filtering
- **Data flow**: Generated solutions → Execution → Pass rate → Filtering
- **Connection to workflow**: Improves training data

### 50. `scripts/upload_details.py`
**Priority: LOW**
- **What**: Upload evaluation details to Hub
- **Why annotate**: Result sharing and transparency
- **Key concepts to explain**:
  - Hub API usage
  - Result formatting
  - Metadata attachment
- **Data flow**: Evaluation results → Formatting → Hub upload
- **Connection to workflow**: Result sharing

---

## Summary Statistics

### By Priority:
- **CRITICAL**: 11 files (core training, rewards, GRPO, configs, key recipes)
- **HIGH**: 16 files (competitive programming, data, evaluation, important configs)
- **MEDIUM**: 17 files (utilities, infrastructure, supporting scripts)
- **LOW**: 6 files (documentation, minor utilities)

### By Category:
- **Core Training**: 10 files (grpo.py, sft.py, generate.py, rewards.py, configs.py, utils)
- **Competitive Programming**: 7 files (IOI, Codeforces, clients)
- **Infrastructure**: 8 files (Hub, callbacks, routers)
- **Recipes & Configs**: 12 files (training configurations)
- **SLURM Scripts**: 6 files (orchestration)
- **Utility Scripts**: 7 files (generation, filtering, benchmarking)

### Estimated Annotation Effort:
- **CRITICAL files**: ~40 hours (detailed annotation)
- **HIGH files**: ~50 hours (comprehensive annotation)
- **MEDIUM files**: ~35 hours (moderate annotation)
- **LOW files**: ~10 hours (basic annotation)
- **Total**: ~135 hours for complete annotation

---

## Annotation Template

Each annotated file will follow this structure:

```python
"""
FILE: <filename>
CATEGORY: <Core Training / Competitive Programming / Infrastructure / etc.>
PRIORITY: <CRITICAL / HIGH / MEDIUM / LOW>
DEPENDENCIES: <list of key imports and their purposes>

=============================================================================
OVERVIEW
=============================================================================

[High-level summary of the file's purpose and role in the system]

ROLE IN DEEPSEEK R1:
[How this component relates to the DeepSeek R1 paper and methodology]

KEY INNOVATIONS:
[Novel aspects or important design decisions]

DATA FLOW:
[Where data comes from, how it's transformed, where it goes]

=============================================================================
"""

# ... annotated code with inline comments following the format:

def function_name(param1, param2):
    """
    WHAT: [What this function does]

    WHY: [Why it's needed, design rationale]

    HOW: [Key algorithm or implementation approach]

    PROXIMAL CONTEXT:
    - Input: [Immediate data sources]
    - Output: [Immediate data destinations]

    DISTAL CONTEXT:
    - Originates from: [Where data ultimately comes from]
    - Flows to: [Where results ultimately go]

    Args:
        param1: [Description with context]
        param2: [Description with context]

    Returns:
        [Description with context]
    """

    # STEP 1: [What and why for this step]
    # HOW: [Implementation approach]
    code_here

    # STEP 2: [What and why]
    # WHY: [Design decision explanation]
    code_here

    # STEP 3: [What and why]
    # DATA FLOW: [How data transforms here]
    code_here
```

---

## Next Steps

1. **Phase 1**: Annotate CRITICAL files (11 files)
2. **Phase 2**: Annotate HIGH priority files (16 files)
3. **Phase 3**: Annotate MEDIUM priority files (17 files)
4. **Phase 4**: Annotate LOW priority files (6 files)
5. **Phase 5**: Review and cross-reference all annotations for consistency

Each phase will produce:
- Fully annotated source files
- Cross-reference documentation
- Data flow diagrams
- Integration guides
