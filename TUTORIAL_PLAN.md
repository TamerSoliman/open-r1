# Comprehensive Tutorial Plan for DeepSeek R1 Advanced Features

This document outlines 30 tutorials for beginner/intermediate AI scientists, grounded in the actual code implementation.

---

## Tutorial Structure

Each tutorial will include:

1. **Conceptual Overview**: Theory and background (accessible to beginners)
2. **Code Walkthrough**: Step-by-step code explanation with line references
3. **Hands-on Example**: Runnable code with expected outputs
4. **Advanced Topics**: Deeper dives for intermediate learners
5. **Common Pitfalls**: Mistakes to avoid and debugging tips
6. **Exercise**: Practice problem with solution

---

## Part 1: Foundational Concepts (5 tutorials)

### Tutorial 1: Understanding the DeepSeek R1 Three-Stage Pipeline
**Target Audience**: Beginner
**Duration**: 45 minutes
**Files**: `README.md`, `src/open_r1/sft.py`, `src/open_r1/grpo.py`

**Topics**:
- What is DeepSeek R1 and why is it significant?
- Stage 1: Distillation from strong model (DeepSeek-R1 → smaller model)
- Stage 2: Reinforcement Learning with GRPO
- Stage 3: Combined distillation + RL (optional)
- Why this approach works: intuition and empirical results

**Code Examples**:
```bash
# Stage 1: Distillation
accelerate launch src/open_r1/sft.py \
  --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml

# Stage 2: GRPO
accelerate launch src/open_r1/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml
```

**Hands-on**:
- Run a small-scale SFT on a subset of data
- Understand checkpoint outputs and metrics

**Exercise**:
- Modify system prompt to change reasoning format
- Compare outputs with different prompts

---

### Tutorial 2: Structured Reasoning with `<think>` and `<answer>` Tags
**Target Audience**: Beginner
**Duration**: 30 minutes
**Files**: `src/open_r1/rewards.py:format_reward`, `recipes/OpenR1-Distill-7B/sft/config_distill.yaml`

**Topics**:
- Why structured reasoning improves performance
- The `<think>` tag: internal reasoning chain
- The `<answer>` tag: final response
- How format rewards enforce this structure
- Connection to Chain-of-Thought (CoT) and other prompting techniques

**Code Walkthrough**:
```python
# From rewards.py:format_reward
def format_reward(completions: List[str]) -> List[float]:
    """
    WHAT: Checks if completion has proper <think>...</think><answer>...</answer> format
    WHY: Enforces structured reasoning, separates thinking from final answer
    """
    # Extract tags and verify structure
    # Return 1.0 if valid, 0.0 if invalid
```

**Hands-on**:
- Write completions with different formats
- Test with `format_reward` function
- See how reward signals affect GRPO training

**Exercise**:
- Create custom format reward for different tag structure
- Experiment with nested tags or multi-step answers

---

### Tutorial 3: Group Relative Policy Optimization (GRPO) Explained
**Target Audience**: Intermediate
**Duration**: 60 minutes
**Files**: `src/open_r1/grpo.py`, `src/open_r1/configs.py:GRPOConfig`

**Topics**:
- What is policy optimization in RL?
- PPO vs RLHF vs GRPO: key differences
- Why "group relative"? Advantage over absolute rewards
- How GRPO reduces variance in advantage estimation
- Mathematical formulation (simplified)
- Connection to DeepSeek R1 paper

**Code Walkthrough**:
```python
# From grpo.py (conceptual flow)
# 1. Generate multiple completions per prompt (num_generations=16)
for prompt in dataset:
    completions = vllm_generate(prompt, num_generations=16)

# 2. Compute rewards for all completions
    rewards = [reward_fn(c) for c in completions]

# 3. Group relative advantage: compare within group
    advantages = rewards - mean(rewards)  # Simplified

# 4. Policy gradient update
    loss = -log_prob(completion) * advantage
```

**Hands-on**:
- Run toy GRPO example with synthetic rewards
- Visualize advantage distribution
- Compare with absolute reward baseline

**Advanced Topics**:
- Advantage normalization techniques
- KL divergence constraints
- Learning rate scheduling for RL

**Exercise**:
- Implement simplified GRPO loop from scratch
- Experiment with different num_generations (4, 8, 16, 32)
- Analyze variance reduction

---

### Tutorial 4: Reward Functions: The Heart of RL Training
**Target Audience**: Intermediate
**Duration**: 45 minutes
**Files**: `src/open_r1/rewards.py`, `src/open_r1/configs.py:GRPOConfig`

**Topics**:
- What makes a good reward function?
- Multi-objective RL: combining multiple rewards
- Reward shaping and sparse vs dense rewards
- Common pitfalls: reward hacking, shortcut learning
- How DeepSeek R1 combines math, code, and format rewards

**Code Walkthrough**:
```python
# Reward composition in grpo.py
reward_funcs = get_reward_funcs(
    reward_names=["accuracy", "format", "tag_count"],
    reward_weights=[1.0, 0.5, 0.25]
)

total_reward = sum(w * r for w, r in zip(weights, rewards))
```

**Hands-on**:
- Implement custom reward function
- Test different reward combinations
- Visualize reward distributions

**Advanced Topics**:
- Reward scaling and normalization
- Curriculum learning with rewards
- Adversarial reward design

**Exercise**:
- Create reward function for specific task (e.g., code style)
- Combine with existing rewards
- Analyze impact on model behavior

---

### Tutorial 5: Dataset Mixtures and Weighted Sampling
**Target Audience**: Beginner
**Duration**: 30 minutes
**Files**: `src/open_r1/utils/data.py`, `src/open_r1/configs.py:ScriptArguments`

**Topics**:
- Why mix multiple datasets?
- Weighted sampling for balanced training
- Train/test splitting strategies
- Column standardization across datasets
- Connection to multi-task learning

**Code Walkthrough**:
```python
# From data.py
dataset_mixture = [
    {"path": "dataset1", "name": "subset1", "columns": ["prompt", "response"], "weight": 0.6},
    {"path": "dataset2", "name": "subset2", "columns": ["question", "answer"], "weight": 0.4},
]
mixed_dataset = get_dataset(ScriptArguments(dataset_mixture=dataset_mixture))
```

**Hands-on**:
- Create dataset mixture config
- Load and inspect mixed dataset
- Verify sampling proportions

**Exercise**:
- Experiment with different weights
- Add third dataset to mixture
- Implement custom column mapping

---

## Part 2: Training Infrastructure (7 tutorials)

### Tutorial 6: Distributed Training with DeepSpeed ZeRO
**Target Audience**: Intermediate
**Duration**: 60 minutes
**Files**: `recipes/accelerate_configs/zero3.yaml`, `slurm/train.slurm`

**Topics**:
- Why distributed training for large models?
- ZeRO stages: 1, 2, 3 (optimizer, gradient, parameter sharding)
- When to use each stage (model size guide)
- Memory savings vs communication overhead
- How ZeRO-3 enables training 7B+ models on consumer GPUs

**Code Walkthrough**:
```yaml
# From zero3.yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_multinode_launcher: standard
  zero3_init_flag: true
  zero_stage: 3
```

**Hands-on**:
- Profile memory usage with DDP vs ZeRO-2 vs ZeRO-3
- Train same model with different strategies
- Compare training speed and memory

**Advanced Topics**:
- ZeRO-Offload to CPU/NVMe
- Gradient accumulation with ZeRO
- ZeRO-Infinity for trillion-parameter models

**Exercise**:
- Calculate memory requirements for different ZeRO stages
- Choose optimal strategy for given hardware
- Debug common ZeRO issues (init failures, OOM)

---

### Tutorial 7: Fully Sharded Data Parallel (FSDP) for 32B+ Models
**Target Audience**: Intermediate
**Duration**: 45 minutes
**Files**: `recipes/accelerate_configs/fsdp.yaml`, `recipes/OlympicCoder-32B/`

**Topics**:
- FSDP vs DeepSpeed ZeRO-3: similarities and differences
- Sharding strategies (full, hybrid, no shard)
- Mixed precision with FSDP
- When to use FSDP (PyTorch native, 32B+ models)
- Backward prefetch for performance

**Code Walkthrough**:
```yaml
# From fsdp.yaml
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
```

**Hands-on**:
- Train 32B model with FSDP
- Compare checkpoint sizes (sharded vs full)
- Benchmark training throughput

**Advanced Topics**:
- CPU offloading with FSDP
- Activation checkpointing
- Hybrid sharding (FSDP + model parallelism)

**Exercise**:
- Port ZeRO-3 config to FSDP
- Optimize FSDP hyperparameters for specific hardware
- Debug activation checkpointing issues

---

### Tutorial 8: vLLM Integration for Fast Generation During GRPO
**Target Audience**: Intermediate
**Duration**: 45 minutes
**Files**: `src/open_r1/grpo.py`, `slurm/train.slurm` (vLLM launch)

**Topics**:
- Why vLLM for RL training?
- PagedAttention for efficient KV cache management
- Tensor parallelism for large models
- Continuous batching for throughput
- Integration with TRL's GRPOTrainer

**Code Walkthrough**:
```python
# From grpo.py (simplified)
from vllm import LLM

vllm_model = LLM(
    model=model_name,
    tensor_parallel_size=2,
    max_model_len=32768,
)

# Generate during training
completions = vllm_model.generate(
    prompts,
    sampling_params=SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=8192,
    )
)
```

**Hands-on**:
- Launch vLLM server for GRPO training
- Benchmark generation throughput
- Compare with HuggingFace `generate()`

**Advanced Topics**:
- Pipeline parallelism with vLLM
- Speculative decoding
- Multi-LoRA serving

**Exercise**:
- Tune vLLM parameters for optimal throughput
- Profile GPU utilization during generation
- Implement custom sampling strategy

---

### Tutorial 9: LoRA and QLoRA: Parameter-Efficient Fine-Tuning
**Target Audience**: Beginner
**Duration**: 45 minutes
**Files**: `src/open_r1/sft.py`, `recipes/` (LoRA configs)

**Topics**:
- What is LoRA (Low-Rank Adaptation)?
- Why LoRA reduces memory and training cost
- Rank selection: balancing capacity and efficiency
- QLoRA: 4-bit quantization + LoRA
- When to use LoRA vs full fine-tuning

**Code Walkthrough**:
```python
# From sft.py (conceptual)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,
)

model = get_peft_model(base_model, lora_config)
```

**Hands-on**:
- Train with different LoRA ranks (4, 8, 16, 32)
- Compare memory usage and convergence
- Merge LoRA adapters with base model

**Advanced Topics**:
- Rank-adaptive LoRA
- LoRA for GRPO training
- Multi-adapter inference

**Exercise**:
- Implement LoRA from scratch (simplified version)
- Experiment with target_modules selection
- Analyze rank vs performance trade-off

---

### Tutorial 10: Flash Attention 2 and Memory Optimization
**Target Audience**: Intermediate
**Duration**: 30 minutes
**Files**: `src/open_r1/utils/model_utils.py`, `recipes/` (attention configs)

**Topics**:
- Why attention is the bottleneck
- Flash Attention: IO-aware algorithm
- Flash Attention 2 improvements
- Memory savings for long context (32k tokens)
- SDPA (Scaled Dot Product Attention) as fallback

**Code Walkthrough**:
```python
# From model_utils.py
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # or "sdpa" or "eager"
)
```

**Hands-on**:
- Benchmark attention implementations (Flash vs SDPA vs Eager)
- Profile memory usage for different context lengths
- Measure training speedup

**Advanced Topics**:
- Flash Attention for inference
- Custom attention patterns
- Block-sparse attention

**Exercise**:
- Calculate memory complexity for standard vs Flash Attention
- Implement simplified Flash Attention (1D case)
- Compare empirical vs theoretical speedup

---

### Tutorial 11: Gradient Checkpointing: Trading Compute for Memory
**Target Audience**: Beginner
**Duration**: 30 minutes
**Files**: `src/open_r1/sft.py`, training configs

**Topics**:
- What is gradient checkpointing?
- Why it enables training larger models
- Memory vs compute trade-off (2x slower, 10x less memory)
- Reentrant vs non-reentrant checkpointing
- When to use it

**Code Walkthrough**:
```python
# From sft.py
training_args = SFTConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

**Hands-on**:
- Train with and without checkpointing
- Measure memory usage and training time
- Find optimal checkpoint frequency

**Advanced Topics**:
- Selective gradient checkpointing
- Checkpointing for custom layers
- Automatic checkpoint policy

**Exercise**:
- Calculate memory savings for specific model
- Implement gradient checkpointing from scratch (toy example)
- Profile GPU memory timeline

---

### Tutorial 12: W&B Integration and Experiment Tracking
**Target Audience**: Beginner
**Duration**: 20 minutes
**Files**: `src/open_r1/utils/wandb_logging.py`, training scripts

**Topics**:
- Why experiment tracking is critical
- W&B dashboard overview
- Logging metrics during training
- Comparing runs and hyperparameter sweeps
- Reproducibility best practices

**Code Walkthrough**:
```python
# From wandb_logging.py
import wandb

wandb.init(
    entity="huggingface",
    project="open-r1",
    name=f"{model_name}-{task}-{timestamp}",
)

# Auto-logged by Trainer
```

**Hands-on**:
- Set up W&B account and project
- Launch training with W&B logging
- Analyze learning curves and reward distributions

**Exercise**:
- Create custom W&B plots for RL metrics
- Log gradient norms and activations
- Set up alerts for training anomalies

---

## Part 3: Reward Engineering (6 tutorials)

### Tutorial 13: Math Reward Functions: Verifying Correctness
**Target Audience**: Intermediate
**Duration**: 45 minutes
**Files**: `src/open_r1/rewards.py:accuracy_reward`, test files

**Topics**:
- Why math verification is challenging (symbolic vs numeric)
- LaTeX parsing with `latex2sympy2`
- Answer extraction from free-form text
- Exact match vs approximate match
- Handling edge cases (multiple answers, units)

**Code Walkthrough**:
```python
# From rewards.py:accuracy_reward
def accuracy_reward(completions, ground_truths):
    """
    WHAT: Extracts LaTeX answers and compares with ground truth
    HOW:
      1. Extract <answer>...</answer> content
      2. Parse LaTeX to SymPy expressions
      3. Compare symbolically (not string match)
      4. Return 1.0 if match, 0.0 otherwise
    """
    for completion, gt in zip(completions, ground_truths):
        pred_answer = extract_answer(completion)  # Parse <answer> tag
        pred_expr = latex_to_sympy(pred_answer)   # LaTeX → SymPy
        gt_expr = latex_to_sympy(gt)
        if symbolic_equals(pred_expr, gt_expr):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
```

**Hands-on**:
- Test with various math expressions
- Debug parsing failures
- Handle special cases (fractions, surds)

**Advanced Topics**:
- Approximate numerical comparison
- Multi-step problem rewards (partial credit)
- Domain-specific rewards (geometry, algebra)

**Exercise**:
- Implement partial credit for close answers
- Create reward for showing work (intermediate steps)
- Handle multi-part math problems

---

### Tutorial 14: Code Execution Rewards with Sandboxing
**Target Audience**: Intermediate
**Duration**: 60 minutes
**Files**: `src/open_r1/rewards.py:code_reward`, `src/open_r1/utils/code_providers.py`

**Topics**:
- Why code execution is essential for code RL
- Sandboxing for security (E2B, MorphCloud)
- Test case evaluation (inputs → expected outputs)
- Timeout and error handling
- Language-specific considerations

**Code Walkthrough**:
```python
# From rewards.py:code_reward
def code_reward(completions, test_cases, provider="e2b"):
    """
    WHAT: Executes code in sandbox and checks outputs
    HOW:
      1. Extract code from completion (markdown blocks)
      2. For each test case: run code with input
      3. Compare output with expected
      4. Return pass rate (% of tests passed)
    """
    provider = CodeExecutionProvider.get(provider)

    for completion, tests in zip(completions, test_cases):
        code = extract_code(completion)
        results = []
        for test in tests:
            output = provider.execute(code, test.input)
            results.append(output == test.expected)
        reward = sum(results) / len(results)
```

**Hands-on**:
- Set up E2B sandbox
- Test code execution with simple examples
- Handle execution errors gracefully

**Advanced Topics**:
- Multi-language support
- Competitive programming evaluation (IOI, Codeforces)
- Performance-based rewards (time, memory)

**Exercise**:
- Create custom code execution provider
- Implement rewards for code quality (style, efficiency)
- Build test suite generator

---

### Tutorial 15: Format Rewards: Enforcing Structure
**Target Audience**: Beginner
**Duration**: 30 minutes
**Files**: `src/open_r1/rewards.py:format_reward`, `src/open_r1/rewards.py:tag_count_reward`

**Topics**:
- Why structured outputs matter
- Regex-based format checking
- Tag counting as soft reward
- Balancing format vs content rewards
- Evolution of format during training

**Code Walkthrough**:
```python
# From rewards.py:format_reward
def format_reward(completions):
    """
    WHAT: Checks <think>...</think><answer>...</answer> structure
    HOW:
      1. Regex match for both tags
      2. Check order (think before answer)
      3. Check single occurrence
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    rewards = []
    for completion in completions:
        if re.search(pattern, completion, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards
```

**Hands-on**:
- Test various completion formats
- Implement custom format reward
- Visualize format compliance over training

**Exercise**:
- Create format reward for JSON output
- Implement hierarchical format (nested tags)
- Combine format with content rewards

---

### Tutorial 16: Length-Based Rewards and Cosine Scaling
**Target Audience**: Intermediate
**Duration**: 30 minutes
**Files**: `src/open_r1/rewards.py:len_reward`, `src/open_r1/rewards.py:get_cosine_scaled_reward`

**Topics**:
- Why length matters (verbosity vs thoroughness)
- Kimi 1.5 approach: reward longer reasoning
- Cosine scaling for smooth length rewards
- Preventing reward hacking (excessive verbosity)
- Balancing length with quality

**Code Walkthrough**:
```python
# From rewards.py:get_cosine_scaled_reward
def get_cosine_scaled_reward(min_length, max_length, reward_value):
    """
    WHAT: Creates length-based reward with cosine scaling
    WHY: Encourages reasoning length without hard thresholds
    HOW: Cosine function from 0 (at min_length) to reward_value (at max_length)
    """
    def cosine_scaled_reward(completions):
        rewards = []
        for completion in completions:
            length = len(completion)
            if length < min_length:
                reward = 0.0
            elif length > max_length:
                reward = reward_value
            else:
                # Cosine interpolation
                progress = (length - min_length) / (max_length - min_length)
                reward = reward_value * (1 - math.cos(progress * math.pi)) / 2
            rewards.append(reward)
        return rewards
    return cosine_scaled_reward
```

**Hands-on**:
- Plot cosine scaling curve
- Test with completions of varying lengths
- Combine with quality rewards

**Advanced Topics**:
- Adaptive length targets during training
- Per-task length preferences
- Detecting and penalizing padding

**Exercise**:
- Implement linear length reward for comparison
- Create reward that prefers concise answers
- Analyze length distributions across training

---

### Tutorial 17: Repetition Penalty: Preventing Degenerate Outputs
**Target Audience**: Intermediate
**Duration**: 30 minutes
**Files**: `src/open_r1/rewards.py:get_repetition_penalty_reward`

**Topics**:
- Why models repeat during RL training
- N-gram based repetition detection
- Exponential penalties for repeated n-grams
- Balancing natural repetition vs degenerate loops
- Connection to generation parameters

**Code Walkthrough**:
```python
# From rewards.py:get_repetition_penalty_reward
def get_repetition_penalty_reward(n=3, penalty_factor=0.5):
    """
    WHAT: Penalizes repeated n-grams
    WHY: Prevents mode collapse during RL
    HOW: Count repeated n-grams, apply exponential penalty
    """
    def repetition_penalty(completions):
        rewards = []
        for completion in completions:
            ngrams = extract_ngrams(completion, n)
            unique_ngrams = set(ngrams)
            repetition_rate = 1 - (len(unique_ngrams) / len(ngrams))
            penalty = math.exp(-penalty_factor * repetition_rate)
            rewards.append(penalty)
        return rewards
    return repetition_penalty
```

**Hands-on**:
- Generate completions with repetition
- Test penalty with different n-gram sizes
- Visualize repetition over training

**Exercise**:
- Implement character-level repetition detection
- Create penalty for phrase-level repetition
- Combine with other quality rewards

---

### Tutorial 18: Reasoning Steps Reward: Detecting Structured Thinking
**Target Audience**: Intermediate
**Duration**: 30 minutes
**Files**: `src/open_r1/rewards.py:reasoning_steps_reward`

**Topics**:
- What constitutes a reasoning step?
- Pattern matching for step indicators (numbered lists, transitions)
- Counting vs evaluating step quality
- Encouraging explicit reasoning chains
- Limitations of heuristic approaches

**Code Walkthrough**:
```python
# From rewards.py:reasoning_steps_reward
def reasoning_steps_reward(completions):
    """
    WHAT: Counts structured reasoning steps
    WHY: Encourages explicit step-by-step thinking
    HOW: Regex patterns for "Step 1:", "First,", "Therefore", etc.
    """
    step_patterns = [
        r"Step \d+:",
        r"\d+\.",
        r"First,|Second,|Third,",
        r"Therefore,|Thus,|Hence,",
    ]
    rewards = []
    for completion in completions:
        step_count = 0
        for pattern in step_patterns:
            step_count += len(re.findall(pattern, completion))
        # Normalize to [0, 1] range
        reward = min(1.0, step_count / 5.0)  # 5 steps → max reward
        rewards.append(reward)
    return rewards
```

**Hands-on**:
- Test with various reasoning styles
- Adjust patterns for specific domains
- Combine with accuracy rewards

**Exercise**:
- Implement semantic step detection (using embeddings)
- Create reward for logical flow between steps
- Evaluate step quality, not just quantity

---

## Part 4: Code Evaluation and Competitive Programming (4 tutorials)

### Tutorial 19: Setting Up Code Execution Backends (E2B, MorphCloud)
**Target Audience**: Beginner
**Duration**: 45 minutes
**Files**: `src/open_r1/utils/code_providers.py`, setup documentation

**Topics**:
- Overview of code execution providers
- E2B: cloud sandboxes, security model
- MorphCloud: API-based execution
- Piston: open-source local option
- Choosing the right provider for your use case

**Code Walkthrough**:
```python
# From code_providers.py
class CodeExecutionProvider(ABC):
    @abstractmethod
    def execute_scripts(self, scripts, languages):
        """Execute code and return pass/fail for each"""
        pass

# E2B implementation
class E2BProvider(CodeExecutionProvider):
    def execute_scripts(self, scripts, languages):
        sandboxes = [E2BSandbox(language=lang) for lang in languages]
        results = []
        for sandbox, script in zip(sandboxes, scripts):
            try:
                output = sandbox.run_code(script)
                results.append(1.0 if output.exit_code == 0 else 0.0)
            finally:
                sandbox.close()
        return results
```

**Hands-on**:
- Set up E2B account and API key
- Run "Hello World" in multiple languages
- Compare execution times across providers

**Exercise**:
- Implement timeout handling
- Add custom language support
- Build cost monitoring for API usage

---

### Tutorial 20: Evaluating on IOI (International Olympiad) Problems
**Target Audience**: Intermediate
**Duration**: 60 minutes
**Files**: `src/open_r1/utils/competitive_programming/ioi_scoring.py`, IOI datasets

**Topics**:
- What is IOI? (Olympic-level programming)
- Subtask-based evaluation: why it matters
- Test result types: AC, WA, TLE, RE, MLE, CE
- Weighted scoring by subtask difficulty
- OlympicCoder model and results

**Code Walkthrough**:
```python
# From ioi_scoring.py
def evaluate_ioi_submission(code, problem):
    """
    WHAT: Evaluates code on IOI problem with subtasks
    HOW:
      1. Load problem subtasks and test cases
      2. For each subtask: run all tests
      3. Subtask passes only if ALL tests pass
      4. Final score = sum(subtask_points * subtask_pass)
    """
    subtasks = load_subtasks(problem)
    total_score = 0

    for subtask in subtasks:
        all_passed = True
        for test in subtask.tests:
            result = execute_with_limits(
                code,
                input=test.input,
                time_limit=subtask.time_limit,
                memory_limit=subtask.memory_limit,
            )
            if result != "AC":  # Accepted
                all_passed = False
                break

        if all_passed:
            total_score += subtask.points

    return total_score / problem.total_points  # Normalize to [0, 1]
```

**Hands-on**:
- Run evaluation on sample IOI problems
- Understand subtask dependencies
- Debug common failure modes (TLE, MLE)

**Advanced Topics**:
- Partial credit strategies
- Multi-language evaluation
- Performance optimization for RL training

**Exercise**:
- Create custom IOI-style problem
- Implement subtask grouping
- Analyze model performance by subtask difficulty

---

### Tutorial 21: Codeforces Integration and Generated Test Cases
**Target Audience**: Intermediate
**Duration**: 45 minutes
**Files**: `src/open_r1/utils/competitive_programming/cf_scoring.py`, Codeforces datasets

**Topics**:
- Codeforces problem format
- Generated test cases (parquet format)
- Scoring modes: pass_fail, partial, weighted_sum
- Language-specific compilation and execution
- Building training datasets from Codeforces

**Code Walkthrough**:
```python
# From cf_scoring.py
def evaluate_codeforces(code, problem_dataset, scoring_mode="pass_fail"):
    """
    WHAT: Evaluates code on Codeforces-style problems
    HOW:
      1. Load generated test cases from parquet
      2. Compile code (if needed)
      3. Run on all test inputs
      4. Compare outputs (exact match)
      5. Compute score based on mode
    """
    test_cases = load_parquet(problem_dataset)

    # Compile once if needed
    executable = compile_code(code, language)

    results = []
    for test in test_cases:
        output = run_code(executable, test.input)
        passed = (output.strip() == test.expected_output.strip())
        results.append(passed)

    if scoring_mode == "pass_fail":
        return 1.0 if all(results) else 0.0
    elif scoring_mode == "partial":
        return sum(results) / len(results)
    elif scoring_mode == "weighted_sum":
        return compute_weighted_score(results, test_cases.weights)
```

**Hands-on**:
- Download Codeforces dataset
- Generate test cases for custom problems
- Run evaluation pipeline

**Exercise**:
- Implement custom scoring mode
- Add test case generation logic
- Create difficulty-aware rewards

---

### Tutorial 22: Code Patching and Auto-Completion
**Target Audience**: Intermediate
**Duration**: 30 minutes
**Files**: `src/open_r1/utils/competitive_programming/code_patcher.py`

**Topics**:
- Why models produce incomplete code
- Pattern matching for code completion
- Template insertion for common structures
- Language-specific patching strategies
- Improving success rate with patching

**Code Walkthrough**:
```python
# From code_patcher.py
def patch_code(code_snippet, language="python"):
    """
    WHAT: Auto-completes partial code from model output
    WHY: Models often omit boilerplate (main function, imports)
    HOW:
      1. Detect missing elements (imports, main, etc.)
      2. Insert appropriate templates
      3. Preserve model-generated core logic
    """
    if language == "python":
        # Check for missing main execution
        if "if __name__" not in code_snippet:
            code_snippet += "\n\nif __name__ == '__main__':\n    main()"

        # Check for missing imports
        if uses_math_functions(code_snippet) and "import math" not in code_snippet:
            code_snippet = "import math\n" + code_snippet

    elif language == "cpp":
        # Add includes if missing
        if "#include" not in code_snippet:
            code_snippet = "#include <iostream>\n#include <vector>\n" + code_snippet

        # Wrap in main if needed
        if "int main" not in code_snippet:
            code_snippet = f"int main() {{\n{indent(code_snippet)}\nreturn 0;\n}}"

    return code_snippet
```

**Hands-on**:
- Test patching on incomplete code samples
- Implement custom patching rules
- Measure impact on pass rate

**Exercise**:
- Add patching for more languages
- Implement intelligent import detection
- Handle edge cases (nested functions, classes)

---

## Part 5: Data Generation and Dataset Management (3 tutorials)

### Tutorial 23: Synthetic Data Generation with Distilabel
**Target Audience**: Intermediate
**Duration**: 45 minutes
**Files**: `src/open_r1/generate.py`, `scripts/generate_reasoning.py`

**Topics**:
- Why synthetic data for reasoning models?
- Distilabel pipeline architecture
- Async generation with vLLM backend
- Sampling strategies (temperature, top_p, num_generations)
- Quality control and filtering

**Code Walkthrough**:
```python
# From generate.py:build_distilabel_pipeline
def build_distilabel_pipeline(model, base_url, temperature, top_p, max_new_tokens, num_generations):
    """
    WHAT: Creates Distilabel pipeline for data generation
    WHY: Scales generation across prompts with async execution
    HOW:
      1. Define pipeline with vLLM backend
      2. Set sampling parameters
      3. Configure batching and parallelism
    """
    pipeline = Pipeline()

    generator = vLLMGenerator(
        model=model,
        base_url=base_url,
        generation_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "n": num_generations,  # Multiple completions per prompt
        },
    )

    pipeline.add_step(generator)
    return pipeline

# Usage
pipeline = build_distilabel_pipeline(...)
dataset = pipeline.run(input_prompts)
```

**Hands-on**:
- Generate synthetic reasoning traces
- Experiment with temperature values
- Filter by quality metrics

**Advanced Topics**:
- Multi-stage generation pipelines
- Self-consistency filtering
- Curriculum data generation

**Exercise**:
- Build custom generation pipeline
- Implement quality-based filtering
- Create diverse prompt templates

---

### Tutorial 24: Dataset Decontamination for Fair Evaluation
**Target Audience**: Intermediate
**Duration**: 30 minutes
**Files**: `scripts/decontaminate.py`

**Topics**:
- Why decontamination is critical
- N-gram overlap detection
- Benchmark leakage risks
- Removal strategies (examples vs entire dataset)
- Reporting decontamination results

**Code Walkthrough**:
```python
# From decontaminate.py
def decontaminate_dataset(training_data, benchmark_data, ngram_size=8):
    """
    WHAT: Removes training examples that overlap with benchmarks
    WHY: Prevents overfitting to benchmark, ensures fair evaluation
    HOW:
      1. Extract n-grams from benchmark examples
      2. For each training example, check for overlap
      3. Remove if overlap exceeds threshold
    """
    benchmark_ngrams = set()
    for example in benchmark_data:
        ngrams = extract_ngrams(example.text, ngram_size)
        benchmark_ngrams.update(ngrams)

    clean_data = []
    for example in training_data:
        example_ngrams = set(extract_ngrams(example.text, ngram_size))
        overlap = len(example_ngrams & benchmark_ngrams)
        overlap_ratio = overlap / len(example_ngrams)

        if overlap_ratio < 0.1:  # Less than 10% overlap
            clean_data.append(example)

    return clean_data
```

**Hands-on**:
- Run decontamination on sample dataset
- Visualize overlap statistics
- Validate with benchmark performance

**Exercise**:
- Implement semantic decontamination (embeddings)
- Experiment with different n-gram sizes
- Create decontamination report

---

### Tutorial 25: Pass Rate Filtering for Dataset Quality
**Target Audience**: Intermediate
**Duration**: 45 minutes
**Files**: `scripts/pass_rate_filtering/compute_pass_rate.py`, filtering recipes

**Topics**:
- Pass@k metric for code problems
- Computing pass rates from multiple generations
- Threshold-based filtering
- Balancing dataset size and quality
- Impact on downstream training

**Code Walkthrough**:
```python
# From compute_pass_rate.py
def compute_pass_rate(dataset, k=10):
    """
    WHAT: Computes pass@k for code problems
    WHY: Filters out problems that are too hard/ambiguous
    HOW:
      1. For each problem: count successful solutions
      2. pass@k = 1 - (combinations without success / total combinations)
      3. Filter problems below threshold
    """
    from scipy.special import comb

    pass_rates = []
    for problem in dataset:
        n_total = len(problem.solutions)
        n_correct = sum(s.passed for s in problem.solutions)

        if n_total < k:
            pass_rate = n_correct / n_total
        else:
            # pass@k formula
            pass_rate = 1.0 - comb(n_total - n_correct, k) / comb(n_total, k)

        pass_rates.append(pass_rate)

    # Filter by threshold
    filtered_dataset = [
        problem for problem, pass_rate in zip(dataset, pass_rates)
        if pass_rate >= 0.3  # Example threshold
    ]

    return filtered_dataset, pass_rates
```

**Hands-on**:
- Compute pass rates for code dataset
- Visualize distribution
- Filter and compare training results

**Exercise**:
- Implement pass@k for math problems
- Create adaptive threshold selection
- Analyze difficulty vs pass rate correlation

---

## Part 6: Evaluation and Benchmarking (3 tutorials)

### Tutorial 26: LightEval Integration for Standardized Benchmarks
**Target Audience**: Beginner
**Duration**: 45 minutes
**Files**: `src/open_r1/utils/evaluation.py`, `src/open_r1/__init__.py`

**Topics**:
- What is LightEval?
- Standard benchmarks: MATH-500, AIME, GPQA, LCB
- Task specification format
- Running evaluations with vLLM backend
- Interpreting results and comparing with baselines

**Code Walkthrough**:
```python
# From evaluation.py
LIGHTEVAL_TASKS = {
    "math_500": "lighteval|math_500|0|0",
    "aime24": "lighteval|aime24|0|0",
    "aime25": "lighteval|aime25|0|0",
    "gpqa": "lighteval|gpqa:diamond|0|0",
    "lcb": "extended|lcb:codegeneration|0|0",
}

def run_evaluation(model, task_name):
    """
    WHAT: Runs LightEval benchmark on trained model
    HOW:
      1. Load model and task
      2. Run inference on all examples
      3. Evaluate answers
      4. Aggregate metrics
    """
    task_spec = LIGHTEVAL_TASKS[task_name]

    # Launch vLLM for large models
    if model.num_parameters > 30B:
        vllm_backend = launch_vllm(model, tensor_parallel=2)

    # Run evaluation
    results = lighteval.evaluate(
        model=model,
        tasks=[task_spec],
        backend=vllm_backend,
    )

    return results
```

**Hands-on**:
- Run evaluation on sample model
- Compare with baseline results
- Analyze failure cases

**Exercise**:
- Add custom LightEval task
- Create comparative leaderboard
- Implement early stopping based on eval metrics

---

### Tutorial 27: Reproducing DeepSeek R1 Evaluation Results
**Target Audience**: Intermediate
**Duration**: 45 minutes
**Files**: `README.md` (evaluation section), evaluation scripts

**Topics**:
- DeepSeek paper benchmark suite
- Prompt formats for each benchmark
- Few-shot vs zero-shot evaluation
- Reproducibility challenges and solutions
- Comparing with paper results

**Code Walkthrough**:
```bash
# Reproducing MATH-500 results
make evaluate TASK=math_500 MODEL=open-r1/OpenR1-Distill-7B

# Reproducing AIME results
make evaluate TASK=aime24 MODEL=open-r1/OpenR1-Distill-7B

# Reproducing code benchmarks
make evaluate TASK=lcb MODEL=open-r1/OlympicCoder-7B
```

**Hands-on**:
- Run full evaluation suite
- Compare results with paper
- Investigate discrepancies

**Exercise**:
- Create reproduction report
- Identify factors affecting reproducibility
- Suggest improvements to evaluation protocol

---

### Tutorial 28: Custom Benchmarks and Task Registration
**Target Audience**: Intermediate
**Duration**: 30 minutes
**Files**: `src/open_r1/__init__.py`, LightEval task creation

**Topics**:
- Creating custom LightEval tasks
- Task registry pattern
- Metric definition and computation
- Prompt engineering for tasks
- Validating custom benchmarks

**Code Walkthrough**:
```python
# From __init__.py and custom task creation
from lighteval.tasks import Task

def create_custom_task():
    """
    WHAT: Creates custom evaluation task
    HOW:
      1. Define prompt template
      2. Implement metric computation
      3. Register with LightEval
    """
    task = Task(
        name="custom_math_task",
        prompt_function=lambda x: f"Problem: {x['problem']}\nSolution:",
        metric="exact_match",
        evaluation_splits=["test"],
    )

    return task

# Register task
CUSTOM_TASKS.register(create_custom_task())
```

**Hands-on**:
- Create custom task for specific domain
- Test on sample data
- Integrate with evaluation pipeline

**Exercise**:
- Build task suite for specific capability
- Implement custom metric
- Create leaderboard for custom tasks

---

## Part 7: Advanced Topics and Optimization (3 tutorials)

### Tutorial 29: Scaling to 32B Models and Beyond
**Target Audience**: Advanced
**Duration**: 60 minutes
**Files**: `recipes/OlympicCoder-32B/`, FSDP configs, multi-node training

**Topics**:
- Challenges at scale (32B+ parameters)
- FSDP for model parallelism
- Multi-node training with SLURM
- Communication optimization
- Checkpoint strategies for large models

**Code Walkthrough**:
```yaml
# From OlympicCoder-32B config
model: Qwen/Qwen2.5-Coder-32B-Instruct
accelerator: fsdp
num_nodes: 16
gpus_per_node: 8

# FSDP config
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
```

**Hands-on**:
- Estimate memory requirements for 32B model
- Configure multi-node FSDP
- Profile communication overhead

**Advanced Topics**:
- Pipeline parallelism
- Activation checkpointing strategies
- Mixed precision training at scale

**Exercise**:
- Calculate optimal parallelism strategy for given hardware
- Implement checkpoint sharding
- Debug multi-node training issues

---

### Tutorial 30: Curriculum Learning and Multi-Stage Training
**Target Audience**: Advanced
**Duration**: 45 minutes
**Files**: Training recipes, configuration examples

**Topics**:
- Why curriculum learning works
- Designing learning curricula (easy→hard)
- Multi-stage training strategies
- Combining distillation, SFT, and GRPO
- Adaptive difficulty progression

**Code Walkthrough**:
```python
# Conceptual curriculum learning pipeline
stages = [
    {
        "name": "Distillation",
        "method": "sft",
        "dataset": "Mixture-of-Thoughts",
        "model": "Qwen2.5-Math-7B",
        "epochs": 5,
    },
    {
        "name": "Easy GRPO",
        "method": "grpo",
        "dataset": "OpenR1-Math-Easy",
        "model": "checkpoint-from-stage-1",
        "reward_funcs": ["accuracy", "format"],
        "epochs": 2,
    },
    {
        "name": "Hard GRPO",
        "method": "grpo",
        "dataset": "OpenR1-Math-Hard",
        "model": "checkpoint-from-stage-2",
        "reward_funcs": ["accuracy", "format", "reasoning_steps"],
        "epochs": 3,
    },
]

for stage in stages:
    model = train_stage(stage)
    evaluate(model, benchmarks)
```

**Hands-on**:
- Design curriculum for specific task
- Implement multi-stage training
- Compare with single-stage baseline

**Advanced Topics**:
- Automatic difficulty estimation
- Adaptive curriculum based on performance
- Multi-task curriculum learning

**Exercise**:
- Create curriculum for code generation
- Implement dynamic difficulty adjustment
- Analyze learning curves across stages

---

## Summary Statistics

### Tutorials by Audience Level:
- **Beginner**: 10 tutorials (fundamentals, infrastructure basics)
- **Intermediate**: 16 tutorials (training, rewards, evaluation)
- **Advanced**: 4 tutorials (scaling, optimization, curriculum)

### Tutorials by Topic Area:
- **Foundational Concepts**: 5 tutorials
- **Training Infrastructure**: 7 tutorials
- **Reward Engineering**: 6 tutorials
- **Code Evaluation**: 4 tutorials
- **Data Management**: 3 tutorials
- **Evaluation**: 3 tutorials
- **Advanced Topics**: 2 tutorials

### Estimated Time:
- **Total**: ~21 hours of tutorial content
- **Beginner path**: ~6 hours
- **Intermediate path**: ~12 hours
- **Advanced path**: ~3 hours

### Prerequisites by Tutorial:
- Tutorials 1-5: Python basics, ML fundamentals
- Tutorials 6-12: PyTorch, distributed training concepts
- Tutorials 13-18: RL basics, reward shaping
- Tutorials 19-22: Systems programming, competitive programming
- Tutorials 23-25: Data science, statistics
- Tutorials 26-28: Evaluation methodologies
- Tutorials 29-30: Advanced ML, optimization

---

## Tutorial Delivery Format

Each tutorial will include:

### 1. Markdown Documentation
- Conceptual overview
- Code walkthroughs with line references
- Diagrams and visualizations

### 2. Jupyter Notebooks
- Interactive code cells
- Runnable examples
- Inline visualizations
- Exercises with solutions

### 3. Standalone Scripts
- Complete runnable examples
- Command-line interfaces
- Configuration files

### 4. Video Walkthroughs (Optional)
- Screen recordings of code execution
- Explanation of key concepts
- Debugging demonstrations

### 5. Exercise Solutions
- Detailed solutions with explanations
- Alternative approaches
- Common mistakes and fixes

---

## Next Steps

1. **Phase 1**: Create tutorials 1-5 (Foundational Concepts)
2. **Phase 2**: Create tutorials 6-12 (Training Infrastructure)
3. **Phase 3**: Create tutorials 13-18 (Reward Engineering)
4. **Phase 4**: Create tutorials 19-25 (Code & Data)
5. **Phase 5**: Create tutorials 26-30 (Evaluation & Advanced)

Each tutorial will be thoroughly tested with actual code execution and validated by beginner/intermediate reviewers.
