# Tutorial 6: DeepSpeed ZeRO for Distributed Training

**Target Audience:** Intermediate to Advanced
**Duration:** 60 minutes
**Prerequisites:** Tutorial 1 (Three-Stage Pipeline), Basic understanding of distributed training

## Table of Contents
1. [Overview](#overview)
2. [The Memory Problem](#the-memory-problem)
3. [ZeRO Optimization Stages](#zero-optimization-stages)
4. [ZeRO-2 Deep Dive](#zero-2-deep-dive)
5. [ZeRO-3 Deep Dive](#zero-3-deep-dive)
6. [Configuration Files](#configuration-files)
7. [Hands-On Example](#hands-on-example)
8. [Performance Analysis](#performance-analysis)
9. [Common Pitfalls](#common-pitfalls)
10. [Exercise](#exercise)

---

## Overview

**DeepSpeed ZeRO** (Zero Redundancy Optimizer) is a memory optimization technique that enables training large language models across multiple GPUs. It's essential for DeepSeek R1 training, allowing 7B models to train on 8×A100 GPUs efficiently.

**What you'll learn:**
- Why standard data parallelism wastes memory
- How ZeRO partitions optimizer states, gradients, and parameters
- Differences between ZeRO-2 and ZeRO-3
- Configuring DeepSpeed for SFT and GRPO
- Performance tradeoffs and when to use each stage

**Key Innovation:**
ZeRO eliminates memory redundancy by **partitioning** model states across GPUs instead of replicating them, enabling:
- ✅ 4-8× memory reduction
- ✅ Larger batch sizes → better convergence
- ✅ Support for larger models on same hardware
- ✅ Near-linear scaling efficiency

---

## The Memory Problem

### Standard Data Parallelism

**Traditional Training (No ZeRO):**

```
GPU 0:                      GPU 1:
┌────────────────────┐      ┌────────────────────┐
│ Model Parameters   │      │ Model Parameters   │  DUPLICATED
│ Gradients          │      │ Gradients          │  DUPLICATED
│ Optimizer States   │      │ Optimizer States   │  DUPLICATED
│ (Adam: momentum +  │      │ (Adam: momentum +  │  DUPLICATED
│  variance)         │      │  variance)         │
└────────────────────┘      └────────────────────┘
```

**Problem: Massive Redundancy!**

For a 7B parameter model with Adam optimizer:
- Parameters: 7B × 4 bytes (FP32) = 28 GB
- Gradients: 7B × 4 bytes = 28 GB
- Optimizer states: 7B × 8 bytes (momentum + variance) = 56 GB
- **Total per GPU: 112 GB**

**On 8 GPUs: 896 GB total, but only 112 GB unique data!**

### Memory Breakdown

**Components:**

| Component | Size (7B model) | Why Needed |
|-----------|-----------------|------------|
| Parameters (FP32) | 28 GB | Forward/backward pass |
| Parameters (FP16) | 14 GB | Mixed precision training |
| Gradients | 28 GB | Backpropagation |
| Optimizer States | 56 GB | Adam momentum + variance |
| Activations | 10-40 GB | Depends on batch size |
| **Total** | **136-166 GB** | **Per GPU without ZeRO!** |

**A100 80GB GPU:** Can't fit 7B model with reasonable batch size! ❌

---

## ZeRO Optimization Stages

### ZeRO Overview

**Key Idea:** Partition memory instead of replicating

```
Standard DP: Each GPU has full copy (redundant)
ZeRO: Each GPU has partition of model states (efficient)
```

### Three Stages

**ZeRO-1: Optimizer State Partitioning**

```
GPU 0: Optimizer states for layers 0-19
GPU 1: Optimizer states for layers 20-39
GPU 2: Optimizer states for layers 40-59
GPU 3: Optimizer states for layers 60-79

Parameters and gradients: STILL REPLICATED
```

**Memory Savings:** ~4× for optimizer states (56 GB → 14 GB per GPU)
**Total Savings:** ~33%

**ZeRO-2: + Gradient Partitioning**

```
GPU 0: Gradients for layers 0-19, Optimizer states for layers 0-19
GPU 1: Gradients for layers 20-39, Optimizer states for layers 20-39
GPU 2: Gradients for layers 40-59, Optimizer states for layers 40-59
GPU 3: Gradients for layers 60-79, Optimizer states for layers 60-79

Parameters: STILL REPLICATED
```

**Memory Savings:** ~4× for optimizer + gradients (84 GB → 21 GB per GPU)
**Total Savings:** ~50%

**ZeRO-3: + Parameter Partitioning**

```
GPU 0: Parameters (layers 0-19), Gradients (layers 0-19), Optimizer (layers 0-19)
GPU 1: Parameters (layers 20-39), Gradients (layers 20-39), Optimizer (layers 20-39)
GPU 2: Parameters (layers 40-59), Gradients (layers 40-59), Optimizer (layers 40-59)
GPU 3: Parameters (layers 60-79), Gradients (layers 60-79), Optimizer (layers 60-79)

EVERYTHING partitioned!
```

**Memory Savings:** ~4× for all states (112 GB → 28 GB per GPU)
**Total Savings:** ~75%

---

## ZeRO-2 Deep Dive

### How ZeRO-2 Works

**Training Step:**

```python
# Forward pass (all GPUs)
for gpu in [0, 1, 2, 3]:
    # Each GPU has full model parameters
    output = model(batch[gpu])
    loss = loss_fn(output, labels[gpu])

# Backward pass
for gpu in [0, 1, 2, 3]:
    loss.backward()  # Compute gradients

# Gradient reduction (ZeRO-2 magic!)
# Instead of all-reduce, use reduce-scatter

# Standard DP (all-reduce):
all_reduce(gradients)  # Each GPU gets full gradients
# Result: All GPUs have complete gradients (redundant!)

# ZeRO-2 (reduce-scatter):
reduce_scatter(gradients)  # Each GPU gets 1/4 of gradients
# Result: GPU 0 has grads[0:25%], GPU 1 has grads[25:50%], etc.

# Optimizer step (each GPU updates its partition)
GPU 0: optimizer.step(params[0:25%], grads[0:25%])
GPU 1: optimizer.step(params[25:50%], grads[25:50%])
GPU 2: optimizer.step(params[50:75%], grads[50:75%])
GPU 3: optimizer.step(params[75:100%], grads[75:100%])

# All-gather updated parameters
all_gather(params)  # Each GPU gets full updated model
```

### Memory Layout (ZeRO-2, 4 GPUs)

**GPU 0:**
```
Parameters (full):     7B × 2 bytes (FP16) = 14 GB
Gradients (partition): 7B/4 × 4 bytes = 7 GB
Optimizer (partition): 7B/4 × 8 bytes = 14 GB
Activations:           ~10 GB
─────────────────────────────────────────
Total:                 45 GB ✅ Fits in 80GB GPU!
```

**Compared to Standard DP:**
```
Parameters:  14 GB
Gradients:   28 GB  ← ZeRO-2 saves 21 GB here
Optimizer:   56 GB  ← ZeRO-2 saves 42 GB here
Activations: 10 GB
─────────────────────────────────────────
Total:       108 GB ❌ Doesn't fit!
```

### Configuration (ZeRO-2)

**From `recipes/accelerate_configs/zero2.yaml`:**

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_multinode_launcher: standard
  zero3_init_flag: false
  zero_stage: 2  # ZeRO-2

  # Optimizer state partitioning
  offload_optimizer_device: none  # Keep optimizer on GPU

  # Gradient partitioning
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0

  # Communication optimization
  overlap_comm: true  # Overlap communication with computation
  reduce_scatter: true  # Use reduce-scatter instead of all-reduce

  # Memory optimization
  allgather_bucket_size: 5e8
  reduce_bucket_size: 5e8

distributed_type: DEEPSPEED
```

---

## ZeRO-3 Deep Dive

### How ZeRO-3 Works

**Key Difference:** Parameters are also partitioned!

**Training Step:**

```python
# Forward pass (parameters gathered on-demand)
for layer in model.layers:
    # Before layer computation: gather parameters
    all_gather(layer.parameters)

    # Compute forward pass
    output = layer(input)

    # After layer computation: discard parameters
    # (will re-gather in backward pass)
    discard(layer.parameters)

# Backward pass (parameters gathered again)
for layer in reversed(model.layers):
    # Gather parameters for backward
    all_gather(layer.parameters)

    # Compute gradients
    gradients = backward(layer)

    # Reduce-scatter gradients
    reduce_scatter(gradients)

    # Discard parameters again
    discard(layer.parameters)

# Optimizer step (each GPU updates its partition)
for gpu in [0, 1, 2, 3]:
    optimizer.step(params_partition[gpu], grads_partition[gpu])
```

**Memory Savings:** Parameters only materialized when needed!

### Memory Layout (ZeRO-3, 4 GPUs)

**GPU 0:**
```
Parameters (partition):  7B/4 × 2 bytes (FP16) = 3.5 GB
Parameters (temp gather): 7B × 2 bytes = 14 GB (during layer computation)
Gradients (partition):   7B/4 × 4 bytes = 7 GB
Optimizer (partition):   7B/4 × 8 bytes = 14 GB
Activations:             ~10 GB
─────────────────────────────────────────
Peak:                    48.5 GB (during layer with gathered params)
Average:                 34.5 GB (most of the time)
```

**Compared to ZeRO-2:**
```
ZeRO-2 peak: 45 GB
ZeRO-3 peak: 48.5 GB (slightly higher due to gather overhead)
ZeRO-3 average: 34.5 GB (lower when parameters discarded)
```

**Benefit:** Can train larger models (14B+) on same hardware

### Configuration (ZeRO-3)

**From `recipes/accelerate_configs/zero3.yaml`:**

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  zero_stage: 3  # ZeRO-3

  # Parameter partitioning
  stage3_prefetch_bucket_size: 5e8
  stage3_param_persistence_threshold: 1e6
  stage3_gather_16bit_weights_on_model_save: true

  # Offloading (optional)
  offload_optimizer_device: none  # Can use "cpu" to save GPU memory
  offload_param_device: none  # Can use "cpu" for even larger models

  # Communication optimization
  overlap_comm: true
  stage3_max_live_parameters: 1e9
  stage3_max_reuse_distance: 1e9

distributed_type: DEEPSPEED
```

---

## Configuration Files

### Using ZeRO with Accelerate

**SFT Training with ZeRO-2:**

```bash
# Training command
accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  src/open_r1/sft.py \
  --config recipes/config_distill.yaml
```

**GRPO Training with ZeRO-3:**

```bash
# Training command
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  src/open_r1/grpo.py \
  --config recipes/config_demo.yaml
```

### When to Use Each Stage

| Stage | Best For | Memory Savings | Speed | Complexity |
|-------|----------|----------------|-------|------------|
| **ZeRO-1** | Small models (<3B) | Low (~33%) | Fastest | Low |
| **ZeRO-2** | Medium models (3-13B) | Medium (~50%) | Fast | Medium |
| **ZeRO-3** | Large models (13B+) | High (~75%) | Slower | High |

**Rule of Thumb:**

- 7B model, 8×A100: **ZeRO-2** (good balance)
- 13B model, 8×A100: **ZeRO-3** (necessary)
- 70B model, 8×A100: **ZeRO-3 + CPU offload** (fits, but slow)

---

## Hands-On Example

### Example 1: ZeRO-2 SFT Training

```bash
# 1. Create ZeRO-2 config
cat > zero2_config.yaml << EOF
compute_environment: LOCAL_MACHINE
deepspeed_config:
  zero_stage: 2
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0
  overlap_comm: true
distributed_type: DEEPSPEED
num_processes: 8
machine_rank: 0
num_machines: 1
mixed_precision: bf16
EOF

# 2. Run training
accelerate launch \
  --config_file zero2_config.yaml \
  src/open_r1/sft.py \
  --config recipes/config_distill.yaml \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --output_dir ./sft_zero2
```

**Expected Output:**

```
[2025-01-15 10:00:00] DeepSpeed info: [Rank 0] DeepSpeed Engine instantiated
[2025-01-15 10:00:01] DeepSpeed ZeRO-2 enabled
[2025-01-15 10:00:01] Total parameters: 7.0B
[2025-01-15 10:00:01] Memory per GPU: 42 GB
[2025-01-15 10:00:01] Training...

Step 100/10000: loss=2.34, lr=5e-6, memory=43GB
Step 200/10000: loss=2.01, lr=5e-6, memory=43GB
```

### Example 2: ZeRO-3 GRPO Training

```bash
# 1. Create ZeRO-3 config
cat > zero3_config.yaml << EOF
compute_environment: LOCAL_MACHINE
deepspeed_config:
  zero_stage: 3
  stage3_prefetch_bucket_size: 5e8
  stage3_param_persistence_threshold: 1e6
  overlap_comm: true
  offload_optimizer_device: none
distributed_type: DEEPSPEED
num_processes: 8
mixed_precision: bf16
EOF

# 2. Run GRPO
accelerate launch \
  --config_file zero3_config.yaml \
  src/open_r1/grpo.py \
  --config recipes/config_demo.yaml \
  --output_dir ./grpo_zero3
```

### Example 3: Memory Profiling

```python
# Add to training script
import torch

def print_memory_stats():
    """Print GPU memory usage"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# Call during training
print_memory_stats()
```

**Output (ZeRO-2):**

```
GPU 0: 42.31 GB allocated, 44.00 GB reserved
GPU 1: 41.98 GB allocated, 44.00 GB reserved
...
```

**Output (ZeRO-3):**

```
GPU 0: 34.12 GB allocated, 36.00 GB reserved
GPU 1: 33.87 GB allocated, 36.00 GB reserved
...
```

---

## Performance Analysis

### Throughput Comparison

**Setup:** 7B model, 8×A100, SFT training

| Configuration | Batch Size | Throughput (samples/sec) | Memory/GPU |
|---------------|------------|--------------------------|------------|
| Standard DP | 2 | OOM | N/A |
| ZeRO-1 | 4 | OOM | N/A |
| ZeRO-2 | 8 | 12.5 | 43 GB |
| ZeRO-2 | 16 | 18.2 | 58 GB |
| ZeRO-3 | 16 | 15.1 | 49 GB |
| ZeRO-3 | 32 | 22.4 | 72 GB |

**Key Observations:**

1. **ZeRO-2** allows 4× larger batch size vs standard DP
2. **ZeRO-3** allows 8× larger batch size
3. **ZeRO-2 is faster** at same batch size (less communication overhead)
4. **ZeRO-3 enables larger batches** → better throughput overall

### Communication Overhead

**ZeRO-2 (per training step):**

```
Forward: No extra communication (params replicated)
Backward: reduce-scatter gradients (1× gradient size)
Optimizer: all-gather parameters (1× parameter size)

Total: 2× model size per step
```

**ZeRO-3 (per training step):**

```
Forward: all-gather parameters for each layer (1× parameter size)
Backward: all-gather parameters again (1× parameter size)
Backward: reduce-scatter gradients (1× gradient size)
Optimizer: No extra communication (params already partitioned)

Total: 3× model size per step
```

**Result:** ZeRO-3 has ~1.5× more communication overhead

---

## Common Pitfalls

### 1. **Using ZeRO-3 When ZeRO-2 Suffices**

❌ **Wrong:**

```yaml
# For 7B model on 8×A100
zero_stage: 3  # Overkill!
```

**Problem:** Extra communication overhead, slower training

✅ **Correct:**

```yaml
zero_stage: 2  # Sufficient for 7B
```

### 2. **Not Tuning Bucket Sizes**

❌ **Wrong:**

```yaml
# Using defaults
deepspeed_config:
  zero_stage: 3
  # No bucket size configuration
```

**Problem:** Suboptimal communication efficiency

✅ **Correct:**

```yaml
deepspeed_config:
  zero_stage: 3
  stage3_prefetch_bucket_size: 5e8  # Tune for your model
  allgather_bucket_size: 5e8
  reduce_bucket_size: 5e8
```

### 3. **Forgetting stage3_gather_16bit_weights_on_model_save**

❌ **Wrong:**

```yaml
# ZeRO-3 config
zero_stage: 3
# Missing: stage3_gather_16bit_weights_on_model_save
```

**Problem:** Saved checkpoint is partitioned, can't load for inference!

✅ **Correct:**

```yaml
zero_stage: 3
stage3_gather_16bit_weights_on_model_save: true  # Gather before saving
```

### 4. **CPU Offloading Without NVMe**

❌ **Wrong:**

```yaml
offload_optimizer_device: cpu  # Slow without fast storage!
```

**Problem:** CPU-GPU transfers bottleneck training

✅ **Correct:** Only use CPU offload with NVMe SSD

```yaml
offload_optimizer_device: cpu
offload_optimizer_nvme_path: /nvme/offload  # Fast NVMe required
```

### 5. **Incompatible Gradient Accumulation**

❌ **Wrong:**

```yaml
# Accelerate config
gradient_accumulation_steps: 4

# Training config (different!)
--gradient_accumulation_steps 8
```

**Problem:** Mismatch causes incorrect gradient accumulation

✅ **Correct:** Must match

```yaml
# Both configs
gradient_accumulation_steps: 8
```

---

## Exercise

### Task 1: Compare ZeRO-2 vs ZeRO-3

```bash
# 1. Train with ZeRO-2
accelerate launch \
  --config_file zero2.yaml \
  src/open_r1/sft.py \
  --config recipes/config_distill.yaml \
  --max_steps 100 \
  --output_dir ./sft_zero2

# Record: memory usage, throughput, training time

# 2. Train with ZeRO-3
accelerate launch \
  --config_file zero3.yaml \
  src/open_r1/sft.py \
  --config recipes/config_distill.yaml \
  --max_steps 100 \
  --output_dir ./sft_zero3

# Record: memory usage, throughput, training time
```

### Task 2: Profile Memory Usage

```python
# Add to src/open_r1/sft.py
import torch

class MemoryTracker:
    def __init__(self):
        self.stats = []

    def log(self, step):
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        self.stats.append({
            "step": step,
            "allocated": allocated,
            "reserved": reserved,
        })

    def plot(self):
        import matplotlib.pyplot as plt
        steps = [s["step"] for s in self.stats]
        allocated = [s["allocated"] for s in self.stats]

        plt.plot(steps, allocated)
        plt.xlabel("Step")
        plt.ylabel("Memory (GB)")
        plt.title("GPU Memory Usage")
        plt.savefig("memory_profile.png")

tracker = MemoryTracker()

# In training loop
for step in range(max_steps):
    # ... training ...
    tracker.log(step)

tracker.plot()
```

### Task 3: Test Batch Size Limits

```python
# Find maximum batch size for each ZeRO stage
for zero_stage in [2, 3]:
    for batch_size in [4, 8, 16, 32, 64]:
        try:
            # Run 10 steps
            run_training(zero_stage, batch_size, max_steps=10)
            print(f"ZeRO-{zero_stage}, batch={batch_size}: SUCCESS")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"ZeRO-{zero_stage}, batch={batch_size}: OOM")
                break
```

**Expected Output:**

```
ZeRO-2, batch=4: SUCCESS
ZeRO-2, batch=8: SUCCESS
ZeRO-2, batch=16: SUCCESS
ZeRO-2, batch=32: OOM

ZeRO-3, batch=4: SUCCESS
ZeRO-3, batch=8: SUCCESS
ZeRO-3, batch=16: SUCCESS
ZeRO-3, batch=32: SUCCESS
ZeRO-3, batch=64: OOM
```

**Deliverable:**

- Table comparing ZeRO-2 vs ZeRO-3 (memory, throughput, max batch size)
- Memory profile graph for both configurations
- Recommendation for which to use for 7B model

---

## Summary

**Key Takeaways:**

1. **ZeRO partitions** optimizer states, gradients, and parameters across GPUs
2. **ZeRO-2** partitions optimizer + gradients (50% memory savings)
3. **ZeRO-3** partitions everything (75% memory savings, but slower)
4. **Use ZeRO-2** for models that fit (faster, less communication)
5. **Use ZeRO-3** for large models that don't fit in ZeRO-2

**Configuration Checklist:**

- ✅ Choose appropriate `zero_stage` (2 or 3)
- ✅ Set `gradient_accumulation_steps` consistently
- ✅ Enable `overlap_comm` for performance
- ✅ Set `stage3_gather_16bit_weights_on_model_save: true` for ZeRO-3
- ✅ Tune bucket sizes for your model

**Memory Formula:**

```
ZeRO-2 memory per GPU ≈ (Params + Grads/N + Optimizer/N + Activations)
ZeRO-3 memory per GPU ≈ (Params/N + Grads/N + Optimizer/N + Activations)

Where N = number of GPUs
```

**Next Tutorial:** FSDP as Alternative to DeepSpeed

---

## Additional Resources

- [DeepSpeed ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [DeepSpeed Documentation](https://www.deepspeed.ai/docs/)
- [Annotated Config: ZeRO-2](../../annotated_code/configs/zero2_ANNOTATED.yaml)
- [Annotated Config: ZeRO-3](../../annotated_code/configs/zero3_ANNOTATED.yaml)
- [HuggingFace Accelerate + DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)

**Questions?** Open an issue on GitHub!
