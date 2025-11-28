# Tutorial 9: Multi-GPU Training Strategies

**Target Audience:** Intermediate
**Duration:** 35 minutes
**Prerequisites:** Tutorial 6 (DeepSpeed ZeRO) or Tutorial 7 (FSDP)

## Table of Contents
1. [Overview](#overview)
2. [Data Parallelism Fundamentals](#data-parallelism-fundamentals)
3. [Tensor Parallelism](#tensor-parallelism)
4. [Pipeline Parallelism](#pipeline-parallelism)
5. [Hybrid Strategies](#hybrid-strategies)
6. [Choosing the Right Strategy](#choosing-the-right-strategy)
7. [Hands-On Example](#hands-on-example)
8. [Summary](#summary)

---

## Overview

**Multi-GPU training** is essential for DeepSeek R1. Different parallelism strategies offer different tradeoffs between memory, speed, and scalability.

**What you'll learn:**
- Data parallelism (standard approach)
- Tensor parallelism (split model layers)
- Pipeline parallelism (split model stages)
- When to use each strategy

**Parallelism Types:**

| Type | Splits | Best For | Complexity |
|------|--------|----------|------------|
| **Data Parallel** | Data across GPUs | Most use cases | Low |
| **Tensor Parallel** | Layers across GPUs | Very large models | Medium |
| **Pipeline Parallel** | Stages across nodes | Multi-node | High |
| **Hybrid** | Combination | 100B+ models | Very High |

---

## Data Parallelism Fundamentals

### How It Works

```
GPU 0: Full Model, Batch[0:4]
GPU 1: Full Model, Batch[4:8]
GPU 2: Full Model, Batch[8:12]
GPU 3: Full Model, Batch[12:16]

Each GPU processes different data
Gradients averaged across GPUs
```

**With ZeRO-3/FSDP:**
- Model sharded across GPUs
- Still processes different data per GPU
- Best of both worlds!

### Configuration

```yaml
# For DeepSeek R1 7B models
num_processes: 8  # 8 GPUs
distributed_type: DEEPSPEED  # or FSDP

# Effective batch size
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
# Total = 4 × 8 GPUs × 8 accum = 256 samples
```

---

## Tensor Parallelism

### Concept

**Split individual layers across GPUs:**

```
Standard (1 GPU):
  Layer: [W0 W1 W2 W3]  (14GB)

Tensor Parallel (4 GPUs):
  GPU 0: [W0]  (3.5GB)
  GPU 1: [W1]  (3.5GB)
  GPU 2: [W2]  (3.5GB)
  GPU 3: [W3]  (3.5GB)
```

### When to Use

- ✅ Model doesn't fit on single GPU (even with ZeRO-3)
- ✅ Need faster inference per request
- ❌ Communication overhead within node

**For DeepSeek R1 7B:** Usually NOT needed (ZeRO-3/FSDP sufficient)

**For larger models (70B+):** Essential

---

## Pipeline Parallelism

### Concept

**Split model stages across devices:**

```
GPU 0: Layers 0-7
GPU 1: Layers 8-15
GPU 2: Layers 16-23
GPU 3: Layers 24-31

Data flows: GPU 0 → GPU 1 → GPU 2 → GPU 3
```

### Microbatching

**Problem:** GPUs idle while waiting for previous stage

**Solution:** Split batch into microbatches

```
Time:   0ms   10ms   20ms   30ms   40ms
GPU 0:  [MB1] [MB2]  [MB3]  [MB4]
GPU 1:        [MB1]  [MB2]  [MB3]  [MB4]
GPU 2:               [MB1]  [MB2]  [MB3]
GPU 3:                      [MB1]  [MB2]

All GPUs busy most of the time!
```

### When to Use

- ✅ Multi-node training with slow interconnect
- ✅ Very large models (100B+)
- ❌ Additional implementation complexity

**For DeepSeek R1 7B:** Not needed

---

## Hybrid Strategies

### ZeRO + Data Parallel

**Default for DeepSeek R1:**

```yaml
# 8 GPUs, single node
num_processes: 8
distributed_type: DEEPSPEED
zero_stage: 3

# Result: Data parallel + ZeRO-3 sharding
```

**Scaling:**

| GPUs | Strategy | Batch Size | Throughput |
|------|----------|------------|------------|
| 1 | No parallel | OOM | N/A |
| 4 | ZeRO-3 | 64 | 48 samples/sec |
| 8 | ZeRO-3 | 128 | 92 samples/sec |

**Efficiency:** ~96% (nearly linear scaling!)

### Tensor + Data Parallel (for 70B models)

```bash
# vLLM configuration
vllm serve deepseek-ai/DeepSeek-R1-70B \
  --tensor-parallel-size 8 \  # Split each layer across 8 GPUs
  --pipeline-parallel-size 1   # No pipeline parallelism

# Within single node: tensor parallelism
# Across nodes: data parallelism
```

---

## Choosing the Right Strategy

### Decision Tree

```
Model size < 13B?
├─ Yes → Use ZeRO-3 or FSDP (data parallel)
│         Most efficient for DeepSeek R1 7B
│
└─ No → Model size < 70B?
    ├─ Yes → Use ZeRO-3 + multi-node data parallel
    │
    └─ No → Use tensor parallel (8 GPUs) + data parallel (across nodes)
```

### For DeepSeek R1 (7B models)

**Recommended:** ZeRO-3 or FSDP with data parallelism

```yaml
# 8× A100 80GB
num_processes: 8
distributed_type: FSDP  # or DEEPSPEED with zero_stage: 3
fsdp_sharding_strategy: FULL_SHARD
```

**Why:**
- ✅ Simplest configuration
- ✅ Best performance (96% efficiency)
- ✅ No communication overhead within node
- ✅ Scales to 64+ GPUs if needed

---

## Hands-On Example

### Example: Multi-GPU Training

```bash
# Single node, 8 GPUs
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  --num_processes 8 \
  src/open_r1/sft.py \
  --config recipes/config_distill.yaml
```

**Expected scaling:**

```
1 GPU: OOM
2 GPUs: 12 samples/sec
4 GPUs: 46 samples/sec (96% efficient)
8 GPUs: 92 samples/sec (96% efficient)
```

---

## Summary

**Key Takeaways:**

1. **Data parallelism** is default and most efficient
2. **ZeRO-3/FSDP** enable data parallelism for large models
3. **Tensor parallelism** for models >70B
4. **Pipeline parallelism** for multi-node with slow interconnect
5. **DeepSeek R1 7B:** Use ZeRO-3 or FSDP (data parallel)

**Configuration:**

```yaml
# Recommended for 7B models
num_processes: 8
distributed_type: FSDP
fsdp_sharding_strategy: FULL_SHARD
```

**Next Tutorial:** Slurm Job Submission

---

## Resources

- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [DeepSpeed Parallelism](https://www.deepspeed.ai/training/)

**Questions?** Open an issue on GitHub!
