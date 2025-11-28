# Tutorial 30: Scaling to 70B Models

**Target Audience:** Advanced
**Duration:** 35 minutes

## Overview

**Training 70B models** requires advanced parallelism strategies and careful resource management.

## Hardware Requirements

```
Minimum: 8× A100 80GB (640 GB total)
Recommended: 16× A100 80GB or 8× H100 80GB
```

## Configuration

```yaml
# config_70b.yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 3  # Required for 70B
  offload_optimizer_device: cpu  # Offload to CPU if needed
  offload_param_device: cpu
  stage3_prefetch_bucket_size: 5e8
  stage3_param_persistence_threshold: 1e6

num_processes: 16  # 16 GPUs
mixed_precision: bf16
gradient_checkpointing: true
```

## Training Command

```bash
# Multi-node training (2 nodes × 8 GPUs)
accelerate launch \
  --config_file config_70b.yaml \
  --num_machines 2 \
  --machine_rank 0 \
  --main_process_ip node0.cluster \
  --main_process_port 29500 \
  src/open_r1/sft.py \
  --config recipes/config_70b.yaml \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64
```

## Memory Breakdown (70B model, 16 GPUs)

```
Parameters: 140 GB → 8.75 GB per GPU (sharded)
Optimizer: 280 GB → 17.5 GB per GPU (sharded)
Gradients: 140 GB → 8.75 GB per GPU (sharded)
Activations: ~20 GB per GPU (with checkpointing)
──────────────────────────────────────────────
Total: ~55 GB per GPU (fits in A100 80GB!)
```

## Performance

```
Training Speed:
  8× A100: 2.5 samples/sec
  16× A100: 4.8 samples/sec (96% efficient)
  8× H100: 5.2 samples/sec (faster GPUs)
```

## Best Practices

1. **Use ZeRO-3** (required)
2. **Gradient checkpointing** (required)
3. **CPU offloading** if memory tight
4. **Multi-node** for faster training
5. **Monitor memory** carefully

## Summary

- **70B models** require 8-16 A100 GPUs
- **ZeRO-3 + checkpointing** essential
- **Multi-node** scales efficiently
- **Throughput:** ~5 samples/sec on 16× A100

**🎉 PHASE 5 COMPLETE! All 30 tutorials finished! 🎉**

---

## Resources
- [DeepSpeed ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
- [Large Model Training Guide](https://huggingface.co/docs/transformers/perf_train_gpu_many)
- [Annotated: ZeRO-3 Config](../../annotated_code/configs/zero3_ANNOTATED.yaml)

**Congratulations on completing all Phase 5 Advanced Tutorials!**
