# Tutorial 10: Slurm Job Submission for Training

**Target Audience:** Intermediate
**Duration:** 30 minutes
**Prerequisites:** Tutorial 9 (Multi-GPU Strategies)

## Table of Contents
1. [Overview](#overview)
2. [Slurm Basics](#slurm-basics)
3. [Job Script Structure](#job-script-structure)
4. [Multi-Node Training](#multi-node-training)
5. [Common Commands](#common-commands)
6. [Hands-On Example](#hands-on-example)
7. [Summary](#summary)

---

## Overview

**Slurm** (Simple Linux Utility for Resource Management) is a cluster job scheduler used for running large-scale training jobs.

**What you'll learn:**
- Slurm basics and terminology
- Writing job scripts for training
- Multi-node distributed training
- Monitoring and managing jobs

---

## Slurm Basics

### Key Concepts

**Node:** Physical server with GPUs
**Job:** Task submitted to run on cluster
**Partition:** Group of nodes (e.g., "gpu", "cpu")
**Allocation:** Resources assigned to your job

### Common Slurm Directives

```bash
#!/bin/bash
#SBATCH --job-name=deepseek-r1      # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --gpus-per-node=8            # GPUs per node
#SBATCH --cpus-per-task=64           # CPUs per task
#SBATCH --time=48:00:00              # Max runtime (48 hours)
#SBATCH --partition=gpu              # Queue/partition
#SBATCH --output=logs/%x-%j.out      # Output file
#SBATCH --error=logs/%x-%j.err       # Error file
```

---

## Job Script Structure

### Basic Training Job

```bash
#!/bin/bash
#SBATCH --job-name=sft-training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Load modules
module load cuda/12.1
module load python/3.10

# Activate environment
source ~/envs/deepseek/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

# Run training
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  src/open_r1/sft.py \
  --config recipes/config_distill.yaml \
  --output_dir /scratch/$USER/output_${SLURM_JOB_ID}
```

### Submit Job

```bash
# Submit
sbatch train_sft.slurm

# Check status
squeue -u $USER

# Cancel job
scancel <job_id>
```

---

## Multi-Node Training

### Multi-Node Script

```bash
#!/bin/bash
#SBATCH --nodes=4                    # 4 nodes
#SBATCH --gpus-per-node=8            # 8 GPUs each = 32 total
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00

# Get node list
NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
MASTER_NODE=$(echo $NODELIST | cut -d' ' -f1)
MASTER_PORT=29500

# Export for all nodes
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=$MASTER_PORT

# Run training (Slurm handles multi-node)
srun accelerate launch \
  --config_file recipes/accelerate_configs/zero3_multinode.yaml \
  --num_processes 32 \
  --num_machines 4 \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  src/open_r1/sft.py \
  --config recipes/config_distill.yaml
```

---

## Common Commands

### Job Management

```bash
# Submit job
sbatch script.slurm

# Check queue
squeue -u $USER

# Job details
scontrol show job <job_id>

# Cancel job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

### Monitoring

```bash
# Watch job output
tail -f logs/sft-training-12345.out

# Check GPU usage
srun --jobid=<job_id> nvidia-smi

# SSH to allocated node
srun --jobid=<job_id> --pty bash
```

---

## Hands-On Example

### Example 1: Submit SFT Job

```bash
# Create job script
cat > train_sft.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=sft-7b
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

module load cuda python
source ~/env/bin/activate

accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  src/open_r1/sft.py \
  --config recipes/config_distill.yaml
EOF

# Submit
sbatch train_sft.slurm

# Output:
# Submitted batch job 12345

# Check status
squeue -j 12345
```

### Example 2: Monitor Training

```bash
# Watch logs
tail -f logs/sft-7b-12345.out

# Check GPU usage
watch -n 1 'srun --jobid=12345 nvidia-smi'
```

---

## Summary

**Key Takeaways:**

1. **Slurm manages cluster resources** for training jobs
2. **Job scripts specify resources** (nodes, GPUs, time)
3. **Multi-node training** requires coordination (MASTER_ADDR/PORT)
4. **Monitor jobs** with squeue, scontrol, logs

**Basic Workflow:**

```bash
# 1. Write job script
vim train.slurm

# 2. Submit
sbatch train.slurm

# 3. Monitor
squeue -u $USER
tail -f logs/job.out

# 4. Results in output_dir
```

**Next Tutorial:** WandB Integration

---

## Resources

- [Slurm Documentation](https://slurm.schedmd.com/)
- [Annotated Script: generate.slurm](../../annotated_code/slurm/generate_ANNOTATED.slurm)

**Questions?** Open an issue on GitHub!
