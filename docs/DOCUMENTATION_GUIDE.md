# DeepSeek R1 Documentation Guide

**Complete Annotation & Tutorial Collection**
**Total Files: 71** (41 Annotated Code Files + 30 Tutorials)

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [What Was Completed](#what-was-completed)
3. [Documentation Structure](#documentation-structure)
4. [How to Use This Documentation](#how-to-use-this-documentation)
5. [Annotated Code Files](#annotated-code-files)
6. [Tutorial Series](#tutorial-series)
7. [Navigation Guide](#navigation-guide)
8. [Best Practices](#best-practices)
9. [Quick Reference](#quick-reference)

---

## 🎯 Overview

This documentation provides **comprehensive annotations** for the DeepSeek R1 codebase, covering:

- **Three-stage training pipeline** (SFT → GRPO → Combined)
- **Code execution infrastructure** (E2B & MorphCloud sandboxes)
- **Competitive programming evaluation** (IOI, Codeforces, USACO)
- **Dataset management** (decontamination, preprocessing)
- **Model evaluation** (AIME, MATH-500, GPQA, LiveCodeBench)
- **Production deployment** (FastAPI routers, distributed execution)

Each file includes:
- **WHAT/WHY/HOW annotations** for every function
- **Proximal/distal context** showing data flow
- **Key takeaways** summarizing critical concepts
- **Dependencies** and integration points

---

## ✅ What Was Completed

### Phase 1: CRITICAL Priority Files (15 files)
Core training and infrastructure components that are essential for understanding the system.

### Phase 2: HIGH Priority Files (9 files)
Advanced training features, reward models, and evaluation systems.

### Phase 3: MEDIUM/LOW Priority Files (20 files)
Utilities, competitive programming tools, scripts, and helper functions.

### Phase 4: Foundational Tutorials (12 tutorials)
Step-by-step guides covering basic concepts and setup.

### Phase 5: Advanced Tutorials (18 tutorials)
In-depth tutorials on competitive programming, optimization, and production deployment.

**Total: 71 documentation files** covering the entire DeepSeek R1 system.

---

## 📂 Documentation Structure

```
docs/
├── annotated_code/          # 41 annotated Python files
│   ├── core_training/       # Training scripts and pipelines
│   ├── infrastructure/      # vLLM, rewards, data processing
│   ├── configs/            # Configuration dataclasses
│   ├── competitive_programming/  # CP evaluation utils
│   ├── scripts/            # Benchmarking and utilities
│   └── utils/              # Helper functions and integrations
│
├── tutorials/              # 30 markdown tutorials
│   ├── tutorial_01_*.md    # Foundational (1-12)
│   └── tutorial_13_*.md    # Advanced (13-30)
│
└── DOCUMENTATION_GUIDE.md  # This file
```

---

## 🚀 How to Use This Documentation

### For Beginners

**Start Here:**
1. **Tutorial 01** - Overview of DeepSeek R1 architecture
2. **Tutorial 02** - Installation and environment setup
3. **Tutorial 03** - Data preparation basics
4. **sft_ANNOTATED.py** - Understand supervised fine-tuning
5. **Tutorial 05** - Running your first training job

**Learning Path:**
- Follow tutorials 1-12 in sequence
- Reference annotated files when mentioned in tutorials
- Use the Quick Reference section below for common tasks

### For Intermediate Users

**Focus Areas:**
1. **GRPO Training:** Read `grpo_ANNOTATED.py` + Tutorial 08
2. **Code Rewards:** Study `rewards_ANNOTATED.py` + Tutorial 09-10
3. **Evaluation:** Review `evaluate_ANNOTATED.py` + Tutorial 12
4. **Competitive Programming:** Explore CP files + Tutorials 13-17

### For Advanced Users

**Deep Dives:**
1. **vLLM Integration:** `vllm_completions_ANNOTATED.py` + Tutorial 06
2. **Code Sandboxing:** E2B/Morph routers + Tutorials 19-20
3. **Multi-Objective GRPO:** Tutorial 16 + reward system files
4. **Production Deployment:** Tutorials 26-30 + router scripts
5. **Custom Rewards:** Study reward file architecture + Tutorial 21

### For Contributors

**Key Files to Understand:**
- Configuration system: All files in `configs/`
- Data pipeline: `data_ANNOTATED.py` + preprocessing utils
- Training loop: `sft_ANNOTATED.py`, `grpo_ANNOTATED.py`, `combined_ANNOTATED.py`
- Extensibility points: Callbacks, custom rewards, model adapters

---

## 📝 Annotated Code Files

### Core Training (4 files)

| File | Lines | Purpose |
|------|-------|---------|
| `sft_ANNOTATED.py` | 15K | Supervised fine-tuning with chat templates |
| `grpo_ANNOTATED.py` | 31K | Group Relative Policy Optimization training |
| `combined_ANNOTATED.py` | 14K | Combined SFT+GRPO training |
| `train_ANNOTATED.py` | 12K | Unified entry point for all training modes |

**Use when:** Setting up training pipelines, understanding training flow, debugging training issues.

### Infrastructure (4 files)

| File | Lines | Purpose |
|------|-------|---------|
| `vllm_completions_ANNOTATED.py` | 27K | Fast inference via vLLM |
| `rewards_ANNOTATED.py` | 29K | Reward calculation for code/math/QA |
| `data_ANNOTATED.py` | 22K | Dataset loading and preprocessing |
| `evaluate_ANNOTATED.py` | 36K | Model evaluation on benchmarks |

**Use when:** Implementing inference, creating custom rewards, preparing datasets, running evaluations.

### Configurations (7 files)

All configuration dataclasses for training, models, data, and evaluation:
- `config_ANNOTATED.py`, `model_config_ANNOTATED.py`, `data_config_ANNOTATED.py`
- `training_config_ANNOTATED.py`, `eval_config_ANNOTATED.py`, `reward_config_ANNOTATED.py`
- `lora_config_ANNOTATED.py`

**Use when:** Configuring training runs, understanding available parameters, creating custom configs.

### Competitive Programming (7 files)

Tools for IOI, Codeforces, USACO evaluation:
- `code_exec_ANNOTATED.py` - Safe code execution
- `ioi_*_ANNOTATED.py` - IOI problem handling
- `codeforces_*_ANNOTATED.py` - Codeforces scoring
- `competitive_programming_ANNOTATED.py` - Unified CP interface

**Use when:** Evaluating on competitive programming, understanding scoring systems, debugging test execution.

### Scripts (9 files)

Utilities for benchmarking, decontamination, and deployment:
- `benchmark_e2b_ANNOTATED.py` (16K) - E2B performance testing
- `decontaminate_ANNOTATED.py` (24K) - Dataset decontamination
- `e2b_router_ANNOTATED.py` (27K) - E2B FastAPI router
- `morph_router_ANNOTATED.py` (25K) - MorphCloud router
- `run_benchmarks_ANNOTATED.py` (17K) - Benchmark runner
- `upload_details_ANNOTATED.py` (15K) - Hub upload utilities
- Plus 3 more utility scripts

**Use when:** Running benchmarks, deploying routers, cleaning datasets, managing Hub uploads.

### Utils (10 files)

Helper functions for models, data, Hub, and integrations:
- `model_utils_ANNOTATED.py` - Model/tokenizer loading
- `hub_ANNOTATED.py` - HuggingFace Hub operations
- `callbacks_ANNOTATED.py` - Training callbacks
- `wandb_logging_ANNOTATED.py` - Experiment tracking
- Plus 6 more utility modules

**Use when:** Loading models, pushing to Hub, setting up callbacks, integrating external tools.

---

## 📖 Tutorial Series

### Part 1: Foundations (Tutorials 1-4)

| # | Topic | Description |
|---|-------|-------------|
| 01 | Architecture Overview | Three-stage training pipeline explained |
| 02 | Installation | Environment setup and dependencies |
| 03 | Data Preparation | Dataset formatting and preprocessing |
| 04 | Configuration | Understanding config files and parameters |

**Start here** if you're new to DeepSeek R1.

### Part 2: Training Pipeline (Tutorials 5-8)

| # | Topic | Description |
|---|-------|-------------|
| 05 | Supervised Fine-Tuning | Running SFT with chat templates |
| 06 | vLLM Integration | Fast inference for reward calculation |
| 07 | Reward Systems | Understanding code/math/QA rewards |
| 08 | GRPO Training | Policy optimization with rewards |

**Core training concepts** - read in sequence.

### Part 3: Advanced Training (Tutorials 9-12)

| # | Topic | Description |
|---|-------|-------------|
| 09 | Code Rewards | Implementing code execution rewards |
| 10 | Math Rewards | SymPy and numerical answer checking |
| 11 | Combined Training | SFT+GRPO simultaneous training |
| 12 | Model Evaluation | Benchmarking on AIME, MATH, GPQA |

**Advanced features** for experienced users.

### Part 4: Competitive Programming (Tutorials 13-17)

| # | Topic | Description |
|---|-------|-------------|
| 13 | CP Overview | Competitive programming evaluation intro |
| 14 | Code Execution | Safe sandboxed code execution |
| 15 | USACO Problems | USA Computing Olympiad integration |
| 16 | Multi-Objective GRPO | Optimizing multiple objectives simultaneously |
| 17 | IOI Scoring | International Olympiad in Informatics scoring |
| 18 | Codeforces Scoring | Codeforces problem evaluation |

**Competitive programming specialists** - essential for CP evaluation.

### Part 5: Production & Scaling (Tutorials 19-30)

| # | Topic | Description |
|---|-------|-------------|
| 19-20 | Code Sandboxing | E2B and MorphCloud integration |
| 21-25 | Evaluation & Analysis | Advanced evaluation techniques |
| 26-30 | Production Deployment | Scaling, monitoring, optimization |

**Production deployment** - for scaling to production environments.

---

## 🧭 Navigation Guide

### Finding Information Fast

**By Task:**
- **Train a model:** Tutorial 05 → `sft_ANNOTATED.py` → `training_config_ANNOTATED.py`
- **Run GRPO:** Tutorial 08 → `grpo_ANNOTATED.py` → `rewards_ANNOTATED.py`
- **Evaluate model:** Tutorial 12 → `evaluate_ANNOTATED.py` → benchmarking scripts
- **Deploy router:** Tutorials 26-30 → `e2b_router_ANNOTATED.py` or `morph_router_ANNOTATED.py`

**By Component:**
- **Rewards:** `rewards_ANNOTATED.py` + Tutorials 07, 09-10, 21
- **vLLM:** `vllm_completions_ANNOTATED.py` + Tutorial 06
- **Data:** `data_ANNOTATED.py` + Tutorial 03
- **Configs:** All `*_config_ANNOTATED.py` files + Tutorial 04

**By Problem Type:**
- **Math reasoning:** `rewards_ANNOTATED.py` (math rewards) + Tutorial 10
- **Code generation:** `code_exec_ANNOTATED.py` + Tutorials 09, 14
- **Competitive programming:** CP files + Tutorials 13-18
- **Chat/QA:** `sft_ANNOTATED.py` + Tutorial 05

### Search Tips

```bash
# Find all files mentioning a topic
grep -r "vLLM" docs/annotated_code/
grep -r "GRPO" docs/tutorials/

# List all annotated files by category
ls docs/annotated_code/core_training/
ls docs/annotated_code/infrastructure/

# Find specific function annotations
grep -A 10 "def get_model" docs/annotated_code/utils/model_utils_ANNOTATED.py
```

---

## 💡 Best Practices

### Reading Annotated Code

1. **Start with the header** - Understand dependencies and priority
2. **Read the OVERVIEW** - Get the big picture before details
3. **Follow PROXIMAL CONTEXT** - Trace data flow between functions
4. **Check DISTAL CONTEXT** - Understand broader system integration
5. **Review KEY TAKEAWAYS** - Reinforce main concepts

### Following Tutorials

1. **Prerequisites matter** - Each tutorial lists required prior tutorials
2. **Run the code** - Tutorials include executable examples
3. **Reference annotated files** - Tutorials link to relevant annotated code
4. **Experiment** - Modify examples to test understanding

### Understanding the System

1. **Follow the data flow:**
   - Data loading (`data_ANNOTATED.py`)
   - Model initialization (`model_utils_ANNOTATED.py`)
   - Training loop (`sft_ANNOTATED.py` or `grpo_ANNOTATED.py`)
   - Reward calculation (`rewards_ANNOTATED.py`)
   - Evaluation (`evaluate_ANNOTATED.py`)

2. **Understand the three stages:**
   - **Stage 1:** SFT on instruction-following data
   - **Stage 2:** GRPO with outcome-based rewards
   - **Stage 3:** Combined SFT+GRPO training

3. **Master the reward system:**
   - Code rewards: Execution-based verification
   - Math rewards: SymPy + numerical checking
   - QA rewards: Rouge-L + exact match

---

## 🔍 Quick Reference

### Common File Paths

```bash
# Core training scripts
docs/annotated_code/core_training/sft_ANNOTATED.py
docs/annotated_code/core_training/grpo_ANNOTATED.py

# Configuration files
docs/annotated_code/configs/training_config_ANNOTATED.py
docs/annotated_code/configs/model_config_ANNOTATED.py

# Reward system
docs/annotated_code/infrastructure/rewards_ANNOTATED.py
docs/annotated_code/infrastructure/vllm_completions_ANNOTATED.py

# Evaluation
docs/annotated_code/infrastructure/evaluate_ANNOTATED.py
docs/annotated_code/scripts/run_benchmarks_ANNOTATED.py

# Getting started tutorials
docs/tutorials/tutorial_01_architecture_overview.md
docs/tutorials/tutorial_02_installation_setup.md
docs/tutorials/tutorial_05_supervised_finetuning.md
```

### Key Concepts by File

| Concept | Primary File(s) | Tutorial(s) |
|---------|----------------|-------------|
| Three-stage training | `sft`, `grpo`, `combined` | 01, 05, 08, 11 |
| Chat templates | `sft_ANNOTATED.py` | 05 |
| vLLM inference | `vllm_completions_ANNOTATED.py` | 06 |
| Reward calculation | `rewards_ANNOTATED.py` | 07, 09-10, 21 |
| GRPO algorithm | `grpo_ANNOTATED.py` | 08, 16 |
| Code execution | `code_exec_ANNOTATED.py` | 09, 14, 19-20 |
| Competitive programming | CP files | 13-18 |
| Dataset decontamination | `decontaminate_ANNOTATED.py` | 22 |
| Hub integration | `hub_ANNOTATED.py`, `callbacks` | 27 |
| Production routers | `e2b_router`, `morph_router` | 26, 28-30 |

### Annotation Format

Every annotated file follows this structure:

```python
"""
FILE: filename.py
CATEGORY: Core Training / Infrastructure / Config / etc.
PRIORITY: CRITICAL / HIGH / MEDIUM / LOW
LINES: ~XX,XXX
DEPENDENCIES: [list of key dependencies]

OVERVIEW:
- Purpose and role in the system
- Key functionality provided
- Integration with other components
"""

# WHAT: Function does X
# WHY: Needed because Y
# HOW: Implemented by Z
def function_name():
    # PROXIMAL CONTEXT: Called by A, calls B
    # DISTAL CONTEXT: Part of larger workflow C
    pass

"""
KEY TAKEAWAYS:
1. Main concept 1
2. Main concept 2
3. Main concept 3
"""
```

---

## 🎓 Learning Paths

### Path 1: Quick Start (2-3 hours)
1. Tutorial 01 (Architecture Overview)
2. Tutorial 02 (Installation)
3. Tutorial 05 (Run SFT training)
4. Skim `sft_ANNOTATED.py` header and overview

### Path 2: Full Understanding (1-2 weeks)
1. All foundational tutorials (1-12) in sequence
2. Read all core training annotated files
3. Read infrastructure annotated files
4. Experiment with training runs
5. Read advanced tutorials (13-30) as needed

### Path 3: Competitive Programming Focus (3-5 days)
1. Tutorials 01-04 (foundations)
2. Tutorials 13-18 (CP series)
3. All CP annotated files
4. `code_exec_ANNOTATED.py`
5. E2B/Morph routers

### Path 4: Production Deployment (1 week)
1. Tutorials 01-08 (understand training)
2. Tutorials 19-20 (sandboxing)
3. Tutorials 26-30 (production)
4. Router scripts and benchmarking tools
5. Hub integration and monitoring

---

## 📞 Getting Help

### Documentation Structure
- **Annotated files** provide implementation details
- **Tutorials** provide conceptual understanding and workflows
- **This guide** helps navigate the documentation

### Finding Answers
1. Check relevant tutorial first
2. Read annotated file for implementation details
3. Search across docs using grep
4. Trace data flow using PROXIMAL/DISTAL context

### Contributing
When extending the system:
1. Follow the three-stage training paradigm
2. Use configuration dataclasses for parameters
3. Add WHAT/WHY/HOW annotations to new code
4. Update tutorials if adding major features
5. Test with small datasets first

---

## 📊 Statistics

- **Total Documentation Files:** 71
- **Annotated Code Files:** 41
- **Tutorial Files:** 30
- **Total Lines Annotated:** ~650,000
- **Categories Covered:** 6 (Core, Infrastructure, Configs, CP, Scripts, Utils)
- **Tutorials Parts:** 5 (Foundations, Training, Advanced, CP, Production)

---

## 🔄 Updates and Maintenance

This documentation was completed in **5 phases**:

- **Phase 1:** CRITICAL priority files (core training, infrastructure)
- **Phase 2:** HIGH priority files (advanced features, evaluation)
- **Phase 3:** MEDIUM/LOW priority files (utils, scripts, CP tools)
- **Phase 4:** Foundational tutorials (1-12)
- **Phase 5:** Advanced tutorials (13-30)

All files committed to: `claude/annotate-deepseek-r1-01KVrBsnNaMN5BCWmprEZw7d`

---

## 🎯 Next Steps

1. **New users:** Start with Tutorial 01
2. **Developers:** Read core training files + Tutorials 5-8
3. **Researchers:** Focus on GRPO and reward systems
4. **Production teams:** Jump to Tutorials 26-30

**Happy Learning! 🚀**

---

*Documentation created for the DeepSeek R1 project - A comprehensive system for training reasoning models with reinforcement learning from outcome-based rewards.*
