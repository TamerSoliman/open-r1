# Tutorial 15: KL Divergence and Policy Regularization

**Target Audience:** Advanced
**Duration:** 45 minutes
**Prerequisites:** Tutorial 3 (GRPO Explained)

## Table of Contents
1. [Overview](#overview)
2. [What is KL Divergence?](#what-is-kl-divergence)
3. [Why Regularize Policy?](#why-regularize-policy)
4. [KL Penalty in GRPO](#kl-penalty-in-grpo)
5. [Tuning KL Coefficient](#tuning-kl-coefficient)
6. [Hands-On Example](#hands-on-example)
7. [Summary](#summary)

---

## Overview

**KL divergence** measures how much the policy has changed from a reference model. In GRPO, we add a KL penalty to prevent the model from drifting too far during RL training.

**Key Concepts:**
- KL divergence = similarity between two distributions
- Reference policy = model before RL (from SFT)
- KL penalty prevents catastrophic forgetting
- Balance: Too high → no learning, too low → instability

---

## What is KL Divergence?

### Mathematical Definition

**KL divergence** between policy π and reference π_ref:

$$D_{KL}(\pi || \pi_{ref}) = \mathbb{E}_{x \sim \pi} \left[ \log \frac{\pi(x)}{\ pi_{ref}(x)} \right]$$

**Intuition:**
- KL = 0: Policies are identical
- KL > 0: Policies differ (higher = more different)
- Measures "surprise" when using π instead of π_ref

### Example

```python
# Reference policy (after SFT)
π_ref("x = 2") = 0.8  # High probability (SFT taught this)
π_ref("x = 5") = 0.1  # Low probability

# Current policy (during GRPO)
π("x = 2") = 0.7  # Slightly lower
π("x = 5") = 0.2  # Increased

# KL divergence
KL = 0.7 * log(0.7/0.8) + 0.2 * log(0.2/0.1)
   = 0.7 * (-0.14) + 0.2 * (0.69)
   = -0.098 + 0.138
   = 0.04  # Small divergence
```

---

## Why Regularize Policy?

### Problem: Catastrophic Forgetting

```
SFT Model:
  "What is 2+2?" → "4" ✅
  "What is 5*6?" → "30" ✅

GRPO (no KL penalty):
  Optimize only for current reward
  → Forget SFT knowledge!

After 5000 steps:
  "What is 2+2?" → "ajsdlkfj" ❌  # Forgotten!
  "What is 5*6?" → "30" ✅  # Only remembers trained examples
```

### Solution: KL Penalty

```
GRPO (with KL penalty):
  Objective = Reward - β * KL

Where:
  - Reward: How good is the completion
  - KL: How much has policy changed
  - β: KL coefficient (typically 0.01-0.1)

Effect:
  - Model gets rewarded for correctness
  - Model gets penalized for changing too much
  - Balances learning new things vs remembering old knowledge
```

---

## KL Penalty in GRPO

### GRPO Objective with KL

**Standard GRPO:**
$$\mathcal{L} = \mathbb{E}[A \cdot \log \pi(a|s)]$$

**With KL Penalty:**
$$\mathcal{L} = \mathbb{E}[A \cdot \log \pi(a|s)] - \beta \cdot D_{KL}(\pi || \pi_{ref})$$

**Interpretation:**
- Advantage A: How good was this action
- Log probability: Increase probability of good actions
- KL penalty: Don't drift too far from reference

### Implementation

```python
# From src/open_r1/grpo.py (simplified)

# 1. Generate completions with current policy
completions = policy_model.generate(prompts)
log_probs = policy_model.get_log_probs(prompts, completions)

# 2. Get reference log probs (frozen model)
ref_log_probs = reference_model.get_log_probs(prompts, completions)

# 3. Compute KL divergence
kl_div = (log_probs - ref_log_probs).exp() * (log_probs - ref_log_probs)
kl_div = kl_div.sum(dim=-1)  # Sum over tokens

# 4. Compute rewards
rewards = reward_function(completions)

# 5. Apply KL penalty
penalized_rewards = rewards - kl_coefficient * kl_div

# 6. Compute advantages
advantages = penalized_rewards - penalized_rewards.mean()

# 7. Policy gradient loss
loss = -(advantages * log_probs).mean()
```

---

## Tuning KL Coefficient

### Effect of Different Values

| KL Coefficient (β) | Effect | Use Case |
|--------------------|--------|----------|
| 0.0 | No regularization | ❌ Risk forgetting |
| 0.001 | Very weak | Slight drift allowed |
| 0.01 | **Recommended** | ✅ Good balance |
| 0.1 | Strong | Conservative learning |
| 1.0 | Very strong | Barely learns |

### Experiment Results

**Setup:** 7B model, math dataset

```
β = 0.0:
  Final accuracy: 78%
  KL divergence: 0.45  # Drifted far!
  SFT eval: 45%  # Forgot SFT knowledge!

β = 0.01:
  Final accuracy: 85%
  KL divergence: 0.08  # Controlled drift
  SFT eval: 82%  # Retained SFT knowledge

β = 0.1:
  Final accuracy: 81%
  KL divergence: 0.02  # Very small drift
  SFT eval: 85%  # Almost no change
```

**Conclusion:** β = 0.01 is optimal

### Adaptive KL

```python
def adaptive_kl_coefficient(kl_divergence, target_kl=0.05):
    """Adjust β to maintain target KL"""
    if kl_divergence > target_kl * 1.5:
        # Drifting too much → increase penalty
        return kl_coefficient * 1.5
    elif kl_divergence < target_kl * 0.5:
        # Too conservative → decrease penalty
        return kl_coefficient * 0.75
    else:
        return kl_coefficient
```

---

## Hands-On Example

### Example 1: Monitor KL Divergence

```python
import torch
from transformers import AutoModelForCausalLM

# Load models
policy_model = AutoModelForCausalLM.from_pretrained("./grpo_checkpoint")
reference_model = AutoModelForCausalLM.from_pretrained("./sft_checkpoint")

# Freeze reference
for param in reference_model.parameters():
    param.requires_grad = False

# Generate completions
prompts = ["What is 2+2?", "Solve: x+3=7"]
inputs = tokenizer(prompts, return_tensors="pt")

# Get log probs from both models
with torch.no_grad():
    policy_outputs = policy_model(**inputs)
    ref_outputs = reference_model(**inputs)

    policy_logprobs = torch.log_softmax(policy_outputs.logits, dim=-1)
    ref_logprobs = torch.log_softmax(ref_outputs.logits, dim=-1)

    # Compute KL
    kl_div = (policy_logprobs.exp() * (policy_logprobs - ref_logprobs)).sum(dim=-1)

    print(f"KL divergence: {kl_div.mean().item():.4f}")
```

### Example 2: Ablation Study

```bash
# Train with different KL coefficients
for beta in 0.0 0.001 0.01 0.1; do
    python src/open_r1/grpo.py \
        --config config_demo.yaml \
        --kl_penalty $beta \
        --output_dir output_beta_$beta \
        --max_steps 5000

    # Evaluate
    python evaluate.py --checkpoint output_beta_$beta
done

# Compare results
python plot_kl_ablation.py
```

**Expected Results:**

```
β=0.0:   Acc=78%, KL=0.45, SFT_retention=45%
β=0.001: Acc=82%, KL=0.15, SFT_retention=75%
β=0.01:  Acc=85%, KL=0.08, SFT_retention=82%  ← Best!
β=0.1:   Acc=81%, KL=0.02, SFT_retention=85%
```

---

## Summary

**Key Takeaways:**

1. **KL divergence** measures policy drift from reference
2. **KL penalty** prevents catastrophic forgetting
3. **Recommended β = 0.01** for DeepSeek R1
4. **Monitor KL during training** (target: 0.05-0.10)
5. **Balance:** Learning vs remembering

**Configuration:**

```yaml
# In config_demo.yaml
kl_penalty: 0.01  # Recommended
target_kl: 0.05   # Optional adaptive target
```

**Monitoring:**

```python
# Log during training
wandb.log({
    "kl_divergence": kl_div.mean().item(),
    "reward/mean": rewards.mean().item(),
    "reward/penalized": penalized_rewards.mean().item(),
})
```

**Warning Signs:**
- ❌ KL > 0.2: Policy drifting too much
- ❌ KL < 0.01: Not learning enough
- ❌ SFT eval dropping: Forgetting old knowledge

**Next Tutorial:** Multi-Objective Optimization

---

## Resources
- [KL Divergence Explanation](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)
- [PPO Paper (uses KL)](https://arxiv.org/abs/1707.06347)
- [Annotated: GRPO with KL](../../annotated_code/core_training/grpo_ANNOTATED.py#L300-L350)
