# Tutorial 29: Knowledge Distillation Best Practices

**Target Audience:** Advanced
**Duration:** 30 minutes

## Overview

**Knowledge distillation** transfers knowledge from large teacher (DeepSeek-R1 70B) to small student (7B).

## Basic Distillation

```python
# Generate training data from teacher
teacher = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-70B")

training_data = []
for problem in problems:
    # Teacher generates solution
    solution = teacher.generate(problem, max_tokens=2048)
    training_data.append({
        "problem": problem,
        "solution": solution,
    })

# Train student on teacher's outputs
student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B")
trainer = Trainer(model=student, train_dataset=training_data)
trainer.train()
```

## Advanced: Temperature Scaling

```python
# Generate with higher temperature for diversity
solutions = teacher.generate(
    problem,
    num_return_sequences=5,
    temperature=0.8,  # More diverse
)

# Filter by quality
good_solutions = [s for s in solutions if verify(s)]
```

## Summary

- **Teacher model** (70B) generates training data
- **Student model** (7B) learns from teacher
- **Temperature scaling** for diversity
- **Quality filtering** improves results

**Next Tutorial:** Scaling to 70B Models (FINAL)

## Resources
- [Distillation Paper](https://arxiv.org/abs/1503.02531)
- [DeepSeek Distillation](https://arxiv.org/abs/2401.14196)
