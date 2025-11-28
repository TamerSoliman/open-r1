# Tutorial 24: Leaderboard Submission and Reproducibility

**Target Audience:** Advanced
**Duration:** 25 minutes

## Overview

**Leaderboard submission** requires:
- Reproducible results
- Proper documentation
- Model card with details

## Submission Checklist

```markdown
## Model Information
- Base model: Qwen/Qwen-7B
- Training: SFT + GRPO
- Dataset: OpenR1-Math-220k + OpenR1-Code-100k
- Training steps: 30,000
- Hardware: 8×A100 80GB

## Benchmark Results
- MMLU: 68.2%
- GSM8K: 82.1%
- HumanEval: 45.3%
- MATH: 38.7%

## Reproducibility
- Random seed: 42
- Exact config: config.yaml (attached)
- Checkpoint: model-30000/
```

## Uploading to Hub

```python
# Push model
model.push_to_hub("username/deepseek-r1-7b-math")

# Push eval results
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="eval_results.json",
    path_in_repo="eval_results.json",
    repo_id="username/deepseek-r1-7b-math",
)
```

## Leaderboard Submission

```bash
# Open LLM Leaderboard
python submit_to_leaderboard.py \
  --model username/deepseek-r1-7b-math \
  --tasks mmlu,gsm8k,humaneval \
  --precision float16
```

## Summary

- **Document everything** (config, data, hardware)
- **Include eval results** in model card
- **Upload to Hub** for visibility
- **Submit to leaderboards** for ranking

**Next Tutorial:** Error Analysis

## Resources
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
