# Tutorial 26: Model Quantization (AWQ, GPTQ)

**Target Audience:** Advanced
**Duration:** 30 minutes

## Overview

**Quantization** reduces model size and increases speed by using lower precision (4-bit instead of 16-bit).

## Quantization Methods

### AWQ (Activation-aware Weight Quantization)

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load model
model = AutoAWQForCausalLM.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")

# Quantize to 4-bit
model.quantize(
    tokenizer,
    quant_config={"zero_point": True, "q_group_size": 128}
)

# Save quantized model
model.save_quantized("./my_model_awq")
```

### GPTQ

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
)

# Quantize
model = AutoGPTQForCausalLM.from_pretrained("./my_model", quantize_config)
model.quantize(calibration_dataset)
model.save_quantized("./my_model_gptq")
```

## Comparison

| Method | Speed | Quality | Memory |
|--------|-------|---------|--------|
| FP16 | 1.0× | 100% | 14 GB |
| AWQ 4-bit | 1.8× | 98% | 4 GB |
| GPTQ 4-bit | 1.7× | 97% | 4 GB |

## Summary

- **AWQ/GPTQ** reduce memory 3-4×
- **Speed increase** 1.7-1.8×
- **Quality loss** <3%
- **Essential** for deployment

**Next Tutorial:** vLLM Production Serving

## Resources
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
