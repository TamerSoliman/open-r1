# Tutorial 27: Serving with vLLM in Production

**Target Audience:** Advanced
**Duration:** 30 minutes

## Overview

**vLLM production serving** for high-throughput inference at scale.

## Production Setup

```bash
# Start vLLM server with production settings
vllm serve model/path \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \  # High concurrency
  --trust-remote-code
```

## Load Balancing

```python
# nginx config for load balancing
upstream vllm_backend {
    server vllm-0:8000;
    server vllm-1:8000;
    server vllm-2:8000;
    server vllm-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://vllm_backend;
    }
}
```

## Monitoring

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

requests_total = Counter('vllm_requests_total', 'Total requests')
latency = Histogram('vllm_latency_seconds', 'Request latency')

@app.route('/generate')
def generate():
    with latency.time():
        result = vllm_client.generate(...)
        requests_total.inc()
    return result
```

## Summary

- **vLLM** for production inference
- **Load balancing** across multiple servers
- **Monitoring** with Prometheus
- **Auto-scaling** based on load

**Next Tutorial:** Long-Context Training

## Resources
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
