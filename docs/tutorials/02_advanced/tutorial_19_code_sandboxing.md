# Tutorial 19: Code Execution Sandboxing (E2B/Morph)

**Target Audience:** Advanced
**Duration:** 35 minutes
**Prerequisites:** Annotated code_providers.py

## Table of Contents
1. [Overview](#overview)
2. [Why Sandboxing?](#why-sandboxing)
3. [E2B Provider](#e2b-provider)
4. [Morph Provider](#morph-provider)
5. [Timeout Layers](#timeout-layers)
6. [Summary](#summary)

---

## Overview

**Sandboxing** isolates code execution to prevent:
- ❌ System damage
- ❌ Infinite loops
- ❌ Memory exhaustion
- ❌ Network access

**Providers:** E2B, Morph (abstracted via unified interface)

---

## Why Sandboxing?

### Unsafe Code Examples

```python
# Example 1: Infinite loop
while True:
    pass  # Never terminates!

# Example 2: Fork bomb
import os
while True:
    os.fork()  # Crash system!

# Example 3: File deletion
import os
os.system("rm -rf /")  # DON'T RUN THIS!
```

**Solution:** Run in isolated sandbox with:
- Time limits (e.g., 10 seconds)
- Memory limits (e.g., 512 MB)
- No file system access
- No network access

---

## E2B Provider

### Setup

```python
from e2b import Sandbox

# Create sandbox
sandbox = Sandbox(
    template="python",
    timeout=10,  # 10 second limit
)

# Execute code
result = sandbox.run_code("""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

print(factorial(5))
""")

print(result.stdout)  # "120"
sandbox.close()
```

### Features

- ✅ Cloud-based sandboxes
- ✅ Multiple language support
- ✅ Fast startup (<1s)
- ✅ Automatic cleanup

---

## Morph Provider

### Setup

```python
from morph import execute_code

result = execute_code(
    code="print(2 + 2)",
    language="python",
    timeout=5,
)

print(result["output"])  # "4"
```

---

## Timeout Layers

### Three-Layer Timeout

```python
async def execute_with_timeout(code, timeout=10):
    """
    Layer 1: Provider timeout (10s)
    Layer 2: Async timeout (12s, allows cleanup)
    Layer 3: Process timeout (15s, hard kill)
    """
    try:
        result = await asyncio.wait_for(
            provider.execute(code, timeout=10),  # Layer 1
            timeout=12,  # Layer 2
        )
        return result
    except asyncio.TimeoutError:
        return {"error": "Timeout", "output": ""}
```

**Why 3 layers?**
- Layer 1: Provider's internal timeout
- Layer 2: Our async timeout (graceful)
- Layer 3: OS-level timeout (forceful)

---

## Summary

**Key Takeaways:**

1. **Always sandbox** untrusted code
2. **E2B/Morph** provide secure execution
3. **Multiple timeout layers** for reliability
4. **10-second timeout** typical for competitive programming

**Configuration:**
```yaml
code_execution:
  provider: e2b  # or morph
  timeout: 10
  memory_limit: 512  # MB
```

**Part 4 COMPLETE!** Moving to Part 5...

---

## Resources
- [E2B Documentation](https://e2b.dev/docs)
- [Annotated: code_providers.py](../../annotated_code/infrastructure/code_providers_ANNOTATED.py)
