"""
==============================================================================
FILE: scripts/generate_reasoning.py
CATEGORY: Scripts - Synthetic Reasoning Generation
PRIORITY: HIGH
LINES: 175
DEPENDENCIES:
    - aiohttp: Async HTTP client
    - aiofiles: Async file I/O
    - uvloop: Fast asyncio event loop
    - datasets: HuggingFace datasets
==============================================================================

OVERVIEW:
Async script for generating synthetic reasoning traces by querying an LLM API
(typically vLLM server) in parallel. Supports resume from interruption and
concurrent generation with configurable limits.

ROLE IN DEEPSEEK R1:
- Generates Mixture-of-Thoughts dataset for SFT training
- Queries DeepSeek-R1 API to produce reasoning traces
- Handles large-scale generation (millions of examples)
- Robust to failures with retry logic and resumption

KEY FEATURES:
1. Async I/O: Process 1000+ requests concurrently
2. Resumable: Tracks processed examples, resumes from interruption
3. Retry Logic: 10 retries per request with backoff
4. Progress Tracking: tqdm progress bar with active task count
5. JSONL Output: Streaming writes, no memory overhead

DATA FLOW:
Dataset → Load examples → Generate completions (async) → Write JSONL
   → Mixture-of-Thoughts dataset → SFT training

TYPICAL USAGE:
python scripts/generate_reasoning.py \
  --dataset-name open-r1/math-prompts \
  --output-file mixture_of_thoughts.jsonl \
  --prompt-column problem \
  --uuid-column id \
  --num-generations 4 \
  --max-concurrent 1000
==============================================================================
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
from asyncio import Lock
from typing import Set

from datasets import load_dataset
from tqdm.asyncio import tqdm

import aiofiles
import aiohttp
import uvloop


file_lock = Lock()  # Protects concurrent writes to output file


async def generate_completion(session, prompt, args):
    """
    WHAT: Generates single completion from vLLM API with retry logic

    WHY: Network requests can fail, need robust retry mechanism

    HOW:
        1. Send POST request to vLLM /v1/chat/completions endpoint
        2. If fails, retry up to 10 times with 10s backoff
        3. Return JSON response or None on total failure

    Args:
        session: aiohttp ClientSession for connection pooling
        prompt: Prompt string to send to API
        args: Command-line arguments (API address, temperature, etc.)

    Returns:
        dict: API response with completion or None on failure

    RETRY STRATEGY:
        - Budget: 10 retries
        - Backoff: 10 seconds between retries
        - Random jitter: 0-0.1s to avoid thundering herd

    API FORMAT (OpenAI-compatible):
        POST /v1/chat/completions
        {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16384,
            "temperature": 0.6,
            "top_p": 0.95
        }
    """
    retry_budget = 10
    while retry_budget > 0:
        try:
            # Random jitter to avoid thundering herd
            await asyncio.sleep(random.uniform(0.0, 0.1))

            async with session.post(
                f"http://{args.api_addr}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
                headers={"Authorization": "Bearer EMPTY"},
            ) as response:
                return await response.json(content_type=None)
        except Exception as e:
            print(f"API error (will retry): {e}")
            retry_budget -= 1
            await asyncio.sleep(10)  # Backoff before retry
    return None  # All retries exhausted


async def process_example(example, session, args, output_file, pbar):
    """
    WHAT: Processes single example: generate N completions and save

    WHY: Core unit of work, processes one prompt with multiple generations

    HOW:
        1. Format prompt using template
        2. Generate N completions in parallel (via asyncio.gather)
        3. Collect generations, finish reasons, API metadata
        4. Write to JSONL file (with file lock)
        5. Update progress bar

    Args:
        example: Dataset example (dict with prompt, id, etc.)
        session: aiohttp ClientSession
        args: Command-line arguments
        output_file: Path to JSONL output file
        pbar: tqdm progress bar

    Returns:
        dict: Result with original fields + generations, or None on error

    OUTPUT FORMAT (JSONL):
        {
            ...original dataset fields...,
            "generations": ["<think>...</think>answer", ...],
            "finish_reasons": ["stop", "stop", ...],
            "api_metadata": [{"prompt_tokens": 100, "completion_tokens": 500}, ...]
        }

    FILE LOCKING:
        - Multiple tasks write concurrently
        - file_lock ensures atomic writes
        - Prevents corrupted JSONL
    """
    # Format prompt using template
    # Example: "You will be given a problem. Please reason step by step:\n{prompt}"
    prompt = args.prompt_template.format(prompt=example[args.prompt_column])

    try:
        # Generate N completions in parallel
        # WHY: Faster than sequential (utilize concurrent API capacity)
        tasks = [generate_completion(session, prompt, args) for _ in range(args.num_generations)]

        completions = await asyncio.gather(*tasks)

        # Check for failures
        if any(completion is None for completion in completions):
            print(f"Error processing example")
            pbar.update(1)
            return None

        # Extract generations, finish reasons, metadata
        generations = []
        finish_reasons = []
        api_metadata = []

        for completion in completions:
            generations.append(completion["choices"][0]["message"]["content"])
            finish_reasons.append(completion["choices"][0]["finish_reason"])
            api_metadata.append(completion["usage"])

        # Combine original dataset fields with generations
        # WHY: Preserve all context (problem statement, metadata, etc.)
        result = {
            **example,  # Preserve all original dataset fields
            "generations": generations,
            "finish_reasons": finish_reasons,
            "api_metadata": api_metadata,
        }

        # Write to file with lock
        # WHY: Multiple tasks write concurrently, need atomic writes
        async with file_lock:
            async with aiofiles.open(output_file, mode="a") as f:
                await f.write(json.dumps(result) + "\n")
                await f.flush()  # Ensure written to disk

        pbar.set_postfix(active=len(pbar.active_tasks), refresh=False)
        pbar.update(1)

        return result
    except Exception as e:
        print(f"Error processing example: {e}")
        pbar.update(1)
        return None


async def load_processed_uuids(output_file, uuid_column):
    """
    WHAT: Loads UUIDs of already-processed examples from output file

    WHY: Enables resumption from interruption
         Avoid re-generating completions for already-processed examples

    HOW:
        1. Read JSONL file line by line (async)
        2. Parse each line, extract UUID
        3. Compute MD5 hash of UUID (for consistent comparison)
        4. Return set of processed UUIDs

    Args:
        output_file: Path to JSONL output file
        uuid_column: Column name for unique identifier

    Returns:
        set: MD5 hashes of processed example UUIDs

    RESUME LOGIC:
        - On start: Load processed UUIDs
        - During generation: Skip examples with UUID in set
        - Result: Only generate missing examples

    UUID HASHING:
        - MD5 hash ensures consistent comparison
        - Handles various UUID formats
    """
    processed_uuids = set()
    if os.path.exists(output_file):
        async with aiofiles.open(output_file, mode="r") as f:
            async for line in f:
                try:
                    data = json.loads(line)
                    # Hash UUID for consistent comparison
                    processed_uuids.add(hashlib.md5(str(data[uuid_column]).encode()).hexdigest())
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
    return processed_uuids


async def main():
    """
    WHAT: Main async function orchestrating generation workflow

    WORKFLOW:
        1. Parse command-line arguments
        2. Load dataset and shuffle
        3. Load processed UUIDs (for resume)
        4. Create aiohttp session with connection pooling
        5. Iterate dataset, process unprocessed examples
        6. Manage concurrency (max 1000 concurrent tasks)
        7. Wait for completion

    CONCURRENCY MANAGEMENT:
        - max_concurrent: 1000 (default)
        - If active tasks >= limit, wait for one to complete
        - Prevents overwhelming API server
        - Maximizes throughput without crashing

    PROGRESS TRACKING:
        - tqdm progress bar
        - Shows active task count
        - Updates every 2 seconds (mininterval)

    CONNECTION POOLING:
        - aiohttp ClientSession
        - Reuses connections (faster than creating new)
        - Configurable timeout (1 hour total)
        - Keep-alive for long-running generation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--prompt-column", type=str, required=True)
    parser.add_argument("--uuid-column", type=str, required=True)
    parser.add_argument("--api-addr", type=str, default="localhost:39876")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="You will be given a problem. Please reason step by step, and put your final answer within \\boxed{{}}:\n{prompt}",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-concurrent", type=int, default=1000)
    args = parser.parse_args()

    # Load dataset and shuffle
    dataset = load_dataset(args.dataset_name, split="train").shuffle()

    # Load processed UUIDs for resume
    processed_uuids = await load_processed_uuids(args.output_file, args.uuid_column)
    if processed_uuids:
        print(f"Found {len(processed_uuids)} already processed examples, resuming from there...")

    # Create output file if doesn't exist
    if not os.path.exists(args.output_file):
        async with aiofiles.open(args.output_file, mode="w") as f:
            await f.write("")

    # Track active tasks for concurrency management
    active_tasks: Set[asyncio.Task] = set()

    # Progress bar
    pbar = tqdm(
        total=len(dataset) - len(processed_uuids),
        desc="Generating responses",
        unit="row",
        mininterval=2,  # Update every 2 seconds
        smoothing=0.0001,  # Low smoothing for accurate ETA
    )
    pbar.active_tasks = active_tasks  # Attach for display

    # Create aiohttp session with connection pooling
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60 * 60),  # 1 hour total timeout
        connector=aiohttp.TCPConnector(
            limit=args.max_concurrent,  # Max concurrent connections
            ttl_dns_cache=300,  # DNS cache TTL
            keepalive_timeout=60 * 60,  # Keep connections alive
        ),
    ) as session:
        # Process dataset
        for example in dataset:
            # Compute UUID hash
            uuid = hashlib.md5(str(example[args.uuid_column]).encode()).hexdigest()

            # Skip if already processed (resume logic)
            if uuid not in processed_uuids:
                # Wait if at concurrency limit
                while len(active_tasks) >= args.max_concurrent:
                    # Wait for at least one task to complete
                    done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        try:
                            await task  # Propagate exceptions
                        except Exception as e:
                            print(f"Task failed: {e}")

                # Create task for this example
                task = asyncio.create_task(process_example(example, session, args, args.output_file, pbar))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)  # Auto-remove when done

                pbar.set_postfix(active=len(active_tasks), refresh=True)

        # Wait for remaining tasks
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    pbar.close()


if __name__ == "__main__":
    uvloop.install()  # Use fast uvloop event loop
    asyncio.run(main())


"""
==============================================================================
KEY TAKEAWAYS - REASONING GENERATION
==============================================================================

1. **Async I/O Performance**:
   - Process 1000+ requests concurrently
   - Connection pooling for speed
   - uvloop for maximum performance

2. **Robustness**:
   - Retry logic (10 attempts per request)
   - Resume from interruption
   - File locking for concurrent writes

3. **Scalability**:
   - Handles millions of examples
   - Configurable concurrency limit
   - Streaming writes (no memory overhead)

4. **Use Case**:
   - Generate Mixture-of-Thoughts dataset
   - Input: Math/code prompts
   - Output: Reasoning traces for SFT

5. **API Integration**:
   - OpenAI-compatible format
   - Works with vLLM server
   - Configurable sampling (temperature, top_p)

6. **Progress Tracking**:
   - tqdm progress bar
   - Active task count
   - Accurate ETA

7. **Resume Logic**:
   - UUID-based deduplication
   - Loads processed examples on start
   - Skips already-processed examples

8. **Output Format**:
   - JSONL (newline-delimited JSON)
   - One example per line
   - Preserves original fields + generations

9. **Typical Performance**:
   - 1000 concurrent requests
   - ~1000 completions/minute
   - Scales with vLLM capacity

10. **Configuration**:
    - num_generations: 4 (diversity)
    - max_tokens: 16384 (long reasoning)
    - temperature: 0.6, top_p: 0.95 (sampling)

==============================================================================
"""
