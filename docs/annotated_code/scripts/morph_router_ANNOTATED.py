"""
==============================================================================
FILE: scripts/morph_router.py
CATEGORY: Scripts - MorphCloud FastAPI Router Service
PRIORITY: MEDIUM
LINES: 173
DEPENDENCIES:
    - fastapi: Web framework for API endpoints
    - uvicorn: ASGI server for FastAPI
    - morphcloud: MorphCloud sandbox for code execution
    - asyncio: Asynchronous execution and semaphore-based rate limiting
    - pydantic: Request/response validation
==============================================================================

==============================================================================
OVERVIEW
==============================================================================

This script implements a FastAPI-based HTTP router service for executing code
in MorphCloud sandboxes. It serves as an alternative to the E2B router, providing
the same batch execution interface but using MorphCloud as the backend execution
service.

ROLE IN DEEPSEEK R1:
-------------------
MorphCloud is an alternative code execution sandbox provider that can be used
instead of E2B for computing rewards during GRPO training. This router provides:

1. **Provider Flexibility**: Switch between E2B and MorphCloud without changing
   the training code (both expose the same HTTP API).

2. **Cost Optimization**: Different providers have different pricing models;
   organizations can choose the most cost-effective option.

3. **Redundancy**: If one provider has downtime or rate limits, training can
   continue using the alternative provider.

4. **Feature Comparison**: Evaluate execution speed, reliability, and cost
   across providers to make informed decisions.

MORPHCLOUD VS E2B:
------------------
Both provide isolated code execution sandboxes, but differ in:

1. **API Structure**:
   - E2B: Uses AsyncSandbox with run_code returning Execution object
   - MorphCloud: Uses Sandbox.new() + run_code with different return format

2. **Output Format**:
   - E2B: Execution.stdout, Execution.stderr, Execution.exit_code
   - MorphCloud: Execution.text or Execution.stdout (adapter needed)

3. **Lifecycle**:
   - E2B: AsyncSandbox.create() → run_code() → kill()
   - MorphCloud: Sandbox.new() → run_code() → close() + shutdown()

4. **Threading Model**:
   - E2B: Native async with AsyncSandbox
   - MorphCloud: Sync API wrapped with asyncio.to_thread()

ARCHITECTURE:
-------------
The service mirrors the E2B router architecture:

    GRPO Training Process (Client)
        ↓ HTTP POST /execute_batch
    MorphCloud Router Service (This Script)
        ↓ MorphCloud API Calls
    MorphCloud Sandbox Service
        ↓ Code Execution Results
    MorphCloud Router Service
        ↓ HTTP Response (normalized to E2B format)
    GRPO Training Process

The key insight is that the client (code_reward) doesn't know or care which
backend is used. Both routers expose identical HTTP APIs.

WHY ASYNCIO.TO_THREAD?
----------------------
MorphCloud's Python SDK is synchronous (blocking I/O), but we want to maintain
the async architecture for consistency with E2B router:

    # MorphCloud is sync:
    sandbox = Sandbox.new(client=client)  # Blocks
    result = sandbox.run_code(code)       # Blocks

    # Wrap in asyncio.to_thread for async compatibility:
    sandbox = await asyncio.to_thread(Sandbox.new, client=client)
    result = await asyncio.to_thread(sandbox.run_code, code)

This allows concurrent execution without blocking the event loop.

TYPICAL REQUEST/RESPONSE:
-------------------------
Request (identical to E2B router):
    POST /execute_batch
    {
        "scripts": ["print(1+1)", "print(2+2)"],
        "languages": ["python", "python"],
        "timeout": 30,
        "request_timeout": 60
    }

Response (normalized to match E2B format):
    [
        {"text": "2\n", "exception_str": null},
        {"text": "4\n", "exception_str": null}
    ]

Note: MorphCloud returns "text" not "execution" object. The code_reward
function handles both formats.

DATA FLOW:
----------
    DISTAL ORIGIN (GRPO training loop):
    └─> Model generates code completions
        └─> code_reward function needs to verify correctness

    PROXIMAL PROCESSING (this service):
    1. Receive batch of code scripts via HTTP POST
    2. Acquire semaphore slots to limit concurrency
    3. Create MorphCloud sandbox for each script
    4. Execute code with timeout
    5. Collect results (text output)
    6. Clean up sandboxes (close + shutdown)
    7. Return results as JSON

    DISTAL DESTINATION (back to GRPO):
    └─> code_reward parses execution results
        └─> Computes scalar rewards based on test case outcomes
        └─> GRPO uses rewards to compute policy gradients

==============================================================================
IMPORTS AND SETUP
==============================================================================
"""

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from fastapi import FastAPI, Request
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()  # WHY: Load MORPH_API_KEY from .env for authentication

"""
==============================================================================
WHAT: BatchRequest
WHY:  Define structure for batch code execution requests (identical to E2B)
HOW:  Pydantic model with validation
==============================================================================

This model is identical to the E2B router's BatchRequest, ensuring API
compatibility. Clients can switch between routers without code changes.

See e2b_router_ANNOTATED.py for detailed documentation of this model.
"""
class BatchRequest(BaseModel):
    """
    BatchRequest is a data model representing a batch processing request.

    Attributes:
        scripts (list[str]): A list of script names or paths to be executed.
        languages (List[str]): The programming languages for each script in the list.
        timeout (int): The maximum allowed execution time for each script in seconds.
        request_timeout (int): The maximum allowed time for the entire batch request in seconds.
    """
    scripts: List[str]
    languages: List[str]
    timeout: int
    request_timeout: int

"""
==============================================================================
WHAT: ScriptResult
WHY:  Define structure for MorphCloud execution results
HOW:  Pydantic model with text output instead of Execution object
==============================================================================

This model differs from E2B's ScriptResult:
- E2B: execution: Optional[Execution] (complex object with stdout/stderr/exit_code)
- MorphCloud: text: Optional[str] (simple string output)

The difference reflects the underlying API structures. The code_reward function
handles both formats by checking which field is present.

Example success:
    {"text": "2\n", "exception_str": null}

Example failure:
    {"text": null, "exception_str": "TimeoutError: Execution exceeded 30 seconds"}
"""
class ScriptResult(BaseModel):
    """
    ScriptResult is a Pydantic model that represents the result of a script execution.
    Attributes:
        text (Optional[str]): The output text from the script execution.
        exception_str (Optional[str]): An optional string that captures the exception
            message or details if an error occurred during the script's execution.
        model_config (ConfigDict): A configuration dictionary that allows arbitrary
            types to be used within the Pydantic model.
    """
    text: Optional[str]
    exception_str: Optional[str]


    model_config = ConfigDict(arbitrary_types_allowed=True)

"""
==============================================================================
WHAT: create_app()
WHY:  Factory function to create configured FastAPI app instance
HOW:  Initialize MorphCloud client, attach to app state, define endpoints
==============================================================================

This factory creates a FastAPI app with MorphCloud integration:

1. Initialize MorphCloud client with API key
2. Store client and Sandbox class in app.state (shared across requests)
3. Create semaphore for concurrency control
4. Define health and execute_batch endpoints

WHY STORE Sandbox CLASS IN APP.STATE?
-------------------------------------
The MorphCloud SDK is imported conditionally (only if this router is used).
Storing the Sandbox class in app.state:
- Avoids import-time dependency on morphcloud package
- Allows mocking in tests
- Makes the dependency explicit

Example:
    app.state.Sandbox = Sandbox  # Store class itself
    # Later in endpoint:
    Sandbox = request.app.state.Sandbox
    sandbox = await asyncio.to_thread(Sandbox.new, client=client)

WHY REQUIRE API KEY?
-------------------
Unlike E2B (which can use E2B_API_KEY env var), MorphCloud requires explicit
API key configuration. The script validates this on startup and fails fast
if not provided.
"""
def create_app(args):
    """
    Creates and configures a FastAPI application instance for the MorphCloud router.

    Args:
        args: An object containing configuration parameters for the application.
              - max_num_sandboxes (int): The maximum number of concurrent sandboxes allowed.
              - api_key (str): The MorphCloud API key to use.

    Returns:
        FastAPI: A configured FastAPI application instance.
    """
    app = FastAPI()

    # WHAT: Import MorphCloud SDK and initialize client
    # WHY: Conditional import allows running E2B router without MorphCloud dependency
    from morphcloud.api import MorphCloudClient
    from morphcloud.sandbox import Sandbox

    # WHAT: Create MorphCloud client with API key
    # WHY: Client handles authentication and API communication
    app.state.client = MorphCloudClient(api_key=args.api_key)

    # WHAT: Store Sandbox class for later use in requests
    # WHY: Avoids re-importing in hot path, enables testing
    app.state.Sandbox = Sandbox

    # WHAT: Create semaphore for concurrency control (same as E2B router)
    # WHY: Prevent overwhelming MorphCloud API with too many concurrent requests
    app.state.sandbox_semaphore = asyncio.Semaphore(args.max_num_sandboxes)

    """
    ==============================================================================
    WHAT: GET /health endpoint
    WHY:  Provide healthcheck for orchestration systems (identical to E2B router)
    ==============================================================================
    """
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    """
    ==============================================================================
    WHAT: POST /execute_batch endpoint
    WHY:  Execute multiple code scripts using MorphCloud sandboxes
    HOW:  Asyncio gather with semaphore-controlled concurrency
    ==============================================================================

    This endpoint is functionally identical to E2B router but uses MorphCloud:

    Key differences from E2B implementation:
    1. MorphCloud SDK is synchronous, so we use asyncio.to_thread()
    2. timeout is in seconds for ttl_seconds, but milliseconds for run_code
    3. Return format is "text" instead of "execution" object
    4. Cleanup requires both close() and shutdown()
    """
    @app.post("/execute_batch")
    async def execute_batch(batch: BatchRequest, request: Request):
        # WHAT: Extract shared resources from app state
        # WHY: Semaphore coordinates concurrency, client handles auth
        semaphore = request.app.state.sandbox_semaphore
        client = request.app.state.client
        Sandbox = request.app.state.Sandbox

        languages = batch.languages
        timeout = batch.timeout
        request_timeout = batch.request_timeout
        asyncio_timeout = batch.timeout + 1  # WHY: Client-side timeout buffer

        """
        ==============================================================================
        WHAT: run_script() - Execute a single script in MorphCloud sandbox
        WHY:  Encapsulate per-script execution logic with MorphCloud-specific handling
        ==============================================================================

        This function mirrors the E2B router's run_script but adapts to MorphCloud's API:

        MORPHCLOUD-SPECIFIC ADAPTATIONS:
        --------------------------------

        1. SANDBOX CREATION:
           E2B:        sandbox = await AsyncSandbox.create(timeout=...)
           MorphCloud: sandbox = await asyncio.to_thread(Sandbox.new, client=..., ttl_seconds=...)

           Why to_thread? MorphCloud SDK is synchronous, blocks event loop.

        2. TIMEOUT UNITS:
           E2B:        timeout in seconds (both create and run_code)
           MorphCloud: ttl_seconds in seconds, run_code timeout in milliseconds

           This is why we do timeout * 1000 for run_code.

        3. OUTPUT EXTRACTION:
           E2B:        execution.stdout, execution.stderr
           MorphCloud: execution.text OR execution.stdout (API inconsistency)

           We check both fields with hasattr() to handle variations.

        4. CLEANUP:
           E2B:        await sandbox.kill()
           MorphCloud: await asyncio.to_thread(sandbox.close) + sandbox.shutdown()

           MorphCloud requires both close (stop sandbox) and shutdown (cleanup resources).

        5. SANDBOX ID TRACKING:
           For debugging, we try to extract sandbox ID from multiple locations:
           - getattr(sandbox, 'id', None)
           - getattr(sandbox._instance, 'id', 'unknown')

           This helps identify which sandbox failed in logs.
        """
        async def run_script(script: str, language: str) -> ScriptResult:
            sandbox = None
            sandbox_id = "unknown"

            async with semaphore:  # WHY: Limit concurrent sandboxes
                try:
                    # WHAT: Create MorphCloud sandbox with TTL (time-to-live)
                    # WHY: Sandbox automatically terminates after TTL to prevent resource leaks
                    # HOW: Wrap synchronous Sandbox.new() in to_thread for async compatibility
                    sandbox = await asyncio.to_thread(
                        Sandbox.new,
                        client=client,
                        ttl_seconds=timeout
                    )

                    # WHAT: Extract sandbox ID for debugging
                    # WHY: Helps identify which sandbox failed in logs
                    # HOW: Try multiple locations due to API structure variability
                    sandbox_id = getattr(sandbox, 'id', None) or getattr(sandbox._instance, 'id', 'unknown')

                    # WHAT: Execute code with timeout
                    # WHY: Prevent hanging on infinite loops or long-running code
                    # HOW: timeout*1000 converts seconds to milliseconds (MorphCloud API requirement)
                    execution = await asyncio.wait_for(
                        asyncio.to_thread(
                            sandbox.run_code,
                            script,
                            language=language,
                            timeout=timeout * 1000  # WHY: MorphCloud expects milliseconds
                        ),
                        timeout=asyncio_timeout,  # WHY: Client-side timeout for network failures
                    )

                    # WHAT: Extract output text from execution result
                    # WHY: MorphCloud API has inconsistent field names (.text vs .stdout)
                    # HOW: Check both fields and return first non-empty one
                    if hasattr(execution, 'text') and execution.text:
                        return ScriptResult(text=execution.text, exception_str=None)
                    elif hasattr(execution, 'stdout') and execution.stdout:
                        return ScriptResult(text=execution.stdout, exception_str=None)
                    else:
                        # WHAT: Handle case where execution succeeds but produces no output
                        # WHY: Some scripts legitimately produce no output (e.g., variable assignments)
                        return ScriptResult(text="", exception_str="No output from execution")

                except Exception as e:
                    # WHAT: Catch any exception and return as error result
                    # WHY: One failing script shouldn't break the entire batch
                    return ScriptResult(text=None, exception_str=str(e))

                finally:
                    # WHAT: Clean up sandbox resources
                    # WHY: Prevent resource leaks and reduce costs
                    # HOW: Both close() and shutdown() required for complete cleanup
                    if sandbox:
                        try:
                            await asyncio.to_thread(sandbox.close)    # Stop sandbox
                            await asyncio.to_thread(sandbox.shutdown) # Cleanup resources
                        except Exception:
                            pass  # Best effort cleanup

        # WHAT: Create async tasks for all scripts
        # WHY: Execute all scripts concurrently (up to semaphore limit)
        tasks = [run_script(script, lang) for script, lang in zip(batch.scripts, batch.languages)]

        # WHAT: Execute all tasks and return results
        # WHY: asyncio.gather runs concurrently and preserves order
        return await asyncio.gather(*tasks)

    return app

"""
==============================================================================
WHAT: parse_args()
WHY:  Parse command-line arguments for MorphCloud router configuration
HOW:  argparse with MorphCloud-specific API key handling
==============================================================================

Command-line arguments (similar to E2B router but with API key requirement):
- --host: IP address to bind to (default: 0.0.0.0)
- --port: TCP port (default: 8001, different from E2B's 8000)
- --max_num_sandboxes: Concurrent sandbox limit (default: 20)
- --api_key: MorphCloud API key (or from MORPH_API_KEY env var)

WHY DIFFERENT DEFAULT PORT?
---------------------------
Port 8001 allows running both E2B and MorphCloud routers simultaneously for
A/B testing or redundancy. Training code can failover between providers.

WHY VALIDATE API KEY ON STARTUP?
--------------------------------
Unlike E2B (which fails at first request), we fail fast on startup if API key
is missing. This prevents silent failures during training.
"""
def parse_args():
    """
    Parse command-line arguments for the morph_router script.

    Arguments:
        --host (str): The hostname or IP address to bind the server to. Defaults to "0.0.0.0".
        --port (int): The port number on which the server will listen. Defaults to 8001.
        --max_num_sandboxes (int): The maximum number of sandboxes that can be created simultaneously. Defaults to 20.
        --api_key (str): The MorphCloud API key. If not provided, it will be read from the MORPH_API_KEY environment variable.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)  # WHY: 8001 to avoid conflict with E2B router (8000)
    parser.add_argument("--max_num_sandboxes", type=int, default=20)
    parser.add_argument("--api_key", default=os.getenv("MORPH_API_KEY"))
    args = parser.parse_args()

    # WHAT: Validate API key is provided before starting server
    # WHY: Fail fast instead of silently accepting requests that will fail
    if not args.api_key:
        raise ValueError("MorphCloud API key not provided. Please set MORPH_API_KEY environment variable or use --api_key.")

    return args

"""
==============================================================================
WHAT: Main entry point
WHY:  Start the MorphCloud router service
HOW:  Parse args, create app, run with uvicorn
==============================================================================
"""
if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)

    print(f"Starting MorphCloud Router on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


"""
==============================================================================
KEY TAKEAWAYS
==============================================================================

1. **PURPOSE**: This service provides a MorphCloud-backed alternative to the
   E2B router, exposing the same HTTP API for batch code execution.

2. **PROVIDER ABSTRACTION**: By implementing the same API as E2B router, clients
   (code_reward) can switch providers without code changes:
   - E2B: POST localhost:8000/execute_batch
   - MorphCloud: POST localhost:8001/execute_batch
   - Same request/response format (modulo execution vs text field)

3. **SYNC-TO-ASYNC ADAPTATION**: MorphCloud SDK is synchronous, so we use
   asyncio.to_thread() to wrap blocking calls:
   - Sandbox.new() → await asyncio.to_thread(Sandbox.new, ...)
   - sandbox.run_code() → await asyncio.to_thread(sandbox.run_code, ...)

   This maintains async architecture for consistency with E2B router.

4. **API DIFFERENCES HANDLED**:
   - Timeout units: seconds (ttl_seconds) vs milliseconds (run_code timeout)
   - Output fields: execution.text OR execution.stdout (check both)
   - Cleanup: close() + shutdown() (two-step) vs kill() (one-step)

5. **PORT SEPARATION**: Default port 8001 (vs E2B's 8000) enables:
   - Running both routers simultaneously
   - A/B testing between providers
   - Failover/redundancy during training

6. **API KEY VALIDATION**: Unlike E2B router, validates API key on startup:
   - Fails fast with clear error message
   - Prevents silent failures during training
   - Ensures service is ready before accepting requests

7. **SEMAPHORE-BASED RATE LIMITING**: Same concurrency control as E2B router:
   - max_num_sandboxes controls concurrent executions
   - Prevents API overload and rate limit errors
   - Typical: 20 concurrent (adjust based on provider limits)

8. **IDENTICAL CLIENT INTERFACE**: From training code perspective, both routers
   are interchangeable. The choice is made at deployment time (which service to
   start), not in training configuration.

==============================================================================
USAGE EXAMPLE
==============================================================================

Starting the MorphCloud router:

    $ export MORPH_API_KEY="your_api_key_here"
    $ python scripts/morph_router.py --host 0.0.0.0 --port 8001 --max_num_sandboxes 20

    Starting MorphCloud Router on 0.0.0.0:8001
    INFO:     Started server process [12346]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)

Making a request (identical to E2B):

    import requests

    response = requests.post("http://localhost:8001/execute_batch", json={
        "scripts": ["print(1+1)", "print(2+2)"],
        "languages": ["python", "python"],
        "timeout": 30,
        "request_timeout": 60
    })

    results = response.json()
    # [
    #   {"text": "2\n", "exception_str": null},
    #   {"text": "4\n", "exception_str": null}
    # ]

Configuring code_reward to use MorphCloud:

    # In training script or config:
    code_reward_kwargs = {
        "execution_service_url": "http://localhost:8001",  # MorphCloud instead of E2B
        "num_parallel": 20
    }

Running both routers for redundancy:

    # Terminal 1: E2B router
    $ E2B_API_KEY=... python scripts/e2b_router.py --port 8000

    # Terminal 2: MorphCloud router
    $ MORPH_API_KEY=... python scripts/morph_router.py --port 8001

    # Training code: Try E2B first, fallback to MorphCloud on failure
    try:
        results = requests.post("http://localhost:8000/execute_batch", ...)
    except:
        results = requests.post("http://localhost:8001/execute_batch", ...)

==============================================================================
"""
