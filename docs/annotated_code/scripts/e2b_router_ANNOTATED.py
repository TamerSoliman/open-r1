"""
==============================================================================
FILE: scripts/e2b_router.py
CATEGORY: Scripts - E2B FastAPI Router Service
PRIORITY: HIGH
LINES: 161
DEPENDENCIES:
    - fastapi: Web framework for API endpoints
    - uvicorn: ASGI server for FastAPI
    - e2b_code_interpreter: E2B sandbox for code execution
    - asyncio: Asynchronous execution and semaphore-based rate limiting
    - pydantic: Request/response validation
==============================================================================

==============================================================================
OVERVIEW
==============================================================================

This script implements a FastAPI-based HTTP router service that provides a REST
API for executing code in E2B sandboxes. It acts as a middleware layer between
the GRPO training pipeline and the E2B cloud execution service, handling batch
code execution requests with parallelization and resource management.

ROLE IN DEEPSEEK R1:
-------------------
During GRPO training on coding tasks, the training pipeline needs to execute
potentially hundreds of code solutions per batch to compute rewards. This router
service provides:

1. **Batch Processing**: Execute multiple code samples in a single HTTP request,
   reducing network overhead and improving throughput.

2. **Resource Management**: Semaphore-based concurrency control prevents
   overwhelming the E2B API with too many simultaneous sandbox requests.

3. **Fault Isolation**: Runs as a separate service, isolating execution failures
   from the training process. If a sandbox crashes, the training loop continues.

4. **Async Execution**: Uses asyncio for efficient concurrent execution, maximizing
   throughput within rate limits.

ARCHITECTURE:
-------------
The service has a client-server architecture:

    GRPO Training Process (Client)
        ↓ HTTP POST /execute_batch
    E2B Router Service (This Script)
        ↓ E2B API Calls
    E2B Cloud Sandbox Service
        ↓ Code Execution Results
    E2B Router Service
        ↓ HTTP Response
    GRPO Training Process

This indirection enables:
- Centralized rate limiting (one semaphore for all training workers)
- Connection pooling (reuse HTTP connections to E2B)
- Logging and monitoring (all executions go through one service)
- Multi-language support (Python and JavaScript execution)

WHY FASTAPI + ASYNC?
--------------------
FastAPI with async/await is ideal for I/O-bound workloads like code execution:

1. **Non-blocking**: While waiting for E2B sandbox creation/execution, the
   server can handle other requests.

2. **High Throughput**: Single process can handle hundreds of concurrent
   executions with minimal memory overhead.

3. **Built-in Validation**: Pydantic models ensure request/response correctness.

4. **Easy Deployment**: Standard ASGI interface works with any cloud provider.

TYPICAL REQUEST/RESPONSE:
-------------------------
Request:
    POST /execute_batch
    {
        "scripts": ["print(1+1)", "print(2+2)", "print(3+3)"],
        "languages": ["python", "python", "python"],
        "timeout": 30,
        "request_timeout": 60
    }

Response:
    [
        {"execution": {...}, "exception_str": null},
        {"execution": {...}, "exception_str": null},
        {"execution": null, "exception_str": "TimeoutError"}
    ]

Each result corresponds to one script in the input array.

DATA FLOW:
----------
    DISTAL ORIGIN (GRPO training loop):
    └─> Model generates code completions
        └─> code_reward function needs to verify correctness

    PROXIMAL PROCESSING (this service):
    1. Receive batch of code scripts via HTTP POST
    2. Acquire semaphore slots to limit concurrency
    3. Create E2B sandbox for each script
    4. Execute code with timeout
    5. Collect results (stdout, stderr, exit code)
    6. Clean up sandboxes
    7. Return results as JSON

    DISTAL DESTINATION (back to GRPO):
    └─> code_reward parses execution results
        └─> Computes scalar rewards based on test case outcomes
        └─> GRPO uses rewards to compute policy gradients

CONCURRENCY MANAGEMENT:
-----------------------
The semaphore (max_num_sandboxes) controls how many sandboxes can exist
simultaneously:

- Too high: Exceeds E2B rate limits, requests fail
- Too low: Underutilizes available capacity, slow throughput
- Typical: 20 for free tier, 100 for PRO tier

Each execution acquires a semaphore slot before creating a sandbox, ensuring
the limit is never exceeded.

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
from typing import  Optional
from fastapi import FastAPI, Request
import argparse
import asyncio
from fastapi import FastAPI
import uvicorn
from e2b_code_interpreter.models import Execution
from dotenv import load_dotenv
from e2b_code_interpreter import AsyncSandbox

load_dotenv()  # WHY: Load E2B_API_KEY from .env for authentication

"""
==============================================================================
WHAT: BatchRequest
WHY:  Define structure for batch code execution requests
HOW:  Pydantic model with validation
==============================================================================

This Pydantic model defines the expected structure for POST /execute_batch:

Fields:
- scripts: List of code strings to execute
- languages: Programming language for each script ("python" or "javascript")
- timeout: Per-script execution timeout in seconds
- request_timeout: Overall E2B request timeout in seconds

Example:
    {
        "scripts": ["print(1+1)", "console.log(2+2)"],
        "languages": ["python", "javascript"],
        "timeout": 30,
        "request_timeout": 60
    }

VALIDATION:
- scripts and languages must have same length (paired execution)
- timeout/request_timeout must be positive integers
- FastAPI automatically validates and returns 422 for invalid requests

DISTAL CONTEXT: This matches the interface expected by open_r1.rewards.code_reward
"""
class BatchRequest(BaseModel):
    """
    BatchRequest is a data model representing a batch processing request.

    Attributes:
        scripts (list[str]): A list of script names or paths to be executed.
        languages (list[str]): The programming languages for each script in the list.
        timeout (int): The maximum allowed execution time for each script in seconds.
        request_timeout (int): The maximum allowed time for the entire batch request in seconds.
    """
    scripts: list[str]
    languages: list[str]
    timeout: int
    request_timeout: int

"""
==============================================================================
WHAT: ScriptResult
WHY:  Define structure for individual execution results
HOW:  Pydantic model with optional execution and error fields
==============================================================================

This model represents the result of executing a single script:

Fields:
- execution: E2B Execution object (stdout, stderr, exit_code, etc.) if successful
- exception_str: Error message if execution failed

States:
1. Success: execution != null, exception_str == null
2. Failure: execution == null, exception_str contains error message

Example success:
    {
        "execution": {
            "stdout": "2\n",
            "stderr": "",
            "exit_code": 0
        },
        "exception_str": null
    }

Example failure:
    {
        "execution": null,
        "exception_str": "TimeoutError: Execution exceeded 30 seconds"
    }

WHY OPTIONAL FIELDS?
- Execution can fail before sandbox creation (network error)
- Execution can timeout (no result returned)
- Exception can occur in sandbox creation/cleanup

The model_config allows arbitrary types (Execution class) in Pydantic,
which normally only supports primitive types.
"""
class ScriptResult(BaseModel):
    """
    ScriptResult is a Pydantic model that represents the result of a script execution.
    Attributes:
        execution (Optional[Execution]): An optional instance of the `Execution` class
            that contains details about the script's execution, such as status, output,
            or any other relevant metadata.
        exception_str (Optional[str]): An optional string that captures the exception
            message or details if an error occurred during the script's execution.
        model_config (ConfigDict): A configuration dictionary that allows arbitrary
            types to be used within the Pydantic model. This is necessary to support
            custom types like `Execution` within the model.
    """
    execution: Optional[Execution]
    exception_str: Optional[str]

    # required to allow arbitrary types in pydantic models such as Execution
    model_config = ConfigDict(arbitrary_types_allowed=True)

"""
==============================================================================
WHAT: create_app()
WHY:  Factory function to create configured FastAPI app instance
HOW:  Attaches semaphore to app state and defines endpoints
==============================================================================

PROXIMAL CONTEXT: Called from __main__ with parsed command-line arguments
DISTAL CONTEXT: Standard FastAPI factory pattern for testability

This factory function creates and configures a FastAPI application:

1. Creates FastAPI instance
2. Attaches semaphore to app.state (shared across all requests)
3. Defines two endpoints:
   - GET /health: Healthcheck for load balancers
   - POST /execute_batch: Batch code execution

WHY FACTORY PATTERN?
- Testability: Can create app with different configurations
- Dependency Injection: args passed in, not global variables
- Clean separation: Configuration vs application logic

WHY ATTACH SEMAPHORE TO APP.STATE?
- Shared resource: All requests must coordinate via same semaphore
- Request-scoped access: Each request can access via request.app.state
- Lifecycle management: Semaphore created once, persists across requests

The semaphore limits concurrent sandbox creation to prevent exceeding E2B
rate limits (default: 20 concurrent sandboxes).
"""
def create_app(args):
    """
    Creates and configures a FastAPI application instance.
    Args:
        args: An object containing configuration parameters for the application.
              - num_sandboxes (int): The maximum number of concurrent sandboxes allowed.
    Returns:
        FastAPI: A configured FastAPI application instance.
    The application includes the following endpoints:
        1. GET /health:
            - Returns the health status of the application.
            - Response: {"status": "ok"}
        2. POST /execute_batch:
            - Executes a batch of scripts in an isolated sandbox environment.
            - Request Body: BatchRequest object containing:
                - languages (list[str]): The programming languages of the scripts (python or javascript).
                - timeout (int): The maximum execution time for each script.
                - request_timeout (int): The timeout for the request itself.
                - scripts (List[str]): A list of scripts to execute.
            - Response: A list of ScriptResult objects for each script, containing:
                - execution: The result of the script execution.
                - exception_str: Any exception encountered during execution.
    Notes:
        - A semaphore is used to limit the number of concurrent sandboxes.
        - Each script execution is wrapped in a timeout to prevent hanging.
        - Sandboxes are cleaned up after execution, even in case of errors.
    """
    app = FastAPI()

    # WHAT: Create semaphore to limit concurrent sandbox creation
    # WHY: E2B has rate limits; exceeding them causes 429 errors
    # HOW: Semaphore(N) allows at most N concurrent acquisitions
    # Instantiate semaphore and attach it to app state
    app.state.sandbox_semaphore = asyncio.Semaphore(args.max_num_sandboxes)

    """
    ==============================================================================
    WHAT: GET /health endpoint
    WHY:  Provide healthcheck for orchestration systems (Kubernetes, Docker)
    HOW:  Returns static JSON indicating service is running
    ==============================================================================

    Load balancers and container orchestrators use this endpoint to verify
    the service is healthy and can accept requests.

    Response: {"status": "ok"}

    This doesn't validate E2B connectivity (that would be slow for healthchecks).
    It only confirms the FastAPI server is running and responsive.
    """
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    """
    ==============================================================================
    WHAT: POST /execute_batch endpoint
    WHY:  Execute multiple code scripts in parallel with resource limits
    HOW:  Asyncio gather with semaphore-controlled concurrency
    ==============================================================================

    This is the main endpoint for batch code execution. It:

    1. Receives a batch of code scripts with languages and timeouts
    2. Spawns async tasks to execute each script in parallel
    3. Uses semaphore to limit concurrent E2B sandbox creation
    4. Returns results for all scripts (success or failure)

    CONCURRENCY MODEL:
    -----------------
    The endpoint uses asyncio.gather to execute all scripts concurrently:

        tasks = [run_script(s, l) for s, l in zip(scripts, languages)]
        return await asyncio.gather(*tasks)

    Each task:
    1. Acquires semaphore slot (blocks if all slots taken)
    2. Creates E2B sandbox
    3. Executes code with timeout
    4. Releases semaphore slot
    5. Returns result

    Example timeline with 3 scripts and max_num_sandboxes=2:

        Time  | Task 1         | Task 2         | Task 3
        ------|----------------|----------------|----------------
        0s    | Acquire sem    | Acquire sem    | Wait for sem
        1s    | Create sandbox | Create sandbox | Wait for sem
        2s    | Execute code   | Execute code   | Wait for sem
        5s    | Complete       | Still running  | Acquire sem
        5s    | Release sem    | Still running  | Create sandbox
        5s    |                | Still running  | Execute code
        8s    |                | Complete       | Still running
        8s    |                | Release sem    | Still running
        10s   |                |                | Complete
        10s   |                |                | Release sem

    Task 3 waits until Task 1 finishes before starting (semaphore limit).

    TIMEOUT HANDLING:
    ----------------
    The code uses two timeout mechanisms:

    1. E2B sandbox timeout (timeout parameter):
       - Set when creating sandbox
       - E2B enforces this on the server side
       - If exceeded, sandbox is killed

    2. Asyncio timeout (asyncio_timeout = timeout + 1):
       - Client-side timeout for waiting on E2B response
       - Set slightly higher than sandbox timeout to account for network latency
       - Prevents hanging if E2B doesn't respond

    This dual-timeout ensures requests eventually complete even if E2B fails.

    ERROR HANDLING:
    --------------
    Exceptions are caught and returned as ScriptResult with exception_str:

    - Sandbox creation failure: Network error, auth error, rate limit
    - Execution timeout: Code runs too long
    - Asyncio timeout: E2B doesn't respond
    - Any other exception: Unexpected errors

    All exceptions are converted to strings and returned to the client,
    preventing the entire batch from failing due to one bad script.

    CLEANUP:
    -------
    The finally block ensures sandboxes are always cleaned up:

        finally:
            try:
                await sandbox.kill()
            except Exception:
                pass  # Best effort cleanup

    This prevents resource leaks even when exceptions occur.
    """
    @app.post("/execute_batch")
    async def execute_batch(batch: BatchRequest, request: Request):
        # WHAT: Extract semaphore and request parameters
        # WHY: Shared semaphore coordinates across all concurrent executions
        semaphore = request.app.state.sandbox_semaphore
        languages = batch.languages
        timeout = batch.timeout
        request_timeout = batch.request_timeout
        asyncio_timeout = batch.timeout + 1  # WHY: Slightly higher than sandbox timeout for network overhead

        """
        ==============================================================================
        WHAT: run_script() - Execute a single script in E2B sandbox
        WHY:  Encapsulate per-script execution logic with error handling
        HOW:  Async function that acquires semaphore, creates sandbox, executes code
        ==============================================================================

        This nested async function handles execution of one script:

        Flow:
        1. Acquire semaphore (wait if at limit)
        2. Create E2B sandbox with timeout
        3. Execute code with asyncio timeout wrapper
        4. Return result (success or exception)
        5. Clean up sandbox (in finally block)

        SEMAPHORE USAGE:
        ---------------
        async with semaphore:
            # Only max_num_sandboxes tasks can be here simultaneously
            # Other tasks wait at the async with statement

        This ensures we never exceed the E2B concurrent sandbox limit.

        SANDBOX CREATION:
        ----------------
        sandbox = await AsyncSandbox.create(
            timeout=timeout,
            request_timeout=request_timeout,
        )

        Creates a new E2B code interpreter sandbox with:
        - timeout: Max execution time for code (E2B enforced)
        - request_timeout: Max time to wait for sandbox creation

        EXECUTION:
        ---------
        execution = await asyncio.wait_for(
            sandbox.run_code(script, language=language),
            timeout=asyncio_timeout,
        )

        Runs the code with a client-side timeout (asyncio.wait_for) to prevent
        hanging if E2B doesn't respond.

        ERROR RECOVERY:
        --------------
        All exceptions are caught and returned as ScriptResult with exception_str.
        This prevents one failing script from breaking the entire batch.

        Cleanup happens in finally block to prevent resource leaks.
        """
        async def run_script(script: str, language: str) -> ScriptResult:

            async with semaphore:  # WHY: Acquire semaphore slot before creating sandbox
                try:
                    # WHAT: Create E2B sandbox for isolated code execution
                    # WHY: Each script needs its own environment to prevent interference
                    sandbox = await AsyncSandbox.create(
                        timeout=timeout,
                        request_timeout=request_timeout,
                    )

                    # WHAT: Execute code with client-side timeout
                    # WHY: Prevents hanging if E2B doesn't respond within expected time
                    execution = await asyncio.wait_for(
                        sandbox.run_code(script, language=language),
                        timeout=asyncio_timeout,
                    )
                    return ScriptResult(execution=execution, exception_str=None)

                except Exception as e:
                    # WHAT: Catch any exception and return as error result
                    # WHY: One failing script shouldn't break the entire batch
                    return ScriptResult(execution=None, exception_str=str(e))

                finally:
                    # WHAT: Clean up sandbox regardless of success/failure
                    # WHY: Prevent resource leaks and reduce E2B costs
                    try:
                        await sandbox.kill()
                    except Exception:
                        pass  # Best effort cleanup - sandbox might already be gone

        # WHAT: Create async tasks for all scripts
        # WHY: Execute all scripts concurrently (up to semaphore limit)
        # HOW: zip pairs each script with its language
        tasks = [run_script(script, lang) for script, lang in zip(batch.scripts, batch.languages)]

        # WHAT: Execute all tasks concurrently and wait for all to complete
        # WHY: asyncio.gather runs tasks in parallel and collects results
        # HOW: Returns list of ScriptResult in same order as input scripts
        return await asyncio.gather(*tasks)

    return app


"""
==============================================================================
WHAT: parse_args()
WHY:  Parse command-line arguments for server configuration
HOW:  argparse with defaults for host, port, and max sandboxes
==============================================================================

Command-line arguments:
- --host: IP address to bind to (default: 0.0.0.0 = all interfaces)
- --port: TCP port for HTTP server (default: 8000)
- --max_num_sandboxes: Concurrent sandbox limit (default: 20)

Example usage:
    python scripts/e2b_router.py --host 127.0.0.1 --port 8080 --max_num_sandboxes 50

WHY THESE DEFAULTS?
- 0.0.0.0: Accessible from other machines (required for distributed training)
- 8000: Standard HTTP alternate port (8080 is also common)
- 20: Conservative limit for E2B free tier (PRO tier supports 100)
"""
def parse_args():
    """
    Parse command-line arguments for the e2b_router script.

    Arguments:
        --host (str): The hostname or IP address to bind the server to. Defaults to "0.0.0.0" (binds to all interfaces).
        --port (int): The port number on which the server will listen. Defaults to 8000.
        --max_num_sandboxes (int): The maximum number of sandboxes that can be created or managed simultaneously. Defaults to 20.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_num_sandboxes", type=int, default=20)
    return parser.parse_args()

"""
==============================================================================
WHAT: Main entry point
WHY:  Start the FastAPI server when script is run directly
HOW:  Parse args, create app, run with uvicorn
==============================================================================

Startup sequence:
1. Parse command-line arguments
2. Create FastAPI app with configuration
3. Run uvicorn ASGI server

Uvicorn is the standard ASGI server for FastAPI, handling:
- HTTP protocol parsing
- WebSocket support
- Request/response buffering
- Worker process management (if --workers > 1)

The server runs until killed (Ctrl+C or SIGTERM).
"""
if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)

    # WHAT: Start uvicorn ASGI server
    # WHY: Production-ready server for FastAPI applications
    # HOW: Binds to specified host:port and serves the app
    uvicorn.run(app, host=args.host, port=args.port)


"""
==============================================================================
KEY TAKEAWAYS
==============================================================================

1. **PURPOSE**: This service provides a REST API for batch code execution in
   E2B sandboxes, enabling GRPO training to compute rewards for coding tasks.

2. **ARCHITECTURE**: FastAPI + asyncio for high-throughput, non-blocking I/O:
   - Single process handles hundreds of concurrent executions
   - Semaphore-based rate limiting prevents API overload
   - Async/await maximizes CPU utilization during I/O waits

3. **CRITICAL FEATURE**: Semaphore-controlled concurrency ensures we never
   exceed E2B rate limits:
   - max_num_sandboxes=20 (free tier)
   - max_num_sandboxes=100 (PRO tier)
   - Tasks wait at semaphore acquisition if limit reached

4. **FAULT TOLERANCE**: Multiple layers of error handling:
   - Try/except around sandbox creation and execution
   - Finally block ensures cleanup even on failure
   - Client-side timeout (asyncio.wait_for) prevents hanging
   - Per-script isolation: one failure doesn't break the batch

5. **DUAL TIMEOUT STRATEGY**:
   - E2B timeout: Server-side enforcement (sandbox killed if exceeded)
   - Asyncio timeout: Client-side fallback (return error if E2B doesn't respond)
   - asyncio_timeout = timeout + 1 (accounts for network latency)

6. **BATCH PROCESSING**: Executing multiple scripts in one HTTP request:
   - Reduces network overhead (one request vs hundreds)
   - Amortizes connection setup cost
   - Simplifies client code (one call instead of loop)

7. **DEPLOYMENT**: Production-ready features:
   - Health endpoint for load balancers
   - Pydantic validation for request/response
   - Configurable host/port/concurrency
   - Standard ASGI interface (works with any cloud provider)

8. **DISTAL INTEGRATION**: This service is called by open_r1.rewards.code_reward
   during GRPO training to verify code correctness. It's a critical component
   of the reward computation pipeline for coding tasks.

==============================================================================
USAGE EXAMPLE
==============================================================================

Starting the service:

    $ export E2B_API_KEY="your_api_key_here"
    $ python scripts/e2b_router.py --host 0.0.0.0 --port 8000 --max_num_sandboxes 20

    INFO:     Started server process [12345]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

Making a request (from Python):

    import requests

    response = requests.post("http://localhost:8000/execute_batch", json={
        "scripts": ["print(1+1)", "print(2+2)", "x = invalid syntax"],
        "languages": ["python", "python", "python"],
        "timeout": 30,
        "request_timeout": 60
    })

    results = response.json()
    # [
    #   {"execution": {"stdout": "2\n", ...}, "exception_str": null},
    #   {"execution": {"stdout": "4\n", ...}, "exception_str": null},
    #   {"execution": {"stderr": "SyntaxError: ...", ...}, "exception_str": null}
    # ]

Health check:

    $ curl http://localhost:8000/health
    {"status":"ok"}

==============================================================================
"""
