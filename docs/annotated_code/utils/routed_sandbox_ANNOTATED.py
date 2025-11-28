# ==============================================================================
# FILE: src/open_r1/utils/routed_sandbox.py
# CATEGORY: Utilities - Code Execution Routing
# PRIORITY: MEDIUM
# LINES: 110
# DEPENDENCIES: requests, e2b_code_interpreter
# ==============================================================================
#
# OVERVIEW:
# Client for routing code execution requests to a centralized E2B router service.
# Mimics the E2B Sandbox API but adds batch processing support and delegates
# sandbox lifecycle management to the router.
#
# KEY FUNCTIONALITY:
# - RoutedSandbox: Client that sends batch execution requests to E2B router
# - run_code(): Execute multiple scripts in a single request
# - Result parsing into E2B Execution objects
#
# This pattern centralizes sandbox management, reducing overhead and improving
# resource utilization for code verification tasks.
# ==============================================================================

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

from typing import List, Optional

import requests
from e2b_code_interpreter.models import Execution, ExecutionError, Result


class RoutedSandbox:
    """
    WHAT: Sandbox client that routes code execution to centralized E2B router

    WHY:
    - Centralize E2B sandbox lifecycle management (create, execute, cleanup)
    - Enable batch processing of multiple scripts in one request
    - Reduce overhead compared to creating individual sandboxes
    - Simplify client code by delegating complexity to router

    HOW:
    - Maintains router URL configuration
    - Sends batch execution requests via HTTP
    - Parses responses into E2B Execution objects

    ARCHITECTURE:
    ```
    Client → RoutedSandbox → HTTP → E2B Router → E2B Sandbox → Execute
    ```

    VS DIRECT E2B:
    - Direct: Client creates sandbox, executes code, cleans up
    - Routed: Router manages entire sandbox lifecycle

    BENEFITS:
    - Batch processing (multiple scripts per request)
    - Resource pooling at router level
    - Simpler client code
    - Centralized error handling and retry logic

    A sandbox environment that routes code execution requests to the E2B Router.
    This class is designed for batched execution of scripts, primarily for Python code.
    It mimics the usage of 'Sandbox' from 'e2b_code_interpreter', but adds support for batch processing.

    Attributes:
        router_url (str): The URL of the E2B Router to which code execution requests are sent.
    """

    def __init__(self, router_url: str):
        """
        WHAT: Initialize routed E2B sandbox client

        WHY:
        - Store router URL for all future requests
        - Simple initialization (router handles complexity)

        PARAMETERS:
        - router_url: URL of E2B router (e.g., "192.168.1.100:8000")

        EXAMPLE:
        ```python
        sandbox = RoutedSandbox(router_url="10.0.0.5:8000")
        ```

        Initializes the RoutedSandbox with the specified router URL.

        Args:
            router_url (str): The URL of the E2B Router.
        """
        self.router_url = router_url

    def run_code(
        self,
        scripts: list[str],
        languages: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
    ) -> list[Execution]:
        """
        WHAT: Execute batch of scripts via E2B router

        WHY:
        - Batch execution reduces overhead (one request for many scripts)
        - Router manages sandbox lifecycle transparently
        - Returns E2B-compatible Execution objects

        HOW:
        1. Apply default timeouts if not provided
        2. Default to Python for all scripts if languages not specified
        3. Build HTTP request payload
        4. Send POST request to router's /execute_batch endpoint
        5. Parse response into Execution objects
        6. Handle empty executions (timeouts/failures)

        PROXIMAL CONTEXT:
        - INPUT: List of scripts and optional configuration
        - OUTPUT: List of Execution objects (E2B format)

        DISTAL CONTEXT:
        - ORIGIN: Called by code_reward or verification functions
        - DESTINATION: Router executes via E2B, returns results

        EXAMPLE 1: Simple execution
        ```python
        sandbox = RoutedSandbox(router_url="10.0.0.5:8000")
        scripts = ["print('hello')", "print(1+1)"]
        executions = sandbox.run_code(scripts)
        # Result: [
        #   Execution(results=[Result(text="hello\n")], logs=...),
        #   Execution(results=[Result(text="2\n")], logs=...)
        # ]
        ```

        EXAMPLE 2: With timeout
        ```python
        scripts = ["import time; time.sleep(10); print('done')"]
        executions = sandbox.run_code(scripts, timeout=20)
        # Allows 20 seconds for execution
        ```

        EXAMPLE 3: Error handling
        ```python
        scripts = ["print(undefined_variable)"]
        executions = sandbox.run_code(scripts)
        # Result: Execution with error field populated
        ```

        RESULT FORMAT (E2B Execution):
        - results: List of Result objects (text, formats)
        - logs: Execution logs (stdout, stderr)
        - error: ExecutionError if failed, None if succeeded
        - execution_count: Number of cells executed

        TIMEOUT HANDLING:
        - execution returns None: Creates empty Execution()
        - Indicates timeout or fatal error

        Executes a batch of scripts in the sandbox environment.

        Args:
            scripts (list[str]): A list of code scripts to execute.
            languages (list[str], optional): List of programming languages for each script. If None, defaults to Python for all scripts.
            timeout (Optional[int], optional): The maximum execution time for each script in seconds. Defaults to 300 seconds.
            request_timeout (Optional[int], optional): The timeout for the HTTP request in seconds. Defaults to 30 seconds.

        Returns:
            list[Execution]: A list of Execution objects containing the results, logs, and errors (if any) for each script.
        """
        # SET DEFAULT TIMEOUTS
        # 300s (5 min) for execution, 30s for HTTP request
        if timeout is None:
            timeout = 300  # Default to 5 minutes
        if request_timeout is None:
            request_timeout = 30  # Default to 30 seconds

        # DEFAULT TO PYTHON FOR ALL SCRIPTS
        # WHY: Most common use case, simplifies API
        if languages is None:
            languages = ["python"] * len(scripts)

        # PREPARE REQUEST PAYLOAD
        payload = {
            "scripts": scripts,
            "languages": languages,
            "timeout": timeout,
            "request_timeout": request_timeout,
        }

        # SEND REQUEST TO E2B ROUTER
        # Uses http:// prefix (router expected on local network)
        response = requests.post(f"http://{self.router_url}/execute_batch", json=payload)

        # CHECK RESPONSE STATUS
        if not response.ok:
            print(f"Request failed with status code: {response.status_code}")

        # PARSE RESPONSE AND CONSTRUCT EXECUTION OBJECTS
        results = response.json()
        output = []

        for result in results:
            if result["execution"] is None:
                # HANDLE EMPTY EXECUTION
                # This can happen when:
                # - Script times out
                # - Fatal error during execution
                # - Sandbox creation fails
                # Create empty Execution object as placeholder
                execution = Execution()
            else:
                # BUILD FULL EXECUTION OBJECT
                # Parse all fields from response into E2B models
                execution = Execution(
                    # PARSE RESULTS
                    # Each result contains text output and metadata
                    results=[Result(**r) for r in result["execution"]["results"]],

                    # PARSE LOGS
                    # Execution logs (stdout, stderr, etc.)
                    logs=result["execution"]["logs"],

                    # PARSE ERROR (if present)
                    # None if execution succeeded
                    # ExecutionError if exception occurred
                    error=(ExecutionError(**result["execution"]["error"]) if result["execution"]["error"] else None),

                    # EXECUTION COUNT
                    # Number of code cells executed
                    execution_count=result["execution"]["execution_count"],
                )
            output.append(execution)

        return output


if __name__ == "__main__":
    """
    EXAMPLE USAGE:
    Demonstrates how to use RoutedSandbox locally

    PREREQUISITES:
    - E2B router running on localhost:8000
    - Start with: python scripts/e2b_router.py

    TEST CASES:
    1. Valid Python code: Should print "hello world"
    2. Syntax error: Should return execution with error
    """
    # for local testing launch an E2B router with: python scripts/e2b_router.py
    sbx = RoutedSandbox(router_url="0.0.0.0:8000")

    # TEST SCRIPTS
    # First: Valid code
    # Second: Syntax error (missing closing quote)
    codes = ["print('hello world')", "print('hello world)"]

    # EXECUTE BATCH
    executions = sbx.run_code(codes)  # Execute Python inside the sandbox

    print(executions)

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================
# 1. **Router Pattern**:
#    - Centralizes E2B sandbox management
#    - Client sends requests, router handles lifecycle
#    - Reduces overhead and complexity
#
# 2. **Batch Processing**:
#    - Execute multiple scripts in one HTTP request
#    - More efficient than individual requests
#    - Results returned in same order as scripts
#
# 3. **E2B Compatibility**:
#    - Returns E2B Execution objects
#    - Compatible with code expecting E2B Sandbox
#    - Easy to swap between direct and routed execution
#
# 4. **Result Structure**:
#    - results: Output from code execution
#    - logs: Execution logs (stdout/stderr)
#    - error: Exception details if failed
#    - execution_count: Number of cells executed
#
# 5. **Error Handling**:
#    - HTTP errors: Prints warning, continues parsing
#    - Execution timeouts: Returns empty Execution()
#    - Syntax errors: Returns Execution with error field
#
# 6. **Default Configuration**:
#    - timeout: 300s (5 minutes) for code execution
#    - request_timeout: 30s for HTTP request
#    - language: Python if not specified
#
# 7. **Use Cases**:
#    - Code verification in GRPO training
#    - Batch evaluation of solutions
#    - Testing code against test cases
#
# 8. **Main Block**:
#    - Example usage for local testing
#    - Requires E2B router running on localhost:8000
#    - Demonstrates both success and error cases
# ==============================================================================
