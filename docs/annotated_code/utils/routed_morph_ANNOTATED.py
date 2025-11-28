# ==============================================================================
# FILE: src/open_r1/utils/routed_morph.py
# CATEGORY: Utilities - Code Execution Routing
# PRIORITY: MEDIUM
# LINES: 121
# DEPENDENCIES: requests
# ==============================================================================
#
# OVERVIEW:
# Client for routing code execution requests to a centralized MorphCloud router
# service. Mimics the MorphCloud Sandbox API but adds batch processing support
# and delegates sandbox lifecycle management to the router.
#
# KEY FUNCTIONALITY:
# - RoutedMorphSandbox: Client that sends batch execution requests to router
# - run_code(): Execute multiple scripts in a single request
# - Error handling with fallback results
#
# This pattern centralizes sandbox management, reducing overhead and improving
# resource utilization compared to creating individual sandboxes per request.
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


class RoutedMorphSandbox:
    """
    WHAT: Client for executing code via centralized MorphCloud router service

    WHY:
    - Centralize sandbox lifecycle management (create, execute, cleanup)
    - Enable batch processing of multiple scripts in one request
    - Reduce overhead of creating individual sandboxes
    - Simplify client code by delegating complexity to router

    HOW:
    - Maintains router URL and timeout configuration
    - Sends batch execution requests via HTTP
    - Returns results in format compatible with MorphCloud Sandbox API

    ARCHITECTURE:
    ```
    Client → RoutedMorphSandbox → HTTP → Router → MorphCloud Sandbox → Execute
    ```

    VS DIRECT SANDBOX:
    - Direct: Client creates sandbox, executes, cleans up
    - Routed: Router manages all sandbox operations

    BENEFITS:
    - Batch processing (multiple scripts per request)
    - Resource pooling at router level
    - Simpler client code
    - Centralized error handling and retry logic

    Attributes:
        router_url (str): The URL of the MorphCloud router service.
        timeout (int): Execution timeout in seconds.
        request_timeout (int): HTTP request timeout in seconds.
    """

    def __init__(self, router_url: str, timeout: int = 300, request_timeout: int = 60):
        """
        WHAT: Initialize routed MorphCloud sandbox client

        WHY:
        - Store router URL for all future requests
        - Set default timeouts for execution and HTTP requests

        PARAMETERS:
        - router_url: URL including host and port (e.g., "192.168.1.100:8001")
        - timeout: Max execution time per script (default: 5 minutes)
        - request_timeout: Max HTTP request time (default: 60 seconds)

        EXAMPLE:
        ```python
        client = RoutedMorphSandbox(
            router_url="10.0.0.5:8001",
            timeout=180,  # 3 minutes for long-running code
            request_timeout=90  # Wait up to 90s for response
        )
        ```

        Initialize the routed MorphCloud sandbox client.

        Args:
            router_url: The URL of the MorphCloud router, including host and port.
            timeout: Default execution timeout in seconds.
            request_timeout: Default HTTP request timeout in seconds.
        """
        self.router_url = router_url
        self.timeout = timeout
        self.request_timeout = request_timeout

    def run_code(
        self,
        scripts: List[str],
        languages: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
    ) -> List:
        """
        WHAT: Execute multiple scripts using MorphCloud via the router

        WHY:
        - Batch execution reduces overhead (one request for many scripts)
        - Router manages sandbox lifecycle transparently
        - Consistent error handling for all scripts

        HOW:
        1. Apply default timeouts if not provided
        2. Default to Python for all scripts if languages not specified
        3. Build HTTP request payload
        4. Send POST request to router's /execute_batch endpoint
        5. Parse response and return results

        PROXIMAL CONTEXT:
        - INPUT: List of scripts and optional languages/timeouts
        - OUTPUT: List of result objects with text and exception_str

        DISTAL CONTEXT:
        - ORIGIN: Called by reward functions or evaluation scripts
        - DESTINATION: Router executes via MorphCloud, returns results

        EXAMPLE 1: Simple Python execution
        ```python
        client = RoutedMorphSandbox(router_url="10.0.0.5:8001")
        scripts = ["print('hello')", "print(1+1)"]
        results = client.run_code(scripts)
        # Result: [
        #   {text: "hello\n", exception_str: None},
        #   {text: "2\n", exception_str: None}
        # ]
        ```

        EXAMPLE 2: Mixed languages
        ```python
        scripts = [
            "print('Python')",
            "console.log('JavaScript')"
        ]
        languages = ["python", "javascript"]
        results = client.run_code(scripts, languages=languages)
        ```

        EXAMPLE 3: Custom timeouts
        ```python
        scripts = ["import time; time.sleep(10); print('done')"]
        results = client.run_code(
            scripts,
            timeout=20,  # Allow 20 seconds
            request_timeout=30  # Wait up to 30s for HTTP response
        )
        ```

        ERROR HANDLING:
        - Network errors: Returns exception_str with error message
        - Execution errors: Returns in exception_str (from router)
        - Timeout: Handled by router, returned as exception_str

        RESULT FORMAT:
        Each result is an object with:
        - text: Script output (stdout/stderr combined)
        - exception_str: Error message if failed, None if succeeded

        Execute multiple scripts using MorphCloud via the router.

        Args:
            scripts: List of code scripts to execute.
            languages: List of programming languages for each script. If None, defaults to Python for all scripts.
            timeout: Execution timeout in seconds. If None, uses the instance timeout.
            request_timeout: HTTP request timeout in seconds. If None, uses the instance request_timeout.

        Returns:
            List of execution results with text and exception_str properties.
        """

        # APPLY DEFAULT TIMEOUTS
        # Use instance defaults if not overridden
        actual_timeout = timeout if timeout is not None else self.timeout
        actual_request_timeout = request_timeout if request_timeout is not None else self.request_timeout

        # DEFAULT TO PYTHON FOR ALL SCRIPTS
        # WHY: Most common use case, simplifies API
        if languages is None:
            languages = ["python"] * len(scripts)

        # BUILD REQUEST PAYLOAD
        payload = {
            "scripts": scripts,
            "languages": languages,
            "timeout": actual_timeout,
            "request_timeout": actual_request_timeout,
        }

        try:
            # SEND REQUEST TO ROUTER
            # Uses http:// prefix (router expected on local network)
            endpoint = f"http://{self.router_url}/execute_batch"
            response = requests.post(endpoint, json=payload, timeout=actual_request_timeout)

            # CHECK RESPONSE STATUS
            if response.status_code != 200:
                error = f"Request to MorphCloud router failed with status code: {response.status_code}"
                print(error)

                # RETURN ERROR RESULTS FOR ALL SCRIPTS
                # WHY: Caller expects one result per script
                results = []
                for _ in scripts:
                    results.append(type("obj", (object,), {"text": None, "exception_str": error}))
                return results

            # PARSE RESPONSE AND BUILD RESULTS
            response_data = response.json()
            results = []

            for item in response_data:
                # Log the response data to see what we're getting
                # print(f"RoutedMorphSandbox: Got response item: {item}")

                # CREATE RESULT OBJECT
                # Uses type() to create anonymous object with attributes
                # WHY: Mimics MorphCloud Sandbox result format
                result = type(
                    "obj",
                    (object,),
                    {
                        "text": item.get("text"),
                        "exception_str": item.get("exception_str"),
                    },
                )
                results.append(result)

            return results

        except Exception as e:
            # HANDLE NETWORK/COMMUNICATION ERRORS
            error = f"Error communicating with MorphCloud router: {str(e)}"
            print(error)

            # RETURN ERROR RESULTS FOR ALL SCRIPTS
            results = []
            for _ in scripts:
                results.append(type("obj", (object,), {"text": None, "exception_str": error}))
            return results

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================
# 1. **Router Pattern**:
#    - Centralizes sandbox management
#    - Client sends requests, router handles complexity
#    - Reduces overhead and improves resource utilization
#
# 2. **Batch Processing**:
#    - Execute multiple scripts in one HTTP request
#    - More efficient than individual requests
#    - Results returned in same order as scripts
#
# 3. **API Compatibility**:
#    - Mimics MorphCloud Sandbox API
#    - Result format: {text: str, exception_str: Optional[str]}
#    - Easy to swap between direct and routed execution
#
# 4. **Error Handling**:
#    - Network errors: Returns error for all scripts
#    - HTTP errors: Returns error for all scripts
#    - Execution errors: Returned individually per script
#
# 5. **Timeout Configuration**:
#    - timeout: How long each script can run
#    - request_timeout: How long to wait for HTTP response
#    - Both can be overridden per request
#
# 6. **Default Language**:
#    - Defaults to Python if not specified
#    - Simplifies common case
#    - Supports any language MorphCloud supports
#
# 7. **Why Anonymous Objects**:
#    - type("obj", (object,), {...}) creates simple object
#    - Avoids defining Result class
#    - Compatible with existing code expecting these attributes
# ==============================================================================
