# ==============================================================================
# FILE: src/open_r1/utils/competitive_programming/piston_client.py
# CATEGORY: Utilities - Piston Code Execution Client
# PRIORITY: MEDIUM
# LINES: 225
# DEPENDENCIES: aiohttp, asyncio, subprocess, re
# ==============================================================================
#
# OVERVIEW:
# Client for executing code using Piston (open-source code execution engine).
# Supports load balancing across multiple Piston workers with automatic failover,
# retry logic, and health monitoring.
#
# KEY FUNCTIONALITY:
# - PistonClient: Load-balanced client for multiple Piston endpoints
# - get_piston_client_from_env(): Create client from environment variables
# - get_slurm_piston_endpoints(): Discover Piston workers from SLURM
# - Automatic retry with exponential backoff
# - Health monitoring and endpoint failover
#
# Piston is used for IOI/Codeforces problems that require the custom cms_ioi package.
# This client provides production-grade reliability for code execution at scale.
# ==============================================================================

import asyncio
import os
import random
import re
import subprocess
from collections import Counter
from functools import lru_cache

import aiohttp


class PistonError(Exception):
    """
    WHAT: Custom exception for Piston-specific errors

    WHY:
    - Distinguish Piston errors from general exceptions
    - Enable specific error handling for Piston issues
    - Provide clear error categorization

    RAISED WHEN:
    - All endpoints are unhealthy
    - Server returns non-200 status
    - Response is empty or invalid
    - Resource temporarily unavailable (overload)
    """
    pass


@lru_cache(maxsize=1)
def get_piston_client_from_env(session=None):
    """
    WHAT: Create PistonClient from environment variables with GPU-aware endpoint distribution

    WHY:
    - Centralized client creation from configuration
    - Support for multiple Piston endpoints (load balancing)
    - GPU-aware distribution for multi-GPU training
    - SLURM integration for dynamic endpoint discovery

    HOW:
    1. Read PISTON_ENDPOINTS from environment
    2. Parse comma-separated endpoints or "slurm" keyword
    3. Distribute endpoints across GPUs if in multi-GPU setup
    4. Shuffle endpoints for randomized load balancing
    5. Create and cache client instance

    PROXIMAL CONTEXT:
    - INPUT: Optional aiohttp session
    - OUTPUT: Cached PistonClient instance

    DISTAL CONTEXT:
    - ORIGIN: Called at evaluation script initialization
    - DESTINATION: Used for all code execution requests

    ENVIRONMENT VARIABLES:
    - PISTON_ENDPOINTS: Comma-separated URLs or "slurm"
    - LOCAL_RANK: GPU index (for multi-GPU)
    - WORLD_SIZE: Total number of GPUs
    - PISTON_MAX_REQUESTS_PER_ENDPOINT: Concurrent requests per endpoint

    EXAMPLE 1: Simple setup
    ```python
    os.environ["PISTON_ENDPOINTS"] = "http://10.0.0.1:2000,http://10.0.0.2:2000"
    client = get_piston_client_from_env()
    # Result: Client with 2 endpoints
    ```

    EXAMPLE 2: Multi-GPU setup
    ```python
    os.environ["PISTON_ENDPOINTS"] = "http://10.0.0.1:2000,http://10.0.0.2:2000,http://10.0.0.3:2000,http://10.0.0.4:2000"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "2"
    client = get_piston_client_from_env()
    # Result: GPU 0 gets endpoints 0,2 (even indices)
    #         GPU 1 gets endpoints 1,3 (odd indices)
    ```

    EXAMPLE 3: SLURM integration
    ```python
    os.environ["PISTON_ENDPOINTS"] = "slurm"
    client = get_piston_client_from_env()
    # Result: Endpoints discovered from squeue
    ```

    CACHING:
    - @lru_cache(maxsize=1) ensures single client instance
    - Prevents duplicate client creation
    - Session reuse for connection pooling
    """
    # READ PISTON_ENDPOINTS ENVIRONMENT VARIABLE
    piston_endpoints = os.getenv("PISTON_ENDPOINTS")
    if piston_endpoints is None:
        raise ValueError(
            "For IOI/CF problems Piston endpoints running our IOI package are required. "
            "Please add a list of valid Piston endpoints to a PISTON_ENDPOINTS variable in a `.env` file."
        )

    # PARSE ENDPOINTS
    # Either comma-separated URLs or "slurm" keyword
    piston_endpoints = sorted(
        piston_endpoints.split(",") if piston_endpoints != "slurm" else get_slurm_piston_endpoints()
    )

    # GPU-AWARE ENDPOINT DISTRIBUTION
    # In multi-GPU setup, distribute endpoints across GPUs
    # WHY: Prevents endpoint contention between GPU processes
    gpu_nb = int(os.getenv("LOCAL_RANK", 0))  # per-GPU index
    world = int(os.getenv("WORLD_SIZE", 1))  # total GPUs
    if world > 1:
        print(f"Using a subset of piston endpoints for GPU#{gpu_nb}")
        # Each GPU gets every Nth endpoint (interleaved distribution)
        # GPU 0: endpoints[0::world] (0, world, 2*world, ...)
        # GPU 1: endpoints[1::world] (1, world+1, 2*world+1, ...)
        piston_endpoints = piston_endpoints[gpu_nb::world]

    # RANDOMIZE ENDPOINT ORDER
    # WHY: Distributes load if multiple clients start simultaneously
    random.shuffle(piston_endpoints)

    # GET MAX REQUESTS PER ENDPOINT
    # Controls concurrency level per endpoint
    max_requests_per_endpoint = os.getenv("PISTON_MAX_REQUESTS_PER_ENDPOINT", "1")

    return PistonClient(piston_endpoints, session, max_requests_per_endpoint=int(max_requests_per_endpoint))


class PistonClient:
    """
    WHAT: Load-balanced client for Piston code execution with health monitoring

    WHY:
    - Distribute load across multiple Piston workers
    - Automatic failover when endpoints fail
    - Retry logic with exponential backoff
    - Health monitoring prevents request routing to failed endpoints

    HOW:
    - Token-based concurrency control (semaphore per endpoint)
    - Round-robin endpoint selection via queue
    - Health checks on failures
    - Exponential backoff with jitter

    ARCHITECTURE:
    ```
    PistonClient
    ├── endpoint_tokens (Queue): Available endpoints for requests
    ├── _endpoint_failures (Counter): Failure counts per endpoint
    ├── _unhealthy_endpoints (Set): Failed endpoints
    └── session (aiohttp): Connection pool
    ```

    A client that will automatically load balance across multiple Piston (https://github.com/engineer-man/piston) workers.
    This assumes piston is running our custom cms_ioi package: https://github.com/guipenedo/piston/releases/

    We recommend starting the instances with the following script as otherwise some IOI problems will hit default limits:
    ```
    export PISTON_COMPILE_TIMEOUT=60000
    export PISTON_RUN_TIMEOUT=60000
    export PISTON_OUTPUT_MAX_SIZE=1000000000
    export PISTON_MAX_FILE_SIZE=1000000000
    export PISTON_DISABLE_NETWORKING=true
    export PISTON_REPO_URL=https://github.com/guipenedo/piston/releases/download/pkgs/index
    mkdir /piston

    sed -i '/app.use(body_parser.urlencoded/c\    app.use(body_parser.urlencoded({ extended: true, limit: \"512mb\" }));' src/index.js
    sed -i '/app.use(body_parser.json/c\    app.use(body_parser.json({ limit: \"512mb\" }));' src/index.js

    # Start server in background
    node src
    ```

    Piston docs for API usage: https://piston.readthedocs.io/en/latest/api-v2/
    """

    def __init__(
        self,
        base_endpoint: str | list[str] = "http://ip-10-53-80-65:3223/api/v2",
        session=None,
        max_requests_per_endpoint=1,
    ):
        """
        WHAT: Initialize Piston client with endpoints and concurrency control

        PARAMETERS:
        - base_endpoint: Single endpoint or list of endpoints
        - session: Optional aiohttp ClientSession for connection pooling
        - max_requests_per_endpoint: Concurrent requests per endpoint

        CONCURRENCY MODEL:
        - Uses token-based system (asyncio.Queue)
        - Each endpoint has max_requests_per_endpoint tokens
        - Request acquires token, releases after completion
        - Prevents endpoint overload
        """
        self.max_requests_per_endpoint = max_requests_per_endpoint

        # NORMALIZE ENDPOINTS TO LIST
        self.base_endpoints = [base_endpoint] if isinstance(base_endpoint, str) else base_endpoint
        if len(self.base_endpoints) == 0:
            raise ValueError("No Piston endpoints provided. Please check your PISTON_ENDPOINTS environment variable.")

        # CREATE ENDPOINT ID MAPPING
        # WHY: For logging and debugging
        self.endpoint_ids = {endpoint: i for i, endpoint in enumerate(self.base_endpoints)}

        # SESSION MANAGEMENT
        self._session = session

        # CONCURRENCY CONTROL: TOKEN QUEUE
        # Each endpoint contributes max_requests_per_endpoint tokens
        # Total tokens = max_requests_per_endpoint * num_endpoints
        self.endpoint_tokens = asyncio.Queue(maxsize=max_requests_per_endpoint * len(self.base_endpoints))

        # POPULATE TOKEN QUEUE
        # Each iteration adds max_requests_per_endpoint tokens per endpoint
        for _ in range(max_requests_per_endpoint):
            for base_endpoint in self.base_endpoints:
                self.endpoint_tokens.put_nowait(base_endpoint)

        # HEALTH MONITORING
        self._endpoint_failures = Counter()  # Track failure counts
        self._unhealthy_endpoints = set()    # Endpoints marked as failed
        self._endpoint_failures_lock = asyncio.Lock()  # Thread-safe health updates

    @property
    def session(self):
        """
        WHAT: Lazy-initialized aiohttp session with connection pooling

        WHY:
        - Connection pooling reduces overhead
        - Configured timeouts prevent hangs
        - Keep-alive improves performance

        CONFIGURATION:
        - sock_read timeout: 30s
        - Connection limit: max_requests_per_endpoint * num_endpoints
        - DNS cache: 5 minutes
        - Keep-alive: 5 minutes
        """
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(sock_read=30),
                connector=aiohttp.TCPConnector(
                    limit=self.max_requests_per_endpoint * len(self.base_endpoints),
                    ttl_dns_cache=300,
                    keepalive_timeout=5 * 60,
                ),
            )
        return self._session

    async def _wait_for_endpoint(self):
        """
        WHAT: Acquire endpoint token (blocks if all endpoints busy)

        WHY:
        - Implements concurrency control
        - Prevents endpoint overload
        - Fair distribution via queue

        RETURNS: Available endpoint URL
        """
        endpoint = await self.endpoint_tokens.get()
        return endpoint

    async def _release_endpoint(self, endpoint):
        """
        WHAT: Release endpoint token back to queue

        WHY:
        - Makes endpoint available for next request
        - Maintains token count invariant

        PARAMETERS:
        - endpoint: Endpoint URL to release
        """
        await self.endpoint_tokens.put(endpoint)

    async def _send_request(self, endpoint, route, data=None, method="post"):
        """
        WHAT: Send HTTP request to Piston endpoint

        WHY:
        - Centralized request logic
        - Consistent error handling
        - JSON content type enforcement

        RETURNS: Parsed JSON response
        """
        async with self.session.request(
            method, f"{endpoint.rstrip('/')}/{route}", json=data, headers={"Content-Type": "application/json"}
        ) as response:
            return await response.json(content_type=None)

    async def _send_to_all(self, route, data=None, method="post"):
        """
        WHAT: Send request to all endpoints (broadcast)

        USE CASES:
        - Package installation/uninstallation
        - Runtime queries
        - Health checks

        RETURNS: List of responses (one per endpoint)
        """
        return await asyncio.gather(
            *[self._send_request(endpoint, route, data, method) for endpoint in self.base_endpoints]
        )

    async def _send_to_one(self, endpoint, route, data=None, method="post"):
        """WHAT: Send request to specific endpoint (wrapper for clarity)"""
        return await self._send_request(endpoint, route, data, method)

    async def install_package(self, language, version):
        """WHAT: Install Piston package on all endpoints"""
        return await self._send_to_all("packages", {"language": language, "version": version}, method="post")

    async def uninstall_package(self, language, version):
        """WHAT: Uninstall Piston package from all endpoints"""
        return await self._send_to_all("packages", {"language": language, "version": version}, method="delete")

    async def get_supported_runtimes(self):
        """WHAT: Query supported runtimes from all endpoints"""
        return await self._send_to_all("runtimes", method="get")

    async def _check_failed_endpoint(self, endpoint):
        """
        WHAT: Mark endpoint as unhealthy after failure

        WHY:
        - Prevents routing requests to failed endpoints
        - Raises error when all endpoints fail

        HOW:
        1. Acquire lock (thread-safe)
        2. Skip if already marked unhealthy
        3. Wait 5 seconds
        4. Test endpoint with runtime query
        5. If still failing, mark unhealthy
        6. Raise error if all endpoints unhealthy
        """
        async with self._endpoint_failures_lock:
            if endpoint in self._unhealthy_endpoints:
                return
            try:
                await asyncio.sleep(5)
                await self.get_supported_runtimes()
            except Exception as e:
                print(f"Error checking endpoint {endpoint}, dropping it ({e})")
                self._unhealthy_endpoints.add(endpoint)
                if len(self._unhealthy_endpoints) >= len(self.base_endpoints):
                    raise PistonError("All endpoints are unhealthy. Please check your Piston workers.")

    async def send_execute(self, data, language="cms_ioi", max_retries=5):
        """
        WHAT: Execute code via Piston with retry logic and error handling

        WHY:
        - Handles transient failures (network, overload)
        - Exponential backoff prevents thundering herd
        - Automatic failover to healthy endpoints

        HOW:
        1. Acquire endpoint token
        2. Send execute request
        3. Handle errors with retry logic
        4. Mark endpoints unhealthy on connection failures
        5. Release token after completion

        RETRY STRATEGY:
        - Max retries: 5
        - Exponential backoff: 1s, 2s, 4s, 8s, 10s (capped)
        - Jitter: ±10% to prevent synchronization

        ERROR HANDLING:
        - PistonError: Retry (server error, overload)
        - TimeoutError: Retry (network timeout)
        - ClientConnectionError: Mark unhealthy, retry
        - Other exceptions: Propagate

        PARAMETERS:
        - data: Execution request (files, stdin, etc.)
        - language: Piston runtime (default: cms_ioi)
        - max_retries: Maximum retry attempts

        RETURNS: Execution result JSON
        """
        # ADD LANGUAGE AND VERSION TO REQUEST
        data = data | {
            "language": language,
            "version": "*",  # Use latest available version
        }

        base_delay = 1.0  # Initial retry delay

        status = None
        endpoint = None

        # RETRY LOOP
        for attempt in range(max_retries + 1):
            try:
                # ACQUIRE ENDPOINT TOKEN
                endpoint = await self._wait_for_endpoint()

                # BRIEF DELAY ON RETRY
                if attempt > 0:
                    await asyncio.sleep(1)

                # SEND EXECUTE REQUEST
                async with self.session.post(
                    f"{endpoint.rstrip('/')}/execute", json=data, headers={"Content-Type": "application/json"}
                ) as response:
                    status = response.status
                    res_json = await response.json(content_type=None)

                    # CHECK FOR ERRORS
                    if status != 200:
                        raise PistonError(f"Server error. status={status}. {res_json}")
                    if res_json is None:
                        raise PistonError(f"Empty response. status={status}")

                    # CHECK FOR OVERLOAD
                    # Piston returns "Resource temporarily unavailable" when overloaded
                    if "run" in res_json and "Resource temporarily unavailable" in res_json["run"].get("stderr", ""):
                        raise PistonError(f"Piston overloaded: {res_json['run']['stderr']}")

                    return res_json

            except (PistonError, asyncio.TimeoutError, aiohttp.ClientConnectionError, RuntimeError) as e:
                # RETRY LOGIC
                if attempt < max_retries:
                    # CALCULATE EXPONENTIAL BACKOFF
                    delay = min(base_delay * (2**attempt), 10)  # Cap at 10 seconds
                    jitter = delay * 0.2 * (2 * asyncio.get_event_loop().time() % 1 - 0.5)  # ±10% jitter
                    retry_delay = delay + jitter
                    print(f"Retrying in {retry_delay:.2f} seconds [{self.endpoint_ids[endpoint]}] {endpoint} - {e}")

                    # HANDLE CONNECTION FAILURES
                    # "Connect call failed" indicates endpoint is down
                    if isinstance(e, aiohttp.ClientConnectionError) and "Connect call failed" in str(e):
                        await self._check_failed_endpoint(endpoint)
                    else:
                        # RELEASE ENDPOINT FOR RETRY
                        # Hopefully we won't get this one again
                        await self._release_endpoint(endpoint)
                    endpoint = None

                    await asyncio.sleep(retry_delay)
                else:
                    # MAX RETRIES EXCEEDED
                    await self._check_failed_endpoint(endpoint)

            except Exception as e:
                # UNEXPECTED ERRORS: PROPAGATE
                print(f"Propagating exception {type(e)}: {e}")
                raise e

            finally:
                # ALWAYS RELEASE ENDPOINT TOKEN
                if endpoint is not None:
                    try:
                        await self._release_endpoint(endpoint)
                    except Exception as e:
                        print(f"Error releasing endpoint {endpoint}: {e}")
                    endpoint = None


def get_slurm_piston_endpoints():
    """
    WHAT: Discover Piston worker endpoints from SLURM queue

    WHY:
    - Dynamic endpoint discovery in SLURM environments
    - No need to manually configure endpoints
    - Automatically finds running piston-worker jobs

    HOW:
    1. Run squeue to list running jobs
    2. Filter for jobs named "piston-worker-{port}"
    3. Extract hostname and port
    4. Build endpoint URLs

    PROXIMAL CONTEXT:
    - INPUT: None (reads from squeue)
    - OUTPUT: List of endpoint URLs

    EXAMPLE OUTPUT:
    ```python
    [
        "http://node001:2000/api/v2",
        "http://node002:2000/api/v2",
        "http://node003:2001/api/v2",
    ]
    ```

    JOB NAMING CONVENTION:
    - Format: "piston-worker-{port}"
    - Example: "piston-worker-2000"
    - Port extracted via regex

    SQUEUE COMMAND:
    - Format: "%j %N %T" (job name, hostname, state)
    - Filter: RUNNING state only

    Get list of active piston worker endpoints from squeue output
    """
    # RUN SQUEUE COMMAND
    # Gets job name, hostname, and status for running jobs
    result = subprocess.run(
        ["squeue", '--format="%j %N %T"', "--noheader", "--states=RUNNING"], capture_output=True, text=True
    )

    # PARSE OUTPUT
    lines = result.stdout.strip().split("\n")

    endpoints = []
    for line in lines:
        # PARSE JOB NAME AND HOSTNAME
        fields = line.split()
        job_name = fields[0].strip('"')  # Remove quotes
        hostname = fields[1]

        # EXTRACT PORT FROM JOB NAME
        # Pattern: piston-worker-{port}
        match = re.match(r"piston-worker-(\d+)", job_name)
        if match:
            port = match.group(1)
            endpoints.append(f"http://{hostname}:{port}/api/v2")

    return endpoints

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================
# 1. **Load Balancing**:
#    - Token-based concurrency control
#    - Round-robin via queue
#    - GPU-aware endpoint distribution
#
# 2. **Reliability**:
#    - Automatic retry with exponential backoff
#    - Health monitoring and failover
#    - Jitter prevents thundering herd
#
# 3. **Piston Setup**:
#    - Requires custom cms_ioi package
#    - Increased limits for IOI problems
#    - Multiple workers for load balancing
#
# 4. **SLURM Integration**:
#    - Dynamic endpoint discovery
#    - Automatic job detection
#    - Zero configuration in SLURM environments
#
# 5. **Error Handling**:
#    - Distinguishes transient vs permanent failures
#    - Marks endpoints unhealthy after connection failures
#    - Raises when all endpoints fail
#
# 6. **Multi-GPU Support**:
#    - Distributes endpoints across GPUs
#    - Prevents endpoint contention
#    - Uses LOCAL_RANK and WORLD_SIZE env vars
#
# 7. **Performance Optimizations**:
#    - Connection pooling (aiohttp)
#    - Keep-alive connections
#    - DNS caching
#    - Concurrent requests per endpoint
# ==============================================================================
