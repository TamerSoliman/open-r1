"""
==============================================================================
FILE: src/open_r1/utils/code_providers.py
CATEGORY: Infrastructure - Code Execution Providers
PRIORITY: CRITICAL
LINES: 367
DEPENDENCIES:
    - e2b_code_interpreter: E2B sandbox API
    - morphcloud: MorphCloud sandbox API
    - asyncio: Async execution management
==============================================================================

OVERVIEW:
This module provides pluggable code execution backends for reward functions.
It abstracts away the differences between E2B, MorphCloud, and Piston providers,
allowing reward functions to execute code securely without knowing the details.

ROLE IN DEEPSEEK R1:
- Enables code_reward, ioi_code_reward, cf_code_reward functions
- Critical for GRPO training on code tasks
- Provides secure sandboxed execution
- Handles timeouts, errors, and parallel execution

KEY INNOVATIONS:
1. Abstract Provider Interface: Switch backends without changing reward code
2. Router Pattern: Batch processing for efficiency
3. Async Execution: Parallel code evaluation
4. Multiple Timeout Layers: Sandbox + asyncio + request timeouts
5. Automatic Cleanup: Always kills sandboxes even on error

DATA FLOW:
Reward function (code_reward) → get_provider() → E2BProvider/MorphProvider
    → Async sandbox creation → Code execution → Results → Rewards
==============================================================================
"""

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
# [License omitted for brevity]

"""Code execution providers for executing and evaluating code snippets."""

import abc
import asyncio
from typing import List, Optional

from ..utils import is_e2b_available, is_morph_available


# WHAT: Conditional imports based on availability
# WHY: Not all users have E2B/MorphCloud installed
# HOW: Check if packages available before importing
if is_e2b_available():
    from e2b_code_interpreter import AsyncSandbox
    from e2b_code_interpreter.models import Execution
    from .routed_sandbox import RoutedSandbox
else:
    AsyncSandbox = None
    Execution = None
    RoutedSandbox = None

if is_morph_available():
    from morphcloud.api import MorphCloudClient
    from morphcloud.sandbox import Sandbox
    from .routed_morph import RoutedMorphSandbox
else:
    MorphCloudClient = None
    Sandbox = None
    RoutedMorphSandbox = None


"""
==============================================================================
ABSTRACT BASE CLASS
==============================================================================
"""


class CodeExecutionProvider(abc.ABC):
    """
    WHAT: Abstract base class for code execution providers

    WHY: Enables pluggable backends - switch between E2B, MorphCloud without
         changing reward function code

    HOW: Defines interface that all providers must implement

    USAGE:
        provider = get_provider(provider_type="e2b")
        rewards = provider.execute_scripts(scripts, languages)
    """

    @abc.abstractmethod
    def execute_scripts(self, scripts: List[str], languages: List[str]) -> List[float]:
        """
        WHAT: Execute multiple scripts and return their reward values

        WHY: Batch execution is more efficient than one-at-a-time

        HOW: Provider-specific implementation

        Args:
            scripts: List of code scripts to execute
            languages: The programming language of each script

        Returns:
            List of float rewards (one per script)
            - Typically 0.0 to 1.0 (pass rate or binary)
            - None if execution failed
        """
        pass


"""
==============================================================================
E2B PROVIDER
==============================================================================
"""


class E2BProvider(CodeExecutionProvider):
    """
    WHAT: Provider that executes code using E2B sandboxes

    WHY: E2B provides secure, isolated Python/JS/R/Java execution
         Free tier: 2 concurrent sandboxes per API key

    HOW:
        - Direct mode: AsyncSandbox for each execution
        - Router mode: RoutedSandbox for batch processing

    KEY FEATURES:
        - Async/await for concurrency (num_parallel sandboxes)
        - Multiple timeout layers (sandbox, asyncio, request)
        - Automatic cleanup (sandbox.kill())
        - Router mode scales to many executions

    USAGE:
        provider = E2BProvider(num_parallel=2)
        rewards = provider.execute_scripts(scripts, languages)
    """

    def __init__(self, num_parallel: int = 2, e2b_router_url: Optional[str] = None):
        """
        WHAT: Initialize the E2B provider

        WHY: num_parallel=2 is suitable for free tier (2 concurrent sandboxes)
             Router URL enables batch processing

        Args:
            num_parallel: Number of parallel sandboxes to use
            e2b_router_url: URL for the E2B router (if using router mode)
        """
        if not is_e2b_available():
            raise ImportError(
                "E2B is not available and required for this provider. Please install E2B with "
                "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
            )

        self.num_parallel = num_parallel
        self.e2b_router_url = e2b_router_url

    def execute_scripts(self, scripts: List[str], languages: List[str]) -> List[float]:
        """
        WHAT: Execute scripts using E2B sandboxes

        WHY: Two modes for different use cases:
             - Router: Batch processing for many scripts
             - Direct: Async execution with limited parallelism

        HOW:
            Router mode:
                1. Send batch to router service
                2. Router distributes to worker pool
                3. Collect results
            Direct mode:
                1. Create semaphore for concurrency control
                2. Launch async tasks for each script
                3. Gather results

        DATA FLOW:
            Scripts → Router/Async execution → Sandbox.run_code()
                → Execution results → Float rewards

        Returns:
            List of rewards (float or None per script)
        """
        # ROUTER MODE: Batch processing via router service
        if self.e2b_router_url is not None:
            routed_sandbox = RoutedSandbox(router_url=self.e2b_router_url)

            # WHAT: Send batch request to router
            # WHY: Router handles load balancing and scaling
            executions = routed_sandbox.run_code(
                scripts=scripts,
                languages=languages,
                timeout=30,  # Sandbox timeout
                request_timeout=28,  # HTTP request timeout (slightly less)
            )

            # WHAT: Parse results from router
            rewards = []
            for execution in executions:
                try:
                    # WHAT: execution.text should contain reward value
                    reward = float(execution.text)
                    rewards.append(reward)
                except Exception:
                    # WHAT: If parsing fails, return None (skip this example)
                    rewards.append(None)
            return rewards

        # DIRECT MODE: Async execution with concurrency limit
        try:
            rewards = self._run_async_from_sync(scripts, languages, self.num_parallel)
        except Exception as e:
            print(f"Error from E2B executor: {e}")
            rewards = [0.0] * len(scripts)

        return rewards

    def _run_async_from_sync(self, scripts: List[str], languages: List[str], num_parallel: int) -> List[float]:
        """
        WHAT: Wrapper to run async code from sync context

        WHY: asyncio.run() creates event loop and runs async function

        HOW: Just wraps _run_async() with error handling
        """
        try:
            rewards = asyncio.run(self._run_async(scripts, languages, num_parallel))
        except Exception as e:
            print(f"Error from E2B executor async: {e}")
            raise e

        return rewards

    async def _run_async(self, scripts: List[str], languages: List[str], num_parallel: int) -> List[float]:
        """
        WHAT: Run multiple scripts concurrently with limited parallelism

        WHY: Semaphore limits concurrent sandboxes (respects API limits)

        HOW:
            1. Create semaphore with num_parallel slots
            2. Create async task for each script
            3. asyncio.gather() runs all tasks concurrently
            4. Semaphore ensures max num_parallel at once

        Args:
            scripts: List of scripts to execute
            languages: Programming languages
            num_parallel: Maximum concurrent executions

        Returns:
            List of rewards
        """
        # WHAT: Semaphore limits concurrency
        # WHY: E2B free tier allows 2 concurrent sandboxes
        semaphore = asyncio.Semaphore(num_parallel)

        # WHAT: Create task for each script
        tasks = [self._run_script(script, languages, semaphore) for script in scripts]

        # WHAT: Run all tasks concurrently
        # HOW: asyncio.gather() waits for all tasks to complete
        results = await asyncio.gather(*tasks)
        rewards = list(results)

        return rewards

    async def _run_script(self, script: str, languages: List[str], semaphore: asyncio.Semaphore) -> float:
        """
        WHAT: Execute a single script in an E2B sandbox

        WHY: Async enables concurrent execution while respecting limits

        HOW:
            1. Acquire semaphore (wait if at limit)
            2. Create sandbox
            3. Run code with timeout
            4. Parse result (should be float)
            5. Kill sandbox (cleanup)
            6. Release semaphore

        TIMEOUT STRATEGY (Defense in Depth):
            - SANDBOX_TIMEOUT: 30s (E2B sandbox timeout)
            - REQUEST_TIMEOUT: 28s (HTTP request timeout, slightly less)
            - ASYNCIO_TIMEOUT: 32s (asyncio timeout, slightly more)
            Why? E2B sandbox timeout doesn't always work, so we add layers

        Args:
            script: Code to execute
            languages: Programming language
            semaphore: Concurrency limiter

        Returns:
            Float reward (0.0 on error)
        """
        # TIMEOUT CONFIGURATION
        # WHY: Multiple timeout layers ensure execution doesn't hang
        # These values are based on empirical testing with 256 examples
        SANDBOX_TIMEOUT = 30
        MARGIN = 2
        REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN  # 28s
        ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN  # 32s

        async with semaphore:
            try:
                # STEP 1: Create sandbox
                # WHY: Each execution gets fresh sandbox (isolation)
                sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)

                # STEP 2: Run code with asyncio timeout wrapper
                # WHY: Adds additional timeout layer beyond sandbox timeout
                execution = await asyncio.wait_for(
                    sandbox.run_code(script, languages=languages),
                    timeout=ASYNCIO_TIMEOUT,
                )

                # STEP 3: Parse result
                # WHAT: execution.text should contain reward value (float)
                return float(execution.text)

            except (TypeError, ValueError):
                # WHAT: Parsing failed (not a valid float)
                return 0.0
            except asyncio.TimeoutError:
                print("Operation timed out")
                return 0.0
            except Exception as e:
                print(f"Error in `_run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
                return 0.0
            finally:
                # STEP 4: Cleanup - ALWAYS kill sandbox
                # WHY: Prevents resource leaks
                try:
                    await sandbox.kill()
                except Exception as e:
                    print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")


"""
==============================================================================
MORPHCLOUD PROVIDER
==============================================================================
"""


class MorphProvider(CodeExecutionProvider):
    """
    WHAT: Provider that executes code using MorphCloud's Sandbox API

    WHY: Alternative to E2B with different pricing/limits
         Longer timeouts (90s vs 30s)

    HOW: Similar to E2BProvider but uses MorphCloud API

    KEY DIFFERENCES FROM E2B:
        - Longer timeout (90s vs 30s)
        - Different API (Sandbox.new vs AsyncSandbox.create)
        - Output parsing handles multi-line results
    """

    def __init__(self, num_parallel: int = 2, morph_router_url: Optional[str] = None):
        """Initialize the Morph provider."""
        if not is_morph_available():
            raise ImportError(
                "MorphCloud is not available and required for this provider. Please install MorphCloud with "
                "`pip install morphcloud` and add an API key to a `.env` file."
            )

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("Warning: python-dotenv not installed. Environment variables must be set directly.")

        self.num_parallel = num_parallel
        self.morph_router_url = morph_router_url

        # ROUTER MODE: Use router service
        if self.morph_router_url is not None:
            self.routed_sandbox = RoutedMorphSandbox(router_url=self.morph_router_url)
            return

        # DIRECT MODE: Set up MorphCloud client
        import os
        self.api_key = os.getenv("MORPH_API_KEY")
        if not self.api_key:
            raise ValueError("MorphCloud API key not found. Please set the MORPH_API_KEY environment variable.")

        try:
            self.client = MorphCloudClient(api_key=self.api_key)
            self.Sandbox = Sandbox
        except ImportError as e:
            raise ImportError(f"Required MorphCloud dependencies not installed: {e}")

    def execute_scripts(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Execute scripts using MorphCloud Sandbox API."""

        # ROUTER MODE
        if hasattr(self, "routed_sandbox"):
            try:
                results = self.routed_sandbox.run_code(
                    scripts=scripts,
                    languages=languages,
                    timeout=90,  # MorphCloud supports longer timeouts
                    request_timeout=96,
                )

                rewards = []
                for result in results:
                    try:
                        reward = float(result.text)
                        rewards.append(reward)
                    except (ValueError, AttributeError):
                        rewards.append(0.0)
                return rewards
            except Exception as e:
                print(f"Error from MorphCloud router: {e}")
                return [0.0] * len(scripts)

        # DIRECT MODE
        import asyncio
        try:
            rewards = asyncio.run(self._run_async(scripts, languages, self.num_parallel))
        except Exception as e:
            print(f"Error from MorphCloud executor: {e}")
            rewards = [0.0] * len(scripts)

        return rewards

    async def _run_async(self, scripts: List[str], languages: List[str], num_parallel: int) -> List[float]:
        """Run multiple scripts concurrently with limited parallelism."""
        semaphore = asyncio.Semaphore(num_parallel)
        tasks = [self._run_script(script, languages, semaphore) for script in scripts]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _run_script(self, script: str, languages: List[str], semaphore: asyncio.Semaphore) -> float:
        """
        WHAT: Execute a single script in a MorphCloud Sandbox

        KEY DIFFERENCE FROM E2B:
            - Uses asyncio.to_thread() for sync API
            - Longer timeout (90s)
            - Multi-line output parsing (takes last line)
        """
        SANDBOX_TIMEOUT = 90
        MARGIN = 6
        ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

        sandbox = None
        async with semaphore:
            try:
                # STEP 1: Create sandbox (sync API, wrapped in thread)
                sandbox = await asyncio.to_thread(self.Sandbox.new, client=self.client, ttl_seconds=SANDBOX_TIMEOUT)

                # STEP 2: Run code
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        sandbox.run_code,
                        script,
                        languages=languages,
                        timeout=SANDBOX_TIMEOUT,
                    ),
                    timeout=ASYNCIO_TIMEOUT,
                )

                # STEP 3: Parse result (multi-line handling)
                # WHY: MorphCloud may return multi-line output
                reward = 0.0
                try:
                    if hasattr(result, "text") and result.text:
                        lines = result.text.strip().split("\n")
                        if lines:
                            try:
                                # WHAT: Try last line first (typical convention)
                                reward = float(lines[-1])
                            except ValueError:
                                try:
                                    # WHAT: Fallback to entire text
                                    reward = float(result.text.strip())
                                except ValueError:
                                    pass
                    elif hasattr(result, "stdout") and result.stdout:
                        lines = result.stdout.strip().split("\n")
                        if lines:
                            try:
                                reward = float(lines[-1])
                            except ValueError:
                                pass
                except (ValueError, AttributeError):
                    pass

                return reward

            except asyncio.TimeoutError:
                return 0.0
            except Exception:
                return 0.0
            finally:
                # STEP 4: Cleanup
                if sandbox:
                    try:
                        await asyncio.to_thread(sandbox.close)
                        await asyncio.to_thread(sandbox.shutdown)
                    except Exception:
                        pass


"""
==============================================================================
FACTORY FUNCTION
==============================================================================
"""


def get_provider(provider_type: str = "e2b", **kwargs) -> CodeExecutionProvider:
    """
    WHAT: Factory function to get the appropriate code execution provider

    WHY: Centralizes provider selection logic
         Enables easy switching between backends

    HOW:
        1. Extract provider-specific kwargs
        2. Instantiate appropriate provider class
        3. Return provider instance

    USAGE:
        # E2B with router
        provider = get_provider(
            provider_type="e2b",
            num_parallel=2,
            e2b_router_url="http://localhost:5000"
        )

        # MorphCloud direct
        provider = get_provider(
            provider_type="morph",
            num_parallel=4
        )

    Args:
        provider_type: Type of provider to use ("e2b", "morph")
        **kwargs: Additional arguments to pass to the provider

    Returns:
        An instance of CodeExecutionProvider

    Raises:
        ValueError: If provider_type is unknown
    """
    num_parallel = kwargs.pop("num_parallel", 2)

    if provider_type == "e2b":
        # Extract E2B-specific arguments
        e2b_router_url = kwargs.pop("e2b_router_url", None)
        return E2BProvider(
            num_parallel=num_parallel,
            e2b_router_url=e2b_router_url,
        )
    elif provider_type == "morph":
        # Extract Morph-specific arguments
        morph_router_url = kwargs.pop("morph_router_url", None)
        return MorphProvider(
            num_parallel=num_parallel,
            morph_router_url=morph_router_url,
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


"""
==============================================================================
KEY TAKEAWAYS - CODE EXECUTION PROVIDERS
==============================================================================

1. **Security is Critical**:
   - Never execute untrusted code directly
   - Always use sandboxed environments
   - E2B and MorphCloud provide isolation

2. **Multiple Timeout Layers**:
   - Sandbox timeout (30s/90s)
   - Request timeout (slightly less)
   - Asyncio timeout (slightly more)
   - Defense in depth prevents hangs

3. **Cleanup is Essential**:
   - Always kill sandboxes in finally block
   - Prevents resource leaks
   - Important for long-running training

4. **Router Pattern Scales**:
   - Direct mode: Good for small batches
   - Router mode: Scales to 100s of concurrent executions
   - Trade-off: Complexity vs throughput

5. **Free Tier Limitations**:
   - E2B free: 2 concurrent sandboxes
   - num_parallel=2 respects this limit
   - Paid tiers allow more concurrency

6. **Error Handling**:
   - Return 0.0 on any error
   - None for unparseable results
   - Print errors for debugging

7. **Provider Choice**:
   - E2B: Faster (30s timeout), good for most code
   - MorphCloud: Longer timeout (90s), good for slow code
   - Piston: Self-hosted, for competitive programming

==============================================================================
"""
