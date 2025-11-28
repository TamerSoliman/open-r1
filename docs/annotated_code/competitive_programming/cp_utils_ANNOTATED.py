# ==============================================================================
# FILE: src/open_r1/utils/competitive_programming/utils.py
# CATEGORY: Utilities - General Helpers
# PRIORITY: LOW
# LINES: 12
# DEPENDENCIES: itertools
# ==============================================================================
#
# OVERVIEW:
# Simple utility function for batching iterables. Provides a batched() function
# similar to itertools.batched (Python 3.12+) for older Python versions.
#
# KEY FUNCTIONALITY:
# - batched(): Split iterable into fixed-size chunks
#
# This is a polyfill for itertools.batched which was added in Python 3.12.
# ==============================================================================

from itertools import islice


def batched(iterable, n):
    """
    WHAT: Batch data into lists of length n (last batch may be shorter)

    WHY:
    - Process large datasets in manageable chunks
    - Control memory usage by limiting batch size
    - Enable parallel processing of batches
    - Polyfill for Python 3.12's itertools.batched

    HOW:
    1. Return original iterable if n < 1
    2. Create iterator from iterable
    3. Use islice to extract n items at a time
    4. Yield batches until iterator exhausted

    PROXIMAL CONTEXT:
    - INPUT: Any iterable and batch size
    - OUTPUT: Generator yielding lists of size n

    DISTAL CONTEXT:
    - ORIGIN: Called when processing large lists of problems/tests
    - DESTINATION: Used in evaluation loops

    EXAMPLE 1: Basic batching
    ```python
    items = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for batch in batched(items, 3):
        print(batch)
    # Output:
    # ['A', 'B', 'C']
    # ['D', 'E', 'F']
    # ['G']  # Last batch shorter
    ```

    EXAMPLE 2: Processing in chunks
    ```python
    test_cases = range(100)
    for batch in batched(test_cases, 10):
        # Process 10 test cases at a time
        results = run_parallel(batch)
    ```

    EXAMPLE 3: Invalid batch size
    ```python
    items = [1, 2, 3]
    batched(items, 0)  # Returns original iterable
    batched(items, -1)  # Returns original iterable
    ```

    MEMORY EFFICIENCY:
    - Generator: Doesn't create all batches upfront
    - Only holds current batch in memory
    - Ideal for large datasets

    VS PYTHON 3.12 itertools.batched:
    - Same behavior for valid inputs
    - This version handles n < 1 by returning original iterable
    - Python 3.12 version raises ValueError for n < 1

    Batch data into lists of length n. The last batch may be shorter.
    """
    # batched('ABCDEFG', 3) --> ABC DEF G

    # HANDLE INVALID BATCH SIZE
    # If n < 1, batching doesn't make sense
    # Return original iterable unchanged
    if n < 1:
        return iterable

    # CREATE ITERATOR
    # Needed for islice to work
    it = iter(iterable)

    # YIELD BATCHES USING WALRUS OPERATOR
    # while batch := list(islice(it, n)):
    #   1. islice(it, n) extracts up to n items
    #   2. list() converts to list
    #   3. Assignment to batch
    #   4. If batch is empty (falsy), loop stops
    while batch := list(islice(it, n)):
        yield batch

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================
# 1. **Purpose**:
#    - Polyfill for Python 3.12's itertools.batched
#    - Enables batching in Python 3.8-3.11
#    - Simple and efficient implementation
#
# 2. **Memory Efficiency**:
#    - Generator function (lazy evaluation)
#    - Only one batch in memory at a time
#    - Suitable for large datasets
#
# 3. **Use Cases**:
#    - Batch API requests (avoid rate limits)
#    - Parallel processing (distribute work)
#    - Memory management (process in chunks)
#    - Database queries (batch inserts)
#
# 4. **Edge Cases**:
#    - n < 1: Returns original iterable
#    - Empty iterable: No batches yielded
#    - n > len(iterable): Single batch with all items
#    - Last batch: May be shorter than n
#
# 5. **Common Patterns**:
#    ```python
#    # Parallel processing
#    for batch in batched(items, batch_size):
#        results = process_parallel(batch)
#
#    # API rate limiting
#    for batch in batched(requests, 10):
#        responses = api.batch_request(batch)
#        time.sleep(1)  # Respect rate limit
#
#    # Memory management
#    for batch in batched(huge_dataset, 1000):
#        process_and_save(batch)
#        del batch  # Free memory
#    ```
#
# 6. **Why Walrus Operator**:
#    - := assigns and returns value in one expression
#    - Cleaner than traditional while True with break
#    - Available in Python 3.8+
#
# 7. **Alternative Without Walrus**:
#    ```python
#    def batched_old(iterable, n):
#        it = iter(iterable)
#        while True:
#            batch = list(islice(it, n))
#            if not batch:
#                break
#            yield batch
#    ```
# ==============================================================================
