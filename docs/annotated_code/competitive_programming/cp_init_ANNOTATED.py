# ==============================================================================
# FILE: src/open_r1/utils/competitive_programming/__init__.py
# CATEGORY: Utilities - Competitive Programming Module Exports
# PRIORITY: LOW
# LINES: 20
# DEPENDENCIES: Internal module imports
# ==============================================================================
#
# OVERVIEW:
# Module initialization file that exports key competitive programming utilities.
# Provides a clean public API for code patching, scoring, and sandbox clients.
#
# KEY EXPORTS:
# - Sandbox clients: get_piston_client_from_env, get_morph_client_from_env
# - Code patching: patch_code, add_includes
# - Scoring: score_submission (Codeforces), score_subtask/score_subtasks (IOI)
# - Utilities: get_slurm_piston_endpoints, SubtaskResult
#
# This __init__.py makes imports cleaner:
# from open_r1.utils.competitive_programming import patch_code
# instead of:
# from open_r1.utils.competitive_programming.code_patcher import patch_code
# ==============================================================================

from .cf_scoring import score_submission
from .code_patcher import patch_code
from .ioi_scoring import SubtaskResult, score_subtask, score_subtasks
from .ioi_utils import add_includes
from .morph_client import get_morph_client_from_env
from .piston_client import get_piston_client_from_env, get_slurm_piston_endpoints


"""
MODULE EXPORTS BREAKDOWN:

1. SANDBOX CLIENTS:
   - get_piston_client_from_env: Create Piston client from env vars
   - get_morph_client_from_env: Create MorphCloud client from env vars
   - get_slurm_piston_endpoints: Get Piston endpoints from SLURM

   WHY: Centralized client creation with consistent configuration

2. CODE PATCHING:
   - patch_code: Fix common Python/C++ compatibility issues
   - add_includes: Add IOI problem headers to C++ code

   WHY: Auto-fix common errors in generated code

3. SCORING SYSTEMS:
   - score_submission: Codeforces-style scoring (pass all tests)
   - score_subtask: IOI-style subtask scoring (single subtask)
   - score_subtasks: IOI-style scoring (all subtasks)
   - SubtaskResult: Dataclass for subtask results

   WHY: Support different competitive programming scoring systems

4. IMPORT PATTERN:
   ```python
   # Clean imports from package root
   from open_r1.utils.competitive_programming import (
       patch_code,
       score_submission,
       get_piston_client_from_env
   )
   ```

5. MODULE STRUCTURE:
   ```
   competitive_programming/
   ├── __init__.py (this file)
   ├── cf_scoring.py (Codeforces scoring)
   ├── code_patcher.py (code fixes)
   ├── ioi_scoring.py (IOI scoring)
   ├── ioi_utils.py (IOI utilities)
   ├── morph_client.py (MorphCloud client)
   ├── piston_client.py (Piston client)
   └── utils.py (general utilities)
   ```
"""

__all__ = [
    # Sandbox clients
    "get_piston_client_from_env",
    "get_slurm_piston_endpoints",
    "get_morph_client_from_env",

    # Code patching
    "patch_code",

    # Scoring
    "score_submission",  # Codeforces
    "score_subtask",     # IOI (single)
    "score_subtasks",    # IOI (multiple)

    # IOI utilities
    "add_includes",

    # Data types
    "SubtaskResult",
]

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================
# 1. **Module Organization**:
#    - Clean separation of concerns (scoring, patching, clients)
#    - __all__ defines public API
#    - Shorter import paths for users
#
# 2. **Competitive Programming Support**:
#    - Codeforces: Binary scoring (all tests pass/fail)
#    - IOI: Subtask-based partial credit
#    - Code patching for common errors
#
# 3. **Sandbox Abstraction**:
#    - Piston: Local execution, multiple endpoints
#    - MorphCloud: Cloud execution, managed service
#    - Consistent client creation pattern
#
# 4. **Import Benefits**:
#    - Users don't need to know internal structure
#    - Easy to refactor without breaking imports
#    - Clear API surface via __all__
#
# 5. **Common Usage**:
#    ```python
#    from open_r1.utils.competitive_programming import (
#        patch_code,
#        score_submission,
#        get_piston_client_from_env
#    )
#
#    client = get_piston_client_from_env()
#    code = patch_code(generated_code, "python3")
#    score = score_submission(code, test_cases, client)
#    ```
# ==============================================================================
