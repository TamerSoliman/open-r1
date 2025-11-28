# ==============================================================================
# FILE: src/open_r1/utils/import_utils.py
# CATEGORY: Utilities - Import Checks
# PRIORITY: LOW
# LINES: 31
# DEPENDENCIES: transformers
# ==============================================================================
#
# OVERVIEW:
# Check availability of optional code execution providers (E2B, Morph).
# Enables graceful fallback when providers are not installed.
#
# PROVIDERS:
# - E2B: Cloud-based code sandboxing
# - Morph: Alternative code execution service
#
# USAGE:
# ```python
# from open_r1.utils import is_e2b_available, is_morph_available
#
# if is_e2b_available():
#     from e2b import Sandbox
# elif is_morph_available():
#     from morphcloud import execute
# else:
#     raise RuntimeError("No code execution provider available")
# ```
# ==============================================================================

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

from transformers.utils.import_utils import _is_package_available


# Use same as transformers.utils.import_utils
_e2b_available = _is_package_available("e2b")


def is_e2b_available() -> bool:
    """Check if E2B code execution provider is installed"""
    return _e2b_available


_morph_available = _is_package_available("morphcloud")


def is_morph_available() -> bool:
    """Check if Morph code execution provider is installed"""
    return _morph_available
