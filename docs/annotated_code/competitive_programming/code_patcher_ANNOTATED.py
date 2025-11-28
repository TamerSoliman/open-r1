# ==============================================================================
# FILE: src/open_r1/utils/competitive_programming/code_patcher.py
# CATEGORY: Utilities - Code Patching
# PRIORITY: MEDIUM
# LINES: 124
# DEPENDENCIES: re
# ==============================================================================
#
# OVERVIEW:
# Automatically fixes common compatibility issues in generated code for competitive
# programming problems. Handles Python 3 compatibility issues and C++ includes.
#
# KEY FUNCTIONALITY:
# - fix_python3_imports(): Fix deprecated Python 3 imports and syntax
# - fix_cpp_includes(): Add standard C++ headers and namespace
# - patch_code(): Main entry point for patching
# - is_patchable(): Check if language supports patching
#
# This module is essential for making model-generated code executable, as models
# often generate code with deprecated imports or missing headers.
# ==============================================================================

import re


def fix_python3_imports(source_code):
    """
    WHAT: Fix common import and function changes between Python 3 versions

    WHY:
    - Models trained on older Python code generate deprecated imports
    - fractions.gcd moved to math.gcd in Python 3.5
    - collections.abc imports changed in Python 3.3+
    - Prevents runtime errors from deprecated functionality

    HOW:
    1. Apply regex replacements for known issues
    2. Add missing imports (math.gcd, sys.set_int_max_str_digits)
    3. Remove trailing backslashes
    4. Return patched code

    PROXIMAL CONTEXT:
    - INPUT: Python source code string
    - OUTPUT: Patched Python source code

    DISTAL CONTEXT:
    - ORIGIN: Called by patch_code() for Python submissions
    - DESTINATION: Executed by Piston or E2B sandbox

    EXAMPLE 1: Fix fractions.gcd
    ```python
    code = '''
    from fractions import gcd
    a, b = 12, 8
    print(gcd(a, b))
    '''
    patched = fix_python3_imports(code)
    # Result: gcd import removed, "from math import gcd" added
    ```

    EXAMPLE 2: Fix collections import
    ```python
    code = "from collections import Mapping"
    patched = fix_python3_imports(code)
    # Result: "from collections.abc import Mapping"
    ```

    COMMON FIXES:
    - fractions.gcd → math.gcd
    - collections.Mapping → collections.abc.Mapping
    - os.getlogin() → False (prevents debug checks)
    - Adds sys.set_int_max_str_digits(0) for large integers

    Args:
        source_code (str): The Python source code to update

    Returns:
        str: The updated source code
    """
    # DEFINE REPLACEMENT PATTERNS
    # Each tuple: (regex_pattern, replacement)
    replacements = [
        # Fix collections.abc imports (changed in Python 3.3+)
        # Before: from collections import Mapping
        # After: from collections.abc import Mapping
        (
            r"from collections import (Mapping|Sequence|Set|Container|MutableMapping|MutableSet|MutableSequence)",
            r"from collections.abc import \1",
        ),

        # Fix imp module deprecation (deprecated in 3.4)
        (r"import imp", r"import importlib"),

        # Fix asyncio.async() to asyncio.ensure_future() (renamed in 3.4.4)
        (r"asyncio\.async\(", r"asyncio.ensure_future("),

        # Fix inspect.getargspec to inspect.getfullargspec (deprecated in 3.5)
        (r"inspect\.getargspec", r"inspect.getfullargspec"),

        # Fix array.array 'c' type code to 'b' (removed in 3.9)
        (r"array\.array\('c'", r"array.array('b'"),

        # Fix backslash line continuation with multiple newlines
        (r"\\(\r\n|\r|\n)+", "\\\n"),

        # FIX os.getlogin() CHECKS
        # Some solutions use getlogin() to check if debugging or submitting
        # Replace with False to prevent debug code execution
        (r"(?:os\s*\.\s*)?getlogin\s*\(\s*\)", "False"),

        # FIX fractions.gcd → math.gcd (moved in Python 3.5)
        # Pattern 1: Direct usage fractions.gcd → math.gcd
        (r"\bfractions\.gcd\b", r"math.gcd"),

        # Pattern 2: Fix 'from fractions import gcd, X' → 'from fractions import X'
        (r"(from\s+fractions\s+import\s+(?:\([^)]*)?)\bgcd\s*,\s*", r"\1"),

        # Pattern 3: Fix 'from fractions import X, gcd' → 'from fractions import X'
        (r"(from\s+fractions\s+import\s+.*?\S)\s*,\s*\bgcd(\s*\)?\s*(?:#.*)?)", r"\1\2"),

        # Pattern 4: Fix standalone 'from fractions import gcd' → remove (will add math.gcd)
        (r"from\s+fractions\s+import\s+\(?\s*gcd\s*\)?", r""),
    ]

    # FIND LAST IMPORT LINE
    # WHY: Need to insert new imports after existing imports
    lines = source_code.splitlines()
    last_import = max(
        [
            i
            for i, line in enumerate(lines)
            if line.strip().startswith("import") or (line.strip().startswith("from") and "import" in line)
        ],
        default=0,
    )
    import_section = "\n".join(lines[: last_import + 1])
    main_source = "\n".join(lines[last_import:])

    # ADD MISSING math IMPORT IF NEEDED
    # Case 1: fractions.gcd used but no math import
    if "fractions.gcd" in source_code and "import math" not in source_code:
        import_section += "\nimport math"
    # Case 2: gcd used but no explicit math import
    elif "gcd" in source_code and "from math import gcd" not in source_code:
        import_section += "\nfrom math import gcd"

    # ADD sys.set_int_max_str_digits(0)
    # WHY: Some competitive programming problems require large integer operations
    # This removes the default limit on integer string conversion
    if "set_int_max_str_digits" not in source_code:
        import_section += "\nimport sys\nsys.set_int_max_str_digits(0)"

    # REBUILD SOURCE CODE
    source_code = import_section + "\n" + main_source

    # APPLY ALL REPLACEMENTS
    for pattern, replacement in replacements:
        source_code = re.sub(pattern, replacement, source_code)

    # REMOVE TRAILING BACKSLASHES
    # WHY: Prevents syntax errors from incomplete line continuations
    source_code = source_code.rstrip("\\")

    return source_code


def fix_cpp_includes(source_code):
    """
    WHAT: Add standard C++ headers and namespace declarations

    WHY:
    - Models often forget #include directives
    - bits/stdc++.h includes most standard library headers
    - using namespace std; simplifies code (models forget std:: prefix)

    HOW:
    1. Add #include <bits/stdc++.h> header
    2. Add using namespace std; if not present and no std:: usage
    3. Prepend to source code

    EXAMPLE:
    ```cpp
    // Original generated code:
    int main() {
        vector<int> v;
        cout << v.size();
    }

    // After patching:
    #include <bits/stdc++.h>

    using namespace std;

    int main() {
        vector<int> v;
        cout << v.size();
    }
    ```

    RETURNS: Patched C++ code with headers
    """
    # ADD bits/stdc++.h HEADER
    # WHY: Contains most standard library headers
    # NOTE: Non-standard but widely used in competitive programming
    code_header = "#include <bits/stdc++.h>\n"

    # ADD using namespace std IF NEEDED
    # Skip if already present or if std:: is used explicitly
    if "using namespace std;" not in source_code and "std::" not in source_code:
        code_header += "\nusing namespace std;\n\n"

    return code_header + source_code


def is_patchable(lang):
    """
    WHAT: Check if language supports automatic patching

    WHY:
    - Only Python and C++ have patching logic implemented
    - Prevents errors when trying to patch unsupported languages

    SUPPORTED:
    - Python: python, python3, Python 3, PyPy 3, PyPy 3-64
    - C++: cpp, C++, C++11, C++14, C++17, C++20

    RETURNS: True if language can be patched
    """
    return lang in ("python", "python3", "Python 3", "PyPy 3", "PyPy 3-64", "cpp") or "C++" in lang


def patch_code(text, lang):
    """
    WHAT: Main entry point for patching code based on language

    WHY:
    - Centralized patching logic
    - Language-aware patching
    - Safe no-op for unsupported languages

    HOW:
    1. Return None if text is empty
    2. Check language and apply appropriate patcher
    3. Return patched code

    PROXIMAL CONTEXT:
    - INPUT: code text and language identifier
    - OUTPUT: Patched code or original if not patchable

    DISTAL CONTEXT:
    - ORIGIN: Called by reward functions before execution
    - DESTINATION: Patched code sent to sandbox for execution

    EXAMPLE 1: Python patching
    ```python
    code = "from fractions import gcd\nprint(gcd(12, 8))"
    patched = patch_code(code, "Python 3")
    # Result: gcd import fixed, math.gcd added
    ```

    EXAMPLE 2: C++ patching
    ```python
    code = "int main() { cout << 'hello'; }"
    patched = patch_code(code, "C++17")
    # Result: #include <bits/stdc++.h> and using namespace std; added
    ```

    EXAMPLE 3: Unsupported language
    ```python
    code = "print('hello')"
    patched = patch_code(code, "ruby")
    # Result: Original code unchanged
    ```

    PARAMETERS:
    - text: Source code string
    - lang: Language identifier (from problem metadata)

    RETURNS: Patched code or original if not patchable
    """
    # HANDLE EMPTY CODE
    if not text:
        return text

    # APPLY LANGUAGE-SPECIFIC PATCHING
    if lang in ("python", "python3", "Python 3", "PyPy 3", "PyPy 3-64"):
        return fix_python3_imports(text)
    elif "cpp" in lang or "C++" in lang:
        return fix_cpp_includes(text)

    # NO PATCHING FOR OTHER LANGUAGES
    return text


# TEST CASES
# Used for validation and demonstration
tests = [
    # TEST 1: fractions.gcd usage
    """read = lambda: map(int, input().split())
n, m, z = read()
from fractions import gcd
ans = z // (n * m // gcd(n, m))
print(ans)""",

    # TEST 2: Multiple imports from fractions
    """from fractions import Fraction,gcd

a,b,c,d = [int(x) for x in input().split()]

if a*d > b*c:
    num = a*d-b*c
    denom = a*d
else:
    num = b*c-a*d
    denom = b*c
div = gcd(num,denom)
print('%d/%d'%(num//div,denom//div))""",
]

if __name__ == "__main__":
    """
    DEMO: Show before/after patching for test cases

    Run with: python code_patcher.py
    """
    for test in tests:
        print("ORIGINAL:", test, sep="\n\n")
        print("PATCHED:", patch_code(test, "Python 3"), sep="\n\n")
        print("=" * 50)

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================
# 1. **Why Patching Needed**:
#    - Models trained on older code generate deprecated syntax
#    - Prevents runtime errors in sandboxes
#    - Makes generated code executable
#
# 2. **Python Fixes**:
#    - fractions.gcd → math.gcd (most common)
#    - collections → collections.abc
#    - os.getlogin() → False (prevents debug code)
#    - Adds sys.set_int_max_str_digits(0) for large integers
#
# 3. **C++ Fixes**:
#    - Adds #include <bits/stdc++.h>
#    - Adds using namespace std; if appropriate
#    - Simplifies competitive programming C++
#
# 4. **Design Patterns**:
#    - Regex-based replacements for known issues
#    - Import section detection and augmentation
#    - Language-aware dispatching
#
# 5. **Usage in Pipeline**:
#    ```python
#    # Before execution
#    if is_patchable(language):
#        code = patch_code(generated_code, language)
#    result = sandbox.run_code(code)
#    ```
#
# 6. **Limitations**:
#    - Only handles known patterns
#    - May not fix all generated code issues
#    - Regex can be fragile for complex cases
#
# 7. **Future Improvements**:
#    - AST-based patching for more robust fixes
#    - Support for more languages
#    - Configurable patching rules
# ==============================================================================
