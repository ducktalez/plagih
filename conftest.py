# pytest configuration for plagih test suite
# This file ensures the project root is in Python path

import sys
from pathlib import Path

# Add project root to path for imports - MUST be at top level conftest.py
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# This makes pytest recognize plagih as a package
def pytest_configure(config):
    """Ensure project root is in path before any test collection."""
    pass
