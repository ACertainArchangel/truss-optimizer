"""Master test suite runner for the Bridge v2 project.

Automatically discovers and runs all test modules matching 'test_*.py'
in this directory. Exit code reflects overall pass/fail status so this
script can be used in CI pipelines.

Usage::

    python run_all_tests.py
"""

import os
import sys
import unittest

# Ensure the parent package (Bridge v2) is importable when running from here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # Load all test files in this directory.
    loader = unittest.TestLoader()

    start_dir = os.path.dirname(os.path.abspath(__file__))

    suite = loader.discover(start_dir, pattern="test_*.py")

    # verbosity=2 prints each test name and its pass/fail result.
    runner = unittest.TextTestRunner(verbosity=2)

    result = runner.run(suite)

    # Return a non-zero exit code when any test fails so CI can detect failures.
    sys.exit(0 if result.wasSuccessful() else 1)
