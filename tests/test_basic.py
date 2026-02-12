import pytest
import sys
import os

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        import src
        import models
        import utils
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")

def test_directory_structure():
    """Test that critical directories exist."""
    required_dirs = [
        'src',
        'models',
        'utils',
        'dashboard',
        'data'
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"Directory {d} is missing"
