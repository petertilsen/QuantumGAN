#!/usr/bin/env python3
"""
Script to run tests with coverage reporting
"""
import os
import sys
import pytest

if __name__ == "__main__":
    try:
        # Add current directory to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # First check if basic imports work
        try:
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend
            print("Basic imports successful")
        except ImportError as e:
            print(f"Error importing basic libraries: {e}")
            sys.exit(1)
            
        # Run pytest with coverage
        result = pytest.main([
            "--cov=quantum_gan",
            "--cov=train",
            "--cov-report=term",
            "--cov-report=html",
            "tests/"
        ])
        sys.exit(result)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)
