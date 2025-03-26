"""
Configuration file for pytest
"""
import pytest
import tensorflow as tf
import numpy as np

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility in tests"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
@pytest.fixture
def mock_quantum_circuit_output():
    """Return a mock output for the quantum circuit"""
    return [0.5, 0.5, 0.5, 0.5]
