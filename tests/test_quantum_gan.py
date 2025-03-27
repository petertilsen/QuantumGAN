import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import pennylane as qml

from quantum_gan import (
    quantum_circuit,
    QuantumLayer,
    QuantumGenerator,
    build_discriminator,
    QuantumGAN
)

class TestQuantumCircuit:
    def test_quantum_circuit_output_shape(self):
        """Test that the quantum circuit returns the expected output shape."""
        n_qubits = 4
        n_layers = 2
        
        # Create random inputs and weights
        inputs = tf.random.normal([n_qubits])
        weights = tf.random.normal([n_layers, n_qubits, 3])
        
        # Get output from quantum circuit
        output = quantum_circuit(inputs, weights)
        
        # Check output shape
        assert len(output) == n_qubits, f"Expected output length {n_qubits}, got {len(output)}"
        
    def test_quantum_circuit_output_type(self):
        """Test that the quantum circuit returns real values."""
        n_qubits = 4
        n_layers = 2
        
        # Create random inputs and weights
        inputs = tf.random.normal([n_qubits])
        weights = tf.random.normal([n_layers, n_qubits, 3])
        
        # Get output from quantum circuit
        output = quantum_circuit(inputs, weights)
        
        # Check that output values are within the expected range for PauliZ measurements
        for val in output:
            assert -1.0 <= val <= 1.0, f"Expected value between -1 and 1, got {val}"


class TestQuantumLayer:
    def test_quantum_layer_initialization(self):
        """Test that the QuantumLayer initializes correctly."""
        n_qubits = 4
        n_layers = 2
        
        layer = QuantumLayer(n_qubits, n_layers)
        
        # Check that weights were created with the right shape
        assert layer.quantum_weights.shape == (n_layers, n_qubits, 3)
        
    def test_quantum_layer_call(self):
        """Test that the QuantumLayer call method works correctly."""
        n_qubits = 4
        n_layers = 2
        batch_size = 3
        
        layer = QuantumLayer(n_qubits, n_layers)
        inputs = tf.random.normal([batch_size, n_qubits])
        
        # Call the layer
        outputs = layer(inputs)
        
        # Check output shape
        assert outputs.shape == (batch_size, n_qubits)
        
        # Check output type
        assert outputs.dtype == tf.float32, f"Expected dtype float32, got {outputs.dtype}"


class TestQuantumGenerator:
    def test_generator_initialization(self):
        """Test that the QuantumGenerator initializes correctly."""
        n_qubits = 4
        n_layers = 2
        latent_dim = 8
        
        generator = QuantumGenerator(n_qubits, n_layers, latent_dim)
        
        # Check that the generator has the expected layers
        assert hasattr(generator, 'input_layer')
        assert hasattr(generator, 'quantum_layer')
        assert hasattr(generator, 'output_layer')
        
    def test_generator_call(self):
        """Test that the QuantumGenerator call method works correctly."""
        n_qubits = 4
        n_layers = 2
        latent_dim = 8
        batch_size = 3
        
        generator = QuantumGenerator(n_qubits, n_layers, latent_dim)
        inputs = tf.random.normal([batch_size, latent_dim])
        
        # Call the generator
        outputs = generator(inputs)
        
        # Check output shape
        assert outputs.shape == (batch_size, latent_dim)
        
        # Check output values are in the expected range (tanh activation)
        assert tf.reduce_all(outputs >= -1.0)
        assert tf.reduce_all(outputs <= 1.0)


class TestDiscriminator:
    def test_discriminator_initialization(self):
        """Test that the discriminator initializes correctly."""
        input_dim = 8
        
        discriminator = build_discriminator(input_dim)
        
        # Check that the discriminator has the expected number of layers
        assert len(discriminator.layers) == 4
        
    def test_discriminator_call(self):
        """Test that the discriminator call method works correctly."""
        input_dim = 8
        batch_size = 3
        
        discriminator = build_discriminator(input_dim)
        inputs = tf.random.normal([batch_size, input_dim])
        
        # Call the discriminator
        outputs = discriminator(inputs)
        
        # Check output shape
        assert outputs.shape == (batch_size, 1)
        
        # Check output values are in the expected range (sigmoid activation)
        assert tf.reduce_all(outputs >= 0.0)
        assert tf.reduce_all(outputs <= 1.0)


class TestQuantumGAN:
    def test_gan_initialization(self):
        """Test that the QuantumGAN initializes correctly."""
        n_qubits = 4
        n_layers = 2
        latent_dim = 8
        
        generator = QuantumGenerator(n_qubits, n_layers, latent_dim)
        discriminator = build_discriminator(latent_dim)
        
        gan = QuantumGAN(generator, discriminator)
        
        # Check that the GAN has the expected components
        assert gan.generator is generator
        assert gan.discriminator is discriminator
        
    def test_gan_compile(self):
        """Test that the QuantumGAN compiles correctly."""
        n_qubits = 4
        n_layers = 2
        latent_dim = 8
        
        generator = QuantumGenerator(n_qubits, n_layers, latent_dim)
        discriminator = build_discriminator(latent_dim)
        
        gan = QuantumGAN(generator, discriminator)
        
        # Compile the GAN
        gan.compile(
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss_fn=tf.keras.losses.BinaryCrossentropy()
        )
        
        # Check that the optimizers and loss function were set correctly
        assert isinstance(gan.g_optimizer, tf.keras.optimizers.Adam)
        assert isinstance(gan.d_optimizer, tf.keras.optimizers.Adam)
        assert isinstance(gan.loss_fn, tf.keras.losses.BinaryCrossentropy)
