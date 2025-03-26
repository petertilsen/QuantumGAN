import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_gan import (
    quantum_circuit, 
    QuantumLayer, 
    QuantumGenerator, 
    build_discriminator, 
    QuantumGAN
)

class TestQuantumCircuit(unittest.TestCase):
    @patch('quantum_gan.qml.expval')
    @patch('quantum_gan.qml.PauliZ')
    @patch('quantum_gan.qml.CNOT')
    @patch('quantum_gan.qml.RZ')
    @patch('quantum_gan.qml.RY')
    @patch('quantum_gan.qml.RX')
    def test_quantum_circuit(self, mock_rx, mock_ry, mock_rz, mock_cnot, mock_pauliz, mock_expval):
        # Mock the return value of expval
        mock_expval.return_value = 0.5
        
        # Create test inputs and weights
        inputs = np.array([0.1, 0.2, 0.3, 0.4])
        weights = np.random.random((2, 4, 3))  # 2 layers, 4 qubits, 3 rotation gates
        
        # Call the quantum circuit
        result = quantum_circuit(inputs, weights)
        
        # Check that the result has the expected shape
        self.assertEqual(len(result), 4)  # 4 qubits
        
        # Check that the quantum operations were called
        self.assertEqual(mock_rx.call_count, 4 + 8)  # 4 input encodings + 8 rotations
        self.assertEqual(mock_ry.call_count, 8)  # 8 rotations
        self.assertEqual(mock_rz.call_count, 8)  # 8 rotations
        self.assertEqual(mock_cnot.call_count, 8)  # 8 entangling gates
        self.assertEqual(mock_pauliz.call_count, 4)  # 4 measurements
        self.assertEqual(mock_expval.call_count, 4)  # 4 expectation values


class TestQuantumLayer(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 4
        self.n_layers = 2
        
    @patch('quantum_gan.quantum_circuit')
    def test_call(self, mock_quantum_circuit):
        # Mock the return value of quantum_circuit
        mock_quantum_circuit.return_value = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Create a quantum layer
        layer = QuantumLayer(self.n_qubits, self.n_layers)
        
        # Create test inputs
        inputs = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        
        # Call the layer
        result = layer(inputs)
        
        # Check that the result has the expected shape
        self.assertEqual(result.shape, (2, 4))  # 2 samples, 4 qubits
        
        # Check that quantum_circuit was called twice (once for each input)
        self.assertEqual(mock_quantum_circuit.call_count, 2)
        
    def test_weights_initialization(self):
        # Create a quantum layer
        layer = QuantumLayer(self.n_qubits, self.n_layers)
        
        # Check that the weights have the expected shape
        self.assertEqual(layer.weights.shape, (self.n_layers, self.n_qubits, 3))


class TestQuantumGenerator(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 4
        self.n_layers = 2
        self.latent_dim = 8
        
    @patch.object(QuantumLayer, 'call')
    def test_call(self, mock_quantum_layer_call):
        # Mock the return value of QuantumLayer.call
        mock_quantum_layer_call.return_value = tf.constant([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
        
        # Create a quantum generator
        generator = QuantumGenerator(self.n_qubits, self.n_layers, self.latent_dim)
        
        # Create test inputs
        inputs = tf.random.normal((2, self.latent_dim))
        
        # Call the generator
        result = generator(inputs)
        
        # Check that the result has the expected shape
        self.assertEqual(result.shape, (2, self.latent_dim))  # 2 samples, latent_dim outputs
        
        # Check that QuantumLayer.call was called once
        mock_quantum_layer_call.assert_called_once()


class TestDiscriminator(unittest.TestCase):
    def test_build_discriminator(self):
        # Create a discriminator
        discriminator = build_discriminator(8)
        
        # Check that the discriminator has the expected architecture
        self.assertEqual(len(discriminator.layers), 4)
        self.assertEqual(discriminator.layers[0].units, 64)
        self.assertEqual(discriminator.layers[1].units, 32)
        self.assertEqual(discriminator.layers[2].units, 16)
        self.assertEqual(discriminator.layers[3].units, 1)
        
        # Check that the discriminator produces the expected output shape
        inputs = tf.random.normal((2, 8))
        result = discriminator(inputs)
        self.assertEqual(result.shape, (2, 1))


class TestQuantumGAN(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 4
        self.n_layers = 2
        self.latent_dim = 8
        
        # Create generator and discriminator
        self.generator = MagicMock()
        self.discriminator = MagicMock()
        
        # Create GAN
        self.gan = QuantumGAN(self.generator, self.discriminator)
        
        # Compile GAN
        self.gan.compile(
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss_fn=tf.keras.losses.BinaryCrossentropy()
        )
        
    def test_compile(self):
        # Check that the optimizers and loss function are set correctly
        self.assertIsInstance(self.gan.g_optimizer, tf.keras.optimizers.Adam)
        self.assertIsInstance(self.gan.d_optimizer, tf.keras.optimizers.Adam)
        self.assertIsInstance(self.gan.loss_fn, tf.keras.losses.BinaryCrossentropy)
        
    def test_train_step(self):
        # Mock generator and discriminator outputs
        self.generator.return_value = tf.random.normal((2, self.latent_dim))
        self.discriminator.return_value = tf.constant([[0.7], [0.3]])
        
        # Set latent_dim attribute on generator mock
        self.generator.latent_dim = self.latent_dim
        
        # Create test data
        real_data = tf.random.normal((2, self.latent_dim))
        
        # Call train_step
        result = self.gan.train_step(real_data)
        
        # Check that the result contains the expected keys
        self.assertIn('d_loss', result)
        self.assertIn('g_loss', result)
        
        # Check that the generator and discriminator were called
        self.generator.assert_called()
        self.discriminator.assert_called()


if __name__ == '__main__':
    unittest.main()
