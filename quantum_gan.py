import numpy as np
import tensorflow as tf
import pennylane as qml
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Define constants
n_qubits = 4  # Number of qubits
n_layers = 2  # Number of layers

# Define the quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit for the generator
@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    """
    Quantum circuit representing the generator component of the GAN.
    
    Args:
        inputs: Input values to the quantum circuit
        weights: Trainable weights for the quantum circuit
    
    Returns:
        Expectation values of PauliZ measurements on each qubit
    """
    # Encode the inputs
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
    
    # Apply variational quantum circuit
    for l in range(n_layers):
        # Single qubit rotations
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        
        # Entangling gates
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])  # Connect the last qubit to the first
    
    # Measure the state of each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define the Keras layer for the quantum circuit
class QuantumLayer(Layer):
    def __init__(self, n_qubits, n_layers):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Initialize weights with small random values
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}  # 3 rotation gates per qubit
        self.weights = tf.Variable(
            0.01 * np.random.randn(n_layers, n_qubits, 3),
            dtype=tf.float32,
            trainable=True,
            name="quantum_weights"
        )
    
    def call(self, inputs):
        return tf.stack([quantum_circuit(inputs[i], self.weights) for i in range(inputs.shape[0])])

# Define the Quantum Generator
class QuantumGenerator(tf.keras.Model):
    def __init__(self, n_qubits, n_layers, latent_dim):
        super(QuantumGenerator, self).__init__()
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        
        # Input processing layer to map latent space to qubit inputs
        self.input_layer = Dense(n_qubits, activation='tanh')
        
        # Quantum layer
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)
        
        # Output processing layer
        self.output_layer = Dense(latent_dim, activation='tanh')
    
    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.quantum_layer(x)
        return self.output_layer(x)

# Define the Classical Discriminator
def build_discriminator(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Define the GAN
class QuantumGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(QuantumGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(QuantumGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        latent_dim = self.generator.latent_dim
        
        # Generate random noise for the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        
        # Train the discriminator
        with tf.GradientTape() as tape:
            # Generate fake samples
            generated_data = self.generator(random_latent_vectors, training=True)
            
            # Get discriminator predictions for real and fake data
            real_predictions = self.discriminator(real_data, training=True)
            fake_predictions = self.discriminator(generated_data, training=True)
            
            # Calculate discriminator loss
            d_loss_real = self.loss_fn(tf.ones_like(real_predictions), real_predictions)
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_predictions), fake_predictions)
            d_loss = d_loss_real + d_loss_fake
            
        # Compute and apply gradients for discriminator
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train the generator
        with tf.GradientTape() as tape:
            # Generate fake samples
            random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
            generated_data = self.generator(random_latent_vectors, training=True)
            
            # Get discriminator predictions for fake data
            fake_predictions = self.discriminator(generated_data, training=True)
            
            # Calculate generator loss (we want the discriminator to think these are real)
            g_loss = self.loss_fn(tf.ones_like(fake_predictions), fake_predictions)
            
        # Compute and apply gradients for generator
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return {"d_loss": d_loss, "g_loss": g_loss}
