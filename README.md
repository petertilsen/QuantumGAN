# QuantumGAN

A Quantum Generative Adversarial Network (QuantumGAN) implementation using PennyLane, TensorFlow, and TensorFlow Quantum.

## Project Overview

This project implements a hybrid quantum-classical GAN where:
- The generator uses a quantum circuit implemented with PennyLane
- The discriminator is a classical neural network implemented with TensorFlow
- The model is trained to generate data that mimics a mixture of Gaussian distributions

## Architecture

### Quantum Generator
The generator uses a variational quantum circuit with the following components:
- **Input Encoding**: Classical data is encoded into quantum states using RX rotations
- **Variational Layers**: Multiple layers of parameterized quantum gates (RX, RY, RZ rotations and CNOTs)
- **Measurement**: Expectation values of PauliZ operators are used as outputs
- **Post-processing**: A classical dense layer maps quantum measurements to the output space

### Classical Discriminator
The discriminator is a standard neural network with:
- Multiple dense layers with LeakyReLU activations
- Dropout for regularization
- Binary classification output with sigmoid activation

### Training Process
The GAN training follows these steps:
1. Generate real data from a mixture of Gaussian distributions
2. Train the discriminator to distinguish between real and generated data
3. Train the generator to produce data that fools the discriminator
4. Visualize the results periodically to monitor convergence

## Requirements

- Python 3.9+
- TensorFlow 2.12+
- TensorFlow Quantum 0.7+
- PennyLane 0.30+
- Qiskit 0.43+
- NumPy 1.23+
- Matplotlib 3.7+
- pytest 7.3.1+ (for testing)

## Installation

### Local Installation

1. Clone the repository:
   ```
   git clone https://github.com/petertilsen/QuantumGAN.git
   cd QuantumGAN
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Docker Deployment

This project can be easily deployed using Docker and docker-compose.

### Prerequisites

- Docker
- Docker Compose

### Running with Docker Compose

1. Clone the repository:
   ```
   git clone https://github.com/petertilsen/QuantumGAN.git
   cd QuantumGAN
   ```

2. Build and start the container:
   ```
   docker-compose up --build
   ```

3. The training will start automatically, and output files (images and model weights) will be saved to the `output` directory.

## Usage

### Training the Model

To train the model with default parameters:

```bash
python train.py
```

### Running Tests

To run the test suite:

```bash
python -m pytest tests/
```

### Configuration

You can modify the following files to customize the behavior:
- `quantum_gan.py`: Contains the model architecture
- `train.py`: Contains the training loop and hyperparameters
- `docker-compose.yml`: Contains Docker deployment settings

Key parameters in `train.py` that you can adjust:
- `n_qubits`: Number of qubits in the quantum circuit
- `n_layers`: Number of variational layers in the quantum circuit
- `latent_dim`: Dimension of the latent space for the generator
- `batch_size`: Batch size for training
- `epochs`: Number of training epochs

## Output

The training process generates:
- Images comparing real and generated data every 10 epochs
- A plot of the training losses
- Saved model weights for both the generator and discriminator

All outputs are saved to the `output` directory.

## Project Structure

```
QuantumGAN/
├── quantum_gan.py     # Core model implementation
├── train.py           # Training script
├── requirements.txt   # Project dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── entrypoint.sh      # Docker entrypoint script
├── tests/             # Test suite
│   ├── test_quantum_gan.py  # Tests for quantum_gan.py
│   └── test_train.py        # Tests for train.py
└── output/            # Generated outputs (images, weights)
```

## Notes

TensorFlow Quantum has limited platform support. The Docker container uses a Linux environment where TensorFlow Quantum is supported.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
