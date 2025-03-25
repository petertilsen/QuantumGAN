# QuantumGAN

A Quantum Generative Adversarial Network (QuantumGAN) implementation using PennyLane, TensorFlow, and TensorFlow Quantum.

## Project Overview

This project implements a hybrid quantum-classical GAN where:
- The generator uses a quantum circuit implemented with PennyLane
- The discriminator is a classical neural network implemented with TensorFlow
- The model is trained to generate data that mimics a mixture of Gaussian distributions

## Requirements

- Python 3.9+
- TensorFlow 2.12+
- TensorFlow Quantum 0.7+
- PennyLane 0.30+
- Qiskit 0.43+
- NumPy 1.23+
- Matplotlib 3.7+

## Docker Deployment

This project can be easily deployed using Docker and docker-compose.

### Prerequisites

- Docker
- Docker Compose

### Running with Docker Compose

1. Clone the repository:
   ```
   git clone <repository-url>
   cd QuantumGAN
   ```

2. Build and start the container:
   ```
   docker-compose up --build
   ```

3. The training will start automatically, and output files (images and model weights) will be saved to the `output` directory.

### Configuration

You can modify the following files to customize the behavior:
- `quantum_gan.py`: Contains the model architecture
- `train.py`: Contains the training loop and hyperparameters
- `docker-compose.yml`: Contains Docker deployment settings

## Output

The training process will generate:
- Images comparing real and generated data every 10 epochs
- A plot of the training losses
- Saved model weights for both the generator and discriminator

## Notes

TensorFlow Quantum has limited platform support. The Docker container uses a Linux environment where TensorFlow Quantum is supported.
