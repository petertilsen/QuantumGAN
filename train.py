import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from quantum_gan import QuantumGenerator, build_discriminator, QuantumGAN

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define parameters
n_qubits = 4
n_layers = 2
latent_dim = 8
data_dim = latent_dim  # Dimension of the real data
batch_size = 32
epochs = 100

# Generate some synthetic data for training (e.g., samples from a specific distribution)
# For example, let's generate data from a mixture of Gaussians
def generate_real_data(n_samples):
    # Generate data from a mixture of 4 Gaussians
    centers = [
        (0.2, 0.2),
        (0.2, -0.2),
        (-0.2, 0.2),
        (-0.2, -0.2)
    ]
    
    data = []
    for _ in range(n_samples):
        # Choose one of the centers randomly
        center = centers[np.random.choice(len(centers))]
        # Generate a point around that center
        point = np.random.normal(loc=center, scale=0.05, size=2)
        # Pad with zeros to match data_dim if needed
        if data_dim > 2:
            point = np.pad(point, (0, data_dim - 2), 'constant')
        data.append(point)
    
    return np.array(data, dtype=np.float32)

# Create the dataset
real_data = generate_real_data(1000)
dataset = tf.data.Dataset.from_tensor_slices(real_data).shuffle(1000).batch(batch_size)

# Build the models
generator = QuantumGenerator(n_qubits, n_layers, latent_dim)
discriminator = build_discriminator(data_dim)

# Create the GAN
gan = QuantumGAN(generator, discriminator)

# Compile the GAN
gan.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

# Lists to store loss values
d_losses = []
g_losses = []

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    epoch_d_losses = []
    epoch_g_losses = []
    
    for batch in dataset:
        # Train on batch
        losses = gan.train_step(batch)
        
        # Store losses
        epoch_d_losses.append(losses["d_loss"].numpy())
        epoch_g_losses.append(losses["g_loss"].numpy())
    
    # Calculate average losses for the epoch
    avg_d_loss = np.mean(epoch_d_losses)
    avg_g_loss = np.mean(epoch_g_losses)
    
    d_losses.append(avg_d_loss)
    g_losses.append(avg_g_loss)
    
    print(f"  Discriminator Loss: {avg_d_loss:.4f}")
    print(f"  Generator Loss: {avg_g_loss:.4f}")
    
    # Generate and visualize samples every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Generate samples
        noise = tf.random.normal(shape=(100, latent_dim))
        generated_samples = generator(noise).numpy()
        
        # Plot the first 2 dimensions of the generated samples
        plt.figure(figsize=(10, 8))
        plt.scatter(real_data[:, 0], real_data[:, 1], c='blue', alpha=0.5, label='Real Data')
        plt.scatter(generated_samples[:, 0], generated_samples[:, 1], c='red', alpha=0.5, label='Generated Data')
        plt.legend()
        plt.title(f"Generated vs Real Data - Epoch {epoch+1}")
        plt.savefig(f"/app/output/samples_epoch_{epoch+1}.png")
        plt.close()

# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.savefig('/app/output/loss_curves.png')
plt.close()

# Save the trained models
generator.save_weights('/app/output/quantum_generator_weights.h5')
discriminator.save_weights('/app/output/discriminator_weights.h5')
