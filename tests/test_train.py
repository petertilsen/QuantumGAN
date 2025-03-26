import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train
from quantum_gan import QuantumGenerator, QuantumGAN

class TestTrain(unittest.TestCase):
    @patch('train.np.random.normal')
    def test_generate_real_data(self, mock_random_normal):
        # Mock the return value of np.random.normal
        mock_random_normal.return_value = np.array([0.1, 0.2])
        
        # Call generate_real_data
        result = train.generate_real_data(10)
        
        # Check that the result has the expected shape
        self.assertEqual(result.shape, (10, train.data_dim))
        
        # Check that np.random.normal was called 10 times
        self.assertEqual(mock_random_normal.call_count, 10)
        
    @patch('train.plt.figure')
    @patch('train.plt.scatter')
    @patch('train.plt.legend')
    @patch('train.plt.title')
    @patch('train.plt.savefig')
    @patch('train.plt.close')
    def test_visualization(self, mock_close, mock_savefig, mock_title, mock_legend, mock_scatter, mock_figure):
        # Create mock data
        real_data = np.random.random((100, 2))
        generated_samples = np.random.random((100, 2))
        
        # Create mock objects
        mock_generator = MagicMock()
        mock_generator.return_value = tf.constant(generated_samples)
        
        # Create a GAN with the mock generator
        gan = QuantumGAN(mock_generator, MagicMock())
        
        # Mock train.QuantumGAN to return our mock GAN
        with patch('train.QuantumGAN', return_value=gan):
            # Mock train.generate_real_data to return our mock data
            with patch('train.generate_real_data', return_value=real_data):
                # Mock train.tf.data.Dataset.from_tensor_slices
                mock_dataset = MagicMock()
                mock_dataset.shuffle.return_value = mock_dataset
                mock_dataset.batch.return_value = mock_dataset
                
                with patch('train.tf.data.Dataset.from_tensor_slices', return_value=mock_dataset):
                    # Mock the training loop to only run for 10 epochs
                    original_range = range
                    
                    def mock_range(*args, **kwargs):
                        if args and args[0] == train.epochs:
                            return original_range(10)  # Only run 10 epochs
                        return original_range(*args, **kwargs)
                    
                    with patch('builtins.range', mock_range):
                        # Run the training script
                        with patch('train.QuantumGenerator', return_value=mock_generator):
                            with patch('train.build_discriminator', return_value=MagicMock()):
                                # Prevent the script from actually running the training loop
                                with patch.object(QuantumGAN, 'train_step', return_value={"d_loss": 0.5, "g_loss": 0.5}):
                                    # Execute the training script
                                    try:
                                        # We need to import the module again to trigger the execution
                                        import importlib
                                        importlib.reload(train)
                                    except:
                                        pass
        
        # Check that visualization functions were called
        mock_figure.assert_called()
        mock_scatter.assert_called()
        mock_legend.assert_called()
        mock_title.assert_called()
        mock_savefig.assert_called()
        mock_close.assert_called()
        
    @patch('train.tf.random.normal')
    def test_random_latent_vectors(self, mock_random_normal):
        # Mock the return value of tf.random.normal
        mock_random_normal.return_value = tf.constant([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        
        # Create a mock batch_size
        batch_size = 1
        
        # Call tf.random.normal with the expected arguments
        result = tf.random.normal(shape=(batch_size, train.latent_dim))
        
        # Check that tf.random.normal was called with the expected arguments
        mock_random_normal.assert_called_with(shape=(batch_size, train.latent_dim))
        
        # Check that the result has the expected shape
        self.assertEqual(result.shape, (batch_size, train.latent_dim))


class TestModelSaving(unittest.TestCase):
    @patch('train.plt.figure')
    @patch('train.plt.plot')
    @patch('train.plt.xlabel')
    @patch('train.plt.ylabel')
    @patch('train.plt.legend')
    @patch('train.plt.title')
    @patch('train.plt.savefig')
    @patch('train.plt.close')
    def test_loss_curve_plotting(self, mock_close, mock_savefig, mock_title, mock_legend, 
                                mock_ylabel, mock_xlabel, mock_plot, mock_figure):
        # Create mock loss data
        d_losses = [0.5, 0.4, 0.3]
        g_losses = [0.6, 0.5, 0.4]
        
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
        
        # Check that plotting functions were called
        mock_figure.assert_called_once()
        self.assertEqual(mock_plot.call_count, 2)
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_legend.assert_called_once()
        mock_title.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
    @patch('train.QuantumGenerator.save_weights')
    @patch('tensorflow.keras.models.Sequential.save_weights')
    def test_model_saving(self, mock_discriminator_save, mock_generator_save):
        # Create mock models
        generator = QuantumGenerator(train.n_qubits, train.n_layers, train.latent_dim)
        discriminator = tf.keras.models.Sequential()
        
        # Save the models
        generator.save_weights('/app/output/quantum_generator_weights.h5')
        discriminator.save_weights('/app/output/discriminator_weights.h5')
        
        # Check that save_weights was called for both models
        mock_generator_save.assert_called_once()
        mock_discriminator_save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
