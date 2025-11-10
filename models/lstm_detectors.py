import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
from .detectors import BaseDetector  # Import the contract

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"LSTM Detector using device: {DEVICE}")


# --- Deep Learning Model Core (PyTorch nn.Module) ---

class LSTMAutoencoder(nn.Module):
    """
    This is the core PyTorch model architecture.
    It's a sequence-to-sequence (Seq2Seq) LSTM Autoencoder.
    """

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim  # Hidden size is often set equal to embedding dim

        # --- Encoder ---
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True  # Input shape: [batch_size, seq_len, features]
        )

        # --- Decoder ---
        self.decoder = nn.LSTM(
            input_size=self.hidden_dim,  # Input is the compressed context
            hidden_size=n_features,  # Output must match original n_features
            num_layers=1,
            batch_first=True
        )

        # --- Output Layer ---
        self.output_layer = nn.Linear(n_features, n_features)

    def forward(self, x):
        """Defines the forward pass of the model."""

        # --- Encoding Phase ---
        _, (hidden_state, cell_state) = self.encoder(x)
        context_vector = hidden_state[-1, :, :]

        # --- Decoding Phase ---
        decoder_input = context_vector.unsqueeze(1).repeat(1, self.seq_len, 1)

        # --- DEBUGGED in Notebook ---
        # We do not pass the encoder's hidden state to the decoder,
        # as their hidden_dim sizes are different. The information
        # is already in the `decoder_input` (the context vector).
        decoder_output, _ = self.decoder(decoder_input)

        reconstructed_x = self.output_layer(decoder_output)
        return reconstructed_x


# --- Deep Learning Detector Wrapper (Our Contract) ---

class LSTMAutoencoderDetector(BaseDetector):
    """
    This is the wrapper class that makes the PyTorch model
    conform to our BaseDetector contract.
    """

    def __init__(self, seq_len, n_features, embedding_dim=64, epochs=50, batch_size=64, learning_rate=1e-3, **kwargs):

        self.seq_len = seq_len
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.threshold = 0.0  # Will be set after training

        self.model = LSTMAutoencoder(seq_len, n_features, embedding_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.L1Loss(reduction='mean')
        self.history = {'train_loss': []}

    def fit(self, X):
        """Trains the PyTorch Autoencoder."""

        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        train_dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        print(f"Starting LSTM Autoencoder training for {self.epochs} epochs on {DEVICE}...")

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, in train_loader:
                self.optimizer.zero_grad()
                reconstructed_X = self.model(batch_X)
                loss = self.criterion(reconstructed_X, batch_X)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_X.size(0)

            avg_loss = total_loss / len(X)
            self.history['train_loss'].append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.6f}")

        print("--- LSTM Autoencoder training complete. ---")

        # Set the anomaly threshold based on training data
        self.threshold = self._determine_threshold(X_tensor)
        print(f"Calculated Anomaly Threshold (MAE): {self.threshold:.6f}")

    def _calculate_reconstruction_errors(self, X_tensor):
        """Helper function to calculate the per-sample reconstruction error (MAE)."""
        self.model.eval()
        with torch.no_grad():
            reconstructed_X = self.model(X_tensor)
            errors = torch.mean(torch.abs(reconstructed_X - X_tensor), dim=[1, 2])
        return errors.cpu().numpy()

    def _determine_threshold(self, X_tensor, percentile=95):
        """Calculates the anomaly threshold (e.g., 95th percentile of training errors)."""
        errors = self._calculate_reconstruction_errors(X_tensor)
        threshold = np.percentile(errors, percentile)
        return threshold

    def predict(self, X):
        """
        Predicts anomalies by calculating reconstruction error and comparing
        it to the learned threshold.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        errors = self._calculate_reconstruction_errors(X_tensor)
        anomaly_indices = np.where(errors > self.threshold)[0]
        return anomaly_indices

    def save(self, filepath):
        """Saves the PyTorch model state and key parameters."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'embedding_dim': self.embedding_dim,
            'threshold': self.threshold
        }
        torch.save(state, filepath)
        print(f"LSTM Detector saved to '{filepath}'")

    @staticmethod
    def load(filepath):
        """Loads a trained LSTM detector."""
        state = torch.load(filepath, map_location=DEVICE)

        detector = LSTMAutoencoderDetector(
            seq_len=state['seq_len'],
            n_features=state['n_features'],
            embedding_dim=state['embedding_dim']
        )
        detector.model.load_state_dict(state['model_state_dict'])
        detector.threshold = state['threshold']
        detector.model.eval()

        print(f"LSTM Detector loaded from '{filepath}' (Threshold: {detector.threshold:.6f})")
        return detector


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
from .detectors import BaseDetector  # Import the contract

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"LSTM Detector using device: {DEVICE}")


# --- Deep Learning Model Core (PyTorch nn.Module) ---

class LSTMAutoencoder(nn.Module):
    """
    This is the core PyTorch model architecture.
    It's a sequence-to-sequence (Seq2Seq) LSTM Autoencoder.
    """

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim  # Hidden size is often set equal to embedding dim

        # --- Encoder ---
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True  # Input shape: [batch_size, seq_len, features]
        )

        # --- Decoder ---
        self.decoder = nn.LSTM(
            input_size=self.hidden_dim,  # Input is the compressed context
            hidden_size=n_features,  # Output must match original n_features
            num_layers=1,
            batch_first=True
        )

        # --- Output Layer ---
        self.output_layer = nn.Linear(n_features, n_features)

    def forward(self, x):
        """Defines the forward pass of the model."""

        # --- Encoding Phase ---
        _, (hidden_state, cell_state) = self.encoder(x)
        context_vector = hidden_state[-1, :, :]

        # --- Decoding Phase ---
        decoder_input = context_vector.unsqueeze(1).repeat(1, self.seq_len, 1)

        # --- DEBUGGED in Notebook ---
        # We do not pass the encoder's hidden state to the decoder,
        # as their hidden_dim sizes are different. The information
        # is already in the `decoder_input` (the context vector).
        decoder_output, _ = self.decoder(decoder_input)

        reconstructed_x = self.output_layer(decoder_output)
        return reconstructed_x


# --- Deep Learning Detector Wrapper (Our Contract) ---

class LSTMAutoencoderDetector(BaseDetector):
    """
    This is the wrapper class that makes the PyTorch model
    conform to our BaseDetector contract.
    """

    def __init__(self, seq_len, n_features, embedding_dim=64, epochs=50, batch_size=64, learning_rate=1e-3, **kwargs):

        self.seq_len = seq_len
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.threshold = 0.0  # Will be set after training

        self.model = LSTMAutoencoder(seq_len, n_features, embedding_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.L1Loss(reduction='mean')
        self.history = {'train_loss': []}

    def fit(self, X):
        """Trains the PyTorch Autoencoder."""

        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        train_dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        print(f"Starting LSTM Autoencoder training for {self.epochs} epochs on {DEVICE}...")

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, in train_loader:
                self.optimizer.zero_grad()
                reconstructed_X = self.model(batch_X)
                loss = self.criterion(reconstructed_X, batch_X)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_X.size(0)

            avg_loss = total_loss / len(X)
            self.history['train_loss'].append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.6f}")

        print("--- LSTM Autoencoder training complete. ---")

        # Set the anomaly threshold based on training data
        self.threshold = self._determine_threshold(X_tensor)
        print(f"Calculated Anomaly Threshold (MAE): {self.threshold:.6f}")

    def _calculate_reconstruction_errors(self, X_tensor):
        """Helper function to calculate the per-sample reconstruction error (MAE)."""
        self.model.eval()
        with torch.no_grad():
            reconstructed_X = self.model(X_tensor)
            errors = torch.mean(torch.abs(reconstructed_X - X_tensor), dim=[1, 2])
        return errors.cpu().numpy()

    def _determine_threshold(self, X_tensor, percentile=95):
        """Calculates the anomaly threshold (e.g., 95th percentile of training errors)."""
        errors = self._calculate_reconstruction_errors(X_tensor)
        threshold = np.percentile(errors, percentile)
        return threshold

    def predict(self, X):
        """
        Predicts anomalies by calculating reconstruction error and comparing
        it to the learned threshold.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        errors = self._calculate_reconstruction_errors(X_tensor)
        anomaly_indices = np.where(errors > self.threshold)[0]
        return anomaly_indices

    def save(self, filepath):
        """Saves the PyTorch model state and key parameters."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'embedding_dim': self.embedding_dim,
            'threshold': self.threshold
        }
        torch.save(state, filepath)
        print(f"LSTM Detector saved to '{filepath}'")

    @staticmethod
    def load(filepath):
        """Loads a trained LSTM detector."""
        state = torch.load(filepath, map_location=DEVICE)

        detector = LSTMAutoencoderDetector(
            seq_len=state['seq_len'],
            n_features=state['n_features'],
            embedding_dim=state['embedding_dim']
        )
        detector.model.load_state_dict(state['model_state_dict'])
        detector.threshold = state['threshold']
        detector.model.eval()

        print(f"LSTM Detector loaded from '{filepath}' (Threshold: {detector.threshold:.6f})")
        return detector
