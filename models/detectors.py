import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib


class BaseDetector:
    """The base class 'contract' for all our anomaly detection models."""

    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save(self, filepath):
        """Saves the trained model to a file."""
        joblib.dump(self, filepath)
        print(f"✅ Detector saved to '{filepath}'")

    @staticmethod
    def load(filepath):
        """Loads a trained model from a file."""
        detector = joblib.load(filepath)
        print(f"✅ Detector loaded from '{filepath}'")
        return detector


class IsolationForestDetector(BaseDetector):
    """Anomaly detection using the Isolation Forest algorithm."""

    def __init__(self, contamination=0.01, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, X):
        num_samples, window_size, num_features = X.shape
        X_reshaped = X.reshape(num_samples, window_size * num_features)
        self.model.fit(X_reshaped)

    def predict(self, X):
        num_samples, window_size, num_features = X.shape
        X_reshaped = X.reshape(num_samples, window_size * num_features)
        predictions = self.model.predict(X_reshaped)
        return np.where(predictions == -1)[0]


class OneClassSVMDetector(BaseDetector):
    """Anomaly detection using One-Class SVM."""

    def __init__(self, nu=0.01, kernel="rbf", gamma="auto"):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X):
        num_samples, window_size, num_features = X.shape
        X_reshaped = X.reshape(num_samples, window_size * num_features)
        self.model.fit(X_reshaped)

    def predict(self, X):
        num_samples, window_size, num_features = X.shape
        X_reshaped = X.reshape(num_samples, window_size * num_features)
        predictions = self.model.predict(X_reshaped)
        return np.where(predictions == -1)[0]


# --- This is the class that was missing ---
class EnsembleDetector(BaseDetector):
    """
    Manages a panel of multiple detector models and makes predictions based on a majority vote.
    """

    def __init__(self, models, voting_threshold=None):
        self.models = models
        if voting_threshold is None:
            self.voting_threshold = len(self.models) // 2 + 1
        else:
            self.voting_threshold = voting_threshold
        print(f"Ensemble created with {len(self.models)} models. Anomaly threshold: {self.voting_threshold} votes.")

    def fit(self, X):
        """Trains every model in the panel."""
        print(f"--- Training Ensemble ---")
        for i, model in enumerate(self.models):
            print(f"Training model {i + 1}/{len(self.models)} ({model.__class__.__name__})...")
            model.fit(X)
        print("--- Ensemble training complete. ---")

    def predict(self, X):
        """
        Gets a vote from every model and returns anomalies based on the threshold.
        """
        num_windows = X.shape[0]
        votes = np.zeros(num_windows)
        for model in self.models:
            anomalous_indices = model.predict(X)
            votes[anomalous_indices] += 1
        final_anomalies = np.where(votes >= self.voting_threshold)[0]
        return final_anomalies


