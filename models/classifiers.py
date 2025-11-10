import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


class BaseClassifier:
    """Base class for all supervised classifiers."""

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save(self, filepath):
        joblib.dump(self, filepath)
        print(f"✅ Classifier saved to '{filepath}'")

    @staticmethod
    def load(filepath):
        classifier = joblib.load(filepath)
        print(f"✅ Classifier loaded from '{filepath}'")
        return classifier


class RandomForestClassifierWrapper(BaseClassifier):
    """A wrapper for scikit-learn's RandomForestClassifier."""

    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(random_state=random_state)
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        # Reshape window data into 2D for scikit-learn
        num_samples, window_size, num_features = X.shape
        X_reshaped = X.reshape(num_samples, window_size * num_features)

        # Convert string labels to integers (e.g., 'normal' -> 0, 'syn_flood' -> 1)
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X_reshaped, y_encoded)

    def predict(self, X):
        num_samples, window_size, num_features = X.shape
        X_reshaped = X.reshape(num_samples, window_size * num_features)

        # Get integer predictions from the model
        predictions_encoded = self.model.predict(X_reshaped)

        # Convert integer predictions back to original string labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions
