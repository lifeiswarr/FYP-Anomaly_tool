import shap
import joblib
import pandas as pd


class ModelExplainer:
    """
    A wrapper for the SHAP (SHapley Additive exPlanations) library to explain
    individual predictions of a trained classifier.
    """

    def __init__(self, model, background_data, feature_names=None):
        """
        Initializes the explainer.

        Args:
            model: The trained classifier model (e.g., from scikit-learn).
            background_data (pd.DataFrame or np.array): A sample of data that represents the
                'normal' or typical input to the model. SHAP uses this to calculate
                expected values.
            feature_names (list of str, optional): The names of the features, used for
                clearer explanations.
        """
        if not hasattr(model, 'predict_proba'):
            raise TypeError("Model must have a 'predict_proba' method for this explainer.")

        self.model = model
        self.feature_names = feature_names
        # The KernelExplainer is a good choice for model-agnostic explanations.
        self.explainer = shap.KernelExplainer(self.model.predict_proba, background_data)

    def explain_instance(self, instance):
        """
        Generates SHAP values for a single prediction instance.

        Args:
            instance (np.array): A single data instance (e.g., a window) to be explained.
                                 Should be a 1D or 2D array.

        Returns:
            dict: A dictionary mapping feature names to their SHAP values for the
                  prediction of the most likely class.
        """
        # Reshape instance if it's flat
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)

        # Get the class predicted with the highest probability
        predicted_class_index = self.model.predict(instance)[0]
        if isinstance(predicted_class_index, str):
            # Find the numeric index for the string label
            predicted_class_index = list(self.model.classes_).index(predicted_class_index)


        # Calculate SHAP values for the instance
        shap_values = self.explainer.shap_values(instance)

        # We want the SHAP values for the predicted class
        instance_shap_values = shap_values[predicted_class_index][0]

        return dict(zip(self.feature_names, instance_shap_values))

    def save(self, filepath):
        """Saves the explainer object to a file."""
        print(f"💾 Saving explainer to {filepath}...")
        joblib.dump(self, filepath)
        print("✅ Explainer saved.")

    @staticmethod
    def load(filepath):
        """Loads an explainer object from a file."""
        print(f"🔄 Loading explainer from {filepath}...")
        return joblib.load(filepath)

