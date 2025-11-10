import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted_columns = None
        self.numerical_features = None
        # --- THIS IS THE CRITICAL FIX ---
        # It MUST list all the text-based columns.
        self.categorical_features = ['protocol_type', 'service', 'flag']

    def _apply_specific_transformations(self, df):
        df_copy = df.copy()
        if 'src_port' in df_copy.columns:
            df_copy['src_port'] = df_copy['src_port'].fillna(0)
        if 'dst_port' in df_copy.columns:
            df_copy['dst_port'] = df_copy['dst_port'].fillna(0)
        port_cols = [col for col in ['src_port', 'dst_port'] if col in df_copy.columns]
        if port_cols:
            df_copy[port_cols] = df_copy[port_cols].astype(int)
        return df_copy

    def fit(self, df):
        print("Fitting the preprocessor...")
        df_clean = self._apply_specific_transformations(df)
        cats_to_encode = [col for col in self.categorical_features if col in df_clean.columns]
        df_processed = pd.get_dummies(df_clean, columns=cats_to_encode, dummy_na=False)
        self.fitted_columns = df_processed.columns.tolist()
        self.numerical_features = df_processed.select_dtypes(include=np.number).columns.tolist()
        if self.numerical_features:
            self.scaler.fit(df_processed[self.numerical_features])
        print("Preprocessor fitted successfully.")

    def transform(self, df):
        df_clean = self._apply_specific_transformations(df)
        cats_to_encode = [col for col in self.categorical_features if col in df_clean.columns]
        df_processed = pd.get_dummies(df_clean, columns=cats_to_encode, dummy_na=False)
        current_columns = df_processed.columns.tolist()
        missing_cols = set(self.fitted_columns) - set(current_columns)
        for c in missing_cols:
            df_processed[c] = 0
        extra_cols = set(current_columns) - set(self.fitted_columns)
        if extra_cols:
            df_processed = df_processed.drop(columns=list(extra_cols))
        df_processed = df_processed[self.fitted_columns]
        if self.numerical_features:
            df_processed[self.numerical_features] = self.scaler.transform(df_processed[self.numerical_features])
        df_processed.fillna(0, inplace=True)
        return df_processed

    def save(self, file_path):
        joblib.dump(self, file_path)
        print(f"Preprocessor state saved to '{file_path}'")

    @staticmethod
    def load(file_path):
        preprocessor = joblib.load(file_path)
        print(f"Preprocessor state loaded from '{file_path}'")
        return preprocessor

    @staticmethod
    def create_windows(df, window_size=10):
        if df.empty or len(df) < window_size:
            return np.array([])
        return np.array([df.iloc[i:i + window_size].values for i in range(len(df) - window_size + 1)])

