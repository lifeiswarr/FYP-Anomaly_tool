import pandas as pd
import json
import numpy as np
from preprocessing.preprocessor import Preprocessor


def explain_anomaly_statistically(preprocessor, anomalous_window_df):
    """
    Provides a simple, human-readable statistical explanation for an anomaly.

    Args:
        preprocessor: The fitted Preprocessor object.
        anomalous_window_df (pd.DataFrame): The raw data for the anomalous window.

    Returns:
        str: A human-readable string explaining the likely cause of the anomaly.
    """
    if not hasattr(preprocessor.scaler, 'mean_'):
        return "Explanation not available (preprocessor not fitted)."

    mean_values = pd.Series(preprocessor.scaler.mean_, index=preprocessor.numerical_features)
    std_values = pd.Series(np.sqrt(preprocessor.scaler.var_), index=preprocessor.numerical_features)
    window_mean = anomalous_window_df[preprocessor.numerical_features].mean()
    z_scores = (window_mean - mean_values).abs() / std_values
    z_scores = z_scores.dropna().sort_values(ascending=False)

    if z_scores.empty:
        return "No significant numerical deviations found."

    top_features = z_scores.head(3)
    explanation_parts = []
    for feature, score in top_features.items():
        if score > 2.0:
            direction = "high" if window_mean[feature] > mean_values.get(feature, 0) else "low"
            explanation_parts.append(f"unusually {direction} '{feature}'")

    if not explanation_parts:
        return f"Minor deviations detected, with the most significant being in '{top_features.index[0]}'."

    return "Key contributing factors: " + ", ".join(explanation_parts) + "."


def generate_rich_prediction_report(detector, classifier, preprocessor, data_stream, window_size):
    """
    Analyzes a data stream, detects anomalies, classifies them, and returns a structured report.

    Args:
        detector: The trained anomaly detection model.
        classifier: The trained classification model.
        preprocessor: The fitted preprocessor.
        data_stream: A generator yielding pandas DataFrames (batches of data).
        window_size (int): The size of the sliding window used for analysis.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected anomaly.
    """
    findings = []
    total_rows_processed = 0

    print(f"Processing data stream", end='')
    for raw_batch_df in data_stream:
        print('.', end='', flush=True)
        processed_df = preprocessor.transform(raw_batch_df)
        windows = Preprocessor.create_windows(processed_df, window_size=window_size)

        if windows.shape[0] > 0:
            anomalous_indices = detector.predict(windows)
            if anomalous_indices.size > 0:
                anomalous_windows = windows[anomalous_indices]
                attack_types = classifier.predict(anomalous_windows)

                for i, window_index_in_batch in enumerate(anomalous_indices):
                    attack = attack_types[i]
                    original_row_index = total_rows_processed + window_index_in_batch + window_size - 1
                    raw_window_df = raw_batch_df.iloc[window_index_in_batch: window_index_in_batch + window_size]
                    explanation = explain_anomaly_statistically(preprocessor, raw_window_df)

                    findings.append({
                        "row": original_row_index,
                        "attack": attack,
                        "reason": explanation
                    })

        total_rows_processed += len(raw_batch_df)

    print("\nAnalysis complete.")
    return findings


def generate_prediction_report(detector, preprocessor, input_file_path, output_file_path, window_size=5):
    """
    Analyzes an entire unlabeled dataset and saves a copy with a new 'prediction' column.
    This report is simpler, marking every row as either normal (0) or anomalous (1).

    Args:
        detector: A trained detector object.
        preprocessor: A fitted preprocessor object.
        input_file_path (str): Path to the unlabeled CSV to analyze.
        output_file_path (str): Path to save the new CSV with predictions.
        window_size (int): The window size used during model training.
    """
    print("\n--- ⚙️ Generating Simple Prediction Report (CSV) ---")
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_file_path}'")
        return

    report_df = df.copy()
    processed_df = preprocessor.transform(df)
    windows = Preprocessor.create_windows(processed_df, window_size=window_size)

    if windows.shape[0] == 0:
        print("Not enough data to create any windows. No report generated.")
        return

    anomalous_window_indices = detector.predict(windows)
    report_df['prediction'] = 0

    for idx in anomalous_window_indices:
        report_df.iloc[idx: idx + window_size, report_df.columns.get_loc('prediction')] = 1

    report_df.to_csv(output_file_path, index=False)

    num_anomalies = len(report_df[report_df['prediction'] == 1])
    print(f"--- ✅ Report complete. Found {num_anomalies} potential anomalous records. ---")
    print(f"--- Saved to '{output_file_path}' ---")

