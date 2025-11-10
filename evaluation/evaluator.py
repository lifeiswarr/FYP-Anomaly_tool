import pandas as pd
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, confusion_matrix


def evaluate_model(detector, preprocessor, test_data_path):
    """
    Evaluates a trained model's performance on a labeled dataset.

    Args:
        detector: The trained detector object (e.g., IsolationForestDetector).
        preprocessor: The fitted preprocessor object.
        test_data_path (str): Path to the labeled CSV file.

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    print("\n--- Starting Model Evaluation ---")

    # 1. Load data and separate features (X) from true labels (y)
    df = pd.read_csv(test_data_path)
    X_test = df.drop(columns=['label'])
    y_true = df['label']

    # 2. Preprocess the features
    processed_df = preprocessor.transform(X_test)

    # 3. Create windows
    window_size = 5  # This should match the training window size
    windows = preprocessor.create_windows(processed_df, window_size=window_size)

    if windows.shape[0] == 0:
        return {"error": "Not enough data to create any windows for evaluation."}

    # 4. Make predictions and measure latency
    start_time = time.time()
    anomalous_window_indices = detector.predict(windows)
    end_time = time.time()

    latency = end_time - start_time

    # 5. Calculate metrics (Precision & Recall)
    # This is tricky: predictions are on windows, but labels are on rows.
    # We'll create "true labels" for each window. A window is an anomaly if it contains any anomalous row.
    y_true_windows = []
    for i in range(len(windows)):
        window_labels = y_true.iloc[i: i + window_size]
        is_anomalous_window = 1 if (1 in window_labels.values) else 0
        y_true_windows.append(is_anomalous_window)

    # Create the prediction vector (1 for anomaly, 0 for normal)
    y_pred_windows = np.zeros(len(windows))
    y_pred_windows[anomalous_window_indices] = 1

    precision = precision_score(y_true_windows, y_pred_windows, zero_division=0)
    recall = recall_score(y_true_windows, y_pred_windows, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true_windows, y_pred_windows).ravel()

    metrics = {
        "precision": precision,
        "recall": recall,
        "latency_seconds": latency,
        "total_windows_evaluated": len(windows),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }

    return metrics
