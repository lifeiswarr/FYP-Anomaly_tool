import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime

# --- Configuration ---
PREPROCESSOR_PATH = 'preprocessor.joblib'
DETECTOR_PATH = 'detector.joblib'
CLASSIFIER_PATH = 'classifier.joblib'
EXPLAINER_PATH = 'explainer.joblib'  # Assuming explainer is saved
DATA_PATH = 'live_capture.csv'
# --- FIX: Changed from 10 to 5 ---
# This MUST match the window size used to train the detector.
# The error (200 features given vs 100 expected) implies the preprocessor
# outputs 20 features, and the model was trained with a window of 5 (20 * 5 = 100).
WINDOW_SIZE = 5


def check_files():
    """Checks if all required model and data files exist."""
    files = [PREPROCESSOR_PATH, DETECTOR_PATH, CLASSIFIER_PATH, DATA_PATH]
    # Explainer is optional for some pipelines
    if os.path.exists(EXPLAINER_PATH):
        files.append(EXPLAINER_PATH)

    for f in files:
        if not os.path.exists(f):
            print(f"❌ Error: Required file not found: {f}")
            print("Please make sure all .joblib files and the .csv are in the same directory.")
            return False
    return True


def create_windows(data, window_size):
    """Slides a window over the 2D data to create 3D windows."""
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)


def run_analysis():
    """
    Loads the full pipeline and runs it on the specified CSV file.
    """
    if not check_files():
        return

    print("--- 🚀 Starting Batch Anomaly Analysis ---")

    try:
        # --- Step 1: Load All Models ---
        print(f"Loading {PREPROCESSOR_PATH}...")
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        print(f"Loading {DETECTOR_PATH}...")
        detector = joblib.load(DETECTOR_PATH)

        print(f"Loading {CLASSIFIER_PATH}...")
        classifier = joblib.load(CLASSIFIER_PATH)

        has_explainer = False
        if os.path.exists(EXPLAINER_PATH):
            print(f"Loading {EXPLAINER_PATH}...")
            explainer = joblib.load(EXPLAINER_PATH)
            has_explainer = True
        else:
            print("Warning: Explainer model not found. Skipping explanation stage.")

        # --- Step 2: Load and Preprocess Data ---
        print(f"Loading and transforming data from {DATA_PATH}...")
        raw_data = pd.read_csv(DATA_PATH)

        # We must use the 'transform' method of the LOADED preprocessor
        # This ensures the new data is scaled and encoded identically to the training data
        processed_data = preprocessor.transform(raw_data)

        if processed_data is None:
            print("❌ Error: Preprocessing failed. Check data schema.")
            return

        print(f"Successfully transformed {len(raw_data)} rows.")

        # --- Step 3: Create Time Windows ---
        print(f"Creating {WINDOW_SIZE}-step time windows...")
        data_windows = create_windows(processed_data, WINDOW_SIZE)

        if len(data_windows) == 0:
            print(f"❌ Error: Not enough data to create a single window (Need > {WINDOW_SIZE} rows).")
            return

        print(f"Created {len(data_windows)} windows.")

        # --- Step 4: Execute 3-Stage Pipeline ---
        print("\n--- STAGE 1: DETECTING ANOMALIES ---")
        # The detector (EnsembleDetector) returns the *indices* of anomalous windows
        anomalous_indices = detector.predict(data_windows)

        if len(anomalous_indices) == 0:
            print("✅ Analysis Complete: No anomalies detected in this dataset.")
            return

        print(f"🚨 DETECTION COMPLETE: Found {len(anomalous_indices)} anomalous windows.")
        print("\n--- STAGE 2 & 3: CLASSIFYING & EXPLAINING ---")

        for i, idx in enumerate(anomalous_indices):
            window_data = data_windows[idx:idx + 1]  # Get the specific window (keep 3D shape)

            # --- Stage 2: Classify ---
            # Use .predict() which should return the string label (e.g., 'SYN_FLOOD')
            classification = classifier.predict(window_data)[0]

            # --- Stage 3: Explain ---
            reason = "No explanation available."
            if has_explainer:
                try:
                    # Assuming the explainer has a method 'explain' or 'predict'
                    # This part may need adjustment based on your explainer.py implementation
                    # For this example, let's assume it has an 'explain' method
                    reason = explainer.explain_instance(window_data[0])  # Pass the 2D window
                except Exception as e:
                    reason = f"Explainer failed: {e}"

            print("-----------------------------------------")
            print(f"🚨 ALERT #{i + 1} (at Window Index {idx}):")
            print(f"  Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Classification:  {classification.upper()}")
            print(f"  Reason:          {reason}")
            print("-----------------------------------------")

        print("\n--- ✅ Batch Analysis Complete ---")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_analysis()

