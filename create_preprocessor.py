import pandas as pd
from preprocessing.preprocessor import Preprocessor

# The reference data file
NORMAL_DATA_FILE = 'live_capture.csv'
# Where to save the fitted preprocessor
PREPROCESSOR_PATH = 'preprocessor.joblib'


def main():
    """
    Creates, fits, and saves the preprocessor based on normal data.
    """
    print("--- Creating Preprocessor Blueprint ---")

    try:
        # Load the trusted, normal data
        normal_df = pd.read_csv(NORMAL_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: The training data file '{NORMAL_DATA_FILE}' was not found.")
        print("Please create it in your 'anomaly_tool' directory.")
        return

    # Create a preprocessor instance
    preprocessor = Preprocessor()

    # Fit it to the data
    preprocessor.fit(normal_df)

    # Save the fitted "blueprint" for later use
    preprocessor.save(PREPROCESSOR_PATH)

    print("\n--- ✅ Blueprint created successfully. ---")


if __name__ == "__main__":
    main()

