# File: ingestion/data_source.py

import pandas as pd

def get_raw_data(filepath="live_capture.csv"):
    """Reads network data from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(filepath)
        print(f"✔️ Successfully loaded data from '{filepath}'")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{filepath}' was not found.")
        print("Please run the 'run_sniffer.py' script first to generate the data.")
        return pd.DataFrame()