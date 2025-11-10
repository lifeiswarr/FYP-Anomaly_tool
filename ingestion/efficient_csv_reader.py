import pandas as pd

def stream_csv_data(file_path, batch_size=10):
    """
    Reads a large CSV file in chunks (mini-batches) to save memory.
    """
    try:
        for chunk in pd.read_csv(file_path, chunksize=batch_size):
            yield chunk
    except FileNotFoundError:
        print(f"\n[ERROR in stream_csv_data] File not found: {file_path}")
        return
    except Exception as e:
        print(f"\n[ERROR in stream_csv_data] Could not read file: {e}")
        return

