import sqlite3
import pandas as pd
import os
import sys
import time  # --- FIX: Added the missing 'time' import ---


def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_row_nicely(row_data, processed_row_data, preprocessor):
    """Prints a single row of data in a readable format, showing original and processed values."""
    print("-" * 60)
    print(f"--- Analyzing Row ID: {row_data['id']} (Timestamp: {row_data['timestamp']}) ---")
    print("-" * 60)

    # --- ROBUST FIX: Check for attributes before accessing them ---

    num_features = []
    cat_features = []
    all_feature_names = []

    # 1. Get Numerical features (Scaler should always exist)
    if hasattr(preprocessor, 'scaler') and hasattr(preprocessor.scaler, 'feature_names_in_'):
        num_features = preprocessor.scaler.feature_names_in_
        all_feature_names.extend(list(num_features))

        print("\n--- Original Features (Numerical) ---")
        for col in num_features:
            print(f"  {col:<15}: {row_data.get(col, 'N/A')}")
    else:
        print("\nWarning: Preprocessor has no 'scaler' or 'feature_names_in_' attribute.")

    # 2. Get Categorical features (Encoder is optional)
    if hasattr(preprocessor, 'encoder') and hasattr(preprocessor.encoder, 'feature_names_in_'):
        cat_features = preprocessor.encoder.feature_names_in_

        print("\n--- Original Features (Categorical) ---")
        for col in cat_features:
            print(f"  {col:<15}: {row_data.get(col, 'N/A')}")

        # Get feature names *after* encoding
        if hasattr(preprocessor.encoder, 'get_feature_names_out'):
            encoded_feature_names = preprocessor.encoder.get_feature_names_out(cat_features)
            all_feature_names.extend(list(encoded_feature_names))
    else:
        # No encoder found, which is fine.
        pass

    print("\n--- Processed Features (What the Model Sees) ---")
    # processed_row_data is a 1D numpy array. We need to map it back to names

    if not all_feature_names:
        print("  (Could not determine processed feature names from preprocessor)")
        # Fallback: just print the array
        print(f"  {processed_row_data}")
    else:
        # Print feature names mapped to values
        # --- FIX: Ensure we don't go out of bounds if lists mismatch ---
        for i, col_name in enumerate(all_feature_names):
            if i < len(processed_row_data):
                print(f"  {col_name:<25}: {processed_row_data[i]:.4f}")
            else:
                # This case might happen if preprocessor logic is complex
                print(f"  {col_name:<25}: (Value not found in processed array)")

    print("-" * 60)


def label_data_interactively(db_path, preprocessor):
    """
    Connects to the DB, finds unlabeled rows, and asks the user to label them.

    Args:
        db_path (str): Path to the SQLite database file.
        preprocessor (Preprocessor): The loaded, trained preprocessor instance.
    """

    known_labels = set()

    try:
        with sqlite3.connect(db_path) as conn:

            # --- FIX: Set row_factory *before* creating the cursor ---
            # This ensures the cursor returns dict-like rows instead of tuples
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # First, find out what labels we already have
            cursor.execute("SELECT DISTINCT label FROM traffic_logs WHERE label IS NOT NULL")
            labels_in_db = cursor.fetchall()
            if labels_in_db:
                known_labels = set([label[0] for label in labels_in_db])
                print(f"Known labels in DB: {known_labels}")

            # Fetch all rows that haven't been labeled yet
            cursor.execute("SELECT * FROM traffic_logs WHERE label IS NULL ORDER BY id ASC")

            unlabeled_rows = cursor.fetchall()

            if not unlabeled_rows:
                clear_screen()
                print("✅ No unlabeled data found. All data in the database has been labeled.")
                return

            print(f"Found {len(unlabeled_rows)} unlabeled rows. Starting labeling session...")
            time.sleep(2)  # This line will now work

            for i, row in enumerate(unlabeled_rows):
                clear_screen()
                print(f"--- Labeling Progress: {i + 1} / {len(unlabeled_rows)} ---")

                # --- FIX: Now that 'row' is a sqlite3.Row, we can cast to dict ---
                row_dict = dict(row)

                # We must process this single row just like the model will
                # The preprocessor expects a DataFrame
                row_df = pd.DataFrame([row_dict])

                # --- FIX: .transform() returns a DataFrame. Get the first row's values. ---
                # This returns a DataFrame with one row.
                processed_df = preprocessor.transform(row_df)

                # --- FIX: Check if processed_df is empty before accessing .iloc[0] ---
                if processed_df.empty:
                    print(f"❌ Skipping row {row_dict['id']}: Preprocessing returned no data.")
                    continue

                # Select the first row (.iloc[0]) and get its data as a NumPy array (.values)
                processed_row = processed_df.iloc[0].values

                print_row_nicely(row_dict, processed_row, preprocessor)  # Pass the dict

                print("\nEnter a label for this row.")
                if known_labels:
                    print(f"Known labels: {', '.join(known_labels)}")
                print("(e.g., 'normal', 'syn_flood', 'port_scan', 'skip', 'quit')")

                label = input("➡️ Label: ").lower().strip()

                if label == 'quit':
                    print("Quitting labeling session.")
                    break

                if label == 'skip':
                    # We use row_dict here since 'row' is a sqlite3.Row object
                    print(f"Skipping row {row_dict['id']}...")
                    continue

                if label:
                    # Save the label to the database
                    cursor.execute(
                        "UPDATE traffic_logs SET label = ? WHERE id = ?",
                        (label, row_dict['id'])  # Use row_dict here
                    )
                    conn.commit()  # Save the change

                    # Add this new label to our known set
                    known_labels.add(label)
                    print(f"Saved label '{label}' for row {row_dict['id']}.")
                    time.sleep(0.5)  # This line will now work

            clear_screen()
            print("✅ Labeling session complete.")

    except sqlite3.Error as e:
        print(f"\n❌ A database error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # --- This allows testing this script directly ---
    print("--- Running DB Labeler in Standalone Test Mode ---")

    # We need to find the DB and preprocessor
    # Assumes 'storage/database_manager.py' is in '../storage'
    # and 'preprocessor.joblib' is in '..'

    current_dir = os.path.dirname(__file__)
    BASE_DIR = os.path.abspath(os.path.join(current_dir, '..'))

    TEST_DB_PATH = os.path.join(BASE_DIR, 'storage', 'traffic_data.db')
    TEST_PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessor.joblib')

    if not os.path.exists(TEST_DB_PATH):
        print(f"❌ Test Error: Database not found at {TEST_DB_PATH}")
        sys.exit(1)

    if not os.path.exists(TEST_PREPROCESSOR_PATH):
        print(f"❌ Test Error: Preprocessor not found at {TEST_PREPROCESSOR_PATH}")
        print("   Please run [Step 2] (Train Anomaly Detector) from the main menu first.")
        sys.exit(1)

    # We need to import the Preprocessor class to load it
    try:
        # Add parent directory to path to find 'preprocessing' module
        sys.path.append(BASE_DIR)
        from preprocessing.preprocessor import Preprocessor
    except ImportError:
        print("❌ Test Error: Could not import Preprocessor class.")
        sys.exit(1)

    print(f"Loading preprocessor from {TEST_PREPROCESSOR_PATH}...")
    preprocessor = Preprocessor.load(TEST_PREPROCESSOR_PATH)

    print("Starting interactive labeler...")
    label_data_interactively(TEST_DB_PATH, preprocessor)