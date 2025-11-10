import pandas as pd
import sys


def label_data_interactively(preprocessor, input_file_path, output_file_path):
    """
    An interactive tool to create a labeled dataset for evaluation.

    Args:
        preprocessor: An already-fitted Preprocessor object.
        input_file_path (str): Path to the unlabeled CSV to review.
        output_file_path (str): Path where the new, labeled CSV will be saved.
    """
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_file_path}'")
        return

    # Use the preprocessor just to get the column names for consistent display
    processed_template = preprocessor.transform(df.head(1))
    window_size = 5  # Should match your project's window size

    labeled_records = []

    print("\n--- 🕵️ Starting Interactive Labeling Session ---")
    print("For each window of data, decide if it represents an anomaly.")
    print("Enter 'y' for YES (anomaly), 'n' for NO (normal), 's' to SKIP, or 'q' to QUIT and save.")

    # We iterate through the raw data, not the processed data, to save the original values
    for i in range(len(df) - window_size + 1):
        window_df = df.iloc[i:i + window_size]

        print("\n" + "=" * 80)
        print(f"--- Window #{i + 1}/{len(df) - window_size + 1} ---")
        print(window_df.to_string(index=False))
        print("=" * 80)

        while True:
            choice = input("Is this window an anomaly? (y/n/s/q): ").lower()
            if choice in ['y', 'n', 's', 'q']:
                break
            print("❌ Invalid input. Please enter 'y', 'n', 's', or 'q'.")

        if choice == 'q':
            print("Quitting and saving progress...")
            break
        if choice == 's':
            continue

        # The label is 1 if the user said 'y', otherwise 0
        label = 1 if choice == 'y' else 0

        # We label the *last record* of the window, as that's when the pattern is complete
        record_to_label = window_df.iloc[-1].to_dict()
        record_to_label['label'] = label
        labeled_records.append(record_to_label)

    if not labeled_records:
        print("\nNo records were labeled. Nothing to save.")
        return

    # Save the results
    labeled_df = pd.DataFrame(labeled_records)
    labeled_df.to_csv(output_file_path, index=False)
    print(f"\n--- ✅ Successfully saved {len(labeled_records)} labeled records to '{output_file_path}' ---")
