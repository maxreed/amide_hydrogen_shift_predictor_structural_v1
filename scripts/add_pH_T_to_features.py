import os
import sys
import pandas as pd

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def main(metadata_path, output_dir):
    input_dir = "training_data"
    
    # Load metadata file
    metadata = pd.read_csv(metadata_path)
    
    # Ensure pH and temperature columns exist
    assert "pH" in metadata.columns and "temperature" in metadata.columns, \
        "Metadata must contain 'pH' and 'temperature' columns."

    # Create output dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith("_forTraining_oneHot.csv"):
            continue

        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)

        # Derive base name (e.g., from 'bmr4892_3_forTraining_oneHot.csv' -> 'bmr4892_3.str')
        base_name = filename.replace("_forTraining_oneHot.csv", ".str")

        # Match row in metadata
        row = metadata[metadata.iloc[:, 0] == base_name]
        if row.empty:
            print(f"Warning: No metadata found for {base_name}, skipping.")
            continue

        ph = row.iloc[0]["pH"]
        temp = row.iloc[0]["temperature"]

        # Normalize
        norm_ph = normalize(ph, 6, 8)
        norm_temp = normalize(temp, 288, 310)

        # Add new columns
        df["pH"] = norm_ph
        df["temperature"] = norm_temp

        # Save new file
        new_filename = filename.replace(".csv", "_pH_T.csv")
        new_path = os.path.join(output_dir, new_filename)
        df.to_csv(new_path, index=False)
        print(f"Saved updated file to: {new_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_pH_T_to_features.py metadata.csv output_dir/")
        sys.exit(1)

    metadata_path = sys.argv[1]
    output_dir = sys.argv[2]
    main(metadata_path, output_dir)
