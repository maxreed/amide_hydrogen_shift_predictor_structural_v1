import pandas as pd
import sys

def update_feature_shifts(shift_csv_path, feature_csv_path, output_path):
    # Load the CSV files
    chemical_shifts_df = pd.read_csv(shift_csv_path)
    features_df = pd.read_csv(feature_csv_path)

    # Convert Residue_Index to integer and adjust to 0-based indexing
    chemical_shifts_df["Residue_Index"] = chemical_shifts_df["Residue_Index"].astype(int) - 1

    # Build mapping from Residue_Index to Amide_N_Shift
    shift_map = dict(zip(chemical_shifts_df["Residue_Index"], chemical_shifts_df["Amide_N_Shift"]))

    # Filter to rows that have a corresponding Residue_Index
    features_df_filtered = features_df[features_df["res_index"].isin(shift_map.keys())].copy()

    # Map the shifts
    features_df_filtered["h_shift"] = features_df_filtered["res_index"].map(shift_map)

    # Drop rows where the new h_shift is NaN
    features_df_filtered = features_df_filtered.dropna(subset=["h_shift"])

    # Save the updated file
    features_df_filtered.to_csv(output_path, index=False)
    print(f"Updated feature file saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python update_features.py <chemical_shift_csv> <feature_csv> <output_csv>")
        sys.exit(1)

    shift_csv = sys.argv[1]
    feature_csv = sys.argv[2]
    output_csv = sys.argv[3]

    update_feature_shifts(shift_csv, feature_csv, output_csv)
