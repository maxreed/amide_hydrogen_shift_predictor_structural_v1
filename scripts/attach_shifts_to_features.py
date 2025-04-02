# april 1, 2025.
# this script opens a given star file for a protein, then opens a corresponding csv of features generated from the alphaflow results of
# that same protein. whenever it finds an amide hydrogen shift for a residue, it grabs the row of features for that residue and outputs that
# to a csv. note that this also has a check to make sure the amino acids for the star file and csv match. it won't output results if they don't
# (well it might if they match fortuitously but it won't output results for MOST of the amino acids - so a low number of rows in this output CSV
# is a red flag and should be inspected).

import pandas as pd
import sys

# Residue name to ID mapping
RES_NAMES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
             'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
res_name_to_id = {k: i for i, k in enumerate(RES_NAMES)}

def attach_shifts(shift_csv, features_csv, output_csv):
    # Load chemical shifts and normalize column names
    shifts = pd.read_csv(shift_csv)
    shifts.columns = shifts.columns.str.strip().str.lower()

    # Adjust for 0-indexing and map residue names to IDs
    shifts["res_index"] = shifts["residue_index"] - 1
    shifts["res_name_id"] = shifts["residue"].map(res_name_to_id)
    shifts = shifts.dropna(subset=["amide_h_shift"])

    # Load features
    features = pd.read_csv(features_csv)

    output_rows = []
    for _, row in shifts.iterrows():
        res_index = row["res_index"]
        res_name_id = row["res_name_id"]

        match = features[(features["res_index"] == res_index) &
                         (features["res_name_id_n_1"] == res_name_id)]

        if not match.empty:
            feature_row = match.iloc[0]
            full_row = [row["amide_h_shift"]] + feature_row.tolist()
            output_rows.append(full_row)

    # Create new column names
    columns = ["h_shift"] + features.columns.tolist()
    df_out = pd.DataFrame(output_rows, columns=columns)
    df_out.to_csv(output_csv, index=False)
    print(f"Wrote labeled features to: {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python attach_shifts_to_features.py <shifts.csv> <features.csv> <output.csv>")
    else:
        attach_shifts(sys.argv[1], sys.argv[2], sys.argv[3])
