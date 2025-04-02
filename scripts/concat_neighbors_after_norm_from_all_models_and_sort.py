# april 1, 2025
# this script isn't useful for making output, but it's nice for inspecting output. it takes all the models from alphaflow and then groups all the common
# interaction. this is good for confirming that the sorting of data is being done correctly, and lets one manually calculate the average distance metric
# plus the averaged direction vector from H to the nearest neighbors.

import os
import sys
import pandas as pd

def concatenate_and_process(directory, output_file):
    # Load and concatenate all CSVs in the directory
    all_rows = []
    for file in sorted(os.listdir(directory)):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, file))
            all_rows.append(df)
    
    if not all_rows:
        print("No CSV files found.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    # Combine res_index, neighbor_res_index, and atom_name
    df_all["res_index_neighbor_atom"] = df_all.apply(
        lambda row: f"{row['res_index']}_{row['neighbor_res_index']}_{row['atom_name']}", axis=1
    )

    # Sort by the combined column
    df_all_sorted = df_all.sort_values("res_index_neighbor_atom")

    # Write to output
    df_all_sorted.to_csv(output_file, index=False)
    print(f"Wrote sorted CSV to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python concat_and_sort.py <input_dir> <output_csv>")
    else:
        concatenate_and_process(sys.argv[1], sys.argv[2])
