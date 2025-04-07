# april 1, 2025
# this script IS for making output for model training. it takes all the models from alphaflow and then groups all the common interaction.
# it then averages the distance metrics and direction vectors for the instances of an interaction from different models.

import os
import sys
import pandas as pd

def reduce_and_filter(directory, output_csv, pivoted_csv):
    all_rows = []
    for file in sorted(os.listdir(directory)):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, file))
            all_rows.append(df)

    if not all_rows:
        print("No CSV files found.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    df_all["res_index_neighbor_atom"] = df_all.apply(
        lambda row: f"{row['res_index']}_{row['neighbor_res_index']}_{row['atom_name']}", axis=1
    )

    grouped = df_all.groupby("res_index_neighbor_atom")

    rows = []
    for name, group in grouped:
        avg_dx = group["dx"].mean()
        avg_dy = group["dy"].mean()
        avg_dz = group["dz"].mean()
        avg_metric = group["metric"].sum() / 10.0

        first = group.iloc[0]
        row = {
            "res_index": first["res_index"],
            "atom_type_id": first["atom_type_id"],
            "res_name_id": first["res_name_id"],
            "dx": avg_dx,
            "dy": avg_dy,
            "dz": avg_dz,
            "metric": avg_metric,
            "atom_name": first["atom_name"],
            "neighbor_res_index": first["neighbor_res_index"],
            "res_index_neighbor_atom": name,
            "amber_type": first["amber_type"] # added this in for outputting amber atom type.
        }
        rows.append(row)

    df_result = pd.DataFrame(rows)

    df_filtered = (
        df_result
        .sort_values(["res_index", "metric"], ascending=[True, False])
        .groupby("res_index")
        .head(25)
        .reset_index(drop=True)
    )

    df_filtered.to_csv(output_csv, index=False)
    print(f"Wrote reduced + filtered CSV to: {output_csv}")

    # Pivot into single-row-per-residue format (excluding atom_name and neighbor_res_index)
    pivoted_rows = []
    for res_index, group in df_filtered.groupby("res_index"):
        row = {"res_index": res_index}
        for i, (_, r) in enumerate(group.iterrows(), start=1):
            # i modified this too, taking out atom_type_id and putting in amber_type
            for col in ["res_name_id","amber_type", "dx", "dy", "dz", "metric"]:
                row[f"{col}_n_{i}"] = r[col]
        pivoted_rows.append(row)

    df_pivoted = pd.DataFrame(pivoted_rows)
    df_pivoted.to_csv(pivoted_csv, index=False)
    print(f"Wrote pivoted CSV to: {pivoted_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python reduce_grouped_vectors.py <input_dir> <output_csv> <pivoted_csv>")
    else:
        reduce_and_filter(sys.argv[1], sys.argv[2], sys.argv[3])
