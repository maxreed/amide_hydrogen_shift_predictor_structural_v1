# march 28, 2025.
# updated to read in CSV output from extract_neighbours.py instead of numpy object.
# for each H and its 20 neighbors, re-perform coordinate transformation and output the rotated and normalized features
# including: res_index, atom_type_id, res_name_id, dx, dy, dz (normalized), metric, atom_name, neighbor_res_index

import pandas as pd
import numpy as np
import sys

# Constant used in metric calculation (1.2 makes the maximum about 1, since the closest two atoms come is about 1.2A)
SCALING_CONSTANT = 1.2

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def compute_rotation_matrix(n_coord, c_coord):
    x_axis = normalize(n_coord)
    tmp = c_coord
    z_axis = normalize(np.cross(x_axis, tmp))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.vstack([x_axis, y_axis, z_axis])

def process_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    output_rows = []

    for res_index, group in df.groupby("res_index"):
        if len(group) < 1:
            continue

        # Find H to origin transformation
        origin = np.zeros(3)
        first_row = group.iloc[0]

        # Collect N and C atoms based on neighbor atom name
        h_to_neighbors = group[["dx", "dy", "dz"]].values
        atom_names = group["atom_name"].values

        # Find N and C vectors relative to H (after H at origin)
        try:
            n_vector = group[group["atom_name"] == "N"][["dx", "dy", "dz"]].values[0]
            c_vector = group[group["atom_name"] == "C"][["dx", "dy", "dz"]].values[0]
        except IndexError:
            continue  # Skip if we can't find N or C

        R = compute_rotation_matrix(n_vector, c_vector)

        for _, row in group.iterrows():
            vec = np.array([row["dx"], row["dy"], row["dz"]])
            rotated = R @ vec
            normed = normalize(rotated)
            metric = SCALING_CONSTANT / np.linalg.norm(rotated)

            output_rows.append([
                int(row["res_index"]),
                int(row["atom_type_id"]),
                int(row["res_name_id"]),
                *normed.tolist(),
                metric,
                row["atom_name"],
                int(row["neighbor_res_index"])
            ])

    header = ["res_index", "atom_type_id", "res_name_id", "dx", "dy", "dz", "metric", "atom_name", "neighbor_res_index"]
    pd.DataFrame(output_rows, columns=header).to_csv(output_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python transform_geometry_with_bf.py input.csv output.csv")
    else:
        process_csv(sys.argv[1], sys.argv[2])
