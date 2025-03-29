# march 28, 2025.
# see the description of transform_geometry_with_bf.py - this is a slightly older version that doesn't save beta factors.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import csv

def build_local_frame(h_coord, n_coord, c_coord):
    """Construct orthonormal frame:
    - X axis: H → N
    - Y axis: in H–N–C plane
    - Z axis: orthogonal
    """
    x_axis = n_coord - h_coord
    x_axis /= np.linalg.norm(x_axis)

    temp_y = c_coord - h_coord
    temp_y -= np.dot(temp_y, x_axis) * x_axis  # Remove X component
    y_axis = temp_y / np.linalg.norm(temp_y)

    z_axis = np.cross(x_axis, y_axis)
    return np.stack([x_axis, y_axis, z_axis])  # 3x3 rotation matrix

def transform_neighbors(h_idx, amide_h, neighbors, atom_type_to_id, res_name_to_id, num_neighbors=20):
    h_row = amide_h[h_idx]
    h_coord = h_row[1:4]
    h_res_id = int(h_row[0])
    h_res_type_id = int(h_row[5])
    neighbors_this = neighbors[h_idx]

    # Find backbone N in same residue
    n_idx = next((i for i, a in enumerate(neighbors_this)
                  if a[0] == atom_type_to_id.get('N', -999)
                  and int(a[1]) == h_res_id), None)
    if n_idx is None:
        return None, None, None  # Can't orient

    n_coord = neighbors_this[n_idx][2:5]

    # Find carbonyl C in residue -1
    c_idx = next((i for i, a in enumerate(neighbors_this)
                  if a[0] == atom_type_to_id.get('C', -999)
                  and int(a[1]) == h_res_id - 1
                  and np.linalg.norm(a[2:5] - n_coord) < 2.0), None)
    if c_idx is None:
        return None, None, None  # Can't orient

    c_coord = neighbors_this[c_idx][2:5]
    R = build_local_frame(h_coord, n_coord, c_coord)

    features = []
    transformed_coords = []
    labels = []

    for atom in neighbors_this:
        pos = atom[2:5] - h_coord  # Translate so H is at origin
        dist = np.linalg.norm(pos)
        if dist < 1e-4:
            continue
        direction = pos / dist
        transformed = R @ direction
        scalar = 0.2 / dist

        features.append([
            atom[0],  # atom_type_id
            atom[6],  # res_type_id
            scalar,
            *transformed
        ])
        transformed_coords.append(transformed)
        labels.append((atom[0], int(atom[1]), atom[6]))  # atom_type_id, res_id, res_type_id

    if len(features) != num_neighbors:
        return None, None, None

    return np.array(features), np.array(transformed_coords), labels

def visualize_frame(coords, labels, h_idx, atom_type_id_to_name, res_id_to_name):
    """3D scatter plot of transformed neighborhood"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    ax.scatter(xs, ys, zs, c='blue', s=60, alpha=0.6)
    for x, y, z, (atom_id, res_id, res_type_id) in zip(xs, ys, zs, labels):
        label = f"{atom_type_id_to_name.get(atom_id, '?')} ({res_id_to_name.get(res_type_id, '?')}{res_id})"
        ax.text(x, y, z, label, size=7)

    ax.set_title(f"Transformed frame for amide H {h_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show(block=True)
    print("this runs")

def main(npz_file, out_npz, out_csv, preview_idx=0):
    data = np.load(npz_file, allow_pickle=True)
    amide_h = data["amide_h"]
    neighbors = data["neighbors"]
    atom_type_to_id = data["atom_type_to_id"].item()
    res_name_to_id = data["res_name_to_id"].item()

    # Reverse mappings for labels
    atom_type_id_to_name = {v: k for k, v in atom_type_to_id.items()}
    res_id_to_name = {v: k for k, v in res_name_to_id.items()}

    all_feats = []

    preview_shown = False

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "H_Index", "Atom_Type_ID", "Residue_Type_ID", "Scalar", "dx", "dy", "dz"
        ])
        for i in range(len(amide_h)):
            feats, transformed_coords, labels = transform_neighbors(
                i, amide_h, neighbors, atom_type_to_id, res_name_to_id
            )
            if feats is not None:
                for row in feats:
                    writer.writerow([i] + list(row))
                all_feats.append(feats)

                # Show first successful frame
                # this doesn't seem to quite work... but the csv looks okay.
                if not preview_shown:
                    visualize_frame(transformed_coords, labels, i, atom_type_id_to_name, res_id_to_name)
                    preview_shown = True

    final_array = np.stack(all_feats)  # shape: (N, 20, 6)
    np.savez(out_npz,
             features=final_array,
             atom_type_to_id=atom_type_to_id,
             res_name_to_id=res_name_to_id)

    print(f"✅ Saved {len(all_feats)} entries to {out_npz} and {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python transform_geometry_with_viz.py input.npz output.npz output.csv")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
