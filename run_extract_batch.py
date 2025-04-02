# april 1, 2025
# this does step 5 in how_to_use_this_thing.txt (i.e. it first runs a script to find the nearest neightbors to our amide hydrogens,
# then it rotates them so the N-H(i) bond is the x-axis and the C(i-1) atom is on the xy plane).

import subprocess

for i in range(1, 11):
    model_str = f"model{i}"
    base = f"bmr11103_3_{model_str}_h"

    pdb_path = f"pdb_files/split_models_h/{base}.pdb"
    out_csv = f"cvs_files/bmr11103_3/{base}_output.csv"
    rotated_csv = f"cvs_files/bmr11103_3_rotated/{base}_output_rotated.csv"

    extract_cmd = [
        "python",
        "scripts/extract_neighbours_with_names.py",
        pdb_path,
        out_csv
    ]
    transform_cmd = [
        "python",
        "scripts/transform_geometry_fromCSV_withNames.py",
        out_csv,
        rotated_csv
    ]

    print("Running extract:", " ".join(extract_cmd))
    subprocess.run(extract_cmd, check=True)

    print("Running transform:", " ".join(transform_cmd))
    subprocess.run(transform_cmd, check=True)
