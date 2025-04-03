import os
import subprocess

def run_pipeline(bmrb_id):
    bmrb_id_str = f"bmr{bmrb_id}_3"

    # 1. Extract H shifts
    subprocess.run([
        "python", "scripts/extract_H_shifts_from_str.py",
        f"star_files/{bmrb_id_str}.str",
        f"h_shifts/{bmrb_id_str}_H.csv"
    ], check=True)

    # 2. Add hydrogens to PDB
    os.chdir("pdb_files")
    subprocess.run([
        "python", "../scripts/add_h_with_cleaning.py",
        f"{bmrb_id_str}.str.pdb", "split_models_h"
    ], check=True)

    # 3. Create output directories
    os.chdir("../cvs_files")
    os.makedirs(f"{bmrb_id_str}", exist_ok=True)
    os.makedirs(f"{bmrb_id_str}_rotated", exist_ok=True)

    # 4. Run extraction + transformation for 10 models
    os.chdir("..")
    for i in range(1, 11):
        model_str = f"model{i}"
        base = f"{bmrb_id_str}_{model_str}_h"

        pdb_path = f"pdb_files/split_models_h/{base}.pdb"
        out_csv = f"cvs_files/{bmrb_id_str}/{base}_output.csv"
        rotated_csv = f"cvs_files/{bmrb_id_str}_rotated/{base}_output_rotated.csv"

        extract_cmd = [
            "python", "scripts/extract_neighbours_with_names.py",
            pdb_path, out_csv
        ]
        transform_cmd = [
            "python", "scripts/transform_geometry_fromCSV_withNames.py",
            out_csv, rotated_csv
        ]

        print("Running extract:", " ".join(extract_cmd))
        subprocess.run(extract_cmd, check=True)

        print("Running transform:", " ".join(transform_cmd))
        subprocess.run(transform_cmd, check=True)

    # 5. Concatenate, sort, and average neighbor vectors
    subprocess.run([
        "python", "scripts/concat_neighbors_after_norm_from_all_models_and_sort_and_average_models.py",
        f"cvs_files/{bmrb_id_str}_rotated",
        f"cvs_files/{bmrb_id_str}_rotated/{bmrb_id_str}_allNeighborsAllModels_averagedVectors_only25.csv",
        f"cvs_files/{bmrb_id_str}_rotated/{bmrb_id_str}_allNeighborsAllModels_averagedVectors_only25_groupedFeatures.csv"
    ], check=True)

    # 6. Attach chemical shifts to the features
    subprocess.run([
        "python", "scripts/attach_shifts_to_features.py",
        f"h_shifts/{bmrb_id_str}_H.csv",
        f"cvs_files/{bmrb_id_str}_rotated/{bmrb_id_str}_allNeighborsAllModels_averagedVectors_only25_groupedFeatures.csv",
        f"training_data/{bmrb_id_str}_forTraining.csv"
    ], check=True)

    # 7. One-hot encode the features
    subprocess.run([
        "python", "scripts/one_hot_encode_features.py",
        f"training_data/{bmrb_id_str}_forTraining.csv",
        f"training_data/{bmrb_id_str}_forTraining_oneHot.csv"
    ], check=True)

    print(f"\nPipeline complete for BMRB ID {bmrb_id}.")

if __name__ == "__main__":
    run_pipeline(19752)
