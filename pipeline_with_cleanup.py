import os
import subprocess
import shutil

def run_pipeline_clean(bmrb_id):
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

    # 4. Run extraction + transformation
    os.chdir("..")
    for i in range(1, 11):
        model_str = f"model{i}"
        base = f"{bmrb_id_str}_{model_str}_h"

        pdb_path = f"pdb_files/split_models_h/{base}.pdb"
        out_csv = f"cvs_files/{bmrb_id_str}/{base}_output.csv"
        rotated_csv = f"cvs_files/{bmrb_id_str}_rotated/{base}_output_rotated.csv"

        subprocess.run([
            "python", "scripts/extract_neighbours_with_names.py",
            pdb_path, out_csv
        ], check=True)

        subprocess.run([
            "python", "scripts/transform_geometry_fromCSV_withNames.py",
            out_csv, rotated_csv
        ], check=True)

    # 5. Concatenate, sort, and average
    avg_vec = f"cvs_files/{bmrb_id_str}_rotated/{bmrb_id_str}_allNeighborsAllModels_averagedVectors_only25.csv"
    grouped_vec = f"cvs_files/{bmrb_id_str}_rotated/{bmrb_id_str}_allNeighborsAllModels_averagedVectors_only25_groupedFeatures.csv"
    subprocess.run([
        "python", "scripts/concat_neighbors_after_norm_from_all_models_and_sort_and_average_models.py",
        f"cvs_files/{bmrb_id_str}_rotated", avg_vec, grouped_vec
    ], check=True)

    # 6. Attach chemical shifts
    training_csv = f"training_data/{bmrb_id_str}_forTraining.csv"
    subprocess.run([
        "python", "scripts/attach_shifts_to_features.py",
        f"h_shifts/{bmrb_id_str}_H.csv", grouped_vec, training_csv
    ], check=True)

    # 7. One-hot encode final CSV
    training_csv_final = training_csv.replace(".csv", "_oneHot.csv")
    subprocess.run([
        "python", "scripts/one_hot_encode_features.py",
        training_csv, training_csv_final
    ], check=True)

    print(f"Pipeline complete for BMRB ID {bmrb_id}, now cleaning up...")

    # 8. Cleanup
    for i in range(1, 11):
        model_path = f"pdb_files/split_models_h/{bmrb_id_str}_model{i}_h.pdb"
        if os.path.exists(model_path):
            os.remove(model_path)

    for folder in [f"cvs_files/{bmrb_id_str}", f"cvs_files/{bmrb_id_str}_rotated"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    h_shift_file = f"h_shifts/{bmrb_id_str}_H.csv"
    if os.path.exists(h_shift_file):
        os.remove(h_shift_file)

    if os.path.exists(training_csv):
        os.remove(training_csv)

    print(f"Cleanup complete. Only {training_csv_final} remains.")

if __name__ == "__main__":
    run_these = [26010,17434,18673,10299,30408,16298,30332,11423,34607,
        5038,6072,16396,5060,34535,30870,6365,15785,30807,16386,15211,19938,25449,31170,
        5461,5599,19955,4892]
    for bmrb_number in run_these:
        run_pipeline_clean(bmrb_number)
