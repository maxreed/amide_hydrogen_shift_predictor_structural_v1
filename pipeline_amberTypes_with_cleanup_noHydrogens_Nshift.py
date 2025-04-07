import os
import subprocess
import shutil

def run_pipeline_clean(bmrb_id,pdb_folder):
    bmrb_id_str = f"bmr{bmrb_id}_3"

    # 1. Extract N shifts
    subprocess.run([
        "python", "scripts/extract_N_shifts_from_str.py",
        f"star_files/{bmrb_id_str}.str",
        f"n_shifts/{bmrb_id_str}_N.csv"
    ], check=True)

    # 2. Swap in the N shifts where the H shifts were in the feature CSVs.
    subprocess.run([
        "python", "scripts/update_Hshift_to_Nshift.py",
        f"n_shifts/{bmrb_id_str}_N.csv",
        f"training_data_withAmber_noH/{bmrb_id_str}_forTraining_oneHot.csv",
        f"training_data_withAmber_noH_N/{bmrb_id_str}_trainingFeatures_Nshifts.csv"
    ], check=True)



if __name__ == "__main__":
    pdb_folder = "BMRB_batch3"
    files = os.listdir(pdb_folder)
    run_these = []
    for star_file in files:
        if star_file[-8:]==".str.pdb":
            run_these.append(int(star_file[3:-10]))
    print("Files to extract features from:")
    print(run_these)
    for bmrb_number in run_these:
        run_pipeline_clean(bmrb_number,pdb_folder)
