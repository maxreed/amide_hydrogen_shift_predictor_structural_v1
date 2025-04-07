import os
import pandas as pd

# === Config ===
OFFSET_FILES = ["train_offset_analysis.csv", "test_offset_analysis.csv"]
INPUT_DIR = os.path.join("..", "training_data_pH_T")
OUTPUT_DIR = os.path.join("..", "training_data_pH_T_avgH_corr")

# Create output dir if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Combine offset tables
offset_dfs = []
for f in OFFSET_FILES:
    df = pd.read_csv(f)
    df["source_id"] = df["source_id"].astype(str)
    offset_dfs.append(df[["source_id", "best_x"]])

offset_table = pd.concat(offset_dfs).drop_duplicates(subset="source_id")
offset_map = dict(zip(offset_table["source_id"], offset_table["best_x"]))

print(f"✅ Loaded best_x values for {len(offset_map)} structures.")

processed = 0
skipped_missing = []
skipped_small = []

for source_id, best_x in offset_map.items():
    file_name = f"bmr{source_id}_3_forTraining_oneHot_pH_T.csv"
    input_path = os.path.join(INPUT_DIR, file_name)
    output_path = os.path.join(OUTPUT_DIR, file_name)

    if not os.path.exists(input_path):
        skipped_missing.append(source_id)
        continue

    try:
        df = pd.read_csv(input_path, header=None)

        if len(df) < 11:  # 1 header + <10 data rows
            df.to_csv(output_path, index=False, header=False)
            skipped_small.append(source_id)
        else:
            df.iloc[1:, 0] = df.iloc[1:, 0].astype(float) - best_x
            df.to_csv(output_path, index=False, header=False)
            processed += 1
    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
        skipped_missing.append(source_id)

print(f"\n✅ Correction complete.")
print(f" - Files corrected:         {processed}")
print(f" - Files skipped (missing): {len(skipped_missing)}")
print(f" - Files skipped (too small to correct): {len(skipped_small)}")
if skipped_small:
    print(" - Small files:", ", ".join(skipped_small))
