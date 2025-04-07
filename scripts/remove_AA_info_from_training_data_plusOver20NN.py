import pandas as pd
import argparse
import os
from pathlib import Path

def clean_onehot_columns(df):
    prefixes = [
        "is_ALA_n", "is_ARG_n", "is_ASN_n", "is_ASP_n", "is_CYS_n",
        "is_GLN_n", "is_GLU_n", "is_GLY_n", "is_HIS_n", "is_ILE_n",
        "is_LEU_n", "is_LYS_n", "is_MET_n", "is_PHE_n", "is_PRO_n",
        "is_SER_n", "is_THR_n", "is_TRP_n", "is_TYR_n", "is_VAL_n"
    ]
    suffixes = ["n_1", "n_21", "n_22", "n_23", "n_24", "n_25"]
    
    cols_to_drop = [
        col for col in df.columns 
        if any(col.startswith(pre) for pre in prefixes) or any(col.endswith(suf) for suf in suffixes)
    ]
    
    return df.drop(columns=cols_to_drop)


def process_directory(input_dir, output_dir=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path

    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in the input directory.")
        return

    for file in csv_files:
        print(f"Processing {file.name}...")
        df = pd.read_csv(file)
        cleaned_df = clean_onehot_columns(df)

        output_file = output_path / f"{file.stem}_cleaned.csv"
        cleaned_df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch clean one-hot encoded columns from CSV files in a directory.")
    parser.add_argument("--input_dir", "-i", required=True, help="Path to the directory containing input CSV files.")
    parser.add_argument("--output_dir", "-o", help="Optional output directory for cleaned files.")

    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)
