import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def process_file(file_path, output_dir):
    df = pd.read_csv(file_path)
    for n in range(2, 26):
        dx_col = f"dx_n_{n}"
        dy_col = f"dy_n_{n}"
        dz_col = f"dz_n_{n}"

        if all(col in df.columns for col in [dx_col, dy_col, dz_col]):
            orderPar = np.sqrt(df[dx_col]**2 + df[dy_col]**2 + df[dz_col]**2)
            df[dx_col] = df[dx_col] / orderPar
            df[dy_col] = df[dy_col] / orderPar
            df[dz_col] = df[dz_col] / orderPar
            df[f"orderPar_{n}^2"] = orderPar ** 2

    output_path = output_dir / f"{file_path.stem}_with_orderPar.csv"
    df.to_csv(output_path, index=False)
    print(f"[✓] Saved: {output_path.name}")

def process_directory(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in the input directory.")
        return

    for csv_file in csv_files:
        process_file(csv_file, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add orderPar normalization to all CSV files in a directory.")
    parser.add_argument("--input_dir", "-i", required=True, help="Directory containing input CSV files.")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory to write output files to.")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
