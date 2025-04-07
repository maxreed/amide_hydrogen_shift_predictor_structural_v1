import pandas as pd
import argparse
import os

def assign_amber_atom_types(input_csv_path, mapping_csv_path, output_csv_path=None):
    # Load input and mapping data
    input_df = pd.read_csv(input_csv_path)
    # you need to set keep_default_na=False to prevent it from reading in "NA" (nitrogen aromatic) as NaN lol
    mapping_df = pd.read_csv(mapping_csv_path, keep_default_na=False)

    # Create the res_id_atom_name key in the input DataFrame
    input_df['res_id_atom_name'] = input_df['res_name_id'].astype(str) + input_df['atom_name']

    # Create mapping dictionary
    mapping_dict = dict(zip(mapping_df['res_id_atom_name'], mapping_df['amber_type']))

    # Function to determine amber_type
    def map_atom_type(row):
        key = row['res_id_atom_name']
        if key in mapping_dict:
            return mapping_dict[key]
        elif row['atom_name'] in ['H2', 'H3']:
            return 'H'
        elif row['atom_name'] == 'OXT':
            return 'O'
        else:
            return 'NOT FOUND'

    # Apply mapping
    input_df['amber_type'] = input_df.apply(map_atom_type, axis=1)

    # Define default output path if not specified
    if output_csv_path is None:
        base, ext = os.path.splitext(input_csv_path)
        output_csv_path = f"{base}_typed{ext}"

    # Save output
    input_df.to_csv(output_csv_path, index=False)
    print(f"[✓] Output written to: {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign AMBER14SB atom types to a CSV file.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file.")
    parser.add_argument("--mapping", "-m", required=True, help="Path to the amber14sb mapping CSV file.")
    parser.add_argument("--output", "-o", help="Path to the output CSV file. If not provided, appends '_typed.csv'.")

    args = parser.parse_args()

    assign_amber_atom_types(
        input_csv_path=args.input,
        mapping_csv_path=args.mapping,
        output_csv_path=args.output
    )
