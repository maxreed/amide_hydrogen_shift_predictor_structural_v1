import os
import pandas as pd
import re

# Define valid suffixes
valid_suffixes = {'_n_1', '_n_2', '_n_3'}

# Define amino acids
amino_acids = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

# Regex to match the pattern is_<AA>_n_<#>
aa_pattern = re.compile(rf'^is_({"|".join(amino_acids)})_n_(\d+)$')

input_dir = 'training_data_withAmber_noH_N'  # One directory above the script
output_dir = 'training_data_withAmber_noH_only3AA_N'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        path = os.path.join(input_dir, filename)
        df = pd.read_csv(path)

        # Keep columns that are not amino acid pattern or are valid ones
        keep_columns = [
            col for col in df.columns
            if not aa_pattern.match(col) or aa_pattern.match(col).group(2) in {'1', '2', '3'}
        ]

        filtered_df = df[keep_columns]
        filtered_df.to_csv(os.path.join(output_dir, filename), index=False)

print("Batch processing complete. Filtered files saved to:", output_dir)
