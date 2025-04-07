import pandas as pd
import sys

ATOM_TYPES = ['C', 'CA', 'CR', 'CT', 'CW', 'H', 'H1', 'HA', 'HC', 'HO',
 'HS', 'N', 'N3', 'NB', 'NH1', 'NH2', 'O', 'OH', 'S', 'SH']

RES_NAMES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
             'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

atom_type_id_to_label = {i: f'is_{atom}_n_' for i, atom in enumerate(ATOM_TYPES)}
res_name_id_to_label = {i: f'is_{res}_n_' for i, res in enumerate(RES_NAMES)}

def one_hot_encode(df):
    keep = df.drop(columns=[col for col in df.columns if col.startswith("amber_type") or col.startswith("res_name_id")])
    new_columns = {}

    for col in df.columns:
        if col.startswith("amber_type"):
            suffix = col.split("_n_")[-1]
            for i, atom in enumerate(ATOM_TYPES):
                new_columns[f'is_{atom}_n_{suffix}'] = (df[col] == ATOM_TYPES[i]).astype(int)

        if col.startswith("res_name_id"):
            suffix = col.split("_n_")[-1]
            for i, res in enumerate(RES_NAMES):
                new_columns[f'is_{res}_n_{suffix}'] = (df[col] == i).astype(int)

    encoded = pd.concat([keep, pd.DataFrame(new_columns)], axis=1)
    return encoded

def process(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df_encoded = one_hot_encode(df)
    df_encoded.to_csv(output_csv, index=False)
    print(f"Wrote one-hot encoded features to: {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python one_hot_encode_features.py <input_csv> <output_csv>")
    else:
        process(sys.argv[1], sys.argv[2])
