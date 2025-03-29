# march 28, 2025
# this script finds the 20 nearest atoms to a all the backbone amide hydrogens in a PDB file. these nearest neighbors and
# their locations make up many of the features i'm eventually going to use in the model. this doesn't fully process the
# data, and the output of this file needs to be further processed with transform_geometry_with_bf.py

from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
import numpy as np
import csv

ATOM_TYPES = ['H', 'C', 'N', 'O', 'S']
RES_NAMES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
             'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

atom_type_to_id = {k: i for i, k in enumerate(ATOM_TYPES)}
res_name_to_id = {k: i for i, k in enumerate(RES_NAMES)}

def infer_atom_type(atom):
    """Infer the element from atom name manually, since Biopython isn't reliable here."""
    name = atom.get_name().strip()
    if name[0].isdigit():
        name = name[1:]  # e.g., 1HD1 → HD1
    return name[0].upper()  # H, C, N, O, S

def safe_lookup(d, key, unknown_val=-1):
    return d.get(key.upper(), unknown_val)

def get_amide_hydrogens_and_neighbors(pdb_file, csv_out, npz_out, num_neighbors=20):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_file)
    model = structure[0]

    all_atoms = []
    all_coords = []

    for chain in model:
        for res in chain:
            if res.id[0] != " ":  # Skip hetero/water
                continue
            for atom in res:
                all_atoms.append({
                    "atom": atom,
                    "coord": atom.coord,
                    "atom_type": infer_atom_type(atom),
                    "res_name": res.resname,
                    "res_id": res.id[1],
                    "b_factor": atom.bfactor
                })
                all_coords.append(atom.coord)

    atom_array = np.array(all_coords)
    tree = cKDTree(atom_array)

    results = []
    csv_rows = []

    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            if "N" in res and "H" in res:
                h_atom = res["H"]
                h_coord = h_atom.coord
                h_b = h_atom.bfactor
                res_id = res.id[1]
                res_name = res.resname

                dists, indices = tree.query(h_coord, k=num_neighbors + 1)
                indices = [i for i in indices if not np.allclose(all_coords[i], h_coord)]
                indices = indices[:num_neighbors]

                neighbor_array = []
                csv_row = [res_id, res_name, *h_coord, h_b]

                for idx in indices:
                    a = all_atoms[idx]
                    atom_type_id = safe_lookup(atom_type_to_id, a["atom_type"])
                    res_type_id = safe_lookup(res_name_to_id, a["res_name"])
                    neighbor_array.append([
                        atom_type_id,
                        a["res_id"],
                        *a["coord"],
                        a["b_factor"],
                        res_type_id
                    ])
                    csv_row += [
                        a["atom_type"],
                        a["res_id"],
                        a["res_name"],
                        *a["coord"],
                        a["b_factor"]
                    ]

                results.append({
                    "h": np.array([
                        res_id, *h_coord, h_b,
                        safe_lookup(res_name_to_id, res_name),
                        safe_lookup(atom_type_to_id, "H")
                    ]),
                    "neighbors": np.array(neighbor_array)
                })
                csv_rows.append(csv_row)

    # Write CSV
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["ResID", "ResName", "H_X", "H_Y", "H_Z", "H_B"]
        for i in range(num_neighbors):
            header += [
                f"A{i+1}_Type", f"A{i+1}_ResID", f"A{i+1}_ResName",
                f"A{i+1}_X", f"A{i+1}_Y", f"A{i+1}_Z", f"A{i+1}_B"
            ]
        writer.writerow(header)
        writer.writerows(csv_rows)

    # Write NPZ
    h_data = np.stack([r["h"] for r in results])
    neighbor_data = np.stack([r["neighbors"] for r in results])
    np.savez(npz_out,
             amide_h=h_data,
             neighbors=neighbor_data,
             atom_type_to_id=atom_type_to_id,
             res_name_to_id=res_name_to_id)

    print(f"✅ Wrote {len(results)} entries to {csv_out} and {npz_out}")

# Run from CLI
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python extract_neighbors.py input.pdb output.csv output.npz")
        exit(1)

    get_amide_hydrogens_and_neighbors(sys.argv[1], sys.argv[2], sys.argv[3])
