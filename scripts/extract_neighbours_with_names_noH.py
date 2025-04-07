# March 28, 2025
# This script finds the 20 nearest atoms to all the backbone amide hydrogens (H) in a PDB file.
# These nearest neighbors and their coordinates will be used later as features in a predictive model.
# This script doesn't fully process the data. Its output is further processed by transform_geometry_with_bf.py.

from Bio.PDB import PDBParser  # Parses PDB files to obtain structural information
from scipy.spatial import cKDTree  # Efficient spatial search for nearest neighbor queries
import numpy as np
import csv

# Define accepted atom types and residue names
ATOM_TYPES = ['H', 'C', 'N', 'O', 'S']
RES_NAMES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
             'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

# Mapping from atom type or residue name to numeric ID
atom_type_to_id = {k: i for i, k in enumerate(ATOM_TYPES)}
res_name_to_id = {k: i for i, k in enumerate(RES_NAMES)}

def infer_atom_type(atom):
    """
    Infers the chemical element of an atom from its name, since Biopython may not do this reliably.
    Strips off leading digit (e.g., '1HD1' becomes 'HD1') to extract the element name.
    """
    name = atom.get_name().strip()
    if name[0].isdigit():
        name = name[1:]  # e.g., '1HD1' → 'HD1'
    return name[0]  # Return first letter, e.g., 'H', 'C', etc.

def extract_features(structure):
    """
    Extracts and returns nearest-neighbor features around each backbone amide hydrogen atom.
    Returns a list of feature rows:
    [residue index, neighbor atom type, residue type, dx, dy, dz, atom name, neighbor residue index]
    """
    atoms = []  # List of all atoms in the structure
    coords = []  # Corresponding coordinates of atoms
    residue_indices = {}  # Map from residue ID to its index in sequence
    next_res_index = 0

    # Loop over all atoms in all residues in all chains
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Add only atoms from known residue names and atom types
                    atom_type = infer_atom_type(atom)
                    if atom_type not in atom_type_to_id:
                        continue
                    if residue.get_resname() not in res_name_to_id:
                        continue
                    atoms.append((atom, residue))
                    coords.append(atom.coord)
                    if residue not in residue_indices:
                        residue_indices[residue] = next_res_index
                        next_res_index += 1

    coords = np.array(coords)  # Shape: (num_atoms, 3)
    tree = cKDTree(coords)  # Build KD-tree for fast neighbor search

    feature_rows = []  # Will hold the final features
    residue_index = 0  # Track the index of each residue with an amide H atom

    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip residues not in defined list
                if residue.get_resname() not in res_name_to_id:
                    residue_index += 1 # this probably should never trigger... but just in case.
                    continue

                # Try to get the backbone amide hydrogen atom ('H')
                if 'H' not in residue:
                    residue_index += 1 # need this to not mess up numbering. THIS IS MY OWN FIX.
                    continue  # Some residues may not have H (e.g., missing data)
                h_atom = residue['H']
                h_coord = h_atom.coord

                # Query for 21 closest atoms including itself; later exclude self
                # NOTE: now modified to do 30.
                # ANOTHER NOTE: i now modified to 60 because i'm purging all hydrogens, which i estimate to be about half.
                distances, indices = tree.query(h_coord, k=61)

                for idx in indices:
                    neighbor_atom, neighbor_res = atoms[idx]
                    # Skip self-match (distance = 0)
                    if np.allclose(h_coord, neighbor_atom.coord):
                        continue

                    # Encode features: [residue index, atom type ID, residue ID, dx, dy, dz, atom name, neighbor residue index]
                    atom_type_id = atom_type_to_id[infer_atom_type(neighbor_atom)]
                    # skip all hydrogens (CRITICAL LINE IN THIS MODIFIED VERSION)
                    if atom_type_id == 0:
                        continue
                    res_name_id = res_name_to_id[neighbor_res.get_resname()]
                    displacement = neighbor_atom.coord - h_coord
                    atom_name = neighbor_atom.get_name().strip()
                    neighbor_res_index = residue_indices[neighbor_res]
                    row = [residue_index, atom_type_id, res_name_id, *displacement.tolist(), atom_name, neighbor_res_index]
                    feature_rows.append(row)

                residue_index += 1  # Move to next residue with an amide hydrogen

    return feature_rows

def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_neighbours.py input.pdb output.csv")
        return

    pdb_file = sys.argv[1]
    output_file = sys.argv[2]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    features = extract_features(structure)

    # Write to CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['res_index', 'atom_type_id', 'res_name_id', 'dx', 'dy', 'dz', 'atom_name', 'neighbor_res_index'])
        writer.writerows(features)

if __name__ == '__main__':
    main()
