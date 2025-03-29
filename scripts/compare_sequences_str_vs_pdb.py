# march 28, 2025
# this script mostly exists as a sanity check to make sure the PDB i'll be extracting features from and the STR containing
# the chemical shifts being used as training data actually refer to the same sequences. i likely won't run this in the final
# pipeline.

from Bio.PDB import PDBParser
import pynmrstar

def extract_sequence_from_nmrstar(file_path):
    entry = pynmrstar.Entry.from_file(file_path)
    for sf in entry:
        if sf.category == "entity" and "Polymer_seq_one_letter_code" in sf:
            raw_seq = sf["Polymer_seq_one_letter_code"][0]
            return raw_seq.replace("\n", "").replace(" ", "")
    raise ValueError("Sequence not found in NMR-STAR file.")

def extract_sequence_from_pdb(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", file_path)
    model = structure[0]  # First model
    residues = []

    for chain in model:
        for res in chain:
            if res.id[0] == " " and "CA" in res:  # Skip heteroatoms and water
                resname = res.resname
                aa = three_to_one.get(resname, "X")  # 'X' for unknown
                residues.append(aa)

    return ''.join(residues)

# Mapping from 3-letter to 1-letter codes
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compare_sequences.py yourfile.str yourfile.pdb")
        exit(1)

    star_file = sys.argv[1]
    pdb_file = sys.argv[2]

    try:
        seq_star = extract_sequence_from_nmrstar(star_file)
        seq_pdb = extract_sequence_from_pdb(pdb_file)

        print("NMR-STAR sequence:")
        print(seq_star)
        print("\nPDB sequence:")
        print(seq_pdb)

        if seq_star == seq_pdb:
            print("\n✅ Sequences match exactly.")
        else:
            print("\n❌ Sequences do not match.")
            print(f"STAR length: {len(seq_star)}, PDB length: {len(seq_pdb)}")

    except Exception as e:
        print(f"Error: {e}")
