# march 31, 2025
# this is the good one for processing the direct alphaflow output!!
# well, relatively good. it splits the models, removes the trouble lines "MODEL" and "ENDMDL", add hydrogens, and adds the missing
# C terminal carboxyl oxygen (OXT) with roughly correct geometry.

import os
import sys
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import numpy as np

def split_models(input_pdb):
    models = []
    current = []
    inside_model = False

    with open(input_pdb, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                inside_model = True
                current = []
            elif line.startswith("ENDMDL"):
                inside_model = False
                models.append(current)
            elif inside_model:
                current.append(line)
    return models

def parse_coords(line):
    return np.array([
        float(line[30:38]),
        float(line[38:46]),
        float(line[46:54])
    ])

def format_pdb_line(serial, name, resname, chain, resseq, icode, coords, occupancy="1.00", temp="0.00", element="O"):
    return (
        f"ATOM  {serial:5d} {name:<4}{resname:>4} {chain}"
        f"{resseq}{icode}   "
        f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"
        f"{occupancy:>6}{temp:>6}           {element:>2}\n"
    )

def ensure_oxt_atom(model_lines):
    atom_lines = [line for line in model_lines if line.startswith("ATOM")]
    if not atom_lines:
        return model_lines

    # Find the last residue
    last_atoms = []
    last_key = None
    for line in atom_lines:
        chain = line[21]
        res_seq = line[22:26]
        i_code = line[26]
        key = (chain, res_seq, i_code)
        if key != last_key:
            last_atoms = []
            last_key = key
        last_atoms.append(line)

    # Bail if OXT already present
    if any(line[12:16].strip() == "OXT" for line in last_atoms):
        return model_lines

    # Extract needed atoms
    c_line = next((l for l in last_atoms if l[12:16].strip() == "C"), None)
    o_line = next((l for l in last_atoms if l[12:16].strip() == "O"), None)
    ca_line = next((l for l in last_atoms if l[12:16].strip() == "CA"), None)

    if not (c_line and o_line and ca_line):
        return model_lines

    C = parse_coords(c_line)
    O = parse_coords(o_line)
    CA = parse_coords(ca_line)

    # Define local coordinate system
    CO = (O - C)
    CO /= np.linalg.norm(CO)

    CCA = (CA - C)
    CCA /= np.linalg.norm(CCA)

    # Normal to the plane defined by CO and CCA
    normal = np.cross(CO, CCA)
    normal /= np.linalg.norm(normal)

    # Rotate CO by 120° around the normal using Rodrigues’ rotation
    angle = np.radians(-120)
    OXT_dir = (
        CO * np.cos(angle) +
        np.cross(normal, CO) * np.sin(angle) +
        normal * (np.dot(normal, CO)) * (1 - np.cos(angle))
    )
    OXT = C + 1.25 * OXT_dir  # 1.25 Å from C

    serial = max(int(l[6:11]) for l in atom_lines) + 1
    res_name = c_line[17:20]
    chain_id = c_line[21]
    res_seq = c_line[22:26]
    i_code = c_line[26]

    oxt_line = format_pdb_line(serial, "OXT", res_name, chain_id, res_seq, i_code, OXT)

    # Insert after last ATOM line
    last_index = max(i for i, line in enumerate(model_lines) if line.startswith("ATOM"))
    model_lines.insert(last_index + 1, oxt_line)

    return model_lines


def add_hydrogens_to_model(model_lines, output_path):
    model_lines = ensure_oxt_atom(model_lines)

    temp_input = output_path + "_temp.pdb"
    with open(temp_input, "w") as f:
        f.writelines(model_lines)

    fixer = PDBFixer(filename=temp_input)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)

    with open(output_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    os.remove(temp_input)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_and_add_hydrogens.py <input_pdb> <output_dir>")
        sys.exit(1)

    input_pdb = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    models = split_models(input_pdb)

    for i, model in enumerate(models, start=1):
        out_file = os.path.join(output_dir, input_pdb[:-8] + "_model" + str(i) + "_h.pdb")
        print(f"Processing model {i} → {out_file}")
        add_hydrogens_to_model(model, out_file)

    print(f"Finished processing {len(models)} models.")
