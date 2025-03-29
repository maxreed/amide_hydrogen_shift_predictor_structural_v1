# max reed (well, mostly chatgpt 4o but i directed it and edited its work)
# march 28, 2025
# this script adds hydrogens to a PDB file while also transferring the beta factors from the initial file to the new file.
# there's a very simple way to add hydrogens, but usually you lose the b factors. a quirt here is that the input script (usually)
# has distances in angstroms, and the output of this has distances in nanometers - be aware.
# this script exists because AF output doesn't have hydrogens, and they need to be added because ultimately i'm going to be
# extracting features from the PDB so all atoms need to be included.
# oh yes, and also recall that the b factors aren't really b factors, they're the pLDDT (which i want to use later).

from pdbfixer import PDBFixer
from openmm.app import PDBFile
from Bio.PDB import PDBParser
import numpy as np

def add_hydrogens_with_b_factors(input_pdb, output_pdb, pH=7.4):
    # Step 1: Parse original PDB for B-factors
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("orig", input_pdb)
    b_factors = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    key = (chain.id, residue.id[1], residue.resname.strip(), atom.name)
                    b_factors[key] = atom.bfactor

    # Step 2: Fix with PDBFixer and add hydrogens
    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingHydrogens(pH=pH)

    # Step 3: Write to file with custom B-factors
    with open(output_pdb, "w") as f:
        for atom, pos in zip(fixer.topology.atoms(), fixer.positions):
            res = atom.residue
            key = (res.chain.id, res.index + 1, res.name, atom.name)

            if key in b_factors:
                b = b_factors[key]
            elif atom.element.symbol == 'H':
                # Try to copy B-factor from bonded N
                bonded_N_key = (res.chain.id, res.index + 1, res.name, 'N')
                b = b_factors.get(bonded_N_key, 0.00)
            else:
                b = 0.00

            f.write(
                "ATOM  {:5d} {:^4s} {:>3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00{:6.2f}           {:>2s}\n".format(
                    atom.index % 100000, atom.name, res.name, res.chain.id,
                    res.index + 1, pos.x, pos.y, pos.z, b, atom.element.symbol
                )
            )

    print(f"✅ Hydrogens added and B-factors handled: {output_pdb}")

add_hydrogens_with_b_factors("pdb_files/bmr4023_3_rank1.pdb", "pdb_files/bmr4023_3_rank1_h_3.pdb")

