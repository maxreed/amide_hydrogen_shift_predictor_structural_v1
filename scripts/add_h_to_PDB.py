# march 28, 2025
# a simple way to add hydrogens to AF output, but you lose the b factors (which hold the pLDDT, which i want
# to keep).

from pdbfixer import PDBFixer
from openmm.app import PDBFile

fixer = PDBFixer(filename="pdb_files/bmr4023_3_rank1.pdb")
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingHydrogens(pH=7.4)

with open("fixed_with_h.pdb", "w") as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f)
