# march 28, 2025
# this takes the H shifts out of a star file. this is important because these will be used as the y data during training.

import pynmrstar
import csv

def extract_amide_shifts(star_file, output_file):
    entry = pynmrstar.Entry.from_file(star_file)

    # Find the assigned_chemical_shifts saveframe
    shift_frame = None
    for sf in entry:
        if sf.category == "assigned_chemical_shifts":
            shift_frame = sf
            break
    if not shift_frame:
        raise ValueError("No 'assigned_chemical_shifts' saveframe found.")

    # Look for the loop with Atom_chem_shift.Atom_ID and Atom_chem_shift.Val
    correct_loop = None
    for loop in shift_frame.loops:
        tag_names = [tag.strip().lstrip('_') for tag in loop.get_tag_names()]
        if 'Atom_chem_shift.Atom_ID' in tag_names and 'Atom_chem_shift.Val' in tag_names:
            correct_loop = loop
            break

    if not correct_loop:
        raise ValueError("No chemical shift loop with expected Atom_chem_shift fields found.")

    tags = [tag.strip().lstrip('_') for tag in correct_loop.get_tag_names()]

    def get_index(field_name):
        if field_name not in tags:
            raise ValueError(f"Missing expected tag: {field_name}")
        return tags.index(field_name)

    seq_id_idx = get_index('Atom_chem_shift.Seq_ID')
    atom_id_idx = get_index('Atom_chem_shift.Atom_ID')
    val_idx = get_index('Atom_chem_shift.Val')

    shifts_by_residue = {}

    for row in correct_loop:
        atom_name = row[atom_id_idx].strip()
        if atom_name not in ('N'):
            continue
        try:
            seq_id = int(row[seq_id_idx])
            shift_val = float(row[val_idx])
            shifts_by_residue[seq_id] = shift_val
        except (ValueError, TypeError):
            continue

    # Extract sequence
    sequence = None
    for sf in entry:
        if sf.category == "entity" and "Polymer_seq_one_letter_code" in sf:
            raw_seq = sf["Polymer_seq_one_letter_code"][0]
            sequence = raw_seq.replace("\n", "").replace(" ", "")
            break
    if sequence is None:
        raise ValueError("Sequence not found in entity saveframe.")

    # Write output
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Residue_Index", "Residue", "Amide_N_Shift"])
        for i, aa in enumerate(sequence, start=1):
            shift = shifts_by_residue.get(i, float('nan'))
            writer.writerow([i, aa, shift])

    print(f"✅ Wrote output to {output_file}")


# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_N_shifts_from_str.py input.str output.csv")
        exit(1)

    extract_amide_shifts(sys.argv[1], sys.argv[2])
