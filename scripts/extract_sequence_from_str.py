# march 28, 2025
# this script grabs the sequence from a star file. i'll use this as the basis for a script to get ALL the sequences from ALL
# the star files i have. once i have the sequences, i'll run then through AF2 (somehow...). maybe i'll run then through
# alphaflow - i think that'd actually have a lot of value, since chemical shift predictors often work better when you
# predict for several structures and average the predictions.

import pynmrstar

def extract_sequence_from_nmrstar(file_path):
    entry = pynmrstar.Entry.from_file(file_path)

    for sf in entry:
        if sf.category == "entity":
            if "Polymer_seq_one_letter_code" in sf:
                raw_seq = sf["Polymer_seq_one_letter_code"][0]
                sequence = raw_seq.replace("\n", "").replace(" ", "")
                return sequence

    raise ValueError("Could not find sequence in 'Polymer_seq_one_letter_code'.")

# Example usage
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "example.str"
    try:
        seq = extract_sequence_from_nmrstar(path)
        print(f"\nExtracted sequence:\n{seq}")
    except Exception as e:
        print(f"Error: {e}")
