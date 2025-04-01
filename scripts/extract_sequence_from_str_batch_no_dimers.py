import os
import csv
import pynmrstar

def extract_sequence_from_nmrstar(file_path):
    entry = pynmrstar.Entry.from_file(file_path)

    # make the title lower case for simplicity.
    this_title = entry[0]['title'][0].lower()
    
    sequence = ''
    
    for sf in entry:
        if sf.category == "entity":
            if "Polymer_seq_one_letter_code" in sf:
                raw_seq = sf["Polymer_seq_one_letter_code"][0]
                if len(sequence)>1:
                    # uncomment this line if you want things with more than 1 molecule.
                    #sequence = sequence + ":"
                    # comment out this next line if you want things with more than 1 molecule.
                    raise ValueError("Star file contained more than a single polypeptide")
                sequence = sequence + raw_seq.replace("\n", "").replace(" ", "")
    
    # i put the test for the title dimer here so i can see what is a dimer but isn't marked as one (for my curiosity as to how bad the
    # BMRB is for this).
    if 'dimer' in this_title:
    	print(this_title)
    	raise ValueError("This star file is for a dimer.")

    return sequence

    raise ValueError("Could not find sequence in 'Polymer_seq_one_letter_code'.")

def extract_sequences_from_directory(directory_path, output_csv):
    data = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".str"):
            full_path = os.path.join(directory_path, filename)
            try:
                sequence = extract_sequence_from_nmrstar(full_path)
                # i added this next line to purge sequences with unusual amino acids (which are marked with X).
                # the purge against U gets rid of two random RNA sequences that are for some reason present...
                # which is quite odd because the list i got from the BMRB was explicitly protein-only.
                if "X" not in sequence and "U" not in sequence:
                    data.append({"filename": filename, "sequence": sequence})
            except Exception as e:
                print(f"Failed to extract from {filename}: {e}")

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "sequence"])
        writer.writeheader()
        writer.writerows(data)

# Example usage
if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    output_file = sys.argv[2] if len(sys.argv) > 2 else "sequences.csv"
    extract_sequences_from_directory(directory, output_file)
