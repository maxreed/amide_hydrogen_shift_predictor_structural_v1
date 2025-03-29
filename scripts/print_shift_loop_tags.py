# march 28, 2025
# purely diagnostic script to see what's actually in a given star file.

import pynmrstar

def print_shift_loop_tags(star_file):
    entry = pynmrstar.Entry.from_file(star_file)

    for sf in entry:
        if sf.category == "assigned_chemical_shifts":
            print("Found assigned_chemical_shifts saveframe.\n")
            for i, loop in enumerate(sf.loops):
                tag_names = [tag.strip().lstrip('_') for tag in loop.get_tag_names()]
                print(f"Loop {i + 1} tags:")
                for tag in tag_names:
                    print(f"  - {tag}")
                print()
            return

    print("No assigned_chemical_shifts saveframe found.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python print_shift_loop_tags.py input.str")
        exit(1)

    print_shift_loop_tags(sys.argv[1])
