import os
import csv
import shutil
import pynmrstar

def extract_conditions_and_title(filepath):
    entry = pynmrstar.Entry.from_file(filepath)
    conditions = {
        "pH": None,
        "temperature": None,
        "ionic_strength": None,
        "pressure": None
    }
    title = None

    try:
        title_sf = entry.get_saveframes_by_category('entry_information')[0]
        title = title_sf.get_tag('_Entry.Title')
    except Exception:
        title = "N/A"

    for sf in entry.get_saveframes_by_category('sample_conditions'):
        try:
            loop = sf.get_loop_by_category('sample_condition_variable')
        except KeyError:
            continue

        headers = loop.get_tag_names()
        if '_Sample_condition_variable.Type' not in headers or '_Sample_condition_variable.Val' not in headers:
            continue

        idx_type = headers.index('_Sample_condition_variable.Type')
        idx_val = headers.index('_Sample_condition_variable.Val')

        for row in loop.data:
            if isinstance(row, dict):
                var_name = row.get('_Sample_condition_variable.Type', '').lower()
                val_str = row.get('_Sample_condition_variable.Val')
            else:
                var_name = row[idx_type].lower()
                val_str = row[idx_val]

            if not val_str or val_str in ['.', '?']:
                continue

            try:
                val = float(val_str)
            except ValueError:
                continue

            if var_name == 'ph':
                conditions["pH"] = val
            elif 'ionic strength' in var_name:
                conditions["ionic_strength"] = val
            elif 'press' in var_name:
                conditions["pressure"] = val
            elif 'temp' in var_name:
                conditions["temperature"] = val

    return title, conditions

def filter_files(input_dir, output_dir, report_csv):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for fname in os.listdir(input_dir):
        if not fname.endswith(".str"):
            continue

        fpath = os.path.join(input_dir, fname)
        try:
            title, conds = extract_conditions_and_title(fpath)
        except Exception as e:
            print(f"[!] Failed to parse {fname}: {e}")
            title = "ERROR"
            conds = {
                "pH": None,
                "temperature": None,
                "ionic_strength": None,
                "pressure": None
            }

        result = "PASS"
        pH = conds["pH"]
        temp = conds["temperature"]
        pressure = conds["pressure"]

        if pH is None or pH < 6 or pH > 8:
            result = "FAIL"
        if temp is None or temp < 288 or temp > 310:
            result = "FAIL"
        if pressure is not None and abs(pressure - 1.0) > 0.1:
            result = "FAIL"

        if result == "PASS":
            shutil.copy(fpath, os.path.join(output_dir, fname))

        rows.append({
            "star_id": fname,
            "title": title,
            "pH": pH,
            "temperature": temp,
            "ionic_strength": conds["ionic_strength"],
            "pressure": pressure,
            "result": result
        })

    with open(report_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Filtered {len(rows)} files. {sum(1 for r in rows if r['result'] == 'PASS')} passed and were copied to {output_dir}.")
    print(f"Report written to {report_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', default='good_star_files')
    parser.add_argument('--report_csv', default='star_file_conditions.csv')
    args = parser.parse_args()

    filter_files(args.input_dir, args.output_dir, args.report_csv)
