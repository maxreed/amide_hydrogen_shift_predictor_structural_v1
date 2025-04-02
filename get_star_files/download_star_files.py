import os
import requests

def download_star_files(txt_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split('\t')
            bmrb_id = parts[0]

            if not bmrb_id.isdigit():
                print(f"Skipping invalid BMRB ID: {bmrb_id}")
                continue

            fname = f"bmr{bmrb_id}_3.str"
            url = f"https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr{bmrb_id}/{fname}"
            local_path = os.path.join(output_dir, fname)

            if os.path.exists(local_path):
                print(f"[✓] Already exists: {fname}")
                continue

            try:
                print(f"[↓] Downloading: {fname}...")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(local_path, 'wb') as out_file:
                        out_file.write(response.content)
                    print(f"[✔] Saved: {fname}")
                else:
                    print(f"[✗] Failed ({response.status_code}): {fname}")
            except Exception as e:
                print(f"[!] Error downloading {fname}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Path to CNH_withPDB.txt')
    parser.add_argument('--output_dir', required=True, help='Directory to save .str files')
    args = parser.parse_args()

    download_star_files(args.input_file, args.output_dir)
