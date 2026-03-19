import os
import subprocess
import glob

# --- Configuration ---
# The folder containing all your position subfolders (e.g. 230624DS30_p0001, etc.)
EXPERIMENT_NAME = "230624DS30"  # Change this to your experiment name, e.g. "230624DS30"

# Raw image root containing position folders
BASE_RAW_DIR = f"/root/data/{EXPERIMENT_NAME}"

# Segmentation root containing matching position folders
BASE_SEG_DIR = f"/root/data/Segmentation_SAM3/{EXPERIMENT_NAME}"

# Where you want tracking outputs for each position to go
BASE_OUTPUT_DIR = f"/root/data/Tracking_Ultrack/{EXPERIMENT_NAME}"

# Path to your tracking script
SCRIPT_PATH = "track_cells.py"

# Optional behavior
OVERWRITE_DB = True

def main():
    # 1. Find all position folders (looking for pattern *_p*)
    # This finds folders like /root/data/230624DS30/230624DS30_p0001
    search_pattern = os.path.join(BASE_RAW_DIR, "*_p*")
    position_folders = sorted(glob.glob(search_pattern))

    # Filter to ensure they are actually directories
    position_folders = [p for p in position_folders if os.path.isdir(p)]  # and "270" in os.path.basename(p)]

    if not position_folders:
        print(f"No position folders found in {BASE_RAW_DIR}!")
        return

    print(f"Found {len(position_folders)} positions to process.")
    print("-" * 50)

    # 2. Iterate and Run
    for raw_path in position_folders:
        folder_name = os.path.basename(raw_path)  # e.g., "230624DS30_p0001"

        # Match segmentation folder by position name
        seg_path = os.path.join(BASE_SEG_DIR, folder_name)

        if not os.path.isdir(seg_path):
            print(f"⚠️  Segmentation folder not found for {folder_name}: {seg_path}")
            print("Skipping this position.\n")
            continue
        
        # Construct the specific output path for this position
        output_path = os.path.join(BASE_OUTPUT_DIR, folder_name)

        print(f"Processing: {folder_name}")
        print(f"Raw:    {raw_path}")
        print(f"Seg:    {seg_path}")
        print(f"Output: {output_path}")

        # Construct the command
        cmd = [
            "python",
            "-W",
            r"ignore::SyntaxWarning:higra(\\..*)?$",
            SCRIPT_PATH,
            "--raw-dir", raw_path,
            "--seg-dir", seg_path,
            "--output-dir", output_path,
        ]

        if OVERWRITE_DB:
            cmd.append("--overwrite-db")

        try:
            # Run the command and wait for it to finish
            # check=True raises an error if the script fails, stopping the batch
            subprocess.run(cmd, check=True)
            print(f"✅ Finished {folder_name}\n")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {folder_name}. Stopping batch.")
            print(f"Error details: {e}")
            break
        except KeyboardInterrupt:
            print("\n🛑 Batch processing stopped by user.")
            break

    print("-" * 50)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()