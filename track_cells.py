import logging
import re
import shutil
from pathlib import Path
from typing import List, Optional
import os
import warnings
# Force the solver to use only 1 thread to prevent segfaults
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MIP_THREADS"] = "1"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MIP_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

# Silence known docstring escape warnings emitted by higra on Python 3.12+.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"higra(\\..*)?$")

import numpy as np
import pandas as pd
import tifffile
from skimage.io import imread
from ultrack import MainConfig, track, to_tracks_layer, to_ctc
from ultrack.utils import estimate_parameters_from_labels
import argparse

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger(__name__)

# # ──────────────────────────── CONFIGURATION ────────────────────────────

# # Paths (Update these as needed)
# RAW_DIR = Path(r"D:/AG/230624DS30/230624DS30_p0001")
# SEG_DIR = Path(r"D:/AG/Segmentation/230624DS30/230624DS30_p0001")
# OUTPUT_DIR = Path(r"D:/AG/Tracking_Ultrack/230624DS30/230624DS30_p0001")

# # Clean up previous runs? (Ultrack uses an sqlite db that persists)
# OVERWRITE_DB = True 

def parse_args():
    parser = argparse.ArgumentParser(description="Cell tracking pipeline using Ultrack")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Path to raw images directory")
    parser.add_argument("--seg-dir", type=Path, required=True, help="Path to segmentation masks directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--overwrite-db", action="store_true", help="Clean up previous Ultrack database")
    return parser.parse_args()

# ──────────────────────────── HELPER FUNCTIONS ────────────────────────────

def get_timepoint(path: Path) -> int:
    """Extracts timepoint from filename (e.g., 'img_t005.tif' -> 5)."""
    # Looks for _t followed by digits
    match = re.search(r'_t(\d+)', path.name)
    if match:
        return int(match.group(1))
    
    # Fallback: try to find just a sequence of digits at the end
    match = re.search(r'(\d+)', path.stem)
    if match:
        return int(match.group(1))
    
    return 0

def load_time_series(folder: Path, filter_str: Optional[str] = None) -> np.ndarray:
    """
    Robustly loads a time series from a folder.
    - Supports .png, .tif, .tiff
    - Filters filenames containing `filter_str` (e.g. "w00" for BF)
    - Sorts by time index using regex
    - Handles RGB/RGBA by taking the first channel
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # 1. Gather files
    extensions = ("*.png", "*.tif", "*.tiff")
    files: List[Path] = []
    for ext in extensions:
        files.extend(folder.glob(ext))
    
    # 2. Filter (e.g., keep only 'w00' files)
    if filter_str:
        files = [f for f in files if filter_str in f.name]
    
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with filter '{filter_str}'")

    # 3. Sort by time
    files.sort(key=get_timepoint)
    
    LOG.info(f"Loading {len(files)} frames from {folder.name}...")

    # 4. Load and Stack
    stack = []
    for f in files:
        # Use tifffile for tifs (safer), skimage for pngs
        if f.suffix in ['.tif', '.tiff']:
            img = tifffile.imread(str(f))
        else:
            img = imread(str(f))

        # Handle Channel Dimensions (Y, X, C) -> (Y, X)
        if img.ndim == 3:
            # Assuming channel is last dimension for PNG/TIF
            img = img[..., 0] 
        
        stack.append(img)

    return np.stack(stack, axis=0)

# ──────────────────────────── MAIN PIPELINE ────────────────────────────

def main():
    args = parse_args()
    RAW_DIR = args.raw_dir
    SEG_DIR = args.seg_dir
    OUTPUT_DIR = args.output_dir
    OVERWRITE_DB = args.overwrite_db

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # 1. Load Data
    try:
        # Load Raw Images (Brightfield usually has "w00")
        images = load_time_series(RAW_DIR, filter_str="w00")
        
        # Load Segmentation Masks (No filter usually, or modify if specific tag exists)
        # Note: Ensure these are integer masks (0=background, 1..N=cells)
        labels = load_time_series(SEG_DIR, filter_str=None).astype(np.int32)
        
    except Exception as e:
        LOG.error(f"Error loading data: {e}")
        return

    if images.shape != labels.shape:
        LOG.warning(f"Shape mismatch! Images: {images.shape}, Labels: {labels.shape}")
        # Determine strict minimum length to proceed
        min_t = min(images.shape[0], labels.shape[0])
        images = images[:min_t]
        labels = labels[:min_t]
        LOG.info(f"Cropped data to matching length: {min_t}")

    # 2. Configure Ultrack
    config = MainConfig()
    
    # Working directory for the SQLite database and intermediate files
    config.data_config.working_dir = OUTPUT_DIR / "ultrack_db"
    
    if OVERWRITE_DB and config.data_config.working_dir.exists():
        shutil.rmtree(config.data_config.working_dir)
        LOG.info("Cleaned up previous Ultrack database.")
    config.data_config.working_dir.mkdir(parents=True, exist_ok=True)

    # Auto-config parameters based on your labels
    LOG.info("Estimating parameters from labels...")
    df_params = estimate_parameters_from_labels(labels, True)
    
    # We take the 1st and 99th percentiles to avoid outliers (tiny dots or huge artifacts)
    # setting the limits too strictly.
    estimated_min = df_params['area'].quantile(0.01)
    estimated_max = df_params['area'].quantile(0.99)
    
    config.segmentation_config.min_area = int(estimated_min)
    config.segmentation_config.max_area = int(estimated_max)
    
    LOG.info(f"Auto-configured Area: min={config.segmentation_config.min_area}, max={config.segmentation_config.max_area}")
    
    # Manual overrides (Adjust these if tracking is too loose/strict)
    config.linking_config.max_distance = 50.0  # Max pixels a cell moves per frame
    config.tracking_config.appear_weight = -0.5
    config.tracking_config.disappear_weight = -0.5
    config.tracking_config.division_weight = -0.1
    # config.tracking_config.solver_name = 'HEURISTIC' # 'HEURISTIC' is faster and often sufficient for simple datasets. Use 'MIP' for optimal but slower results.

    # 3. Run Tracking
    LOG.info("Running Ultrack...")
    
    # Note: passing 'detection=images' allows ultrack to use pixel intensity 
    # for better edge scoring, but it's optional.
    track(config, 
          labels=labels, 
        #   detection=images, 
          overwrite=True)
    
    LOG.info("Tracking finished.")

    # 4. Export Results
    
    # A. Napari Tracks (Graph & CSV)
    tracks_df, graph = to_tracks_layer(config)
    
    # Save as .npz for the viewer script
    npz_path = OUTPUT_DIR / "napari_tracks_data.npz"
    np.savez_compressed(
        npz_path, 
        tracks=tracks_df[["track_id", "t", "y", "x"]].values, 
        graph=graph
    )
    LOG.info(f"Saved Napari tracks to: {npz_path}")
    
    # Save CSV for analysis
    tracks_df.to_csv(OUTPUT_DIR / "tracks.csv", index=False)
    
    # B. Tracked Labels (Lineage-colored masks)
    # This converts the internal graph results back into a label stack
    # where pixel value = track_id
    LOG.info("Generating tracked label stack...")
    
    # Using CTC (Cell Tracking Challenge) format exporter to get the stack
    # This creates a folder of TIFs where ID is consistent over time
    ctc_output_dir = OUTPUT_DIR / "tracked_labels"
    to_ctc(config, ctc_output_dir, overwrite=True)
    
    # If you want a single stacked TIFF of the results:
    tracked_stack_files = sorted(ctc_output_dir.glob("mask*.tif"))
    if tracked_stack_files:
        tracked_stack = np.stack([tifffile.imread(str(f)) for f in tracked_stack_files])
        tifffile.imwrite(OUTPUT_DIR / "tracked_labels_stacked.tif", tracked_stack)
        LOG.info(f"Saved stacked tracked labels to {OUTPUT_DIR / 'tracked_labels_stacked.tif'}")

    print("\n" + "=" * 50)
    print(f"DONE. Results in {OUTPUT_DIR}")
    print(f"  - Tracks NPZ: {npz_path}")
    print(f"  - Tracks CSV: {OUTPUT_DIR / 'tracks.csv'}")
    print(f"  - Tracked Labels: {ctc_output_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()