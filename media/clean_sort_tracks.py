# clean_sort_tracking_csv.py
#
# Post-process result CSV from video_tracking_speed_lane_label_speed.py:
#   1) Remove track IDs that appear in fewer than MIN_FRAMES rows
#   2) Sort remaining data by ID, then by time (frame index)

from pathlib import Path
from typing import Optional

import pandas as pd

# === USER SETTINGS ============================================================
# Path to the CSV produced by video_tracking_speed_lane_speed.py
INPUT_CSV = "outputs/result_video1.csv"

# Minimum number of frames an ID must have to be kept
MIN_FRAMES = 10

# Optional custom output path.
# If None, will create "<stem>_clean_min<MIN_FRAMES>.csv" in the same folder.
OUTPUT_CSV: Optional[str] = None
# ==============================================================================


def clean_and_sort_csv(input_csv: str,
                       min_frames: int = 10,
                       output_csv: Optional[str] = None) -> Path:
    """
    - Load tracking CSV
    - Remove IDs with fewer than min_frames rows
    - Sort by id, then by frame (time)
    - Save new CSV

    Returns the output CSV path.
    """
    input_path = Path(input_csv)

    if output_csv is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_clean_min{min_frames}.csv"
        )
    else:
        output_path = Path(output_csv)

    # Read CSV
    df = pd.read_csv(input_path)

    # Ensure numeric types for sorting/filtering
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)

    # 1) Filter out IDs with fewer than min_frames rows
    df_clean = df.groupby("id").filter(lambda g: len(g) >= min_frames)

    # 2) Sort by ID, then by frame (frame corresponds to time order)
    df_clean = df_clean.sort_values(by=["id", "frame"]).reset_index(drop=True)

    # Save cleaned CSV
    df_clean.to_csv(output_path, index=False)

    print("=== Cleaning & Sorting Done ===")
    print(f"Input file:       {input_path}")
    print(f"Output file:      {output_path}")
    print(f"Min frames / ID:  {min_frames}")
    print(f"Track IDs kept:   {df_clean['id'].nunique()}")
    print(f"Total rows kept:  {len(df_clean)}")

    return output_path


if __name__ == "__main__":
    clean_and_sort_csv(INPUT_CSV, MIN_FRAMES, OUTPUT_CSV)
