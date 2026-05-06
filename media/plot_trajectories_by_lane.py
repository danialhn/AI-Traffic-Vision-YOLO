# plot_trajectories_by_lane.py
#
# Plot trajectories (cx over time, in meters) for vehicles in selected lanes.
# Uses the CSV from video_tracking_speed_lane_speed.py (or its cleaned version).
# One PNG file per lane will be saved.

from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

# ================== USER SETTINGS ============================================
# Path to the CSV file (raw or cleaned)
INPUT_CSV = "outputs/result_video1_clean_min10.csv"

# Lanes you want to plot, e.g. [0], [1], [4], etc.
LANES_TO_PLOT: List[int] = [1]

# Image width in pixels (for cx flip)
IMAGE_WIDTH = 1920

# Road length in meters that corresponds to the full image width
ROAD_LENGTH_M = 70.0
# ============================================================================


def transform_position_m(cx: float, lane: int, image_width: int, meters_per_pixel: float) -> float:
    """
    Compute the position along the road in meters from cx, depending on lane.

    For lanes 0-3: position = (image_width - cx) * meters_per_pixel
    For lanes 4-7: position = cx * meters_per_pixel
    """
    # --- based on lane ---
    if lane in (0, 1, 2, 3):
        cx_effective = image_width - cx
    else:
        cx_effective = cx

    return cx_effective * meters_per_pixel

def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    # Folder where plots will be saved (same as CSV)
    output_dir = input_path.parent

    # Load data
    df = pd.read_csv(input_path)

    # Ensure correct dtypes
    df["id"] = df["id"].astype(int)
    df["lane"] = df["lane"].astype(int)
    df["cx"] = df["cx"].astype(float)
    df["time_s"] = df["time_s"].astype(float)

    # Sort globally to be safe
    df = df.sort_values(by=["id", "time_s"])

    # Pixel → meter factor
    meters_per_pixel = ROAD_LENGTH_M / float(IMAGE_WIDTH)  # 70 / 1920

    for lane in LANES_TO_PLOT:
        df_lane = df[df["lane"] == lane].copy()
        if df_lane.empty:
            print(f"No data for lane {lane}, skipping.")
            continue

        # Compute position in meters for this lane
        df_lane["pos_m"] = df_lane.apply(
            lambda row: transform_position_m(
                row["cx"],
                int(row["lane"]),
                IMAGE_WIDTH,
                meters_per_pixel,
            ),
            axis=1,
        )

        # Create plot for this lane
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each vehicle's trajectory in this lane
        for vid, group in df_lane.groupby("id"):
            group_sorted = group.sort_values("time_s")
            ax.plot(group_sorted["time_s"], group_sorted["pos_m"], label=f"ID {vid}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position along road (m)")
        ax.set_title(f"Vehicle trajectories in lane {lane}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True)
        fig.tight_layout()

        # Save figure
        out_name = f"{input_path.stem}_lane{lane}_trajectories.png"
        out_path = output_dir / out_name
        fig.savefig(out_path, dpi=300)
        print(f"Saved plot for lane {lane} to: {out_path}")

    # Show all figures (optional; comment out if you only want saving)
    plt.show()


if __name__ == "__main__":
    main()
