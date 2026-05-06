# plot_speed_pdf_all_lanes.py
#
# Plot speed *probability distribution functions* (km/h) for selected lanes
# in ONE plot. Uses CSV from video_tracking_speed_lane_speed.py
# (or its cleaned version).

from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================== USER SETTINGS ============================================
# Path to the CSV file (raw or cleaned)
INPUT_CSV = "outputs/result_video1_clean_min10.csv"

# Lanes you want to analyze, e.g. [0], [1, 2, 3], [4, 5], etc.
LANES_TO_PLOT: List[int] = [5, 6]

# Histogram bin size in km/h
BIN_SIZE_KMH = 5.0
# ============================================================================


def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    output_dir = input_path.parent

    # Load data
    df = pd.read_csv(input_path)

    # Ensure correct dtypes
    df["lane"] = df["lane"].astype(int)
    df["speed_m_s"] = df["speed_m_s"].astype(float)

    # Convert speed to km/h
    df["speed_kmh"] = df["speed_m_s"] * 3.6

    # Keep only rows for the lanes we care about
    df = df[df["lane"].isin(LANES_TO_PLOT)]

    # Filter out zero speeds (often "not yet computed")
    df = df[df["speed_kmh"] > 0]

    if df.empty:
        raise ValueError("No non-zero speeds found for the selected lanes.")

    # Determine global bins based on maximum speed across all selected lanes
    max_speed = df["speed_kmh"].max()
    max_bin_edge = BIN_SIZE_KMH * (int(max_speed // BIN_SIZE_KMH) + 1)
    bins = np.arange(0, max_bin_edge + BIN_SIZE_KMH, BIN_SIZE_KMH)

    # Create a single figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram for each lane on the same axes, normalized as PDF
    for lane in LANES_TO_PLOT:
        df_lane = df[df["lane"] == lane]
        if df_lane.empty:
            print(f"No data for lane {lane}, skipping.")
            continue

        speeds = df_lane["speed_kmh"].values

        ax.hist(
            speeds,
            bins=bins,
            alpha=0.5,              # transparency so overlaps are visible
            label=f"Lane {lane}",
            edgecolor="black",
            density=True             # <-- PDF: area under curve = 1
        )

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Speed PDF for lanes {LANES_TO_PLOT}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Lanes")
    fig.tight_layout()

    # Save combined figure
    lanes_str = "_".join(str(l) for l in LANES_TO_PLOT)
    out_name = f"{input_path.stem}_lanes_{lanes_str}_speed_pdf.png"
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=300)
    print(f"Saved combined speed PDF plot to: {out_path}")

    # Optional: show the plot (comment out if you don't want it)
    plt.show()


if __name__ == "__main__":
    main()
