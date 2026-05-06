# video_tracking_speed_lane_speed.py
# Track vehicles, assign lanes, estimate speed over a 0.5 s window,
# draw speed on the video, and export per-frame CSV.

from pathlib import Path
import csv
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import math

# Load YOLO model (with the trained weights)
model = YOLO("best.pt")

# --- User / model settings ---
VIDEO_PATH = "video1.mp4"     # input video
CONF = 0.65                   # detection confidence threshold
IOU = 0.5                     # NMS IoU threshold
IMGSZ = 1088                  # inference resolution
MAX_DET = 300                 # max detections per frame
DEVICE = "mps"                # computation device (e.g., "cpu", "0", "mps")
TRACKER = "bytetrack.yaml"    # tracker configuration
PERSIST = True                # keep track IDs across frames

print("Using device:", DEVICE)

# === Video info ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

# Get video metadata: FPS + frame size
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Video size:", width, "x", height, "FPS:", fps)
cap.release()

# === Pixel → meter conversion ===
# Assumption: full image width corresponds to 70 m of road.
ROAD_LENGTH_M = 70.0
meters_per_pixel = ROAD_LENGTH_M / float(width)

# Use a 0.5 s window for speed estimation (in number of frames)
frames_per_half_second = int(round(fps / 2.0))

# === Output paths ===
out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)
stem = Path(VIDEO_PATH).stem
video_out = out_dir / f"result_{stem}.mp4"
csv_out = out_dir / f"result_{stem}.csv"

# Prepare video writer for annotated output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))

print(f"Saving annotated video to: {video_out}")
print(f"Saving per-frame CSV to:   {csv_out}")


def get_lane_from_y(cy: float) -> int:
    """
    Map the vertical center of the box (cy) to a lane index based on y-coordinates.
    Lane definitions:
      0: cy < 155
      1: 155 <= cy < 255
      2: 255 <= cy < 350
      3: 350 <= cy < 455
      4: 455 <= cy < 600
      5: 600 <= cy < 705
      6: 705 <= cy < 810
      7: cy >= 810
    Returns -1 if out of range.
    """
    if cy < 155:
        return 0
    elif 155 <= cy < 255:
        return 1
    elif 255 <= cy < 350:
        return 2
    elif 350 <= cy < 455:
        return 3
    elif 455 <= cy < 600:
        return 4
    elif 600 <= cy < 705:
        return 5
    elif 705 <= cy < 810:
        return 6
    elif cy >= 810:
        return 7
    else:
        return -1


# Position history:
# For each track ID (tid), we store a deque of (frame_idx, cx, cy)
# limited to approximately the last 0.5 seconds of motion.
pos_history = defaultdict(lambda: deque())


def compute_speed_m_s(
    tid: int,
    frame_idx: int,
    cx: float,
    cy: float,
    pos_history,
    frames_per_half_second: int,
    fps: float,
    meters_per_pixel: float,
) -> float:
    """
    Update position history for this track ID and return speed (m/s)
    computed over the last ~0.5 seconds.

    Logic:
      - If tid is invalid (< 0), return 0.
      - Append current (frame_idx, cx, cy) to the history.
      - Drop any history entries older than 0.5 s (using frames_per_half_second).
      - If we have at least 2 points in the window:
           speed = distance_between_first_and_last / time_between_first_and_last
        where distance is computed in meters using meters_per_pixel.
    """
    # Ignore invalid track IDs
    if tid < 0:
        return 0.0

    # Retrieve deque for this ID and append current position
    hist = pos_history[tid]
    hist.append((frame_idx, cx, cy))

    # Remove entries older than the desired time window (~0.5 s)
    while hist and (frame_idx - hist[0][0]) > frames_per_half_second:
        hist.popleft()

    # Need at least two points to compute a speed
    if len(hist) < 2:
        return 0.0

    # Take oldest and newest samples in the current window
    f0, x0, y0 = hist[0]
    f1, x1h, y1h = hist[-1]
    frame_delta = f1 - f0
    if frame_delta <= 0:
        return 0.0

    # Convert frame difference to seconds
    dt = frame_delta / fps  # seconds

    # Distance in pixels between first and last positions
    dx_px = x1h - x0
    dy_px = y1h - y0
    dist_px = math.sqrt(dx_px * dx_px + dy_px * dy_px)

    # Convert pixel distance to meters and compute speed
    dist_m = dist_px * meters_per_pixel
    speed_m_s = dist_m / dt
    return speed_m_s


# === Main CSV + tracking loop ===
with open(csv_out, "w", newline="") as fcsv:
    w = csv.writer(fcsv)

    # CSV header
    w.writerow([
        "frame", "time_s", "id", "class", "conf",
        "lane", "speed_m_s",
        "x1", "y1", "x2", "y2", "cx", "cy"
    ])

    frame_idx = -1  # manual frame index (independent from internal indices)

    # Ultralytics streaming tracking:
    # model.track yields a result object "r" for each frame.
    for r in model.track(
        source=VIDEO_PATH,
        conf=CONF,
        iou=IOU,
        imgsz=[IMGSZ],
        max_det=MAX_DET,
        device=DEVICE,
        tracker=TRACKER,
        persist=PERSIST,
        stream=True
    ):
        frame_idx += 1
        time_s = frame_idx / fps          # current timestamp in seconds
        names = r.names                   # class name dictionary

        # Start from original frame for our own drawing (no default r.plot())
        annotated = r.orig_img.copy()

        # Iterate over all detected boxes in the current frame
        for b in r.boxes:
            cls_id = int(b.cls)           # numeric class ID
            cls = names[cls_id]           # class name (string)
            conf = float(b.conf[0])       # confidence score

            # Bounding box coordinates (top-left and bottom-right)
            x1, y1, x2, y2 = map(float, b.xyxy[0])

            # Center of the bounding box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Track ID from tracker; -1 if no ID
            tid = int(b.id[0]) if (hasattr(b, "id") and b.id is not None) else -1

            # Lane index based on vertical position
            lane = get_lane_from_y(cy)

            # --- Speed over ~0.5 s window via helper function ---
            speed_m_s = compute_speed_m_s(
                tid, frame_idx, cx, cy,
                pos_history, frames_per_half_second,
                fps, meters_per_pixel
            )

            # Write CSV row (with numeric values formatted to 2 decimals where needed)
            w.writerow([
                frame_idx,
                f"{time_s:.2f}",          # time_s
                tid,
                cls,
                f"{conf:.2f}",            # conf
                lane,
                f"{speed_m_s:.2f}",       # speed_m_s
                f"{x1:.2f}",              # x1
                f"{y1:.2f}",              # y1
                f"{x2:.2f}",              # x2
                f"{y2:.2f}",              # y2
                f"{cx:.2f}",              # cx
                f"{cy:.2f}",              # cy
            ])

            # === Draw bbox and label with SPEED (no probability) ===

            # Convert coordinates to integers for OpenCV drawing
            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1_i, y1_i),
                (x2_i, y2_i),
                (0, 255, 0),  # green box
                2
            )

            # Convert speed to km/h for display
            speed_kmh = speed_m_s * 3.6
            if tid >= 0:
                label = f"{cls} ID{tid} {speed_kmh:.1f} km/h"
            else:
                label = f"{cls} {speed_kmh:.1f} km/h"

            # Draw text background rectangle to improve readability
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x, text_y = x1_i, max(0, y1_i - 5)
            cv2.rectangle(
                annotated,
                (text_x, text_y - th - baseline),
                (text_x + tw, text_y + baseline),
                (0, 255, 0),
                thickness=-1
            )

            # Put text label on top of the box
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),   # black text
                2,
                cv2.LINE_AA
            )

        # Save annotated frame to output video
        writer.write(annotated)

        # Show live preview window; press 'q' to stop early
        cv2.imshow("YOLO Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Cleanup: release video writer and close windows
writer.release()
cv2.destroyAllWindows()
print("Done.")
