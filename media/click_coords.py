import cv2

# ---- Global list to store clicked points ----
clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points

    # Left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked at: (x={x}, y={y})")

        # Draw a small red circle on the clicked point
        frame = param  # the image passed as "param"
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Refresh the window
        cv2.imshow("First Frame - Click to get coordinates", frame)


def main():
    video_path = "video1.mp4"  # <-- change this to your video path

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Read the first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read first frame from video.")
        return

    # Show the first frame
    window_name = "First Frame - Click to get coordinates"
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, frame)

    # Set mouse callback: pass the frame as "param"
    cv2.setMouseCallback(window_name, click_event, frame)

    print("Instructions:")
    print(" - Click on the image to get pixel coordinates.")
    print(" - Press 'q' or ESC to close the window.")

    # Wait for user to press 'q' or ESC to exit
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 = ESC
            break

    cv2.destroyAllWindows()

    # Print all collected points
    print("\nAll clicked points:")
    for i, (x, y) in enumerate(clicked_points, 1):
        print(f"{i}: (x={x}, y={y})")


if __name__ == "__main__":
    main()
