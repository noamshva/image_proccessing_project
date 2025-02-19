
"""
ball_calibration.py

This script performs an interactive calibration of the ball radius at five specific points on the projector wall.
For each ball color (blue and yellow), the script displays a marker at the calibration point and waits for the ball 
to be placed in that position. When the ball is detected (using color thresholding based on HSV ranges defined in 
calibration.yaml) and its center is within a specified tolerance of the marker, the ball's measured radius is recorded.
Calibration data is then saved to ball_calibration.yaml.
"""

import cv2
import numpy as np
import yaml
import time

# Provided projector wall coordinates (in the captured frame's coordinate system)
SCREEN_COORDS = np.array([
    [48, 95],    # top-left
    [542, 104],  # top-right
    [522, 386],  # bottom-right
    [65, 390]    # bottom-left
], dtype="float32")

# Define calibration points using SCREEN_COORDS:
# For the corners, we use the provided coordinates directly.
# The center is computed as the average of the four corner points.
calibration_points = {
    "top_left": tuple(SCREEN_COORDS[0].astype(int)),
    "top_right": tuple(SCREEN_COORDS[1].astype(int)),
    "bottom_right": tuple(SCREEN_COORDS[2].astype(int)),
    "bottom_left": tuple(SCREEN_COORDS[3].astype(int)),
    "center": tuple(np.mean(SCREEN_COORDS, axis=0).astype(int))
}

# Tolerance (in pixels) for the ball's center to be considered "on" the calibration point
CENTER_TOLERANCE = 50

# Minimal area threshold to consider a detection valid
AREA_THRESHOLD = 200

def load_color_calibration(calib_file="calibration.yaml"):
    """
    Loads the color calibration data from a YAML file.
    Expecting a structure with keys 'blue' and 'yellow', each containing a list of lower/upper HSV ranges.
    We'll use the first provided pair for each color.
    """
    try:
        with open(calib_file, "r") as f:
            calib_data = yaml.safe_load(f)
        print("Color calibration data loaded.")
    except Exception as e:
        print("Failed to load calibration data:", e)
        # Fallback defaults if file loading fails
        calib_data = {
            "blue": [[[97, 141, 146], [117, 255, 255]]],
            "yellow": [[[14, 197, 152], [34, 255, 255]]]
        }
    return calib_data

def detect_ball(frame, lower_bound, upper_bound, min_size=5, max_size=100):
    """
    Detects a ball in the given frame based on the provided lower and upper HSV bounds.
    Returns the ball's center and radius if found, otherwise (None, None).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_bound, dtype=np.uint8)
    upper = np.array(upper_bound, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > AREA_THRESHOLD:
            (x, y), radius = cv2.minEnclosingCircle(c)
            if min_size <= radius <= max_size:
                return (int(x), int(y)), int(radius)
    return None, None

def main():
    # Load color calibration thresholds for ball detection
    calib_data = load_color_calibration()
    # Use the first HSV range for each color
    color_thresholds = {
        "blue": calib_data.get("blue", [[[97, 141, 146], [117, 255, 255]]])[0],
        "yellow": calib_data.get("yellow", [[[14, 197, 152], [34, 255, 255]]])[0]
    }
    
    # Initialize the video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    # Dictionary to store calibration results for each color
    ball_calibration = {"blue": {}, "yellow": {}}

    # Create a named window for calibration display
    window_name = "Ball Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # For each color, calibrate at each calibration point
    for color in ["blue", "yellow"]:
        print(f"\nStarting calibration for {color} ball.")
        lower_bound, upper_bound = color_thresholds[color]
        for point_name, marker in calibration_points.items():
            calibrated = False
            print(f"Place the {color} ball at the {point_name} position {marker} and press 'c' to capture when ready.")
            while not calibrated:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Create a copy for display and draw the calibration marker
                display_frame = frame.copy()
                cv2.circle(display_frame, marker, 10, (0, 0, 255), -1)  # Red marker
                cv2.putText(display_frame, f"{color.upper()} {point_name}", (marker[0]+15, marker[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Try detecting the ball in the frame using the specified HSV thresholds
                ball_center, ball_radius = detect_ball(frame, lower_bound, upper_bound)
                if ball_center is not None:
                    # Draw detected ball on display frame
                    cv2.circle(display_frame, ball_center, ball_radius, (255, 0, 0), 2)
                    # Check if the detected ball is near the calibration marker
                    dist = np.linalg.norm(np.array(ball_center) - np.array(marker))
                    cv2.putText(display_frame, f"Dist: {int(dist)}", (ball_center[0] + 15, ball_center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if dist <= CENTER_TOLERANCE:
                        cv2.putText(display_frame, "In Position", (marker[0] - 50, marker[1] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "Adjust Ball", (marker[0] - 50, marker[1] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "No ball detected", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and ball_center is not None:
                    # Only capture if the ball is within tolerance of the marker
                    dist = np.linalg.norm(np.array(ball_center) - np.array(marker))
                    if dist <= CENTER_TOLERANCE:
                        ball_calibration[color][point_name] = ball_radius
                        print(f"Calibrated {color} ball at {point_name}: radius = {ball_radius}")
                        calibrated = True
                    else:
                        print("Ball not in position. Please adjust and try again.")
                elif key == ord('q'):
                    print("Calibration aborted by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            # Small delay before moving to next point
            time.sleep(1)

    # Save calibration results to ball_calibration.yaml
    with open("ball_calibration.yaml", "w") as f:
        yaml.dump(ball_calibration, f)
    print("\nCalibration complete. Data saved to ball_calibration.yaml")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
