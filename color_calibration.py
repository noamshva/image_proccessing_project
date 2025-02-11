import cv2
import numpy as np
import yaml
import os

CALIBRATION_FILE = "calibration.yaml"

def sample_color_from_video(video_path, max_samples=3, color_name="Color"):
    """
    Allows the user to sample the HSV color of the ball from a video.
    Instructions:
      - Press 'p' to pause the video.
      - While paused, left-click on the ball to sample its color.
      - The video resumes immediately after sampling.
      - Press 'q' to quit calibration.

    Parameters:
      video_path (str): Path to the video file.
      max_samples (int): Maximum number of samples to collect.
      color_name (str): Name of the color being sampled (used in messages).

    Returns:
      List of samples. Each sample is a tuple of two lists:
      (lower_bound, upper_bound) representing HSV bounds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return []

    samples = []
    paused = False
    resume_flag = False
    window_name = f"Calibration - {color_name} (Press 'p' to pause, 'q' to quit)"
    frame = None  # To hold the current frame

    def click_handler(event, x, y, flags, param):
        nonlocal resume_flag, samples, frame
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert current frame to HSV and sample the pixel.

            # Define a tolerance range around the sampled HSV value:
            # - Hue: ±10 (ensuring values stay within 0–180)
            # - Saturation: Lower bound set to at least 50 (to avoid noise) and allow a drop of 50,
            #   with the upper bound fixed at 255.
            # - Value: Similarly, lower bound set to at least 50 and upper bound fixed at 255.
            
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            pixel = hsv_frame[y, x]
            h, s, v = int(pixel[0]), int(pixel[1]), int(pixel[2])
            lower_bound = [max(0, h - 10), max(50, s - 50), max(50, v - 50)]
            upper_bound = [min(180, h + 10), 255, 255]
            samples.append((lower_bound, upper_bound))
            print(f"Sampled {color_name} sample {len(samples)}: Lower={lower_bound}, Upper={upper_bound}")
            resume_flag = True
            # Remove the callback after sampling.
            cv2.setMouseCallback(window_name, lambda *args: None)

    while True:
        # Only read a new frame when not paused or after resuming.
        if not paused or resume_flag:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video.")
                break
            resume_flag = False

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('p'):
            # Pause and set mouse callback for sampling.
            paused = True
            cv2.setMouseCallback(window_name, click_handler)
            print(f"Paused for {color_name} calibration. Click on the ball to sample its color.")
            #print("the ball tocuh the screeen:",hit_wall)
        elif key == ord('q'):
            print("Exiting calibration for", color_name)
            break

        # If a sample was taken while paused, resume playback.
        if paused and resume_flag:
            paused = False

        # Exit if we have collected enough samples.
        if len(samples) >= max_samples:
            print(f"Collected {max_samples} samples for {color_name}.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return samples

def save_calibration(color_ranges, filename=CALIBRATION_FILE):
    """
    Save the calibration dictionary to a YAML file.
    Convert tuples to lists before saving.
    """
    data_to_save = {
        color: [[list(lower), list(upper)] for lower, upper in color_ranges[color]]
        for color in color_ranges
    }

    with open(filename, "w") as f:
        yaml.dump(data_to_save, f)
    
    print(f"Calibration data saved to {filename}")

    
def load_calibration(filename=CALIBRATION_FILE):
    """
    Load calibration data from a YAML file.
    Returns the color_ranges dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Calibration file '{filename}' not found.")
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    print(f"Calibration data loaded from {filename}")
    return data

def calibrate_colors(video_path, max_samples=3):
    """
    Calibrate colors for Blue and Yellow by sampling from the video.
    After calibration, the data is saved to a YAML file.
    
    Returns:
      A dictionary with keys 'blue' and 'yellow', each mapping to a list of sampled HSV bounds.
    """
    print("Calibrating Blue Ball Color:")
    blue_samples = sample_color_from_video(video_path, max_samples=max_samples, color_name="Blue")
    if not blue_samples:
        raise ValueError("No blue samples were collected.")

    print("Calibrating Yellow Ball Color:")
    yellow_samples = sample_color_from_video(video_path, max_samples=max_samples, color_name="Yellow")
    if not yellow_samples:
        raise ValueError("No yellow samples were collected.")

    color_ranges = {"blue": blue_samples, "yellow": yellow_samples}
    save_calibration(color_ranges)
    return color_ranges
