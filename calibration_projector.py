import cv2
import yaml
import numpy as np

# Global list to store the calibration points
calib_points = []

def click_event(event, x, y, flags, param):
    global calib_points, frame
    # On left mouse button click, record the point and draw a circle
    if event == cv2.EVENT_LBUTTONDOWN:
        calib_points.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{len(calib_points)}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Calibration", frame)

def main():
    global frame
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", click_event)

    print("Please click 4 points on the projector screen in order (top-left, top-right, bottom-right, bottom-left).")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        # Once 4 points are captured, break out of the loop.
        if len(calib_points) == 4:
            break
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(calib_points) != 4:
        print("Calibration failed: 4 points were not selected.")
        return

    # Optionally, order the points if needed.
    points_array = np.array(calib_points, dtype=np.float32)
    # Save the calibration data to a YAML file.
    calib_data = {"projector_points": points_array.tolist()}
    with open("projector_calibration.yaml", "w") as f:
        yaml.dump(calib_data, f)
    print("Calibration complete. Data saved to projector_calibration.yaml")

if __name__ == "__main__":
    main()

