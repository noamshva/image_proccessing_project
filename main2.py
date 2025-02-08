# main.py
import cv2
import numpy as np
import threading
import time
from queue import Queue
from game2 import Game, WIDTH, HEIGHT

# -------------------- Helper Functions for Camera Processing --------------------
def order_points(pts):
    """
    Order points in the following order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_projector_screen_transform(frame, width, height):
    """
    Detect the projector screen in the frame and return the perspective transform matrix.
    Only returns a valid transform if the detected quadrilateral is sufficiently large.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    frame_area = frame.shape[0] * frame.shape[1]
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            if cv2.contourArea(approx) < 0.5 * frame_area:
                continue
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            return M
    return None

def detect_ball(frame, lower_color, upper_color):
    """
    Detect a ball of a given HSV color range.
    Returns the center coordinate and the radius.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:
            (x, y), radius = cv2.minEnclosingCircle(c)
            return (int(x), int(y)), float(radius)
    return None, None

# --- Updated functions for detecting multiple red and green balloons ---
def detect_red_balloon(frame):
    """
    Detect all red balloons in the frame.
    Returns a list of tuples: [((x, y), radius), ...]
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_balloons = []
    for c in contours:
        if cv2.contourArea(c) > 100:
            (x, y), radius = cv2.minEnclosingCircle(c)
            red_balloons.append(((int(x), int(y)), float(radius)))
    return red_balloons

def detect_green_balloon(frame):
    """
    Detect all green balloons in the frame.
    Returns a list of tuples: [((x, y), radius), ...]
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_balloons = []
    for c in contours:
        if cv2.contourArea(c) > 100:
            (x, y), radius = cv2.minEnclosingCircle(c)
            green_balloons.append(((int(x), int(y)), float(radius)))
    return green_balloons

# -------------------- Camera Processing Function --------------------
def camera_processing(game_instance, event_queue):
    cap = None
    for i in range(5):
        print(f"Trying camera index: {i}")
        cap_candidate = cv2.VideoCapture(i)
        if cap_candidate.isOpened():
            cap = cap_candidate
            print(f"Opened camera index: {i}")
            break
    if cap is None or not cap.isOpened():
        print("Could not open camera")
        return

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    print("Camera window is open. Starting processing immediately.")

    # HSV ranges for blue and yellow (for thrown ball detection)
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([130, 255, 255])
    lower_yellow = np.array([18, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    smoothing_window = 5
    blue_radii_buffer = []
    yellow_radii_buffer = []
    blue_history = {'prev_radius': None, 'prev_time': None, 'speed': 0}
    yellow_history = {'prev_radius': None, 'prev_time': None, 'speed': 0}
    speed_threshold = 0.5

    def smooth_radius(radius_buffer, new_radius):
        radius_buffer.append(new_radius)
        if len(radius_buffer) > smoothing_window:
            radius_buffer.pop(0)
        return np.mean(radius_buffer)

    def calculate_speed(current_radius, history):
        current_time = time.time()
        prev_radius = history['prev_radius']
        prev_time = history['prev_time']
        if prev_radius is not None and prev_time is not None:
            dt = current_time - prev_time
            if dt > 0:
                speed = (prev_radius - current_radius) / dt
                history['speed'] = speed
        history['prev_radius'] = current_radius
        history['prev_time'] = current_time
        return history['speed']

    detection_interval = 30
    frame_counter = 0
    last_M = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        if frame_counter % detection_interval == 0:
            M_new = get_projector_screen_transform(frame, WIDTH, HEIGHT)
            if M_new is not None:
                last_M = M_new
        if last_M is not None:
            try:
                frame_warp = cv2.warpPerspective(frame, last_M, (WIDTH, HEIGHT))
            except cv2.error:
                frame_warp = cv2.resize(frame, (WIDTH, HEIGHT))
        else:
            frame_warp = cv2.resize(frame, (WIDTH, HEIGHT))

        # --- Process Blue Ball (thrown ball detection) ---
        blue_center, blue_radius = detect_ball(frame_warp, lower_blue, upper_blue)
        if blue_center is not None and blue_radius is not None:
            cv2.circle(frame_warp, blue_center, int(blue_radius), (255, 0, 0), 2)
            smoothed_blue = smooth_radius(blue_radii_buffer, blue_radius)
            blue_speed = calculate_speed(smoothed_blue, blue_history)
            cv2.putText(frame_warp, f"Blue Speed: {blue_speed:.2f}", (blue_center[0]+10, blue_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            if abs(blue_speed) < speed_threshold:
                event_queue.put(blue_center)

        # --- Process Yellow Ball (thrown ball detection) ---
        yellow_center, yellow_radius = detect_ball(frame_warp, lower_yellow, upper_yellow)
        if yellow_center is not None and yellow_radius is not None:
            cv2.circle(frame_warp, yellow_center, int(yellow_radius), (0, 255, 255), 2)
            smoothed_yellow = smooth_radius(yellow_radii_buffer, yellow_radius)
            yellow_speed = calculate_speed(smoothed_yellow, yellow_history)
            cv2.putText(frame_warp, f"Yellow Speed: {yellow_speed:.2f}", (yellow_center[0]+10, yellow_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if abs(yellow_speed) < speed_threshold:
                event_queue.put(yellow_center)

        # --- Overlay Red Balloon Detections ---
        red_balloons = detect_red_balloon(frame_warp)
        for red_center, red_radius in red_balloons:
            cv2.circle(frame_warp, red_center, int(red_radius), (0, 0, 255), 2)
            cv2.putText(frame_warp, "Red Balloon", (red_center[0]+10, red_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- Overlay Green Balloon Detections ---
        green_balloons = detect_green_balloon(frame_warp)
        for green_center, green_radius in green_balloons:
            cv2.circle(frame_warp, green_center, int(green_radius), (0, 255, 0), 2)
            cv2.putText(frame_warp, "Green Balloon", (green_center[0]+10, green_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Camera", frame_warp)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

def start_game_and_camera():
    event_queue = Queue()
    game_instance = Game()
    game_instance.event_queue = event_queue
    camera_thread = threading.Thread(target=camera_processing, args=(game_instance, event_queue))
    camera_thread.daemon = True
    camera_thread.start()
    game_instance.run()

if __name__ == "__main__":
    start_game_and_camera()
