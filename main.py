import cv2
import numpy as np
import threading
import time
import os
import yaml
from queue import Queue


from game import Game, WIDTH, HEIGHT, RED, GREEN, SCREEN_COORDS
from video_recorder import VideoRecorder
from Kalman_filter import kalman_filter,Kalman_update,IsBallIsBack




def detect_ball(frame, color_samples, screen_top_left=None, screen_bottom_right=None, params=None):
    """
    Detect a ball in an image using color-based filtering in the HSV color space.
    The function applies color masking, morphological operations, and contour detection to locate the ball and determine its position and radius.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)        
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    mask = None
    for sample in color_samples:
        lower_bound = np.array(sample[0], dtype=np.uint8)
        upper_bound = np.array(sample[1], dtype=np.uint8)
        temp_mask   = cv2.inRange(hsv, lower_bound, upper_bound)
        
        if mask is None:
            mask = temp_mask
        else:
            mask = cv2.bitwise_or(mask, temp_mask)

    if screen_top_left and screen_bottom_right:
        screen_mask = np.zeros_like(mask)
        cv2.rectangle(screen_mask, screen_top_left, screen_bottom_right, 255, -1)
        mask = cv2.bitwise_and(mask, screen_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 200:
            (x, y), radius = cv2.minEnclosingCircle(c)
            
            min_size = 0
            max_size = 100
            if params and params.get('min_size', 0) > 0 and params.get('max_size', 0) > 0:
                min_size = params['min_size']
                max_size = params['max_size']
            
            if min_size <= radius <= max_size:
                return (int(x), int(y)), int(radius)
    return None, None





try:
    with open("calibration.yaml", "r") as f:
        calibration_data = yaml.safe_load(f)
    print("Calibration data loaded from calibration.yaml")
except Exception as e:
    print("Failed to load calibration data:", e)
    calibration_data = {
         "blue": [[[95, 177, 174], [115, 255, 255]]],
         "yellow": [[[4, 195, 135], [24, 255, 255]]]
    }

latest_frame = None
frame_lock = threading.Lock()

def capture_thread_func(cap, stop_event):
    global latest_frame
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame.copy()
    print("Capture thread ending.")



def get_projector_screen_transform(frame, width, height):
    """
    calculates a perspective transformation matrix that will allow the screen identified 
    in the image to be mapped to a rectangular area of ​fixed size (width, height).
    we get the coordinate from calibration projector.
    
    """
    # Draw screen contour in blue to avoid interference with green detection
    screen_points = SCREEN_COORDS.astype(np.int32)
    cv2.polylines(frame, [screen_points], True, (255, 0, 0), 2)
    
    # Calculate transform for the fixed coordinates
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(SCREEN_COORDS, dst)
    return M

def detect_red_balloon(frame):
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
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            red_balloons.append(((int(x), int(y)), float(radius)))
    return red_balloons

def detect_green_balloon(frame):
    """
    same idea like th red balloon
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
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            green_balloons.append(((int(x), int(y)), float(radius)))
    return green_balloons



def Handle_ballons(ballons,frame_display,color):
    """
    this function draw a circle on the baloon its help for debug
    
    """
    for center, radius in ballons:
    # Draw in original space
        if color == "RED":
            cv2.circle(frame_display, center, int(radius), (0, 0, 255), 2)
            cv2.putText(frame_display, "Red Balloon", (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.circle(frame_display, center, int(radius), (0, 255, 0), 2)
            cv2.putText(frame_display, "Green Balloon", (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def update_event_queue(event_queue,center,radius,frame_num,frame_display,ball_color,M,Kalman,last_vx,last_vy):
    """
    Draw a circle on the ball and update event queue each frame to detect a hit
    """
    if ball_color=="BLUE":    
        if center is not None and radius is not None:

            predicted                  =  Kalman_update(Kalman,center,radius)
            IsBack,last_vx,last_vy     =  IsBallIsBack(predicted,last_vx,last_vy)
            predicted_center           =  (int(predicted[0]), int(predicted[1]))

            cv2.circle(frame_display, predicted_center, int(predicted[2]), (255, 0, 0), 2)         # We will draw a circle around the blue ball to verify identification          
            cv2.putText(frame_display, "Blue Ball", (predicted_center[0] + 10, predicted_center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            src_pt = np.array([[[predicted_center[0], predicted_center[1]]]], dtype=np.float32)
            center_warped = cv2.perspectiveTransform(src_pt, M)[0][0]                   # get the prespective of the ball 3D->2D
            event_queue.put(("blue", int(center_warped[0]), int(center_warped[1]), int(predicted[2]),frame_number,IsBack))
        return last_vx,last_vy
    else:      
        if center is not None and radius is not None:
            predicted                  =  Kalman_update(Kalman,center,radius)
            IsBack,last_vx,last_vy     =  IsBallIsBack(predicted,last_vx,last_vy)
            predicted_center           =  (int(predicted[0]), int(predicted[1]))

            cv2.circle(frame_display, predicted_center, int(predicted[2]), (0, 255, 255), 2)
            cv2.putText(frame_display, "Yellow Ball", (predicted_center[0] + 10, predicted_center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            src_pt = np.array([[[predicted_center[0], predicted_center[1]]]], dtype=np.float32)
            center_warped = cv2.perspectiveTransform(src_pt, M)[0][0]
            event_queue.put(("yellow", int(center_warped[0]), int(center_warped[1]), int(radius),frame_number,IsBack))
        return last_vx,last_vy
    return



Yellow_kalman   =  kalman_filter()
blue_kalman     =  kalman_filter()



def camera_processing(game_instance, event_queue):
    """
    This function we connect to the relevanr camera, detect th balloons and the balls and show the video in real time. 
    """
    global latest_frame
    global frame_number   

    cap = None

    for i in range(1,5):            # find the camera
        cap_candidate = cv2.VideoCapture(i)
        if cap_candidate.isOpened():
            cap = cap_candidate
            print(f"Opened camera index: {i}")
            break
    if cap is None or not cap.isOpened():
        print("Could not open camera")
        return

    cam_fps = cap.get(cv2.CAP_PROP_FPS)    #find the FPS 
    if cam_fps <= 0 or cam_fps is None:
        cam_fps = 30
    print(f"Reported camera FPS: {cam_fps}")

    stop_event = threading.Event()
    cap_thread = threading.Thread(target=capture_thread_func, args=(cap, stop_event))
    cap_thread.daemon = True
    cap_thread.start()

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    print("Camera window is open. Starting processing...")

    effective_frame_count = 0
    fps_timer             = time.time()
    effective_fps         = 0

    frame_number = 0        # helps us to followinf the frame

    recording_enabled = False
    contour_recording_enabled = False
    video_recorder = None
    video_recorder_contour = None
    last_vx_yellow = 1  # מהירות נוכחית בציר X
    last_vy_yellow = 1
    last_vx_blue = 1
    last_vy_blue = 1

    while True:
        effective_frame_count += 1
        frame_number += 1 
        current_time = time.time()
        if current_time - fps_timer >= 1.0:
            effective_fps = effective_frame_count / (current_time - fps_timer)
            fps_timer = current_time
            effective_frame_count = 0

        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
            frame_display = frame.copy()

        # Get transform matrix and draw screen contour
        M = get_projector_screen_transform(frame_display, WIDTH, HEIGHT)

        # Detect the radius and caenter of the ball in original frame space 
        blue_center, blue_radius     = detect_ball(frame, calibration_data["blue"], params={"min_size": 5, "max_size": 50})
        yellow_center, yellow_radius = detect_ball(frame, calibration_data["yellow"], params={"min_size": 5, "max_size": 50})

        last_vx_blue,last_vy_blue=update_event_queue(event_queue,blue_center,blue_radius,frame_number,frame_display,"BLUE",M,blue_kalman,last_vx_blue,last_vy_blue)
        last_vx_yellow,last_vy_yellow=update_event_queue(event_queue,yellow_center,yellow_radius,frame_number,frame_display,"YELLOW",M,Yellow_kalman,last_vx_yellow,last_vy_yellow)

        red_balloons = detect_red_balloon(frame)
        Handle_ballons(red_balloons,frame_display,"RED")
        
        green_balloons = detect_green_balloon(frame)
        Handle_ballons(green_balloons,frame_display,"GREEN")


        # Show original frame with overlays
        cv2.imshow("Camera", frame_display)

        # Handle recording video if you dont have camera
        if recording_enabled and video_recorder is not None:
            video_recorder.record_frame(frame.copy())
        if contour_recording_enabled and video_recorder_contour is not None:
            video_recorder_contour.record_frame(frame_display.copy())

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording_enabled = not recording_enabled
            if recording_enabled:
                rec_fps = effective_fps if effective_fps >= 1 else 1
                video_recorder = VideoRecorder(filename="clean_video.avi", fps=rec_fps)
                print(f"Raw recording started at {rec_fps:.2f} FPS.")
            else:
                if video_recorder is not None:
                    video_recorder.stop()
                    video_recorder = None
                print("Raw recording stopped.")
        elif key == ord('c'):
            contour_recording_enabled = not contour_recording_enabled
            if contour_recording_enabled:
                rec_fps = effective_fps if effective_fps >= 1 else 1
                video_recorder_contour = VideoRecorder(filename="contour_video.avi", fps=rec_fps)
                print(f"Contour recording started at {rec_fps:.2f} FPS.")
            else:
                if video_recorder_contour is not None:
                    video_recorder_contour.stop()
                    video_recorder_contour = None
                print("Contour recording stopped.")
       
    stop_event.set()
    cap_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    if video_recorder is not None:
        video_recorder.stop()
    if video_recorder_contour is not None:
        video_recorder_contour.stop()

    game_instance.run()


def start_game_and_camera():
    event_queue = Queue()       # Queue of all the frames' we wiil use it to detect hit 
    game_instance = Game()
    game_instance.event_queue = event_queue
    cam_thread = threading.Thread(target=camera_processing, args=(game_instance, event_queue))
    cam_thread.daemon = True
    cam_thread.start()
    game_instance.run()

if __name__ == "__main__":
    start_game_and_camera()