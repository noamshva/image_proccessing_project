import cv2
import numpy as np
import pygame
import os
from pathlib import Path
from time import time

class BallGameTracker:
    def __init__(self):
        # Initialize camera
        for i in range(2):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                break
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Create parameter window
        cv2.namedWindow('HSV Calibration')
        # Blue ball trackbars
        cv2.createTrackbar('Blue H Min', 'HSV Calibration', 90, 180, lambda x: None)
        cv2.createTrackbar('Blue H Max', 'HSV Calibration', 130, 180, lambda x: None)
        cv2.createTrackbar('Blue S Min', 'HSV Calibration', 80, 255, lambda x: None)
        cv2.createTrackbar('Blue V Min', 'HSV Calibration', 80, 255, lambda x: None)
        # Yellow ball trackbars
        cv2.createTrackbar('Yellow H Min', 'HSV Calibration', 18, 180, lambda x: None)
        cv2.createTrackbar('Yellow H Max', 'HSV Calibration', 35, 180, lambda x: None)
        cv2.createTrackbar('Yellow S Min', 'HSV Calibration', 100, 255, lambda x: None)
        cv2.createTrackbar('Yellow V Min', 'HSV Calibration', 100, 255, lambda x: None)

        # Ball tracking history
        self.blue_history = {
            'prev_radius': None,
            'prev_time': None,
            'speed': 0,
            'hit_wall': False,
            'last_hit_time': 0
        }
        self.yellow_history = {
            'prev_radius': None,
            'prev_time': None,
            'speed': 0,
            'hit_wall': False,
            'last_hit_time': 0
        }

        # Initialize Pygame for the balloon game
        pygame.init()
        if os.name == 'nt':
            os.environ['SDL_VIDEO_WINDOW_POS'] = "2000,0"
        self.game_running = True

    def get_hsv_ranges(self):
        """Get current HSV ranges from trackbars"""
        # Blue ranges
        blue_h_min = cv2.getTrackbarPos('Blue H Min', 'HSV Calibration')
        blue_h_max = cv2.getTrackbarPos('Blue H Max', 'HSV Calibration')
        blue_s_min = cv2.getTrackbarPos('Blue S Min', 'HSV Calibration')
        blue_v_min = cv2.getTrackbarPos('Blue V Min', 'HSV Calibration')
        
        # Yellow ranges
        yellow_h_min = cv2.getTrackbarPos('Yellow H Min', 'HSV Calibration')
        yellow_h_max = cv2.getTrackbarPos('Yellow H Max', 'HSV Calibration')
        yellow_s_min = cv2.getTrackbarPos('Yellow S Min', 'HSV Calibration')
        yellow_v_min = cv2.getTrackbarPos('Yellow V Min', 'HSV Calibration')

        # Create the range arrays
        lower_blue = np.array([blue_h_min, blue_s_min, blue_v_min])
        upper_blue = np.array([blue_h_max, 255, 255])
        
        lower_yellow = np.array([yellow_h_min, yellow_s_min, yellow_v_min])
        upper_yellow = np.array([yellow_h_max, 255, 255])

        return lower_blue, upper_blue, lower_yellow, upper_yellow

    def detect_ball(self, frame, lower_color, upper_color):
        """Enhanced ball detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Enhanced morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 100:  # Minimum area threshold
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                return (int(x), int(y)), int(radius), mask
        return None, None, mask

    def calculate_speed_and_hits(self, current_radius, history, current_time):
        """Calculate speed and detect wall hits"""
        if history['prev_radius'] is not None and history['prev_time'] is not None:
            dt = current_time - history['prev_time']
            
            if dt > 0:
                radius_change = current_radius - history['prev_radius']
                speed = radius_change / dt
                
                # Detect wall hits (when speed changes from positive to negative)
                if speed < 0 and history['speed'] > 0:
                    if current_time - history['last_hit_time'] > 0.5:  # 500ms minimum between hits
                        history['hit_wall'] = True
                        history['last_hit_time'] = current_time
                else:
                    history['hit_wall'] = False
                
                history['speed'] = speed
        
        history['prev_radius'] = current_radius
        history['prev_time'] = current_time
        
        return history['speed'], history['hit_wall']

    def run(self):
        while self.game_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Get current HSV ranges
            lower_blue, upper_blue, lower_yellow, upper_yellow = self.get_hsv_ranges()
            
            # Detect balls
            blue_data = self.detect_ball(frame, lower_blue, upper_blue)
            yellow_data = self.detect_ball(frame, lower_yellow, upper_yellow)
            
            # Create debug view with masks
            mask_display = None
            if blue_data[2] is not None and yellow_data[2] is not None:
                mask_display = cv2.bitwise_or(blue_data[2], yellow_data[2])
            
            # Process blue ball
            if blue_data[0]:
                center, radius = blue_data[0], blue_data[1]
                blue_speed, blue_hit = self.calculate_speed_and_hits(
                    radius, self.blue_history, time()
                )
                
                # Draw ball and info
                cv2.circle(frame, center, radius, (255, 0, 0), 2)
                cv2.putText(frame, f"Blue Speed: {blue_speed:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                if blue_hit:
                    cv2.putText(frame, "BLUE BALL HIT!", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Process yellow ball
            if yellow_data[0]:
                center, radius = yellow_data[0], yellow_data[1]
                yellow_speed, yellow_hit = self.calculate_speed_and_hits(
                    radius, self.yellow_history, time()
                )
                
                # Draw ball and info
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
                cv2.putText(frame, f"Yellow Speed: {yellow_speed:.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                if yellow_hit:
                    cv2.putText(frame, "YELLOW BALL HIT!", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show frames
            cv2.imshow('Ball Tracking', frame)
            if mask_display is not None:
                cv2.imshow('Mask View', mask_display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.game_running = False
            elif key == ord('s'):
                # Save current HSV values
                print("Current HSV Ranges:")
                print(f"Blue: H({lower_blue[0]}-{upper_blue[0]}) "
                      f"S({lower_blue[1]}-255) V({lower_blue[2]}-255)")
                print(f"Yellow: H({lower_yellow[0]}-{upper_yellow[0]}) "
                      f"S({lower_yellow[1]}-255) V({lower_yellow[2]}-255)")

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    tracker = BallGameTracker()
    tracker.run()