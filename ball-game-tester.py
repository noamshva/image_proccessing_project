

import cv2
import numpy as np
from pathlib import Path
import time
from collections import deque


class GameTester:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video file")
        
        # Initialize scores and debugging window
        self.blue_score = 0
        self.yellow_score = 0
        cv2.namedWindow('Parameters')
        
        # Configurable parameters with trackbars
        cv2.createTrackbar('Min Ball Size', 'Parameters', 5, 100, lambda x: None)
        cv2.createTrackbar('Max Ball Size', 'Parameters', 25, 100, lambda x: None)
        cv2.createTrackbar('Size Change Threshold', 'Parameters', 15, 100, lambda x: None)
        
        # Color ranges
        self.lower_blue = np.array([90, 80, 80])   # slightly broader range
        self.upper_blue = np.array([130, 255, 255])
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_red = np.array([0, 120, 100])
        self.upper_red = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 120, 100])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_green = np.array([45, 150, 150])
        self.upper_green = np.array([75, 255, 255])
        
        # Enhanced ball tracking
        self.blue_tracker = {
            'positions': deque(maxlen=10),  # Store last 10 positions
            'sizes': deque(maxlen=10),      # Store last 10 sizes
            'times': deque(maxlen=10),      # Store timestamps
            'last_hit_time': 0,
            'last_score_time': 0,
            'direction': 'unknown',         # Track movement direction
            'hit_wall': False
        }
        self.yellow_tracker = {
            'positions': deque(maxlen=10),
            'sizes': deque(maxlen=10),
            'times': deque(maxlen=10),
            'last_hit_time': 0,
            'last_score_time': 0,
            'direction': 'unknown',
            'hit_wall': False
        }
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter('game_analysis.mp4', fourcc, 30.0, (frame_width, frame_height))

    def get_parameters(self):
        """Get current parameters from trackbars"""
        return {
            'min_size': cv2.getTrackbarPos('Min Ball Size', 'Parameters'),
            'max_size': cv2.getTrackbarPos('Max Ball Size', 'Parameters'),
            'size_threshold': cv2.getTrackbarPos('Size Change Threshold', 'Parameters') / 100.0
        }

    def track_ball_movement(self, current_pos, current_size, tracker, timestamp, params):
        """Enhanced ball tracking with size-based wall hit detection"""
        if current_pos and current_size:
            tracker['positions'].append(current_pos)
            tracker['sizes'].append(current_size)
            tracker['times'].append(timestamp)
            
            if len(tracker['sizes']) >= 3:
                # Calculate size change rate
                recent_sizes = list(tracker['sizes'])
                size_change = (recent_sizes[-1] - recent_sizes[-2]) / recent_sizes[-2]
                
                # Detect direction change based on size
                if size_change > params['size_threshold']:
                    new_direction = 'approaching'
                elif size_change < -params['size_threshold']:
                    new_direction = 'moving_away'
                else:
                    new_direction = tracker.get('direction', 'unknown')
                
                # Detect wall hits
                if (tracker['direction'] == 'approaching' and new_direction == 'moving_away' and
                    timestamp - tracker['last_hit_time'] > 0.5):
                    tracker['hit_wall'] = True
                    tracker['last_hit_time'] = timestamp
                else:
                    tracker['hit_wall'] = False
                
                tracker['direction'] = new_direction
                
                # Debug information
                return {
                    'hit_wall': tracker['hit_wall'],
                    'direction': tracker['direction'],
                    'size_change': size_change,
                    'current_size': current_size
                }
        
        return {'hit_wall': False, 'direction': 'unknown', 'size_change': 0, 'current_size': 0}

    def process_frame(self, frame, timestamp):
        """Process frame with enhanced tracking"""
        # Get current parameters
        params = self.get_parameters()
        
        # Detect screen and balloons
        screen_rect = self.detect_projection_screen(frame)
        if screen_rect:
            x, y, w, h = screen_rect
            screen_roi = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Detect balloons
            red_balloon = self.detect_balloon(screen_roi, self.lower_red, self.upper_red,
                                           self.lower_red2, self.upper_red2)
            green_balloon = self.detect_balloon(screen_roi, self.lower_green, self.upper_green)
            
            # Adjust balloon coordinates
            if red_balloon[0]:
                pos, radius, contour = red_balloon
                red_balloon = ((pos[0] + x, pos[1] + y), radius, contour)
                cv2.circle(frame, (pos[0] + x, pos[1] + y), radius, (0, 0, 255), 2)
            
            if green_balloon[0]:
                pos, radius, contour = green_balloon
                green_balloon = ((pos[0] + x, pos[1] + y), radius, contour)
                cv2.circle(frame, (pos[0] + x, pos[1] + y), radius, (0, 255, 0), 2)
        
        # Detect and track balls
        blue_ball, blue_radius = self.detect_object(frame, self.lower_blue, self.upper_blue)
        yellow_ball, yellow_radius = self.detect_object(frame, self.lower_yellow, self.upper_yellow)
        
        # Track blue ball
        if blue_ball and blue_radius:
            blue_info = self.track_ball_movement(blue_ball, blue_radius, 
                                               self.blue_tracker, timestamp, params)
            
            # Draw blue ball tracking info
            cv2.circle(frame, blue_ball, blue_radius, (255, 0, 0), 2)
            cv2.putText(frame, f"Blue Size: {blue_radius}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Direction: {blue_info['direction']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Handle hits and scoring
            if blue_info['hit_wall']:
                cv2.putText(frame, "BLUE BALL HIT WALL!", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                if (red_balloon[0] and timestamp - self.blue_tracker['last_score_time'] > 1.0 and
                    self.check_balloon_hit(blue_ball, red_balloon, blue_radius)):
                    self.blue_score += 1
                    self.blue_tracker['last_score_time'] = timestamp
                    cv2.putText(frame, "BLUE SCORES!", 
                              (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        
        # Track yellow ball (similar to blue)
        if yellow_ball and yellow_radius:
            yellow_info = self.track_ball_movement(yellow_ball, yellow_radius, 
                                                 self.yellow_tracker, timestamp, params)
            
            cv2.circle(frame, yellow_ball, yellow_radius, (0, 255, 255), 2)
            cv2.putText(frame, f"Yellow Size: {yellow_radius}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Direction: {yellow_info['direction']}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if yellow_info['hit_wall']:
                cv2.putText(frame, "YELLOW BALL HIT WALL!", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                if (green_balloon[0] and timestamp - self.yellow_tracker['last_score_time'] > 1.0 and
                    self.check_balloon_hit(yellow_ball, green_balloon, yellow_radius)):
                    self.yellow_score += 1
                    self.yellow_tracker['last_score_time'] = timestamp
                    cv2.putText(frame, "YELLOW SCORES!", 
                              (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
        # Draw scores
        cv2.putText(frame, f"Blue Score: {self.blue_score}", 
                   (frame.shape[1]-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Yellow Score: {self.yellow_score}", 
                   (frame.shape[1]-300, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Draw parameters
        cv2.putText(frame, f"Min Size: {params['min_size']}", 
                   (frame.shape[1]-300, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Max Size: {params['max_size']}", 
                   (frame.shape[1]-300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {params['size_threshold']:.2f}", 
                   (frame.shape[1]-300, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame


    def detect_balloon(self, frame, lower_color, upper_color, lower_color2=None, upper_color2=None):
        """Enhanced balloon detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask with improved processing
        mask = cv2.inRange(hsv, lower_color, upper_color)
        if lower_color2 is not None and upper_color2 is not None:
            mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
            mask = cv2.bitwise_or(mask, mask2)
        
        # Enhanced mask processing
        kernel = np.ones((3,3), np.uint8)  # Smaller kernel for finer detail
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_balloon = None
        max_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Adjusted minimum area
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(h) / w if w > 0 else 0
                    
                    shape_score = (
                        circularity * 0.4 +
                        solidity * 0.3 +
                        min(aspect_ratio, 1.5) / 1.5 * 0.3
                    )
                    
                    if shape_score > max_score:
                        max_score = shape_score
                        ((cx, cy), radius) = cv2.minEnclosingCircle(contour)
                        best_balloon = ((int(cx), int(cy)), int(radius), contour)
        
        return best_balloon if max_score > 0.6 else (None, None, None)

    def check_balloon_hit(self, ball_pos, balloon_data, ball_radius):
        """Enhanced balloon hit detection"""
        if ball_pos and balloon_data and len(balloon_data) == 3:
            balloon_pos, balloon_radius, balloon_contour = balloon_data
            
            # Check if ball center is within balloon contour
            point = np.array([ball_pos[0], ball_pos[1]])
            if cv2.pointPolygonTest(balloon_contour, (float(point[0]), float(point[1])), False) >= 0:
                return True
                
            # Additional distance-based check
            distance = np.sqrt(
                (ball_pos[0] - balloon_pos[0])**2 + 
                (ball_pos[1] - balloon_pos[1])**2
            )
            return distance < (ball_radius + balloon_radius)
        return False

    
    def detect_projection_screen(self, frame):
        """Detect the white projection screen in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours of white regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest white region
            screen_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(screen_contour) > 10000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(screen_contour)
                return (x, y, w, h)
        return None

    def detect_object(self, frame, lower_color, upper_color):
        """Enhanced ball detection with better noise handling"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask with improved color ranges
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Better morphological operations using elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours with improved filtering
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 100:  # minimum area threshold
                (x, y), radius = cv2.minEnclosingCircle(c)
                return (int(x), int(y)), int(radius)
        return None, None

    def calculate_speed_and_hits(self, current_radius, history, current_time):
        """Enhanced speed calculation with smoothing"""
        # Add current radius to buffer
        history['radius_buffer'].append(current_radius)
        
        if len(history['radius_buffer']) >= 3:  # Need minimum samples for smoothing
            # Use median filtered radius for calculations
            smoothed_radius = np.median(history['radius_buffer'])
            
            if history['prev_radius'] is not None and history['prev_time'] is not None:
                dt = current_time - history['prev_time']
                
                if dt > 0:
                    # Calculate speed with smoothed radius
                    radius_change = smoothed_radius - history['prev_radius']
                    speed = radius_change / dt
                    
                    # Add to speed buffer
                    history['speed_buffer'].append(speed)
                    
                    # Calculate smoothed speed
                    if len(history['speed_buffer']) >= 3:
                        smoothed_speed = np.median(history['speed_buffer'])
                        
                        # Enhanced wall hit detection with hysteresis
                        speed_threshold = 2.0  # Adjust based on your needs
                        if smoothed_speed < -speed_threshold and history['speed'] > speed_threshold:
                            if current_time - history['last_hit_time'] > 0.5:
                                history['hit_wall'] = True
                                history['last_hit_time'] = current_time
                        else:
                            history['hit_wall'] = False
                        
                        history['speed'] = smoothed_speed
            
            history['prev_radius'] = smoothed_radius
            history['prev_time'] = current_time
            
            return history['speed'], history['hit_wall']
            
        history['prev_radius'] = current_radius
        history['prev_time'] = current_time
        return 0, False

    def run(self):
        """Process the entire video"""
        frame_count = 0
        speed_factor = 1.0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = frame_count / (self.cap.get(cv2.CAP_PROP_FPS) * speed_factor)
            processed_frame = self.process_frame(frame, timestamp)
            self.out.write(processed_frame)
            
            # Display frame
            cv2.imshow('Game Analysis', processed_frame)
            
            # Handle keyboard input for speed control
            key = cv2.waitKey(int(1000/(30*speed_factor))) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Slower
                speed_factor = min(2.0, speed_factor + 0.1)
            elif key == ord('f'):  # Faster
                speed_factor = max(0.1, speed_factor - 0.1)
                
            # Display current speed
            cv2.putText(processed_frame, f"Speed: {1/speed_factor:.1f}x", 
                    (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
            
            frame_count += 1

if __name__ == "__main__":
    video_path = "ball_video.mp4"
    tester = GameTester(video_path)
    final_scores = tester.run()
    
    print("Game Analysis Complete!")
    print(f"Final Scores:")
    print(f"Blue Player: {final_scores['blue_score']}")
    print(f"Yellow Player: {final_scores['yellow_score']}")


