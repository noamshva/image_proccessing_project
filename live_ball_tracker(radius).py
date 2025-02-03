import cv2
import numpy as np
from pathlib import Path
import time
from collections import deque

class BallTracker:
    def __init__(self):
        # Initialize camera (try different indices if camera not found)
        for i in range(2):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                break
                
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Initialize tracking windows
        cv2.namedWindow('Parameters')
        cv2.namedWindow('Ball Tracking')
        
        # Configurable parameters with trackbars
        cv2.createTrackbar('Min Ball Size', 'Parameters', 5, 100, lambda x: None)
        cv2.createTrackbar('Max Ball Size', 'Parameters', 25, 100, lambda x: None)
        cv2.createTrackbar('Size Change Threshold', 'Parameters', 15, 100, lambda x: None)
        
        # Color ranges
        self.lower_blue = np.array([90, 80, 80])
        self.upper_blue = np.array([130, 255, 255])
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        
        # Initialize score
        self.blue_score = 0
        self.yellow_score = 0
        
        # Enhanced ball tracking with deque for smooth tracking
        self.blue_tracker = {
            'positions': deque(maxlen=10),
            'sizes': deque(maxlen=10),
            'times': deque(maxlen=10),
            'last_hit_time': 0,
            'last_score_time': 0,
            'direction': 'unknown',
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
        
        self.running = True

    def get_parameters(self):
        """Get current parameters from trackbars"""
        return {
            'min_size': cv2.getTrackbarPos('Min Ball Size', 'Parameters'),
            'max_size': cv2.getTrackbarPos('Max Ball Size', 'Parameters'),
            'size_threshold': cv2.getTrackbarPos('Size Change Threshold', 'Parameters') / 100.0
        }

    def track_ball_movement(self, current_pos, current_size, tracker, timestamp, params):
        """Track ball movement and detect hits"""
        if current_pos and current_size:
            tracker['positions'].append(current_pos)
            tracker['sizes'].append(current_size)
            tracker['times'].append(timestamp)
            
            if len(tracker['sizes']) >= 3:
                recent_sizes = list(tracker['sizes'])
                size_change = (recent_sizes[-1] - recent_sizes[-2]) / recent_sizes[-2]
                
                # Detect direction based on size change
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
                
                return {
                    'hit_wall': tracker['hit_wall'],
                    'direction': tracker['direction'],
                    'size_change': size_change,
                    'current_size': current_size
                }
        
        return {'hit_wall': False, 'direction': 'unknown', 'size_change': 0, 'current_size': 0}

    def detect_object(self, frame, lower_color, upper_color):
        """Detect colored object in frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 100:
                (x, y), radius = cv2.minEnclosingCircle(c)
                return (int(x), int(y)), int(radius)
        return None, None

    def process_frame(self, frame, timestamp):
        """Process a single frame"""
        # Get current parameters
        params = self.get_parameters()
        
        # Detect balls
        blue_ball, blue_radius = self.detect_object(frame, self.lower_blue, self.upper_blue)
        yellow_ball, yellow_radius = self.detect_object(frame, self.lower_yellow, self.upper_yellow)
        
        # Process blue ball
        if blue_ball and blue_radius:
            blue_info = self.track_ball_movement(blue_ball, blue_radius, 
                                               self.blue_tracker, timestamp, params)
            
            cv2.circle(frame, blue_ball, blue_radius, (255, 0, 0), 2)
            cv2.putText(frame, f"Blue Size: {blue_radius}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Direction: {blue_info['direction']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if blue_info['hit_wall']:
                cv2.putText(frame, "BLUE BALL HIT WALL!", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Process yellow ball
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
        
        # Draw parameters
        cv2.putText(frame, f"Min Size: {params['min_size']}", 
                   (frame.shape[1]-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Max Size: {params['max_size']}", 
                   (frame.shape[1]-300, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {params['size_threshold']:.2f}", 
                   (frame.shape[1]-300, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def run(self):
        start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = time.time() - start_time
            processed_frame = self.process_frame(frame, timestamp)
            
            # Show frame
            cv2.imshow('Ball Tracking', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        tracker = BallTracker()
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()