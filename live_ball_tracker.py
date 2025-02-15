import cv2
import numpy as np
from pathlib import Path
import time
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections


class KalmanFilter:
    def __init__(self):
        # State: [x, y, vx, vy, ax, ay, r, vr]
        self.kalman = cv2.KalmanFilter(8, 4)  # 8 states, 3 measurements (x, y, radius)
        
        dt = 1.0 / 60.0  # Assuming 60 FPS
        
        # Measurement Matrix (we measure x, y, and radius)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # Measure x
            [0, 1, 0, 0, 0, 0, 0, 0],  # Measure y
            [0, 0, 0, 0, 0, 0, 1, 0],   # Measure radius
            [0, 0, 0, 0, 0, 0, 0, 1]   # Measure radius velocity
        ], np.float32)

        # Transition Matrix (predicts movement over time)
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0, 0, 0],  # x position = x+vx*dt+0.5*ax*dt^2
            [0, 1, 0, dt, 0, 0.5*dt*dt, 0, 0],  # y positionn= y+vy*dt+0.5*ay*dt^2
            [0, 0, 1, 0, dt, 0, 0, 0],  # x velocity
            [0, 0, 0, 1, 0, dt, 0, 0],  # y velocity
            [0, 0, 0, 0, 1, 0, 0, 0],  # x acceleration
            [0, 0, 0, 0, 0, 1, 0, 0],  # y acceleration
            [0, 0, 0, 0, 0, 0, 1, dt],  # radius
            [0, 0, 0, 0, 0, 0, 0, 1]   # radius velocity
        ], np.float32)

        # Process Noise Covariance (reducing noise)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1

        # Measurement Noise Covariance (controls measurement uncertainty)
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        # Optionally adjust noise for radius velocity measurement
        self.kalman.measurementNoiseCov[3,3] = 0.35 # Higher uncertainty for radius velocity
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32) * 0.25
        self.prediction = np.zeros((2, 1), np.float32)
        self.initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 10
        self.radius_velocity = 0
    def reset(self):
        """Reset the Kalman filter when tracking is lost"""
        self.initialized = False
        self.lost_frames = 0
    def predict(self):
        """
        Predict the next state without measurement update.
        Returns predicted position and size.
        """
        prediction = self.kalman.predict()
        
        # Extract predicted position and size from state
        predicted_pos = (prediction[0][0], prediction[1][0])
        predicted_size = prediction[6][0]
        predicted_size_velocity = prediction[7][0]
        return predicted_pos, predicted_size, predicted_size_velocity
    def update(self, point, radius):
        """Update Kalman filter with new (x, y) and radius measurement"""
        if point is None or radius is None:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.reset()
                return None
            
            self.prediction = self.kalman.predict()
            return (self.prediction[0:2].reshape(-1), self.prediction[6][0],self.prediction[7][0])  # Return (x, y), radius, and radius velocity
        # If radius_velocity is not provided, estimate it from previous state
        radius_velocity = None
        if self.initialized:
            radius_velocity = (radius - self.prediction[6][0]) / (1.0/60.0)  # Estimate using dt
        else:
            radius_velocity = 0
        measurement = np.array([[np.float32(point[0])], [np.float32(point[1])],[np.float32(radius)], [np.float32(radius_velocity)]])
        
        if not self.initialized:
            self.kalman.statePre = np.array([
                [measurement[0, 0]],  # x
                [measurement[1, 0]],  # y
                [0],                  # vx
                [0],                  # vy
                [0],                  # ax
                [0],                  # ay
                [measurement[2, 0]],  # radius
                [measurement[3, 0]]   # radius velocity
            ], np.float32)
            self.initialized = True

        # Predict next state
        self.prediction = self.kalman.predict()

        # Update with measurement
        self.kalman.correct(measurement)
        self.lost_frames = 0
        
        return (self.prediction[0:2].reshape(-1), self.prediction[6][0],self.prediction[7][0])  # Return (x, y), radius, and radius velocity


class BallTracker:
    def __init__(self, color_ranges=None):
        self.radius_change_threshold = 0.05  # 5% change in radius to detect z-direction
        self.min_velocity_threshold = 0.5    # Minimum velocity to detect planar movement
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
        cv2.createTrackbar('Min Ball Size', 'Parameters', 10, 100, lambda x: None)
        cv2.createTrackbar('Max Ball Size', 'Parameters', 40, 100, lambda x: None)
        cv2.createTrackbar('Size Change Threshold', 'Parameters', 15, 100, lambda x: None)
        
        # Initialize Kalman filters for each color
        self.blue_kalman = KalmanFilter()
        self.yellow_kalman = KalmanFilter()
        
        # Initialize trajectory tracking
        #self.blue_trajectory = deque(maxlen=30)
        #self.yellow_trajectory = deque(maxlen=30)
        # Initialize color ranges using calibration data if provided.
        # Expected format for calibration: a dict with keys "blue" and "yellow",
        # where each value is a list of samples. Each sample is a tuple/list like:
        #   [ [lower_bound values], [upper_bound values] ]
        if color_ranges:
            if isinstance(color_ranges, dict):
                self.blue_samples = color_ranges.get("blue", [])
                self.yellow_samples = color_ranges.get("yellow", [])
            else:
                # If color_ranges is a list, assume [ (blue_samples), (yellow_samples) ]
                self.blue_samples, self.yellow_samples = color_ranges[0], color_ranges[1]
        else:
            # Default samples if no calibration data is provided.
            self.blue_samples = [
                ([90, 80, 80], [130, 255, 255])
            ]
            self.yellow_samples = [
                ([20, 100, 100], [30, 255, 255])
            ]

        self.blue_radius_history = collections.deque(maxlen=100)  # Store last 100 radius samples
        self.yellow_radius_history = collections.deque(maxlen=100)

        # Initialize score
        self.blue_score = 0
        self.yellow_score = 0
        
        # Enhanced ball tracking with deque for smooth tracking
        self.blue_tracker = {
            'positions': deque(maxlen=10),
            'sizes': deque(maxlen=10),
            'times': deque(maxlen=10),
            'last_radius': None,
            'radius_velocities': deque(maxlen=5),  # Track recent radius changes
            'last_hit_time': 0,
            'last_score_time': 0,
            'direction': 'unknown',
            'hit_wall': False,
            'velocity': np.array([0.0, 0.0]),
            'last_stable_pos': None,
            'last_stable_size': None,
            'confidence': 1.0,
            'frames_since_good_detection': 0,
            'edge_detection_counter': 0,
            'min_tracking_confidence': 0.3
        }
        self.yellow_tracker = {
            'positions': deque(maxlen=10),
            'sizes': deque(maxlen=10),
            'times': deque(maxlen=10),
            'last_radius': None,
            'radius_velocities': deque(maxlen=5),  # Track recent radius changes
            'last_hit_time': 0,
            'last_score_time': 0,
            'direction': 'unknown',
            'hit_wall': False,
            'velocity': np.array([0.0, 0.0]),
            'last_stable_pos': None,
            'last_stable_size': None,
            'confidence': 1.0,
            'frames_since_good_detection': 0,
            'edge_detection_counter': 0,
            'min_tracking_confidence': 0.3
        }
        
        self.running = True


    def get_parameters(self):
        """Get current parameters from trackbars"""
        return {
            'min_size': cv2.getTrackbarPos('Min Ball Size', 'Parameters'),
            'max_size': cv2.getTrackbarPos('Max Ball Size', 'Parameters'),
            'size_threshold': cv2.getTrackbarPos('Size Change Threshold', 'Parameters') / 100.0
        }
    @staticmethod
    def check_near_edge(current_pos, params):
        x, y = current_pos
        frame_width = params.get('frame_width', 1920)
        frame_height = params.get('frame_height', 1080)
        edge_threshold = params.get('edge_threshold', 50)
        
        near_edge = (x < edge_threshold or 
                    x > frame_width - edge_threshold or
                    y < edge_threshold or 
                    y > frame_height - edge_threshold)
        return near_edge
    @staticmethod
    def update_confidence(tracker, near_edge):
        if 'confidence' not in tracker:
            tracker.update({
                'confidence': 1.0,
                'frames_since_good_detection': 0,
                'edge_detection_counter': 0,
                'last_stable_pos': None,
                'last_stable_size': None,
                'min_tracking_confidence': 0.3
            })
        
        if near_edge:
            tracker['edge_detection_counter'] += 1
            tracker['confidence'] *= 0.9
        else:
            tracker['edge_detection_counter'] = 0
            tracker['confidence'] = min(1.0, tracker['confidence'] * 1.1)
    @staticmethod
    def update_kalman_filter(kalman_filter, current_pos, current_size, tracker):
        result = kalman_filter.update(current_pos, current_size)
        
        if result is None:
            tracker['frames_since_good_detection'] += 1
            tracker['confidence'] *= 0.8
            
            if tracker['confidence'] > tracker['min_tracking_confidence']:
                predicted_state = kalman_filter.predict()
                if predicted_state is not None:
                    predicted_pos, predicted_radius, _ = predicted_state
                    result = (predicted_pos, predicted_radius, 0)
        
        return result
    @staticmethod
    def smooth_transition(predicted_pos, tracker, params):
        if tracker['last_stable_pos'] is not None:
            distance_to_last = np.linalg.norm(np.array(predicted_pos) - np.array(tracker['last_stable_pos']))
            if distance_to_last > params.get('max_position_jump', 100):
                alpha = params.get('smoothing_factor', 0.3)
                smoothed_pos = tuple(alpha * np.array(predicted_pos) + 
                                (1 - alpha) * np.array(tracker['last_stable_pos']))
                predicted_pos = smoothed_pos
        return predicted_pos
    @staticmethod
    def update_tracker(tracker, predicted_pos, predicted_radius, timestamp, kalman_filter, good_new, good_old):
        tracker['positions'].append(predicted_pos)
        tracker['sizes'].append(predicted_radius)
        tracker['times'].append(timestamp)
        
        velocity = np.array([kalman_filter.kalman.statePost[2][0],
                        kalman_filter.kalman.statePost[3][0]])
        tracker['velocity'] = velocity
        tracker['good_new'] = good_new
        tracker['good_old'] = good_old
        return tracker
    @staticmethod
    def detect_direction_and_hits(tracker, predicted_pos, predicted_radius, timestamp, previous_direction, params):
        if tracker['last_radius'] is not None and predicted_radius is not None:
            radius_change = (predicted_radius - tracker['last_radius']) / tracker['last_radius']
            
            if tracker['confidence'] < 0.8:
                radius_change *= tracker['confidence']
            
            tracker['radius_velocities'].append(radius_change)
            if len(tracker['radius_velocities']) > 3:
                tracker['radius_velocities'].popleft()
            
            if len(tracker['radius_velocities']) >= 2:
                prev_radius_change = tracker['radius_velocities'][-2]
                curr_radius_change = tracker['radius_velocities'][-1]
                
                if tracker['confidence'] > 0.6:
                    if prev_radius_change > 0 and curr_radius_change < 0:
                        tracker['direction'] = 'approaching'
                    elif prev_radius_change < 0 and curr_radius_change > 0:
                        tracker['direction'] = 'moving_away'
            
            if (previous_direction == 'approaching' and tracker['direction'] == 'moving_away' and timestamp - tracker['last_hit_time'] > 0.5 and
                tracker['confidence'] > 0.7):
                    tracker['hit_wall'] = True
                    tracker['last_hit_time'] = timestamp
            else:
                    tracker['hit_wall'] = False
        
        if tracker['confidence'] > 0.8:
            tracker['last_stable_pos'] = predicted_pos
            tracker['last_stable_size'] = predicted_radius
        
        tracker['last_radius'] = predicted_radius

        # Integrate optical flow logic
        if ('good_new' in tracker and 'good_old' in tracker and 
            tracker['good_new'] is not None and tracker['good_old'] is not None and 
            len(tracker['good_new']) > 0 and len(tracker['good_old']) > 0):

            motion_vectors = tracker['good_new'] - tracker['good_old']
            velocities = np.linalg.norm(motion_vectors, axis=1)
            directions = np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0])
            
            # Example: Use the average velocity and direction for further processing
            avg_velocity = np.mean(velocities)
            avg_direction = np.mean(directions)
            
            # Update tracker with average velocity and direction
            tracker['avg_velocity'] = avg_velocity
            tracker['avg_direction'] = avg_direction
            
            # Additional logic based on optical flow
            if avg_velocity > params.get('velocity_threshold', 1.0):
                if avg_direction > 0:
                    optical_flow_direction = 'moving_away'
                else:
                    optical_flow_direction = 'approaching'
                
                # Combine confidence and optical flow results
                if tracker['confidence'] > 0.6:
                    if tracker['direction'] == 'approaching' and optical_flow_direction == 'moving_away':
                        tracker['direction'] = 'moving_away'
                    elif tracker['direction'] == 'moving_away' and optical_flow_direction == 'approaching':
                        tracker['direction'] = 'approaching'
            
            # Check for wall hit with combined information
            if (previous_direction == 'approaching' and 
                tracker['direction'] == 'moving_away' and
                timestamp - tracker['last_hit_time'] > 0.5 and
                tracker['confidence'] > 0.7):
                    tracker['hit_wall'] = True
                    tracker['last_hit_time'] = timestamp
            else:
                    tracker['hit_wall'] = False

        return tracker
    def detect_screen(self, frame):
        """Detect the white rectangular screen in the frame"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # Define the color range for white (this may need tuning depending on lighting)
        lower_white = np.array([0, 0, 200])  # Lower bound for white
        upper_white = np.array([180, 40, 255])  # Upper bound for white
    
        # Create a mask for white color
        mask = cv2.inRange(hsv, lower_white, upper_white)
    
        # Use morphological transformations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Check if contours exist and find the largest one (which should be the screen)
        if contours:
            # Sort contours by area and take the largest one
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]
        
            # Get the bounding rectangle for the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y), (x + w, y + h)  # Return top-left and bottom-right coordinates
        else:
            return None, None  # No screen detected
    @staticmethod
    def calculate_optical_flow(prev_frame, current_frame, prev_points, lk_params):
        """Calculate optical flow using the Lucas-Kanade method."""
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        current_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_points, None, **lk_params)
        
        # Select good points
        good_new = current_points[st == 1]
        good_old = prev_points[st == 1]
        
        return good_new, good_old, current_points

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Example usage
# prev_frame = ...  # Previous frame from video
# current_frame = ...  # Current frame from video
# prev_points = ...  # Points to track (e.g., from cv2.goodFeaturesToTrack)
    def track_ball_movement(self, current_pos, current_size, tracker, kalman_filter, timestamp, params,current_frame, previous_direction='unknown'):
        """Track ball movement and detect hits with enhanced stability"""
        
        if current_pos is not None:
            near_edge = self.check_near_edge(current_pos, params)
            self.update_confidence(tracker, near_edge)
        
        result = self.update_kalman_filter(kalman_filter, current_pos, current_size, tracker)
        
        if result is None:
            return None, tracker
        
        predicted_pos, predicted_radius, predicted_radius_velocity = result
        
        predicted_pos = self.smooth_transition(predicted_pos, tracker, params)
        
        good_new, good_old = None, None
        if current_pos and current_size and tracker['confidence'] > tracker['min_tracking_confidence']:
            if 'prev_frame' in tracker and 'prev_points' in tracker:
                good_new, good_old, current_points = self.calculate_optical_flow(tracker['prev_frame'], current_frame, tracker['prev_points'], lk_params)
                tracker['prev_points'] = current_points
                
                # Update tracker with optical flow points
                tracker['good_new'] = good_new
                tracker['good_old'] = good_old
            
            tracker = self.update_tracker(tracker, predicted_pos, predicted_radius, timestamp, kalman_filter, good_new, good_old)
            tracker = self.detect_direction_and_hits(tracker, predicted_pos, predicted_radius, timestamp, previous_direction, params)
        
        tracker['prev_frame'] = current_frame
        
        return (predicted_pos, predicted_radius), tracker
            
        


    def detect_object(self, frame, color_samples, screen_top_left=None, screen_bottom_right=None, params=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = None
        for sample in color_samples:
            lower_bound = np.array(sample[0], dtype=np.uint8)
            upper_bound = np.array(sample[1], dtype=np.uint8)
            temp_mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            if mask is None:
                mask = temp_mask
            else:
                mask = cv2.bitwise_or(mask, temp_mask)

        if screen_top_left and screen_bottom_right:
            screen_mask = np.zeros_like(mask)
            cv2.rectangle(screen_mask, screen_top_left, screen_bottom_right, 255, -1)
            mask = cv2.bitwise_and(mask, screen_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 200:
                (x, y), radius = cv2.minEnclosingCircle(c)
                
                # Check params and apply default values if needed
                min_size = 0
                max_size = 100
                if params and params.get('min_size', 0) > 0 and params.get('max_size', 0) > 0:
                    min_size = params['min_size']
                    max_size = params['max_size']
                
                # Check radius condition
                if min_size <= radius <= max_size:
                    return (int(x), int(y)), int(radius)
        return None, None


    def process_frame(self, frame, timestamp):
        """Process a single frame"""
        # Get current parameters
        params = self.get_parameters()
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

                # Detect the screen
        screen_top_left, screen_bottom_right = self.detect_screen(frame)
        if screen_top_left and screen_bottom_right:
        # Draw the detected screen (white rectangle) on the frame
            cv2.rectangle(frame, screen_top_left, screen_bottom_right, (0, 255, 0), 2)  # Green rectangle
        
        
        # Detect balls using calibrated colors.
        blue_ball, blue_radius = self.detect_object(frame, self.blue_samples, screen_top_left, screen_bottom_right, params)
        yellow_ball, yellow_radius = self.detect_object(frame, self.yellow_samples, screen_top_left, screen_bottom_right, params)

        # Update trackers with Kalman predictions
        blue_result = self.track_ball_movement(
        blue_ball, blue_radius, self.blue_tracker, self.blue_kalman, timestamp, params, frame, self.blue_tracker['direction'])
        yellow_result = self.track_ball_movement(
        yellow_ball, yellow_radius, self.yellow_tracker, self.yellow_kalman, timestamp, params, frame, self.yellow_tracker['direction'])
        if blue_result is not None:
            blue_predicted, self.blue_tracker = blue_result
        else:
            blue_predicted = None

        if yellow_result is not None:
            yellow_predicted, self.yellow_tracker = yellow_result
            # Ensure yellow_predicted is correctly formatted
            if isinstance(yellow_predicted, tuple) and len(yellow_predicted) == 2:
                pos, radius = yellow_predicted  # Extract position and radius

                if isinstance(pos, np.ndarray) and len(pos) == 2:
                    yellow_predicted = (int(pos[0]), int(pos[1]))  # Convert (x, y) to tuple
                else:
                    yellow_predicted = None  # Invalid data, set to None
            else:
                yellow_predicted = None  # Ensure invalid cases are handled
        if blue_ball:
            cv2.circle(frame, blue_ball, blue_radius, (255, 0, 0), 2)
            if blue_predicted is not None:
                blue_predicted, self.blue_tracker = blue_result
        # Format blue_predicted similar to yellow_predicted
                if isinstance(blue_predicted, tuple) and len(blue_predicted) == 2:
                    pos, radius = blue_predicted  # Extract position and radius
                    if isinstance(pos, np.ndarray) and len(pos) == 2:
                        blue_predicted = (int(pos[0]), int(pos[1]))  # Convert (x, y) to tuple
                    else:
                        blue_predicted = None  # Invalid data, set to None
                else:
                    blue_predicted = None  # Ensure invalid cases are handled
            else:
                blue_predicted = None  # Ensure invalid cases are handled

                #self.blue_trajectory.append(tuple(map(int, blue_predicted)))
            
            cv2.putText(frame, f"Blue Speed: {np.linalg.norm(self.blue_tracker['velocity']):.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Direction: {self.blue_tracker['direction']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if self.blue_tracker['hit_wall']:
                cv2.putText(frame, "BLUE BALL HIT!", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Process yellow ball
        if yellow_ball:
            cv2.circle(frame, yellow_ball, yellow_radius, (0, 255, 255), 2)
            if yellow_predicted is not None:
                cv2.circle(frame, tuple(map(int, yellow_predicted)), 3, (0, 255, 255), -1)
                #self.yellow_trajectory.append(tuple(map(int, yellow_predicted)))
            cv2.putText(frame, f"Yellow Speed: {np.linalg.norm(self.yellow_tracker['velocity']):.1f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Direction: {self.yellow_tracker['direction']}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if self.yellow_tracker['hit_wall']:
                cv2.putText(frame, "YELLOW BALL HIT!", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if blue_ball and blue_radius:
            self.blue_radius_history.append(blue_radius)
        if yellow_ball and yellow_radius:
            self.yellow_radius_history.append(yellow_radius)


        # # Draw trajectories
        # if len(self.blue_trajectory) > 1:
        #     pts = np.array(list(self.blue_trajectory), np.int32)
        #     pts = pts.reshape((-1, 1, 2))
        #     cv2.polylines(frame, [pts], False, (255, 0, 0), 2)
            
        # if len(self.yellow_trajectory) > 1:
        #     pts = np.array(list(self.yellow_trajectory), np.int32)
        #     pts = pts.reshape((-1, 1, 2))
        #     cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
        return frame

    def plot_radius_live(self):
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("Ball Radius Over Time")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Radius")
        
        line_blue, = ax.plot([], [], label='Blue Ball Radius', color='blue')
        line_yellow, = ax.plot([], [], label='Yellow Ball Radius', color='yellow')
        ax.legend()

        while self.running:
            line_blue.set_data(range(len(self.blue_radius_history)), list(self.blue_radius_history))
            line_yellow.set_data(range(len(self.yellow_radius_history)), list(self.yellow_radius_history))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)

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