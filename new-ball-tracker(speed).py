import cv2
import numpy as np
import pygame
import os
from pathlib import Path
from time import time

class BallTracker:
    def __init__(self):
        # Try different camera indices
        for i in range(2):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                break
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
            
        # (1) Color ranges for blue and yellow balls
        #    - You might need to adjust these after testing with your specific lighting.
        self.lower_blue = np.array([90, 80, 80])   # slightly broader range
        self.upper_blue = np.array([130, 255, 255])
        
        self.lower_yellow = np.array([18, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])
        
        # (2) Track history for speed calculation:
        self.blue_history = {
            'prev_radius': None,
            'prev_time': None,
            'speed': 0
        }
        self.yellow_history = {
            'prev_radius': None,
            'prev_time': None,
            'speed': 0
        }
        
        # (3) Keep a short “window” of recent radii for smoothing
        self.blue_radii_buffer = []
        self.yellow_radii_buffer = []
        self.smoothing_window = 5  # how many frames to average
        
        # Initialize Pygame
        pygame.init()
        self.game_running = True

    def detect_ball(self, frame, lower_color, upper_color):
        """Detect a ball of specified color range and return (center, radius)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # (A) Create mask using inRange for the given color
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # (B) Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # (C) Find contours and pick the largest one above a certain area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 100:  # filter out very small areas
                (x, y), radius = cv2.minEnclosingCircle(c)
                return (int(x), int(y)), float(radius)
        
        return None, None

    def smooth_radius(self, radius_buffer, new_radius):
        """
        Add the new radius to a rolling buffer and return the average
        to reduce jitter in radius measurements.
        """
        radius_buffer.append(new_radius)
        if len(radius_buffer) > self.smoothing_window:
            radius_buffer.pop(0)
        return np.mean(radius_buffer)

    def calculate_x_speed(self, current_radius, history):
        """Calculate speed based on rate of change in radius (approx. depth movement)."""
        current_time = time()
        prev_radius = history['prev_radius']
        prev_time = history['prev_time']
        
        if prev_radius is not None and prev_time is not None:
            dt = current_time - prev_time
            if dt > 0:
                # Negative speed => ball is moving closer (radius increasing)
                # Positive speed => ball is moving away (radius decreasing)
                radius_change = prev_radius - current_radius
                speed = radius_change / dt
                history['speed'] = speed
        
        # Update history for next frame
        history['prev_radius'] = current_radius
        history['prev_time'] = current_time
        
        return history['speed']

    def run(self):
        while self.game_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # (1) Detect blue and yellow balls
            blue_center, blue_radius = self.detect_ball(frame, self.lower_blue, self.upper_blue)
            yellow_center, yellow_radius = self.detect_ball(frame, self.lower_yellow, self.upper_yellow)
            
            # (2) Process Blue Ball
            if blue_center and blue_radius:
                # Smooth out the radius
                smoothed_blue_radius = self.smooth_radius(self.blue_radii_buffer, blue_radius)
                # Calculate speed based on the smoothed radius
                blue_speed = self.calculate_x_speed(smoothed_blue_radius, self.blue_history)
                
                # Draw the detected circle
                cv2.circle(frame, blue_center, int(smoothed_blue_radius), (255, 0, 0), 2)
                cv2.putText(frame, f"Blue Speed: {blue_speed:.2f}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # (3) Process Yellow Ball
            if yellow_center and yellow_radius:
                # Smooth out the radius
                smoothed_yellow_radius = self.smooth_radius(self.yellow_radii_buffer, yellow_radius)
                # Calculate speed based on the smoothed radius
                yellow_speed = self.calculate_x_speed(smoothed_yellow_radius, self.yellow_history)
                
                # Draw the detected circle
                cv2.circle(frame, yellow_center, int(smoothed_yellow_radius), (0, 255, 255), 2)
                cv2.putText(frame, f"Yellow Speed: {yellow_speed:.2f}", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # (4) Show camera feed
            cv2.imshow('Ball Tracker', frame)
            
            # (5) Handle quit events
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.game_running = False

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    tracker = BallTracker()
    tracker.run()
