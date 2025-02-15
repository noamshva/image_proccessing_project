import cv2
import numpy as np
import pygame
import os
import time
import threading
from live_ball_tracker import BallTracker
from baloon import Game, BALLOON_SIZE, WIDTH, HEIGHT, RED, GREEN
# Import calibration functions from our module.
from color_calibration import calibrate_colors, load_calibration

class CombinedGame:
    def __init__(self, video_path=None, color_ranges=None):
        # Initialize ball tracker with provided color ranges.
        self.tracker = BallTracker(color_ranges=color_ranges)
        self.paused = False
        # Open video file if provided; otherwise, use default camera.
        if video_path:
            self.tracker.cap = cv2.VideoCapture(video_path)
            if not self.tracker.cap.isOpened():
                raise ValueError(f"Error: Cannot open video file: {video_path}")
        else:
            self.tracker.cap = cv2.VideoCapture(0)
            if not self.tracker.cap.isOpened():
                raise ValueError("Error: Cannot access the camera.")
        
        # Initialize balloon game.
        pygame.init()
        if os.name == 'nt':
            os.environ['SDL_VIDEO_WINDOW_POS'] = "2000,0"  # For second monitor.
        
        self.game = Game()
        self.game_running = True
        self.clock = pygame.time.Clock()

    def check_balloon_hit(self, ball_pos, balloon_rect, ball_radius):
        """Check if the ball hit the balloon using distance-based detection."""
        balloon_center = (balloon_rect.x + balloon_rect.width // 2,balloon_rect.y + balloon_rect.height // 2)
        distance = np.sqrt((ball_pos[0] - balloon_center[0])**2 +(ball_pos[1] - balloon_center[1])**2)
        return distance < (ball_radius + balloon_rect.width // 2)

    def run(self):
        while self.game_running:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('p'):
                self.paused = not self.paused
                print("Paused" if self.paused else "Resumed")
                self.game.game_state = "paused" if self.paused else "playing"
            if not self.paused:  # Only read frames when not paused
                ret, frame = self.tracker.cap.read()
                if not ret:
                    print("End of video or error reading frame.")
                    break

            current_time = time.time()
            processed_frame = self.tracker.process_frame(frame, current_time)
        # Process Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.game_running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                        print("Paused" if self.paused else "Resumed")
                    elif event.key == pygame.K_q:
                        print("Quitting game...")
                        self.game_running = False
                        break

            if not self.paused:
                if self.game.game_state == "playing":
                    self.game.update()
                self.game.draw()
                pygame.display.flip()

                cv2.imshow('Ball Tracking', processed_frame)
                # print(f"Frame processed at time {current_time}")

            key = cv2.waitKey(1) & 0xFF  # OpenCV Key Handling
            if key == ord('q'):  # Ensure 'q' quits both OpenCV and the game
                self.game_running = False
                break

            self.clock.tick(60)

        # Cleanup
        self.tracker.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()



def main():
    video_path = "ball_video.mp4"  # enter the video path
    run_on_video = True  # Change to False to use the real-time camera.
    use_saved_calibration = True  # Change to False to force new calibration samples.
    show_live_ball_radius = True  # Plot live ball radius found for both balls (just for debugging)
    try:
        if use_saved_calibration:
            try:
                # Try loading previously saved calibration data.
                color_ranges = load_calibration()
                print("Loaded calibration data from YAML.")
            except Exception as e:
                print("Failed to load calibration data, proceeding with new sampling.")
                color_ranges = calibrate_colors(video_path, max_samples=10)
        else:
            # Calibrate (sample) colors for both Blue and Yellow.
            color_ranges = calibrate_colors(video_path, max_samples=10)
        
        if run_on_video:
            print("Running on video...")
            game = CombinedGame(video_path=video_path, color_ranges=color_ranges)
        else:
            print("Running on real-time camera...")
            game = CombinedGame(color_ranges=color_ranges)

        if show_live_ball_radius:
            threading.Thread(target=game.tracker.plot_radius_live, daemon=True).start()

        game.run()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()
