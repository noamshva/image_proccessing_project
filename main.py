import cv2
import numpy as np
import pygame
import os
from pathlib import Path
import time
from live_ball_tracker import BallTracker
from baloon import Game, BALLOON_SIZE, WIDTH, HEIGHT, RED, GREEN

class CombinedGame:
    def __init__(self):
        # Initialize ball tracker
        self.tracker = BallTracker()
        
        # Initialize balloon game
        pygame.init()
        if os.name == 'nt':
            os.environ['SDL_VIDEO_WINDOW_POS'] = "2000,0"  # For second monitor
        
        self.game = Game()
        self.game_running = True
        self.clock = pygame.time.Clock()

    def check_balloon_hit(self, ball_pos, balloon_rect, ball_radius):
        """Check if ball hit balloon using distance-based detection"""
        balloon_center = (
            balloon_rect.x + balloon_rect.width // 2,
            balloon_rect.y + balloon_rect.height // 2
        )
        
        # Calculate distance between ball and balloon
        distance = np.sqrt(
            (ball_pos[0] - balloon_center[0])**2 + 
            (ball_pos[1] - balloon_center[1])**2
        )
        
        return distance < (ball_radius + balloon_rect.width//2)

    def run(self):
        while self.game_running:
            # Process camera frame
            ret, frame = self.tracker.cap.read()
            if not ret:
                break

            # Get current time for tracking
            current_time = time.time()
            
            # Process frame with ball tracker
            processed_frame = self.tracker.process_frame(frame, current_time)
            
            # Handle Pygame events FIRST
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.game_running = False
                    break
                # Let game handle other events (like difficulty selection)
                if not self.game.handle_events():
                    self.game_running = False
                    break

            # Update and draw game
            if self.game.game_state == "playing":
                self.game.update()
            self.game.draw()
            pygame.display.flip()

            # Show camera feed
            cv2.imshow('Ball Tracking', processed_frame)
            
            # Handle camera window quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.game_running = False

            self.clock.tick(60)

        # Cleanup
        self.tracker.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

def main():
    try:
        game = CombinedGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()