import pygame
import random
import sys
import os
import math
from time import time

pygame.init()

# Constants (same as before)
BALLOON_SIZE = 200
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
# Difficulty settings
DIFFICULTY_SETTINGS = {
    'easy': {'spawn_rate': 210, 'speed_range': (1, 2), 'max_balloons': 4},  # 2 of each color, slow-medium
    'normal': {'spawn_rate': 180, 'speed_range': (2, 3), 'max_balloons': 6},  # 3 of each color, medium speed
    'hard': {'spawn_rate': 150, 'speed_range': (2, 4), 'max_balloons': 8}  # 4 of each color, medium-fast speed
}
GAME_DURATION = 120
POP_DURATION = 5

if os.name == 'nt':
    os.environ['SDL_VIDEO_WINDOW_POS'] = "2000,0"

WIDTH = 1920
HEIGHT = 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Balloon Pop Game")
clock = pygame.time.Clock()

# Particle and PopAnimation classes remain the same
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * 3.14159)
        speed = random.uniform(2, 8)
        self.dx = speed * math.cos(angle)
        self.dy = speed * math.sin(angle)
        self.size = random.randint(3, 8)
        self.life = POP_DURATION

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.dy += 0.2
        self.life -= 1
        self.size = max(0, self.size - 0.2)

class PopAnimation(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.image = pygame.Surface((BALLOON_SIZE * 3, BALLOON_SIZE * 3), pygame.SRCALPHA)
        self.color = color
        self.frame = 0
        self.rect = self.image.get_rect(center=(x + BALLOON_SIZE//2, y + BALLOON_SIZE//2))
        self.particles = [Particle(BALLOON_SIZE * 1.5, BALLOON_SIZE * 1.5, color) for _ in range(20)]
        
    def update(self):
        self.frame += 1
        if self.frame >= POP_DURATION:
            self.kill()
        else:
            self.image.fill((0, 0, 0, 0))
            for particle in self.particles:
                particle.update()
                if particle.size > 0:
                    pygame.draw.circle(self.image, (*self.color[:3], 255), 
                                    (int(particle.x), int(particle.y)), 
                                    int(particle.size))

class Balloon(pygame.sprite.Sprite):
    def __init__(self, color, speed_range):
        super().__init__()
        self.color = color
        
        # Load the appropriate balloon image
        if self.color == RED:
            self.image = pygame.image.load('red_balloon.png').convert_alpha()
        else:
            self.image = pygame.image.load('green_balloon.png').convert_alpha()
            
        # Scale image to desired size
        self.image = pygame.transform.scale(self.image, (BALLOON_SIZE, BALLOON_SIZE))
        
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, WIDTH - BALLOON_SIZE)
        self.rect.y = HEIGHT
        self.speed = random.randint(*speed_range)

    def update(self):
        self.rect.y -= self.speed
        if self.rect.bottom < 0:
            self.kill()

class Game:
    def __init__(self):
        self.red_score = 0
        self.green_score = 0
        self.balloons = pygame.sprite.Group()
        self.pop_animations = pygame.sprite.Group()
        self.frame_count = 0
        self.game_state = "difficulty_select"  # New state for difficulty selection
        self.difficulty = None
        self.font = pygame.font.Font(None, 48)
        self.start_time = None
        self.time_left = GAME_DURATION

    def spawn_balloons(self):
        settings = DIFFICULTY_SETTINGS[self.difficulty]
        
        # Count current balloons of each color
        red_count = len([b for b in self.balloons if b.color == RED])
        green_count = len([b for b in self.balloons if b.color == GREEN])
        max_per_color = settings['max_balloons'] // 2  # Ensure even split between colors
        
        # Only spawn if below max for both colors and if it's spawn time
        if self.frame_count % settings['spawn_rate'] == 0:
            # Determine how many of each color to spawn to maintain balance
            red_to_spawn = max_per_color - red_count
            green_to_spawn = max_per_color - green_count
            
            if red_to_spawn > 0:
                balloon = Balloon(RED, settings['speed_range'])
                balloon.rect.x = random.randint(0, WIDTH//2 - BALLOON_SIZE)
                self.balloons.add(balloon)
                
            if green_to_spawn > 0:
                balloon = Balloon(GREEN, settings['speed_range'])
                balloon.rect.x = random.randint(WIDTH//2, WIDTH - BALLOON_SIZE)
                self.balloons.add(balloon)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                if self.game_state == "difficulty_select":
                    # Check which difficulty was clicked
                    for i, diff in enumerate(['easy', 'normal', 'hard']):
                        rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 - 50 + i*70, 200, 50)
                        if rect.collidepoint(pos):
                            self.difficulty = diff
                            self.game_state = "start"
                
                elif self.game_state == "start":
                    self.game_state = "playing"
                    self.start_time = time()
                
                elif self.game_state == "playing":
                    pos = pygame.mouse.get_pos()
                    for balloon in self.balloons:
                        if balloon.rect.collidepoint(pos):
                            self.pop_animations.add(PopAnimation(balloon.rect.x, balloon.rect.y, balloon.color))
                            if balloon.color == RED:
                                self.red_score += 1
                            else:
                                self.green_score += 1
                            balloon.kill()
        return True

    def update(self):
        if self.game_state == "playing":
            self.time_left = GAME_DURATION - int(time() - self.start_time)
            if self.time_left <= 0:
                self.game_state = "end"
                return

            self.frame_count += 1
            self.spawn_balloons()
            self.balloons.update()
            self.pop_animations.update()

    def draw(self):
        screen.fill(WHITE)
        
        if self.game_state == "difficulty_select":
            title_text = self.font.render("Select Difficulty", True, BLACK)
            title_rect = title_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 120))
            screen.blit(title_text, title_rect)
            
            for i, diff in enumerate(['easy', 'normal', 'hard']):
                # Draw button
                button_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 - 50 + i*70, 200, 50)
                pygame.draw.rect(screen, BLACK, button_rect, 2)
                
                # Draw text
                diff_text = self.font.render(diff.title(), True, BLACK)
                text_rect = diff_text.get_rect(center=button_rect.center)
                screen.blit(diff_text, text_rect)
        
        elif self.game_state == "start":
            start_text = self.font.render("Click to Start Game (ESC to quit)", True, BLACK)
            text_rect = start_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            screen.blit(start_text, text_rect)
            
            instructions = [
                "Player 1 (Green) vs Player 2 (Red)",
                "2 Minutes Competition",
                "Click balloons to score points"
            ]
            for i, text in enumerate(instructions):
                inst_text = pygame.font.Font(None, 36).render(text, True, BLACK)
                inst_rect = inst_text.get_rect(center=(WIDTH/2, HEIGHT/2 + 50 + i*30))
                screen.blit(inst_text, inst_rect)
                
        elif self.game_state == "playing":
            self.balloons.draw(screen)
            self.pop_animations.draw(screen)
            
            red_text = self.font.render(f"Red: {self.red_score}", True, RED)
            green_text = self.font.render(f"Green: {self.green_score}", True, GREEN)
            time_text = self.font.render(f"Time: {self.time_left}s", True, BLACK)
            
            screen.blit(red_text, (10, 10))
            screen.blit(green_text, (WIDTH - 200, 10))
            screen.blit(time_text, (WIDTH//2 - 70, 10))
            
        else:
            winner_color = "Green" if self.green_score > self.red_score else "Red"
            if self.green_score == self.red_score:
                winner_color = "Tie"
                
            end_text = self.font.render(f"Game Over! {winner_color} Wins!", True, BLACK)
            score_text = self.font.render(f"Final Score - Red: {self.red_score}, Green: {self.green_score}", True, BLACK)
            restart_text = self.font.render("Press ESC to quit", True, BLACK)
            
            screen.blit(end_text, (WIDTH//2 - 200, HEIGHT//2 - 50))
            screen.blit(score_text, (WIDTH//2 - 250, HEIGHT//2 + 20))
            screen.blit(restart_text, (WIDTH//2 - 100, HEIGHT//2 + 90))
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            clock.tick(FPS-10)  # Slightly faster than normal FPS to speed up game

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()
    sys.exit()