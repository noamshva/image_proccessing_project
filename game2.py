# game.py
import pygame
import random
import sys
import os
import math
from time import time

pygame.init()

# Set the SDL window position to the projector monitor (adjust as needed)
if os.name == 'nt':
    os.environ['SDL_VIDEO_WINDOW_POS'] = "2000,0"

# Constants
BALLOON_SIZE = 200
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

DIFFICULTY_SETTINGS = {
    'easy': {'spawn_rate': 210, 'speed_range': (1, 2), 'max_balloons': 4},
    'normal': {'spawn_rate': 180, 'speed_range': (2, 3), 'max_balloons': 6},
    'hard': {'spawn_rate': 150, 'speed_range': (2, 4), 'max_balloons': 8}
}
GAME_DURATION = 120
POP_DURATION = 5

# Fixed virtual resolution
WIDTH = 1920
HEIGHT = 1080

# Create a full-screen window on the projector
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Balloon Pop Game")
clock = pygame.time.Clock()

# Particle and PopAnimation classes (unchanged)
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
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
        self.rect = self.image.get_rect(center=(x + BALLOON_SIZE // 2, y + BALLOON_SIZE // 2))
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
        if self.color == RED:
            self.image = pygame.image.load('red_balloon.png').convert_alpha()
        else:
            self.image = pygame.image.load('green_balloon.png').convert_alpha()
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
        self.game_state = "difficulty_select"
        self.difficulty = None
        self.font = pygame.font.Font(None, 48)
        self.start_time = None
        self.time_left = GAME_DURATION
        self.event_queue = None  # This will be set externally

    def spawn_balloons(self):
        settings = DIFFICULTY_SETTINGS[self.difficulty]
        red_count = len([b for b in self.balloons if b.color == RED])
        green_count = len([b for b in self.balloons if b.color == GREEN])
        max_per_color = settings['max_balloons'] // 2
        if self.frame_count % settings['spawn_rate'] == 0:
            if red_count < max_per_color:
                balloon = Balloon(RED, settings['speed_range'])
                balloon.rect.x = random.randint(0, WIDTH // 2 - BALLOON_SIZE)
                self.balloons.add(balloon)
            if green_count < max_per_color:
                balloon = Balloon(GREEN, settings['speed_range'])
                balloon.rect.x = random.randint(WIDTH // 2, WIDTH - BALLOON_SIZE)
                self.balloons.add(balloon)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Convert the actual mouse position to the virtual resolution.
                actual_size = pygame.display.get_surface().get_size()
                raw_pos = pygame.mouse.get_pos()
                # Scale the mouse coordinates:
                pos = (raw_pos[0] * WIDTH / actual_size[0], raw_pos[1] * HEIGHT / actual_size[1])
                
                if self.game_state == "difficulty_select":
                    # The difficulty buttons are defined in the virtual space.
                    for i, diff in enumerate(['easy', 'normal', 'hard']):
                        rect = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 50 + i * 70, 200, 50)
                        if rect.collidepoint(pos):
                            self.difficulty = diff
                            self.game_state = "start"
                elif self.game_state == "start":
                    self.game_state = "playing"
                    self.start_time = time()
                elif self.game_state == "playing":
                    # Process clicks on balloons during play.
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
        # Process events from the camera thread
        if self.event_queue is not None:
            while not self.event_queue.empty():
                point = self.event_queue.get()
                self.pop_balloon_at_point(point)
        if self.game_state == "playing":
            self.time_left = GAME_DURATION - int(time() - self.start_time)
            if self.time_left <= 0:
                self.game_state = "end"
                return
            self.frame_count += 1
            self.spawn_balloons()
            self.balloons.update()
            self.pop_animations.update()

    def draw(self, surface):
        surface.fill(WHITE)
        if self.game_state == "difficulty_select":
            title_text = self.font.render("Select Difficulty", True, BLACK)
            title_rect = title_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 120))
            surface.blit(title_text, title_rect)
            for i, diff in enumerate(['easy','normal','hard']):
                button_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 - 50 + i*70, 200, 50)
                pygame.draw.rect(surface, BLACK, button_rect, 2)
                diff_text = self.font.render(diff.title(), True, BLACK)
                text_rect = diff_text.get_rect(center=button_rect.center)
                surface.blit(diff_text, text_rect)
        elif self.game_state == "start":
            start_text = self.font.render("Click to Start Game (ESC to quit)", True, BLACK)
            text_rect = start_text.get_rect(center=(WIDTH/2, HEIGHT/2))
            surface.blit(start_text, text_rect)
            instructions = ["Player 1 (Green) vs Player 2 (Red)", "2 Minutes Competition", "Click balloons to score points"]
            for i, text in enumerate(instructions):
                inst_text = pygame.font.Font(None, 36).render(text, True, BLACK)
                inst_rect = inst_text.get_rect(center=(WIDTH/2, HEIGHT/2 + 50 + i*30))
                surface.blit(inst_text, inst_rect)
        elif self.game_state == "playing":
            self.balloons.draw(surface)
            self.pop_animations.draw(surface)
            red_text = self.font.render(f"Red: {self.red_score}", True, RED)
            green_text = self.font.render(f"Green: {self.green_score}", True, GREEN)
            time_text = self.font.render(f"Time: {self.time_left}s", True, BLACK)
            surface.blit(red_text, (10, 10))
            surface.blit(green_text, (WIDTH - 200, 10))
            surface.blit(time_text, (WIDTH//2 - 70, 10))
        else:
            if self.green_score == self.red_score:
                winner_text = "Tie"
            else:
                winner_text = "Green" if self.green_score > self.red_score else "Red"
            end_text = self.font.render(f"Game Over! {winner_text} Wins!", True, BLACK)
            score_text = self.font.render(f"Final Score - Red: {self.red_score}, Green: {self.green_score}", True, BLACK)
            restart_text = self.font.render("Press ESC to quit", True, BLACK)
            surface.blit(end_text, (WIDTH//2 - 200, HEIGHT//2 - 50))
            surface.blit(score_text, (WIDTH//2 - 250, HEIGHT//2 + 20))
            surface.blit(restart_text, (WIDTH//2 - 100, HEIGHT//2 + 90))

    def pop_balloon_at_point(self, point):
        for balloon in self.balloons:
            if balloon.rect.collidepoint(point):
                self.pop_animations.add(PopAnimation(balloon.rect.x, balloon.rect.y, balloon.color))
                if balloon.color == RED:
                    self.red_score += 1
                else:
                    self.green_score += 1
                balloon.kill()
                break

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            # Draw the game to a virtual surface with fixed resolution.
            virtual_surface = pygame.Surface((WIDTH, HEIGHT))
            self.draw(virtual_surface)
            # Scale the virtual surface to fit the actual display surface.
            actual_surface = pygame.display.get_surface()
            scaled_surface = pygame.transform.scale(virtual_surface, actual_surface.get_size())
            actual_surface.blit(scaled_surface, (0, 0))
            pygame.display.flip()
            clock.tick(FPS)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
