# Pygame rendering, pixel-array batch draw for large particle coutns
import pygame
import numpy as np
from simulation.body import Body
from collections import deque

class Renderer:
    def __init__(self, width: int, height: int, title: str = "N-Body Simulator"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.trails: dict[str, deque] = {}  # body name -> deque of screen positions
        self.trail_length = 200
        self.zoom = 1.0

    def world_to_screen(self, position: np.ndarray, scale: float, offset: np.ndarray) -> tuple:
        """Convert world coordinates (metres) to screen pixels."""
        x = int(position[0] * scale + offset[0])
        y = int(position[1] * scale + offset[1])
        return (x, y)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.zoom *= 1.1   # scroll up = zoom in
                else:
                    self.zoom *= 0.9   # scroll down = zoom out
                self.trails.clear()  # clear trails when zooming
        return True

    def draw(self, bodies: list[Body], scale: float, offset: np.ndarray, sim_time: float) -> None:
        self.screen.fill((0, 0, 0))

        for body in bodies:
            screen_pos = self.world_to_screen(body.position, scale, offset)

            # update trail
            if body.name not in self.trails:
                self.trails[body.name] = deque(maxlen=self.trail_length)
            self.trails[body.name].append(screen_pos)

            # draw trail
            trail = self.trails[body.name]
            for i in range(1, len(trail)):
                alpha = int(255 * i / len(trail))  # fade older points
                color = tuple(int(c * alpha / 255) for c in body.color)
                pygame.draw.line(self.screen, color, trail[i - 1], trail[i], 1)

            # draw body
            if 0 <= screen_pos[0] <= self.width and 0 <= screen_pos[1] <= self.height:
                pygame.draw.circle(self.screen, body.color, screen_pos, int(body.radius))

            # label
            label = self.font.render(body.name, True, (180, 180, 180))
            self.screen.blit(label, (screen_pos[0] + 15, screen_pos[1] - 6))

        # sim time
        days = sim_time / 86400
        time_label = self.font.render(f"Day: {days:.1f}", True, (255, 255, 255))
        self.screen.blit(time_label, (10, 10))

        pygame.display.flip()
    
    def tick(self, fps: int) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()