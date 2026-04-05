# Pygame rendering, pixel-array batch draw for large particle coutns
import pygame
import numpy as np
from simulation.body import Body
from collections import deque
from config import FPS

class Renderer:
    def __init__(self, width: int, height: int, title: str = "N-Body Simulator"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.trails: dict[str, deque] = {}  # body name -> deque of world positions (metres)
        self.trail_length = 200
        self.zoom = 1.0
        self.steps_per_frame = 30
        self.show_trails = True
        self.cycle_integrator = False  # set True by handle_events when I is pressed
        self.reset_time = False        # set True by handle_events when K is pressed

    def world_to_screen(self, position: np.ndarray, scale: float, offset: np.ndarray) -> tuple:
        """Convert world coordinates (metres) to screen pixels.
        Returns (-1, -1) for NaN/inf positions (treated as off-screen by all callers).
        Clamps finite values to a range pygame can safely handle.
        """
        x = float(position[0]) * scale + float(offset[0])
        y = float(position[1]) * scale + float(offset[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return (-1, -1)
        return (int(np.clip(x, -1_000_000, 1_000_000)),
                int(np.clip(y, -1_000_000, 1_000_000)))

    def handle_events(self) -> bool:
        # reset one-shot flags each frame
        self.cycle_integrator = False
        self.reset_time = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.steps_per_frame = min(self.steps_per_frame * 2, 3840)
                    self.trails.clear()
                if event.key == pygame.K_MINUS:
                    self.steps_per_frame = max(self.steps_per_frame // 2, 1)
                    self.trails.clear()
                if event.key == pygame.K_r:
                    self.steps_per_frame = 30
                    self.trails.clear()
                if event.key == pygame.K_i:
                    self.cycle_integrator = True
                if event.key == pygame.K_k:
                    self.reset_time = True
                if event.key == pygame.K_l:
                    self.show_trails = not self.show_trails
                    self.trails.clear()
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.zoom *= 1.1   # scroll up = zoom in
                else:
                    self.zoom *= 0.9   # scroll down = zoom out
                self.trails.clear()  # clear trails when zooming
        return True

    def draw(self, bodies: list[Body], scale: float, offset: np.ndarray, sim_time: float, integrator_name: str) -> None:
        self.screen.fill((0, 0, 0))

        hud_width = 200
        hud_x = self.width - hud_width

        for body in bodies:
            screen_pos = self.world_to_screen(body.position, scale, offset)

            # update trail with world position (always accumulate so toggling back on shows recent history)
            if body.name not in self.trails:
                self.trails[body.name] = deque(maxlen=self.trail_length)
            self.trails[body.name].append(body.position.copy())

            # draw trail — convert world positions to screen at draw time
            if self.show_trails and self.zoom > 0.5:
                trail = self.trails[body.name]
                for i in range(1, len(trail)):
                    p1 = self.world_to_screen(trail[i - 1], scale, offset)
                    p2 = self.world_to_screen(trail[i], scale, offset)
                    # skip segments where either endpoint is off the safe draw area
                    if not (-10_000 <= p1[0] <= self.width + 10_000 and
                            -10_000 <= p1[1] <= self.height + 10_000 and
                            -10_000 <= p2[0] <= self.width + 10_000 and
                            -10_000 <= p2[1] <= self.height + 10_000):
                        continue
                    alpha = int(255 * i / len(trail))  # fade older points
                    color = tuple(int(c * alpha / 255) for c in body.color)
                    pygame.draw.line(self.screen, color, p1, p2, 1)

            # draw body and label only when on screen (guards against C int overflow for off-screen bodies)
            on_screen = 0 <= screen_pos[0] <= self.width and 0 <= screen_pos[1] <= self.height
            if on_screen:
                pygame.draw.circle(self.screen, body.color, screen_pos, int(body.radius))
                label = self.font.render(body.name, True, (180, 180, 180))
                self.screen.blit(label, (screen_pos[0] + 15, screen_pos[1] - 6))

        # --- HUD panel ---
        panel = pygame.Surface((hud_width, self.height), pygame.SRCALPHA)
        panel.fill((15, 15, 25, 210))
        self.screen.blit(panel, (hud_x, 0))
        pygame.draw.line(self.screen, (60, 60, 90), (hud_x, 0), (hud_x, self.height), 1)

        days = sim_time / 86400
        years = days / 365.25
        days_per_sec = self.steps_per_frame * FPS
        trails_status = "ON" if self.show_trails else "OFF"
        trails_color = (100, 220, 100) if self.show_trails else (200, 100, 100)

        # (text, color, is_header)
        hud_lines = [
            ("SIMULATION",                  (180, 180, 255), True),
            (None, None, False),
            ("Time",                        (130, 130, 200), True),
            (f"Day:  {days:>10,.1f}",       (220, 220, 220), False),
            (f"Year: {years:>10.2f}",       (220, 220, 220), False),
            (None, None, False),
            ("Speed",                       (130, 130, 200), True),
            (f"{self.steps_per_frame} days / frame",  (220, 220, 220), False),
            (f"{days_per_sec} days / sec",            (220, 220, 220), False),
            (None, None, False),
            ("Integrator",                  (130, 130, 200), True),
            (integrator_name,               (220, 220, 220), False),
            (None, None, False),
            ("Display",                     (130, 130, 200), True),
            (f"Trails: {trails_status}",    trails_color,    False),
            (None, None, False),
            ("Controls",                    (130, 130, 200), True),
            ("[+ / =]  2x speed",           (170, 170, 170), False),
            ("[  -  ]  0.5x speed",         (170, 170, 170), False),
            ("[  R  ]  reset speed",        (170, 170, 170), False),
            ("[  I  ]  cycle integrator",   (170, 170, 170), False),
            ("[  K  ]  reset time",         (170, 170, 170), False),
            ("[  L  ]  toggle trails",      (170, 170, 170), False),
        ]

        y = 16
        pad = 12
        line_height = 19
        for text, color, is_header in hud_lines:
            if text is None:
                y += 6
                continue
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (hud_x + pad, y))
            if is_header:
                # underline headers
                line_y = y + surf.get_height() + 1
                pygame.draw.line(self.screen, (60, 60, 90),
                                 (hud_x + pad, line_y),
                                 (self.width - pad, line_y), 1)
                y += line_height + 3
            else:
                y += line_height

        pygame.display.flip()

    def tick(self, fps: int) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
