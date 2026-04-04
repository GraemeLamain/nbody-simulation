# Body Dataclass: mass, position, velocity, force all as numpy arrays
from dataclasses import dataclass       # this basically just writes the init function for us
import numpy as np

@dataclass
class Body:
    name: str
    mass: float
    position: np.ndarray    # [x, y] in meters
    velocity: np.ndarray    # [vx, vy] in m/s
    force: np.ndarray = None
    color: tuple = (255, 255, 255)
    radius: float = 5.0

    def __post_init__(self):
        if self.force is None:
            self.force = np.zeros(2)

    def reset_force(self):
        self.force = np.zeros(2)