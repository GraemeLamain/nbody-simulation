# body.py
from dataclasses import dataclass
import numpy as np

@dataclass
class Body:
    name:     str
    mass:     float
    position: np.ndarray    # [x, y] metres
    velocity: np.ndarray    # [vx, vy] m/s
    force:    np.ndarray = None  # [fx, fy] Newtons
    color:  tuple = (255, 255, 255)
    radius: float = 5.0

    def __post_init__(self):
        # can't use mutable numpy array as a dataclass default, so set it here
        if self.force is None:
            self.force = np.zeros(2)

    def reset_force(self):
        self.force = np.zeros(2)