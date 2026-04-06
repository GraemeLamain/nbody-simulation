# Body dataclass - represents a single point mass in the simulation.
# All values are in SI units (kg, metres, m/s, Newtons) so that the
# physical constants in config.py apply directly without any unit conversion.
from dataclasses import dataclass
import numpy as np

@dataclass
class Body:

    name:     str
    mass:     float
    position: np.ndarray    # [x, y] in metres - where the body is in the 2D plane
    velocity: np.ndarray    # [vx, vy] in m/s  - speed and direction of travel
    force:    np.ndarray = None  # [fx, fy] in Newtons - net gravity from all other bodies
    color:  tuple = (255, 255, 255)  # RGB colour for the renderer - purely visual
    radius: float = 5.0             # display radius in pixels - not a physical size

    def __post_init__(self):
        # dataclass runs __post_init__ automatically after __init__.
        # We set force here rather than as a field default because mutable numpy
        # arrays can't be used as dataclass defaults (they'd be shared across instances).
        if self.force is None:
            self.force = np.zeros(2)

    def reset_force(self):
        # Wipe the force vector before each new gravity calculation.
        # If we didn't do this, forces from the previous timestep would pile up
        # and the simulation would quickly go haywire.
        self.force = np.zeros(2)
