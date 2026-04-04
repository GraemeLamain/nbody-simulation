# Gravity with both the naive version and the Barnes-Hut version
import numpy as np
from simulation.body import Body
from config import G, SOFTENING

def compute_gravity_naive(bodies: list[Body]) -> None:
    '''Compute pairwise gravitational forces between all bodies. O(n^2)'''

    # reset all forces before recomputing
    for body in bodies:
        body.reset_force()
    
    # compute forces between every unique pair
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            b1 = bodies[i]
            b2 = bodies[j]

            displacement = b2.position - b1.position                # vector from b1 to b2
            distance = np.linalg.norm(displacement) + SOFTENING

            magnitude = G * b1.mass * b2.mass / (distance**2)
            force = magnitude * displacement / np.linalg.norm(displacement)

            b1.force += force       # b1 is pulled toward b2
            b2.force -= force       # b2 is pulled toward b1 (Newton's third law)
