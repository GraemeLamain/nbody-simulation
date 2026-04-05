# Gravity with both the naive version and the Barnes-Hut version
import numpy as np
from simulation.body import Body
from config import G, SOFTENING, THETA
from simulation.quadtree import QuadTree, BoundingBox

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

def compute_gravity_barnes_hut(bodies: list[Body], theta: float = THETA, softening: float = SOFTENING) -> None:
    '''Compute gravitational forces using the Barnes-Hut O(n log n) algorithm.

    Builds a fresh quadtree every call, computes the mass distribution bottom-up,
    then walks the tree for each body using the opening criterion theta.
    '''
    # reset all forces before recomputing
    for body in bodies:
        body.reset_force()

    # Build a square bounding box enclosing all bodies with 10% padding
    positions = np.array([b.position for b in bodies])
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)

    extent = max(max_x - min_x, max_y - min_y)
    half_width = (extent / 2) * 1.1
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    if half_width == 0.0:   # edge case: all bodies at same position
        half_width = 1.0

    # Build the quadtree
    root = QuadTree(BoundingBox(cx, cy, half_width))
    for body in bodies:
        root.insert(body)

    # Compute the mass distribution
    root.compute_mass_distribution()

    # Compute the forces
    for body in bodies:
        body.force = root.compute_force(body, theta, G, softening)
