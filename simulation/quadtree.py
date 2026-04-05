# Barnes-Hut QuadTree: insert, centre-of-mass, force walk
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from simulation.body import Body
from config import G


class BoundingBox:
    '''A square region of space centred at (cx, cy).

    Every node in the quadtree owns one of these — it defines the patch of sky
    that node is responsible for. We keep it square (not rectangular) so that
    subdividing always produces four equal quadrants.
    '''

    def __init__(self, cx: float, cy: float, half_width: float) -> None:
        self.cx = cx                        # x-coordinate of the centre of the box
        self.cy = cy                        # y-coordinate of the centre of the box
        self.half_width = half_width        # half the width of the box

    def contains(self, position: np.ndarray) -> bool:
        # Strict inequalities so bodies on the shared edge always fall into
        # exactly one quadrant when we subdivide
        return (self.cx - self.half_width <= position[0] < self.cx + self.half_width and
                self.cy - self.half_width <= position[1] < self.cy + self.half_width)

    def subdivide(self) -> Tuple[BoundingBox, BoundingBox, BoundingBox, BoundingBox]:
        # Cut the box in half along both axes so each child covers one quadrant
        half = self.half_width / 2
        nw = BoundingBox(self.cx - half, self.cy + half, half)
        ne = BoundingBox(self.cx + half, self.cy + half, half)
        sw = BoundingBox(self.cx - half, self.cy - half, half)
        se = BoundingBox(self.cx + half, self.cy - half, half)
        return nw, ne, sw, se


class QuadTree:
    '''A single node in the Barnes-Hut quadtree.

    The tree has two kinds of nodes:
      - Leaf:     holds at most one Body directly (divided=False, body is set)
      - Internal: holds no body itself but has four child nodes (divided=True)

    The key insight behind Barnes-Hut: instead of computing force from every
    single body in the simulation, we only recurse into a region if it is
    "too close" to approximate. If a group of bodies is far enough away, we
    treat their combined mass as a single point at their centre of mass.
    The threshold is the opening angle theta — smaller theta means more
    accurate but slower, larger means faster but less accurate.
    '''

    def __init__(self, boundary: BoundingBox) -> None:
        self.boundary = boundary                # the region of space this node is responsible for
        self.body: Optional[Body] = None       # only set for leaf nodes
        self.total_mass: float = 0.0            # total mass of all bodies in this node and its children
        self.centre_of_mass: np.ndarray = np.zeros(2)   # centre of mass of all bodies in this node and its children
        self.nw: Optional[QuadTree] = None      # child nodes
        self.ne: Optional[QuadTree] = None
        self.sw: Optional[QuadTree] = None
        self.se: Optional[QuadTree] = None
        self.divided: bool = False              # whether this node has been subdivided

    def insert(self, body: Body) -> bool:
        '''Drop a body into the tree. 
        
        Returns False if it falls outside this node.
        '''

        # If the body is outside the boundary, return False
        if not self.boundary.contains(body.position):
            return False

        # If the node is an empty leaf, insert the body here
        if not self.divided and self.body is None:
            self.body = body
            return True

        # Object in node that is not a leaf, we need to subdivide and evict the existing body
        # We must handle the edge case where two bodies are at the same exact position
        # nudge the newcomer by 1 km so they don't fight over the same leaf node forever
        if not self.divided and self.body is not None:
            existing = self.body
            self.body = None
            self.subdivide()
            if np.allclose(existing.position, body.position):
                body.position = body.position.copy()
                body.position[0] += 1e3
            self.insert_into_children(existing)
            self.insert_into_children(body)
            return True

        # Already divided — pass the body down to whichever child owns it
        return self.insert_into_children(body)

    def compute_mass_distribution(self) -> None:
        '''Walk the tree bottom-up and compute total_mass and centre_of_mass for every node.

        Leaf nodes get their values directly from the body they hold.
        Internal nodes aggregate their children — total mass is just a sum,
        centre of mass is a weighted average of the children's centres.
        '''

        # Base case: leaf node, set mass and centre of mass from the body
        if not self.divided:
            if self.body is not None:
                self.total_mass = self.body.mass
                self.centre_of_mass = self.body.position.copy()
            return

        # Internal node: recurse first so children are ready before we read them
        self.total_mass = 0.0
        self.centre_of_mass = np.zeros(2)
        for child in (self.nw, self.ne, self.sw, self.se):
            child.compute_mass_distribution()
            self.total_mass += child.total_mass
            self.centre_of_mass += child.total_mass * child.centre_of_mass

        if self.total_mass > 0.0:
            self.centre_of_mass /= self.total_mass  # weighted average → true CoM

    def compute_force(self, body: Body, theta: float, G: float, softening: float) -> np.ndarray:
        '''Return the gravitational force this node exerts on body.

        This is where the Barnes-Hut approximation actually happens.
        At each internal node we compute the ratio s/d, where s is the width
        of the node and d is how far away the body is from the node's centre
        of mass. If s/d < theta the node is far enough away that we treat the
        whole thing as a single point — no need to recurse deeper.
        '''

        # If the node is empty, return zero force
        if self.total_mass == 0.0:
            return np.zeros(2)

        if not self.divided:
            # Leaf — direct force calculation, but never from a body to itself
            if self.body is body:   # identity check, not equality
                return np.zeros(2)
            displacement = self.body.position - body.position
            raw_dist = np.linalg.norm(displacement)
            dist = raw_dist + softening     # softening prevents division by zero at close range
            if dist == 0.0:
                return np.zeros(2)
            force_mag = G * body.mass * self.body.mass / dist ** 2
            return force_mag * displacement / raw_dist if raw_dist > 0.0 else np.zeros(2)

        # Internal node — check the opening criterion
        displacement = self.centre_of_mass - body.position
        dist_to_com = np.linalg.norm(displacement)

        if dist_to_com == 0.0:
            # Body is sitting right at the centre of mass — can't approximate,
            # must recurse to avoid accidentally applying self-force
            force = np.zeros(2)
            for child in (self.nw, self.ne, self.sw, self.se):
                force += child.compute_force(body, theta, G, softening)
            return force

        # s/d ratio: 2*half_width is the "width" of this node (s), dist_to_com is d
        ratio = (2.0 * self.boundary.half_width) / dist_to_com
        if ratio < theta:
            # Far enough away — approximate the whole node as a point mass
            dist = dist_to_com + softening
            force_mag = G * body.mass * self.total_mass / dist ** 2
            return force_mag * displacement / dist_to_com

        # Too close or too large to approximate — go deeper
        force = np.zeros(2)
        for child in (self.nw, self.ne, self.sw, self.se):
            force += child.compute_force(body, theta, G, softening)
        return force

    def subdivide(self) -> None:
        ''' Create four child nodes and attach them to this node. '''
        nw_box, ne_box, sw_box, se_box = self.boundary.subdivide()
        self.nw = QuadTree(nw_box)
        self.ne = QuadTree(ne_box)
        self.sw = QuadTree(sw_box)
        self.se = QuadTree(se_box)
        self.divided = True

    def insert_into_children(self, body: Body) -> bool:
        ''' Insert a body into the appropriate child node. '''
        for child in (self.nw, self.ne, self.sw, self.se):
            if child.insert(body):
                return True
        return False
