# Barnes-Hut QuadTree: insert, centre-of-mass, force walk
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import math
from simulation.body import Body
from config import G


class BoundingBox:
    '''A square region of space centred at (cx, cy).

    Every node in the quadtree owns one of these - it defines the patch of sky
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
        nw = BoundingBox(self.cx - half, self.cy + half, half)   # top-left
        ne = BoundingBox(self.cx + half, self.cy + half, half)   # top-right
        sw = BoundingBox(self.cx - half, self.cy - half, half)   # bottom-left
        se = BoundingBox(self.cx + half, self.cy - half, half)   # bottom-right
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
        self.boundary       = boundary          # the spatial region this node covers
        self.bodies: list[Body] = []            # body bucket - only populated for leaf nodes
        self.total_mass     = 0.0               # total mass of all bodies in this subtree
        self.centre_of_mass = np.zeros(2)       # mass-weighted average position of the subtree
        self.nw: Optional[QuadTree] = None      # north-west child (top-left quadrant)
        self.ne: Optional[QuadTree] = None      # north-east child (top-right quadrant)
        self.sw: Optional[QuadTree] = None      # south-west child (bottom-left quadrant)
        self.se: Optional[QuadTree] = None      # south-east child (bottom-right quadrant)
        self.divided = False                    # True once this node has been subdivided

        # Pre-computed arrays populated by compute_mass_distribution().
        # These let compute_force() evaluate a leaf bucket with NumPy ops instead of a Python loop.
        self.leaf_positions: Optional[np.ndarray] = None   # shape (n, 2) - body positions in the bucket
        self.leaf_masses:    Optional[np.ndarray] = None   # shape (n,)   - body masses in the bucket
        self.leaf_ids:       Optional[np.ndarray] = None   # shape (n,)   - id(body) for self-exclusion

    def insert(self, body: Body) -> bool:
        '''Place a body into the appropriate leaf node, subdividing if needed.'''

        # If the body is outside this node's region, reject it immediately.
        # The caller (insert_into_children) tries each of the four children in order,
        # so only the one whose boundary contains the body will accept it.
        if not self.boundary.contains(body.position):
            return False

        if not self.divided:
            # This is a leaf node - add the body to the bucket.
            self.bodies.append(body)

            # If the bucket has exceeded the capacity limit, subdivide and push
            # all bodies (including the one we just added) down to the children.
            if len(self.bodies) > self.MAX_CAPACITY:
                self.subdivide()

                # Safety check: if every body in the bucket is at the exact same coordinate,
                # they will all land in the same child and trigger another overflow, causing
                # infinite recursion. We nudge the newest body slightly to break the deadlock.
                if all(np.allclose(b.position, self.bodies[0].position) for b in self.bodies):
                    body.position = body.position.copy()
                    body.position[0] += 1e3     # 1 km offset - negligible compared to simulation scales

                # Re-insert every body into the now-created children.
                for b in self.bodies:
                    self.insert_into_children(b)

                # Clear the bucket - this node is now an internal node, not a leaf.
                self.bodies = []

            return True

        # This node is already subdivided (internal node) - pass the body straight to children.
        return self.insert_into_children(body)

    def compute_mass_distribution(self) -> None:
        '''Walk the tree bottom-up and compute total_mass and centre_of_mass for every node.

        Leaf nodes get their values directly from the body they hold.
        Internal nodes aggregate their children — total mass is just a sum,
        centre of mass is a weighted average of the children's centres.
        '''

        # Base case: leaf node, set mass and centre of mass from the body
        if not self.divided:
            # Leaf node: sum over the individual bodies in the bucket.
            self.total_mass     = 0.0
            self.centre_of_mass = np.zeros(2)

            for b in self.bodies:
                self.total_mass     += b.mass
                # Accumulate the numerator of the CoM formula: Sum(m_i * r_i)
                self.centre_of_mass += b.mass * b.position

            # Divide by total mass to get the mass-weighted average position.
            # Guard against an empty bucket (total_mass = 0) to avoid division by zero.
            if self.total_mass > 0.0:
                self.centre_of_mass /= self.total_mass

            # Pre-compute flat arrays for the vectorised leaf evaluation in compute_force().
            # This small loop (at most 16 iterations) runs once per leaf per frame - not per body.
            # Storing the arrays here avoids allocating new NumPy arrays inside the hot force loop.
            n = len(self.bodies)
            if n > 0:
                self.leaf_positions = np.empty((n, 2), dtype=np.float64)
                self.leaf_masses    = np.empty(n,      dtype=np.float64)
                self.leaf_ids       = np.empty(n,      dtype=np.int64)
                for i, b in enumerate(self.bodies):
                    self.leaf_positions[i] = b.position
                    self.leaf_masses[i]    = b.mass
                    self.leaf_ids[i]       = id(b)   # CPython object id used as a unique integer key
            else:
                self.leaf_positions = None
                self.leaf_masses    = None
                self.leaf_ids       = None
            return

        # Internal node: aggregate from the four children.
        # Because we recurse into children first, their total_mass and centre_of_mass
        # are already computed by the time we reach this line.
        self.total_mass     = 0.0
        self.centre_of_mass = np.zeros(2)

        for child in (self.nw, self.ne, self.sw, self.se):
            child.compute_mass_distribution()
            self.total_mass     += child.total_mass
            # Add each child's contribution to the CoM numerator: m_child * CoM_child
            # This is equivalent to summing m_i * r_i over all individual bodies in the child.
            self.centre_of_mass += child.total_mass * child.centre_of_mass

        if self.total_mass > 0.0:
            self.centre_of_mass /= self.total_mass

    def compute_force(self, body: Body, theta: float, G: float, softening: float) -> np.ndarray:
        '''Iterative Barnes-Hut traversal using pre-computed leaf arrays.

        This is the pure-Python version of the force walk. It is kept for reference
        and correctness testing. In production, _bh_force_kernel() in physics.py
        performs the same traversal on the flattened tree arrays at native speed.
        '''
        fx, fy = 0.0, 0.0
        stack  = [self]
        mu     = G * body.mass   # factor out G*m_i - it multiplies every term

        while stack:
            node = stack.pop()

            if node.total_mass == 0.0:
                continue

            # Leaf node: vectorised force calculation over the bucket.
            if not node.divided:
                if node.leaf_ids is not None:
                    # Build a boolean mask that excludes the body itself.
                    mask = node.leaf_ids != id(body)
                    if mask.any():
                        pos    = node.leaf_positions[mask]
                        masses = node.leaf_masses[mask]

                        dp       = pos - body.position
                        raw_dist = np.sqrt(dp[:, 0]**2 + dp[:, 1]**2)
                        dist     = raw_dist + softening

                        safe_raw  = np.where(raw_dist == 0.0, 1.0, raw_dist)
                        safe_dist = np.where(dist     == 0.0, 1.0, dist)

                        force_mag = np.where(dist == 0.0, 0.0, mu * masses / (safe_dist * safe_dist))
                        direction = dp / safe_raw[:, np.newaxis]

                        fx += np.dot(force_mag, direction[:, 0])
                        fy += np.dot(force_mag, direction[:, 1])
                continue

            # Internal node: Barnes-Hut opening criterion.
            # Compute the vector from this body to the node's centre of mass.
            dx         = node.centre_of_mass[0] - body.position[0]
            dy         = node.centre_of_mass[1] - body.position[1]
            dist_to_com = math.hypot(dx, dy)

            # Body sits exactly on the CoM - the ratio is undefined, so open the node.
            if dist_to_com == 0.0:
                stack.extend([node.nw, node.ne, node.sw, node.se])
                continue

            # s/d < theta → node is far enough away to approximate as a point mass.
            # s = 2 * half_width (the side length of the node's bounding box)
            # d = dist_to_com
            ratio = (2.0 * node.boundary.half_width) / dist_to_com
            if ratio < theta:
                dist       = dist_to_com + softening
                force_mag  = mu * node.total_mass / (dist * dist)
                fx        += force_mag * dx / dist_to_com
                fy        += force_mag * dy / dist_to_com
            else:
                # Too close to approximate - recurse into the four children.
                stack.extend([node.nw, node.ne, node.sw, node.se])

        return np.array([fx, fy])

    def subdivide(self) -> None:
        '''Create four child nodes covering the four quadrants of this node's boundary.'''
        nw_box, ne_box, sw_box, se_box = self.boundary.subdivide()
        self.nw      = QuadTree(nw_box)
        self.ne      = QuadTree(ne_box)
        self.sw      = QuadTree(sw_box)
        self.se      = QuadTree(se_box)
        self.divided = True
        # Clear any leaf arrays - this node is now internal and they will never be read.
        self.leaf_positions = None
        self.leaf_masses    = None
        self.leaf_ids       = None

    def insert_into_children(self, body: Body) -> bool:
        '''Try to insert `body` into whichever child quadrant contains it.'''
        for child in (self.nw, self.ne, self.sw, self.se):
            if child.insert(body):
                return True
        return False

    # -------------------------------------------------------------------------
    # Tree serialisation for Numba
    # -------------------------------------------------------------------------

    def flatten(self, id_to_idx: dict):
        '''Serialise the entire tree into flat NumPy arrays for the Numba force kernel.

        Numba's @njit mode cannot work with Python objects, so we do a single DFS
        pass and pack each node's data into parallel arrays indexed by an integer node ID.
        The root is assigned ID 0. Child links are stored as integer indices (-1 = absent).
        Leaf body data is concatenated into separate flat arrays; node_leaf_start[node]
        gives the offset into those arrays for the bodies in that leaf.

        id_to_idx maps id(body) → original index in the bodies list, which the kernel
        uses to skip the body that is currently having its force computed.
        '''
        # Dynamically allocate space based on the number of bodies currently in the simulation.
        # Multiplying by 4 provides a massive safety buffer for deep trees.
        MAX_NODES = max(4096, len(id_to_idx) * 4)   

        # Per-node data arrays (one entry per node, indexed by the integer node ID assigned below).
        node_cx         = np.empty(MAX_NODES, dtype=np.float64)   # centre-of-mass x
        node_cy         = np.empty(MAX_NODES, dtype=np.float64)   # centre-of-mass y
        node_mass       = np.empty(MAX_NODES, dtype=np.float64)   # total mass of the subtree
        node_half_w     = np.empty(MAX_NODES, dtype=np.float64)   # half-width of the bounding box
        node_child      = np.full((MAX_NODES, 4), -1, dtype=np.int32)  # indices of nw/ne/sw/se children
        node_leaf_start = np.full(MAX_NODES, -1,  dtype=np.int32)  # offset into flat leaf arrays (-1 = internal)
        node_leaf_count = np.zeros(MAX_NODES,      dtype=np.int32)  # number of bodies in this leaf

        # Flat arrays accumulating leaf body data across all leaves.
        leaf_pos_list  = []   # body positions
        leaf_mass_list = []   # body masses
        leaf_idx_list  = []   # original body indices (0..n-1) for self-exclusion in the kernel

        count = 0   # number of nodes assigned an ID so far

        # DFS stack entries: (node object, parent's integer ID, which child slot we are in (0-3)).
        # parent_id = -1 for the root since it has no parent.
        stack = [(self, -1, -1)]

        while stack:
            node, parent_id, slot = stack.pop()

            # Assign this node the next available integer ID.
            idx    = count
            count += 1

            # Tell the parent which integer ID this child received.
            if parent_id >= 0:
                node_child[parent_id, slot] = idx

            # Copy this node's physics data into the flat arrays.
            node_cx[idx]     = node.centre_of_mass[0]
            node_cy[idx]     = node.centre_of_mass[1]
            node_mass[idx]   = node.total_mass
            node_half_w[idx] = node.boundary.half_width

            if not node.divided:
                # Leaf node: record the start offset and body count, then append body data.
                node_leaf_start[idx] = len(leaf_pos_list)
                node_leaf_count[idx] = len(node.bodies)
                for b in node.bodies:
                    leaf_pos_list.append(b.position.copy())
                    leaf_mass_list.append(b.mass)
                    leaf_idx_list.append(id_to_idx[id(b)])
            else:
                # Internal node: push all four children onto the DFS stack.
                # We record the parent ID and child slot so the child can update node_child
                # once it receives its own integer ID (which we don't know yet at push time).
                for s, child in enumerate((node.nw, node.ne, node.sw, node.se)):
                    stack.append((child, idx, s))

        # Slice the pre-allocated arrays down to the actual number of nodes used.
        n = count
        leaf_pos  = np.array(leaf_pos_list,  dtype=np.float64).reshape(-1, 2)
        leaf_mass = np.array(leaf_mass_list, dtype=np.float64)
        leaf_idx  = np.array(leaf_idx_list,  dtype=np.int32)

        return (node_cx[:n], node_cy[:n], node_mass[:n], node_half_w[:n],
                node_child[:n], node_leaf_start[:n], node_leaf_count[:n],
                leaf_pos, leaf_mass, leaf_idx)
