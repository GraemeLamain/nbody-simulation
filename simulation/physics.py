# Three different ways to compute gravity, all based on Newton's law of gravitation.
#
# All three are JIT-compiled by Numba so Python loop overhead doesn't skew any
# performance comparisons between them.
import math
import numpy as np
from numba import njit
from simulation.body import Body
from config import G, SOFTENING, THETA
from simulation.quadtree import QuadTree, BoundingBox


# =============================================================================
# KERNEL 1 - Naive pairwise O(n²/2), Newton's 3rd law
# =============================================================================

@njit(cache=True)
def _naive_force_kernel(body_pos, body_mass, out_forces, G, softening):
    '''The straightforward O(n^2) gravity kernel.

    Loops over every unique pair (i, j) with j > i, computes the mutual force,
    and applies it to both bodies at once using Newton's third law.
    cache=True means Numba writes the compiled binary to disk so it doesn't
    recompile from scratch every run.
    '''
    n = len(body_mass)

    for i in range(n):
        # j starts at i+1 so we visit each pair exactly once.
        # If j started at 0 we'd compute every pair twice, and Newton's 3rd law
        # trick would break - forces would be doubled.
        for j in range(i + 1, n):

            # Vector pointing from body i to body j (in metres).
            dx = body_pos[j, 0] - body_pos[i, 0]
            dy = body_pos[j, 1] - body_pos[i, 1]

            # Straight-line distance between the two bodies.
            raw_dist = math.sqrt(dx*dx + dy*dy)

            # Add the softening length to avoid F blowing up to infinity when two bodies get very close
            dist = raw_dist + softening

            # If softening is 0 and the bodies are exactly on top of each other, just skip this pair rather than crash with a divide-by-zero.
            if dist == 0.0:
                continue

            # safe_raw is only for normalising the direction vector (dx/raw_dist).
            # If raw_dist is 0 we substitute 1.0 - the force will be 0 anyway
            # since dx and dy are both 0 in that case, so no physics is changed.
            safe_raw = raw_dist if raw_dist > 0.0 else 1.0

            # Newton's law: F = G * m1 * m2 / r^2
            # We use the softened distance so the force stays bounded.
            f = G * body_mass[i] * body_mass[j] / (dist * dist)

            # Break the scalar force into x and y components.
            # (dx / safe_raw) gives the x-component of the unit vector from i to j.
            fx = f * dx / safe_raw
            fy = f * dy / safe_raw

            # Apply to both bodies in one shot: i is pulled toward j, j toward i.
            # This is the Newton's 3rd law shortcut that halves the total work.
            out_forces[i, 0] += fx
            out_forces[i, 1] += fy
            out_forces[j, 0] -= fx   # equal and opposite
            out_forces[j, 1] -= fy


# =============================================================================
# KERNEL 2 - Vectorized full N×N matrix, no Newton's 3rd shortcut
# =============================================================================

@njit(cache=True)
def _vectorized_force_kernel(body_pos, body_mass, out_forces, G, softening):
    '''The full NxN kernel - every ordered pair computed independently.

    Unlike the naive kernel, this doesn't exploit Newton's 3rd law. It evaluates
    (i->j) and (j->i) as separate force calculations. The trade-off is that it does
    twice the work, but it uses Plummer softening (epsilon^2 added inside the square root)
    instead of plain additive softening, which is a smoother and more physically
    standard approach.
    '''
    n = len(body_mass)

    for i in range(n):
        for j in range(n):

            # A body can't pull on itself.
            if i == j:
                continue

            dx = body_pos[j, 0] - body_pos[i, 0]
            dy = body_pos[j, 1] - body_pos[i, 1]

            # Plummer softening: tuck epsilon^2 inside the distance before taking the root,
            # giving sqrt(r^2 + epsilon^2) as the effective distance. This is always positive
            # and stays smooth as r -> 0, with no kink in the force law.
            # The naive kernel adds epsilon after the root (r + epsilon), which works but has
            # a slight discontinuity in slope. Plummer is the cosmological standard.
            dist_sq = dx*dx + dy*dy + softening*softening
            dist    = math.sqrt(dist_sq)

            # F = G * m1 * m2 / (r^2 + epsilon^2)
            f = G * body_mass[i] * body_mass[j] / dist_sq

            # Accumulate the force from j onto i.
            # (dx / dist) is the x unit-vector component from i toward j.
            # We don't need to handle j's side here - that happens naturally
            # when the outer loop reaches i = (old j).
            out_forces[i, 0] += f * dx / dist
            out_forces[i, 1] += f * dy / dist


# =============================================================================
# KERNEL 3 - Barnes-Hut O(n log n) tree walk
# =============================================================================

@njit(cache=True)
def _bh_force_kernel(body_pos, body_mass,
                     node_cx, node_cy, node_mass, node_half_w,
                     node_child, node_leaf_start, node_leaf_count,
                     leaf_pos, leaf_mass, leaf_body_idx,
                     theta, G, softening, out_forces):
    '''Barnes-Hut force walk on the flattened quadtree arrays.

    The core idea: groups of bodies that are far enough away get lumped together
    into a single point mass, cutting the cost from O(n²) to O(n log n).
    We can't pass the tree object into Numba (it's a Python class), so
    QuadTree.flatten() serialises everything into plain NumPy arrays beforehand.
    The root node always lives at index 0 in those arrays.
    '''
    n = len(body_mass)

    # We need an explicit stack for tree traversal since Numba doesn't allow
    # Python lists inside @njit. 512 slots is massively overkill - at n=1000
    # the tree is only about 3 levels deep and the stack never goes past ~12.
    stack = np.empty(512, dtype=np.int32)

    for i in range(n):
        # Pull body i's position out once so we're not hitting the array every iteration.
        bx = body_pos[i, 0]
        by = body_pos[i, 1]

        # Pre-multiply G * m_i so we don't redo that multiplication for
        # every node or body we visit during the walk.
        mu = G * body_mass[i]

        fx = 0.0
        fy = 0.0

        # Seed the traversal with the root node (index 0).
        top      = 0
        stack[0] = 0

        while top >= 0:
            # Pop the top node.
            node = stack[top]
            top -= 1

            # Empty nodes can appear when a quadrant was created during subdivision
            # but no bodies ended up in it. Nothing to do here.
            if node_mass[node] == 0.0:
                continue

            # ---- LEAF NODE: compute the force exactly ----
            # Leaves are identified by node_leaf_start >= 0 (internal nodes use -1).
            if node_leaf_start[node] >= 0:
                start = node_leaf_start[node]
                count = node_leaf_count[node]

                # Go through each body in this leaf (bucket holds at most MAX_CAPACITY = 16).
                # No Newton's 3rd law shortcut here - we're only visiting the nodes
                # that matter for body i, not doing a symmetric pair loop.
                for k in range(start, start + count):

                    # Don't let a body attract itself.
                    # leaf_body_idx[k] maps back to the original 0..n-1 body index.
                    if leaf_body_idx[k] == i:
                        continue

                    dx = leaf_pos[k, 0] - bx
                    dy = leaf_pos[k, 1] - by
                    raw_dist = math.sqrt(dx*dx + dy*dy)

                    # Same additive softening as the naive kernel.
                    dist = raw_dist + softening
                    if dist == 0.0:
                        continue

                    safe_raw = raw_dist if raw_dist > 0.0 else 1.0

                    # F = G * m_i * m_k / (r + ε)²
                    # mu already has G * m_i baked in, so just multiply by m_k.
                    f = mu * leaf_mass[k] / (dist * dist)
                    fx += f * dx / safe_raw
                    fy += f * dy / safe_raw
                continue   # leaf handled, move on

            # ---- INTERNAL NODE: decide whether to approximate or go deeper ----
            # Vector from body i to this node's centre of mass.
            dx = node_cx[node] - bx
            dy = node_cy[node] - by
            dist_com = math.sqrt(dx*dx + dy*dy)

            # If the body lands exactly on the node's centre of mass we can't
            # compute the opening ratio (s/d → ∞), so just open the node.
            if dist_com == 0.0:
                for c in range(4):
                    child = node_child[node, c]
                    if child >= 0:
                        top += 1
                        stack[top] = child
                continue

            # Barnes-Hut opening criterion: s / d < theta
            # s = 2 * half_width (the cell's side length)
            # d = dist_com (distance from body i to the cell's centre of mass)
            # When this ratio is small the cell is far enough away - or compact
            # enough - that treating it as one point mass is a good approximation.
            if (2.0 * node_half_w[node]) / dist_com < theta:
                # Good enough to approximate. Use the node's total mass at its
                # centre of mass as a stand-in for the whole subtree.
                dist = dist_com + softening
                f    = mu * node_mass[node] / (dist * dist)
                fx  += f * dx / dist_com
                fy  += f * dy / dist_com
            else:
                # Too close or too spread out to approximate safely.
                # Push the four children so we dig deeper on the next iterations.
                for c in range(4):
                    child = node_child[node, c]
                    if child >= 0:        # -1 means that quadrant is empty
                        top += 1
                        stack[top] = child

        # Store the accumulated force for body i.
        out_forces[i, 0] = fx
        out_forces[i, 1] = fy


# =============================================================================
# Public wrappers - pack arrays, call the kernel, unpack forces back to bodies
# =============================================================================

def compute_gravity_naive(bodies: list[Body]) -> None:
    '''O(n^2/2) pairwise gravity using Newton's 3rd law.

    Visits every unique pair once and applies the force to both bodies in one go.
    Simple and exact, but the work grows as n*(n-1)/2 - gets slow quickly for large n.
    '''
    n = len(bodies)
    if n == 0:
        return

    # Clear forces first - we don't want last frame's values bleeding into this one.
    for body in bodies:
        body.reset_force()

    # Pack positions and masses into contiguous arrays so Numba can digest them.
    body_pos  = np.array([b.position for b in bodies])
    body_mass = np.array([b.mass     for b in bodies])
    out_forces = np.zeros((n, 2))

    _naive_force_kernel(body_pos, body_mass, out_forces, G, SOFTENING)

    # Push the results back onto the Body objects so the integrator can read them.
    for i, b in enumerate(bodies):
        b.force = out_forces[i]


def compute_gravity_barnes_hut(bodies: list[Body], theta: float = THETA, softening: float = SOFTENING) -> None:
    '''O(n log n) gravity via the Barnes-Hut quadtree approximation.

    Rebuilds the quadtree from scratch every call (it goes stale the moment
    bodies move), computes centre-of-mass data bottom-up, then flattens the
    tree into plain NumPy arrays and passes everything to the compiled kernel.
    '''
    if not bodies:
        return

    for body in bodies:
        body.reset_force()

    # Figure out the bounding box of all current body positions.
    positions = np.array([b.position for b in bodies])
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)

    # The root cell needs to be a square that fits every body, with a bit of padding.
    # Using the larger of the two extents keeps it square, and adding 10% stops
    # bodies near the edge from sitting exactly on the boundary, which can cause
    # the boundary.contains() check to fail due to floating-point rounding.
    extent     = max(max_x - min_x, max_y - min_y)
    half_width = (extent / 2) * 1.1
    cx         = (min_x + max_x) / 2.0
    cy         = (min_y + max_y) / 2.0

    # Edge case: if all bodies are at the same point, extent is 0 - give the root
    # some finite size so subsequent geometry doesn't blow up.
    if half_width == 0.0:
        half_width = 1.0

    # Insert every body. The tree splits leaf cells automatically when they fill up.
    root = QuadTree(BoundingBox(cx, cy, half_width))
    for body in bodies:
        root.insert(body)

    # Propagate mass upward so every internal node knows its subtree's total mass
    # and centre of mass. The force walk needs this to apply the BH approximation.
    root.compute_mass_distribution()

    # Build a lookup from Python object id → array index so the Numba kernel
    # can identify which body is "self" and skip its own contribution.
    id_to_idx = {id(b): i for i, b in enumerate(bodies)}

    # Flatten the tree into plain arrays the Numba kernel can actually read.
    node_cx, node_cy, node_mass, node_half_w, node_child, \
    node_leaf_start, node_leaf_count, leaf_pos, leaf_mass, leaf_body_idx = root.flatten(id_to_idx)

    n          = len(bodies)
    body_pos   = np.array([b.position for b in bodies])
    body_mass  = np.array([b.mass     for b in bodies])
    out_forces = np.zeros((n, 2))

    _bh_force_kernel(body_pos, body_mass,
                     node_cx, node_cy, node_mass, node_half_w,
                     node_child, node_leaf_start, node_leaf_count,
                     leaf_pos, leaf_mass, leaf_body_idx,
                     theta, G, softening, out_forces)

    for i, b in enumerate(bodies):
        b.force = out_forces[i]


def compute_gravity_vectorized(bodies: list[Body]) -> None:
    '''O(n^2) gravity, evaluating every ordered pair independently.

    Structurally different from the naive solver: visits all n^2 ordered pairs
    (both (i,j) and (j,i)) rather than the n*(n-1)/2 unique pairs.
    '''
    n = len(bodies)
    if n == 0:
        return

    for body in bodies:
        body.reset_force()

    body_pos   = np.array([b.position for b in bodies])
    body_mass  = np.array([b.mass     for b in bodies])
    out_forces = np.zeros((n, 2))

    _vectorized_force_kernel(body_pos, body_mass, out_forces, G, SOFTENING)

    for i, b in enumerate(bodies):
        b.force = out_forces[i]
