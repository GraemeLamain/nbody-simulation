# Different ways of computing gravity forces
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
    '''O(n^2) gravity kernel using Newton's 3rd law to halve the work.

    j starts at i+1 so each pair is only visited once.
    cache=True means Numba saves the compiled binary so it doesn't recompile every run.
    '''
    n = len(body_mass)

    for i in range(n):
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

            # if bodies are exactly on top of each other, avoid dividing by zero in direction calc
            safe_raw = raw_dist if raw_dist > 0.0 else 1.0

            # Newton's law: F = G * m1 * m2 / r^2
            # We use the softened distance so the force stays bounded.
            f = G * body_mass[i] * body_mass[j] / (dist * dist)
            fx = f * dx / safe_raw
            fy = f * dy / safe_raw

            # apply to both bodies at once - Newton's 3rd law
            out_forces[i, 0] += fx
            out_forces[i, 1] += fy
            out_forces[j, 0] -= fx
            out_forces[j, 1] -= fy


# =============================================================================
# KERNEL 2 - Vectorized full N×N, no Newton's 3rd shortcut
# =============================================================================

@njit(cache=True)
def _vectorized_force_kernel(body_pos, body_mass, out_forces, G, softening):
    '''Full NxN kernel - every ordered pair computed independently.

    Doesn't use Newton's 3rd law so it does twice the work, but uses Plummer
    softening (epsilon^2 inside the sqrt) which is smoother than additive softening.
    '''
    n = len(body_mass)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            dx = body_pos[j, 0] - body_pos[i, 0]
            dy = body_pos[j, 1] - body_pos[i, 1]

            # Plummer softening - always positive, smooth as r -> 0
            dist_sq = dx*dx + dy*dy + softening*softening
            dist    = math.sqrt(dist_sq)

            # F = G * m1 * m2 / (r^2 + epsilon^2)
            f = G * body_mass[i] * body_mass[j] / dist_sq
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

    Groups of distant bodies get treated as a single point mass, cutting cost
    from O(n^2) to O(n log n). We can't pass the tree object into Numba so
    QuadTree.flatten() serializes everything into plain numpy arrays first.
    Root node is always at index 0.
    '''
    n = len(body_mass)

    # explicit stack since Numba doesn't allow Python lists in @njit
    # 512 is way more than needed - tree is ~12 levels deep at n=1000
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

        top      = 0
        stack[0] = 0  # start at root

        while top >= 0:
            # Pop the top node.
            node = stack[top]
            top -= 1

            # Empty nodes can appear when a quadrant was created during subdivision
            # but no bodies ended up in it. Nothing to do here.
            if node_mass[node] == 0.0:
                continue

            # leaf node - compute force exactly
            if node_leaf_start[node] >= 0:
                start = node_leaf_start[node]
                count = node_leaf_count[node]

                # Go through each body in this leaf
                for k in range(start, start + count):
                    if leaf_body_idx[k] == i:
                        continue

                    dx = leaf_pos[k, 0] - bx
                    dy = leaf_pos[k, 1] - by
                    raw_dist = math.sqrt(dx*dx + dy*dy)

                    dist = raw_dist + softening
                    if dist == 0.0:
                        continue

                    safe_raw = raw_dist if raw_dist > 0.0 else 1.0

                    # F = G * m_i * m_k / (r + ε)²
                    # mu already has G * m_i baked in, so just multiply by m_k.
                    f = mu * leaf_mass[k] / (dist * dist)
                    fx += f * dx / safe_raw
                    fy += f * dy / safe_raw
                continue

            # internal node - check opening criterion
            dx = node_cx[node] - bx
            dy = node_cy[node] - by
            dist_com = math.sqrt(dx*dx + dy*dy)

            if dist_com == 0.0:
                # body is exactly on the CoM, can't compute ratio so just open it
                for c in range(4):
                    child = node_child[node, c]
                    if child >= 0:
                        top += 1
                        stack[top] = child
                continue

            # s/d < theta means cell is far enough to approximate as a point mass
            if (2.0 * node_half_w[node]) / dist_com < theta:
                dist = dist_com + softening
                f    = mu * node_mass[node] / (dist * dist)
                fx  += f * dx / dist_com
                fy  += f * dy / dist_com
            else:
                # too close, open the node
                for c in range(4):
                    child = node_child[node, c]
                    if child >= 0:
                        top += 1
                        stack[top] = child

        # Store the accumulated force for body i.
        out_forces[i, 0] = fx
        out_forces[i, 1] = fy


# =============================================================================
# Public wrappers
# =============================================================================

def compute_gravity_naive(bodies: list[Body]) -> None:
    '''O(n^2/2) pairwise gravity. Simple and exact but gets slow fast for large n.'''
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
    '''O(n log n) gravity via Barnes-Hut quadtree.

    Rebuilds the tree every call since bodies move each step.
    Flattens it into numpy arrays for the Numba kernel.
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
    extent     = max(max_x - min_x, max_y - min_y)
    half_width = (extent / 2) * 1.1  # 10% padding so edge bodies don't fall outside
    cx         = (min_x + max_x) / 2.0
    cy         = (min_y + max_y) / 2.0

    if half_width == 0.0:
        half_width = 1.0

    # Insert every body. The tree splits leaf cells automatically when they fill up.
    root = QuadTree(BoundingBox(cx, cy, half_width))
    for body in bodies:
        root.insert(body)

    # Propagate mass upward so every internal node knows its subtree's total mass
    # and centre of mass. The force walk needs this to apply the BH approximation.
    root.compute_mass_distribution()

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
    '''O(n^2) gravity, visiting all ordered pairs independently.'''
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