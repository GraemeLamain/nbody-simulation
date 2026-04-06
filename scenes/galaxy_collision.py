# Two galaxy disks on a collision course, each with a central black hole and a swarm of stars
import numpy as np
from simulation.body import Body
from config import G

# Define the colors for the different galaxies
STAR_COLORS_A = [
    (100, 149, 237),   # cornflower blue
    (135, 206, 235),   # sky blue
    (173, 216, 230),   # light blue
    (255, 255, 255),   # white
    (200, 200, 255),   # lavender
]
STAR_COLORS_B = [
    (255, 200, 50),    # gold
    (255, 140, 0),     # orange
    (255, 220, 120),   # pale gold
    (255, 255, 200),   # warm white
    (255, 170, 80),    # amber
]


def create_galaxy_single(seed: int = 42, colors: list = STAR_COLORS_A, BH_MASS: float = 2e33) -> list[Body]:
    """Single rotating galaxy disk: one central black hole + num_stars stars.

    Stars are placed at random radii in [MIN_R, MAX_R] with circular orbital
    velocities so the disk is rotationally stable around the black hole.
    """
    rng = np.random.default_rng(seed)

    num_stars = np.random.randint(1000, 2500)

    # Define the range of radii for the stars
    MIN_R, MAX_R = 5e10, 4e11

    # Create the black hole
    black_hole = Body(
        name="black_hole",
        mass=BH_MASS,
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        color=(255, 255, 255),
        radius=10.0,
    )

    bodies: list[Body] = [black_hole]

    # Create stars
    angles = rng.uniform(0.0, 2 * np.pi, num_stars)
    radii  = rng.uniform(MIN_R, MAX_R, num_stars)

    for i, (theta, r) in enumerate(zip(angles, radii)):
        pos = np.array([r * np.cos(theta), r * np.sin(theta)])
        v_mag = np.sqrt(G * BH_MASS / r)
        vel = np.array([-np.sin(theta), np.cos(theta)]) * v_mag
        color = colors[rng.integers(len(colors))]
        bodies.append(Body(
            name=f"star_{i}",
            mass=1e20,
            position=pos,
            velocity=vel,
            color=color,
            radius=1.0,
        ))

    return bodies


def create_galaxy_collision(seed: int = 42) -> list[Body]:
    """Two counter-rotating galaxy disks aimed at each other.

    Galaxy 1 is centred at [-6e11, 0] moving right; galaxy 2 is centred at
    [+6e11, 0] moving left, with its rotation reversed so the disks spin
    in opposite directions as they pass through each other.
    """
    OFFSET = 6e11

    galaxy1 = create_galaxy_single(seed=seed,     colors=STAR_COLORS_A, BH_MASS=4e33)
    galaxy2 = create_galaxy_single(seed=seed + 1, colors=STAR_COLORS_B, BH_MASS=2e33)

    # High infall speeds make for a dramatic, chaotic collision.
    infall1 = np.array([ 35000.0, -8000.0])
    infall2 = np.array([-20000.0,  5000.0])

    # Slower alternative for a gentler, more gravitational interaction:
    # infall1 = np.array([ 3000.0, -1000.0])
    # infall2 = np.array([-2000.0,  2000.0])

    for body in galaxy1:
        body.position = body.position.copy() + np.array([-OFFSET, 0.0])
        body.velocity = body.velocity.copy() + infall1

    for body in galaxy2:
        body.position = body.position.copy() + np.array([OFFSET, 0.0])
        # Negate tangential velocity to reverse rotation direction
        body.velocity = -body.velocity.copy() + infall2

    return galaxy1 + galaxy2
