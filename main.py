import argparse
import copy
import pygame
from functools import partial
import numpy as np
from simulation.body import Body
from simulation.physics import compute_gravity_naive, compute_gravity_barnes_hut, compute_gravity_vectorized
from simulation.integrators import EulerIntegration, RK4Integrator, LeapfrogIntegrator
from simulation.simulation import Simulation
from renderer.pygame_renderer import Renderer
from scenes.solar_system import create_solar_system
from scenes.galaxy_collision import create_galaxy_collision
from validation.nasa_fetch import create_solar_system_from_jpl
from validation.compare import run_integrator_comparison
from validation.nasa_fetch import create_solar_system_from_jpl
from config import WIDTH, HEIGHT, FPS, AU, G

# --- CLI arguments ---
parser = argparse.ArgumentParser(description="N-Body Gravitational Simulator")
parser.add_argument("--jpl",     action="store_true", help="Use NASA JPL Horizons initial conditions")
parser.add_argument("--compare", action="store_true", help="Run integrator comparison and exit")
parser.add_argument("--scene",   choices=["solar", "galaxy"], default="solar", help="Scene to load: solar (default) or galaxy collision")
parser.add_argument("--gravity", choices=["naive", "barneshut", 'vectorized'], default=None, help="Gravity algorithm: naive (default for solar) or barneshut (default for galaxy)")
args = parser.parse_args()

# --- Gravity function ---
if args.scene == "solar" and args.gravity is None:
    args.gravity = "naive"
elif args.scene == "galaxy" and args.gravity is None:
    args.gravity = "barneshut"

if args.gravity == "barneshut":
    # Galaxy needs non-zero softening to prevent close encounters from blowing up
    # velocities to NaN. Solar system is fine at 0 since bodies are well-separated.
    softening = 1e9 if args.scene == "galaxy" else 0.0
    compute_forces_fn = partial(compute_gravity_barnes_hut, softening=softening)
elif args.gravity == "vectorized":
    compute_forces_fn = compute_gravity_vectorized
else:
    compute_forces_fn = compute_gravity_naive

# --- Initial conditions ---
if args.scene == "galaxy":
    print("Loading galaxy collision scene...")
    bodies = create_galaxy_collision()
    print(f"Galaxy loaded: {len(bodies)} bodies.")
elif args.jpl:
    # If the --jpl flag is used, fetch the initial conditions from NASA JPL Horizons for today
    print("Fetching initial conditions from NASA JPL Horizons (2026-03-03)...")
    bodies = create_solar_system_from_jpl("2026-03-03")
    print("JPL initial conditions loaded.")
else:
    # Otherwise we use the approximate solar system initial conditions
    print("Using approximate solar system initial conditions.")
    bodies = create_solar_system()

# Zero out net system momentum by adjusting the first body (the Sun).
# Without this the whole system drifts off-screen over time.
total_momentum = np.array([0.0, 0.0])
for body in bodies:
    total_momentum += body.mass * body.velocity

bodies[0].velocity -= total_momentum / bodies[0].mass

initial_bodies = copy.deepcopy(bodies)

# --- Integrator comparison mode ---
if args.compare:
    if args.jpl:
        # If the --jpl flag is used with the --compare flag, we get initial conditions from NASA JPL Horizons for our comparisons
        # We expand the factory into a full function so it can anchor the system
        def jpl_factory():
            # Fetch the raw bodies
            bodies = create_solar_system_from_jpl("2026-03-03")

            # Calculate total system momentum
            total_momentum = np.array([0.0, 0.0])
            for body in bodies:
                total_momentum += body.mass * body.velocity

            # Apply equal and opposite velocity to the Sun
            bodies[0].velocity -= total_momentum / bodies[0].mass

            # Return the anchored system to the integrator
            return bodies

        factory = jpl_factory
    else:
        factory = create_solar_system

    # Run each integrator for 3 simulated years and compare how they track a target planet
    run_integrator_comparison(
        bodies_factory=factory,
        integrators={
            "Euler":    EulerIntegration(),
            "RK4":      RK4Integrator(),
            "Leapfrog": LeapfrogIntegrator(),
        },
        target_planet="Mars",
        years=3,
        dt=86400.0,
    )
    raise SystemExit(0)

# --- Scene-specific settings ---
if args.scene == "galaxy":
    scale = (WIDTH * 0.4) / 9e11
    dt_sim = 3600.0             # 3600 = 1 hour
    default_steps = 1           # 1 physics frame per redraw
else:
    scale = (WIDTH * 0.4) / (32 * AU)
    dt_sim = 86400.0
    default_steps = 30

# --- Simulation setup ---
integrators = [EulerIntegration(), RK4Integrator(), LeapfrogIntegrator()]
integrator_names = ["Euler", "RK4", "Leapfrog"]
integrator_index = 2  # default to Leapfrog

sim = Simulation(
    bodies=bodies,
    integrator=integrators[integrator_index],
    compute_forces=compute_forces_fn,
    dt=dt_sim
)

renderer = Renderer(WIDTH, HEIGHT)
renderer.steps_per_frame = default_steps

# Add this check to prevent Pygame from melting trying to draw trails for 1000 bodies
if args.scene == "galaxy":
    renderer.show_trails = False

def handle_collisions(bodies: list[Body]) -> list[Body]:
    """Checks for bodies that are close enough to collide and merges/deletes them."""
    COLLISION_DISTANCE = 1.5e10  # 15 billion meters (This acts as our event horizon for the black holes)
    MASSIVE_THRESHOLD = 1e30     # Anything heavier than this is considered a Black Hole

    # Find the black holes in the bodies list
    black_holes = [(i, b) for i, b in enumerate(bodies) if b.mass >= MASSIVE_THRESHOLD]
    if not black_holes:
        return bodies

    destroyed_indices = set()
    new_bodies = []

    # Only check collisions that involve a black hole.
    # If we did star-star collisions too then it would slow the simulation down a TON
    for m_idx, m_body in black_holes:
        if m_idx in destroyed_indices: continue

        for i, b in enumerate(bodies):
            if i == m_idx or i in destroyed_indices: continue
                
            # Check a rough square boundary first before doing expensive square roots
            dx = abs(m_body.position[0] - b.position[0])
            if dx > COLLISION_DISTANCE: continue

            dy = abs(m_body.position[1] - b.position[1])
            if dy > COLLISION_DISTANCE: continue
                
            # If they are within the same square, calculate exact circular distance
            dist = np.sqrt(dx*dx + dy*dy)

            if dist < COLLISION_DISTANCE:
                if b.mass >= MASSIVE_THRESHOLD:
                    # Merge the two black holes
                    new_mass = m_body.mass + b.mass
                    new_pos = (m_body.position * m_body.mass + b.position * b.mass) / new_mass
                    new_vel = (m_body.velocity * m_body.mass + b.velocity * b.mass) / new_mass

                    new_bodies.append(Body(
                        name="black_hole_merged", mass=new_mass, position=new_pos,
                        velocity=new_vel, color=(255, 0, 255), radius=8.0
                    ))
                    destroyed_indices.add(m_idx)
                    destroyed_indices.add(i)
                    break # This black hole merged, stop checking it against stars
                    
                # Black Hole vs Star (Black Hole eats it)
                else:
                    # Star gets swallowed by the black hole.
                    destroyed_indices.add(i)

    surviving_bodies = [b for i, b in enumerate(bodies) if i not in destroyed_indices]
    return surviving_bodies + new_bodies

# --- Main Simulation Loop ---
running = True

while running:
    running = renderer.handle_events()

    # Handle all the HUD keybinds
    if renderer.cycle_integrator:
        integrator_index = (integrator_index + 1) % len(integrators)
        sim.integrator = integrators[integrator_index]
        sim.bodies = copy.deepcopy(initial_bodies)
        sim.time = 0.0
        renderer.trails.clear()

    if renderer.reset_time:
        sim.bodies = copy.deepcopy(initial_bodies)
        sim.time = 0.0
        renderer.trails.clear()

    for _ in range(renderer.steps_per_frame):
        sim.step()

        # Remove stars that have mathematically escaped the galaxy AND left the screen
        if args.scene == "galaxy":
            MIN_VISIBLE_DISTANCE = 3e12  # Keep it alive if it's still on screen
            TOTAL_MASS = 4e33            # Rough mass of the two black holes combined
            
            surviving_stars = []
            for b in sim.bodies:
                # Never delete the black holes
                if b.mass >= 1e30:
                    surviving_stars.append(b)
                    continue
                
                # Calculate distance from the center of the universe (0,0)
                r = np.linalg.norm(b.position)
                if r == 0:
                    surviving_stars.append(b)
                    continue
                
                # Calculate escape velocity at this exact distance
                # v_escape = sqrt(2 * G * M / r)
                v_escape = np.sqrt(2 * G * TOTAL_MASS / r)
                
                # Calculate the star's actual speed (magnitude of velocity vector)
                speed = np.linalg.norm(b.velocity)
                
                # Delete ONLY if it is off-screen AND has escaped gravity
                if r > MIN_VISIBLE_DISTANCE and speed > v_escape:
                    pass 
                else:
                    surviving_stars.append(b)

            sim.bodies = surviving_stars

            # Handle black hole mergers and star swallowing.
            sim.bodies = handle_collisions(sim.bodies)

    if args.scene == "galaxy":
        # Galaxy collision is symmetric so there's no single body to follow - just keep the view centred on the origin.
        offset = np.array([WIDTH / 2, HEIGHT / 2])
    else:
        # Track the Sun so it stays in frame as the system's momentum slowly carries it.
        offset = np.array([WIDTH / 2, HEIGHT / 2]) - sim.bodies[0].position * scale * renderer.zoom

    renderer.draw(sim.bodies, scale * renderer.zoom, offset, sim.time, integrator_names[integrator_index])
    renderer.tick(FPS)

renderer.close()
