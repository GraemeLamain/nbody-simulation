import argparse
import copy
from functools import partial
import numpy as np
from simulation.body import Body
from simulation.physics import compute_gravity_naive, compute_gravity_barnes_hut
from simulation.integrators import EulerIntegration, RK4Integrator, LeapfrogIntegrator
from simulation.simulation import Simulation
from renderer.pygame_renderer import Renderer
from scenes.solar_system import create_solar_system
from config import WIDTH, HEIGHT, FPS

AU = 1.496e11

# --- CLI arguments ---
parser = argparse.ArgumentParser(description="N-Body Gravitational Simulator")
parser.add_argument("--jpl",     action="store_true", help="Use NASA JPL Horizons initial conditions")
parser.add_argument("--compare", action="store_true", help="Run integrator comparison and exit")
parser.add_argument("--scene",   choices=["solar", "galaxy"], default="solar", help="Scene to load: solar (default) or galaxy collision")
parser.add_argument("--gravity", choices=["naive", "barneshut"], default=None, help="Gravity algorithm: naive (default for solar) or barneshut (default for galaxy)")
args = parser.parse_args()

# --- Gravity function ---
use_barneshut = (args.gravity == "barneshut") or (args.scene == "galaxy" and args.gravity != "naive")
if use_barneshut:
    # Galaxy needs non-zero softening to prevent close encounters from blowing up
    # velocities to NaN. Solar system is fine at 0 since bodies are well-separated.
    softening = 1e9 if args.scene == "galaxy" else 0.0
    compute_forces_fn = partial(compute_gravity_barnes_hut, softening=softening)
else:
    compute_forces_fn = compute_gravity_naive

# --- Initial conditions ---
if args.scene == "galaxy":
    from scenes.galaxy import create_galaxy_collision
    print("Loading galaxy collision scene...")
    bodies = create_galaxy_collision()
    print(f"Galaxy loaded: {len(bodies)} bodies.")
elif args.jpl:
    # If the --jpl flag is used, fetch the initial conditions from NASA JPL Horizons for today
    from validation.nasa_fetch import create_solar_system_from_jpl
    print("Fetching initial conditions from NASA JPL Horizons (2026-03-03)...")
    bodies = create_solar_system_from_jpl("2026-03-03")
    print("JPL initial conditions loaded.")
else:
    # Otherwise we use the approximate solar system initial conditions
    print("Using approximate solar system initial conditions.")
    bodies = create_solar_system()

# Calculate total system momentum and anchor the first body
total_momentum = np.array([0.0, 0.0])
for body in bodies:
    total_momentum += body.mass * body.velocity

bodies[0].velocity -= total_momentum / bodies[0].mass

initial_bodies = copy.deepcopy(bodies)

# --- Integrator comparison mode ---
if args.compare:
    from validation.compare import run_integrator_comparison
    if args.jpl:
        # If the --jpl flag is used with the --compare flag, we get initial conditions from NASA JPL Horizons for our comparisons
        from validation.nasa_fetch import create_solar_system_from_jpl
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

    # Run the integrator comparison over a 3 year time frame
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
    scale = (WIDTH * 0.4) / 8e11
    dt_sim = 3600.0
    default_steps = 1
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

    # Calculate viewport offset
    if args.scene == "galaxy":
        # Fixed origin — no body to track in a symmetric collision
        offset = np.array([WIDTH / 2, HEIGHT / 2])
    else:
        # Centre on the Sun so it stays on screen as it drifts
        offset = np.array([WIDTH / 2, HEIGHT / 2]) - sim.bodies[0].position * scale * renderer.zoom

    renderer.draw(sim.bodies, scale * renderer.zoom, offset, sim.time, integrator_names[integrator_index])
    renderer.tick(FPS)

renderer.close()
