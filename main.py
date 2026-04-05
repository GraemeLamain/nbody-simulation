import argparse
import copy
import numpy as np
from simulation.body import Body
from simulation.physics import compute_gravity_naive
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
args = parser.parse_args()

# --- Initial conditions ---
if args.jpl:
    # If the --jpl flag is used, fetch the initial conditions from NASA JPL Horizons for today
    from validation.nasa_fetch import create_solar_system_from_jpl
    print("Fetching initial conditions from NASA JPL Horizons (2026-03-03)...")
    bodies = create_solar_system_from_jpl("2026-03-03")
    print("JPL initial conditions loaded.")
else:
    # Otherwise we use the approximate solar system initial conditions
    print("Using approximate solar system initial conditions.")
    bodies = create_solar_system()

# print(f"Bodies: {bodies}")

# Calculate total system momentum
total_momentum = np.array([0.0, 0.0])
for body in bodies:
    total_momentum += body.mass * body.velocity
    
# Apply equal and opposite velocity to the Sun to anchor the simulation
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

# --- Simulation setup ---
integrators = [EulerIntegration(), RK4Integrator(), LeapfrogIntegrator()]
integrator_names = ["Euler", "RK4", "Leapfrog"]
integrator_index = 2  # default to Leapfrog

sim = Simulation(
    bodies=bodies,
    integrator=integrators[integrator_index],
    compute_forces=compute_gravity_naive,
    dt=86400.0
)

# scale and offset to centre the solar system on screen
# the scale is this far out because Neptune is at 30 AU and we want it to fit on screen
scale = (WIDTH * 0.4) / (32 * AU)

renderer = Renderer(WIDTH, HEIGHT)
renderer.steps_per_frame = 30

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

    # Calculate offset based on the sun so it always stays centered there
    offset = np.array([WIDTH / 2, HEIGHT / 2]) - sim.bodies[0].position * scale * renderer.zoom
    renderer.draw(sim.bodies, scale * renderer.zoom, offset, sim.time, integrator_names[integrator_index])
    renderer.tick(FPS)

renderer.close()
