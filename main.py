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

bodies = create_solar_system()
initial_bodies = copy.deepcopy(bodies)

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
