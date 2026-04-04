import numpy as np
from simulation.body import Body
from simulation.physics import compute_gravity_naive
from simulation.integrators import EulerIntegration, RK4Integrator , LeapfrogIntegrator
from simulation.simulation import Simulation
from renderer.pygame_renderer import Renderer
from config import WIDTH, HEIGHT, FPS

AU = 1.496e11

sun = Body("Sun", 1.989e30, np.array([0.0, 0.0]), np.array([0.0, 0.0]), color=(255, 255, 0), radius=12)
earth = Body("Earth", 5.972e24, np.array([AU, 0.0]), np.array([0.0, 29780.0]), color=(0, 100, 255), radius=6)

# sim = Simulation(
#     bodies=[sun, earth],
#     integrator=EulerIntegration(),
#     compute_forces=compute_gravity_naive,
#     dt=3600.0
# )

# sim = Simulation(
#     bodies=[sun, earth],
#     integrator=RK4Integrator(),
#     compute_forces=compute_gravity_naive,
#     dt=3600.0
# )

sim = Simulation(
    bodies=[sun, earth],
    integrator=LeapfrogIntegrator(),
    compute_forces=compute_gravity_naive,
    dt=3600.0
)

# scale and offset to centre the solar system on screen
scale = WIDTH / (3 * AU)
offset = np.array([WIDTH / 2, HEIGHT / 2])

renderer = Renderer(WIDTH, HEIGHT)
running = True

while running:
    running = renderer.handle_events()

    for _ in range(24):   # 24 steps per frame = 1 simulated day per frame
        sim.step()

    renderer.draw(sim.bodies, scale * renderer.zoom, offset, sim.time)
    renderer.tick(FPS)

renderer.close()