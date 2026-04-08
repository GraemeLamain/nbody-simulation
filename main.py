import argparse
import copy
import pygame
from functools import partial
import numpy as np
from simulation.body import Body
from simulation.physics import compute_gravity_naive, compute_gravity_barnes_hut
from simulation.integrators import EulerIntegration, RK4Integrator, LeapfrogIntegrator
from simulation.simulation import Simulation
from renderer.pygame_renderer import Renderer
from scenes.solar_system import create_solar_system
from scenes.galaxy_collision import create_galaxy_collision
from validation.nasa_fetch import create_solar_system_from_jpl
from validation.compare import run_integrator_comparison
from config import WIDTH, HEIGHT, FPS, AU, G

# --- CLI arguments ---
parser = argparse.ArgumentParser(description="N-Body Gravitational Simulator")
parser.add_argument("--jpl",     action="store_true", help="Use NASA JPL Horizons initial conditions")
parser.add_argument("--compare", action="store_true", help="Run integrator comparison and exit")
parser.add_argument("--scene",   choices=["solar", "galaxy"], default="solar", help="Scene to load: solar (default) or galaxy collision")
parser.add_argument("--gravity", choices=["naive", "barneshut"], default=None, help="Gravity algorithm: naive (default for solar) or barneshut (default for galaxy)")
parser.add_argument("--record",  action="store_true", help="Record simulation to a GIF and exit")
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
        target_planet="Earth",
        years=10,
        dt=86400.0 * 5,
    )
    raise SystemExit(0)

# --- Scene-specific settings ---
if args.scene == "galaxy":
    scale = (WIDTH * 0.4) / 9e11
    dt_sim = 3600.0             # 3600 = 1 hour
    default_steps = 1           # 1 physics frame per redraw
else:
    scale = (WIDTH * 0.4) / (36 * AU)
    dt_sim = 86400.0
    default_steps = 30

# --- Recording settings ---
if args.record:
    if args.scene == "solar":
        RECORD_FRAMES   = 600   # total frames to capture
        RECORD_EVERY    = 2     # save every Nth frame to keep GIF size down
        WARMUP_STEPS    = 0     # no warmup needed, solar system is interesting immediately
        GIF_FPS         = 30
        GIF_FILENAME    = "solar_system_full.gif"
    else:
        RECORD_FRAMES   = 600
        RECORD_EVERY    = 3
        WARMUP_STEPS    = 150  # skip ahead so galaxies are mid-collision
        GIF_FPS         = 20
        GIF_FILENAME    = "galaxy_collision.gif"

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

def step_scene():
    """Run one full frame worth of physics steps including galaxy-specific pruning."""
    for _ in range(renderer.steps_per_frame):
        sim.step()
 
        if args.scene == "galaxy":
            MIN_VISIBLE_DISTANCE = 3e12
            TOTAL_MASS = 4e33
            surviving_stars = []
            for b in sim.bodies:
                if b.mass >= 1e30:
                    surviving_stars.append(b)
                    continue
                r = np.linalg.norm(b.position)
                if r == 0:
                    surviving_stars.append(b)
                    continue
                v_escape = np.sqrt(2 * G * TOTAL_MASS / r)
                speed    = np.linalg.norm(b.velocity)
                if r > MIN_VISIBLE_DISTANCE and speed > v_escape:
                    pass
                else:
                    surviving_stars.append(b)
            sim.bodies = surviving_stars
            sim.bodies = handle_collisions(sim.bodies)

def get_offset():
    if args.scene == "galaxy":
        return np.array([WIDTH / 2, HEIGHT / 2])
    else:
        return np.array([WIDTH / 2, HEIGHT / 2]) - sim.bodies[0].position * scale * renderer.zoom

# --- Recording mode ---
if args.record:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required for recording. Install it with: pip install Pillow")
        raise SystemExit(1)
 
    # warmup - run physics without capturing so we start at an interesting point
    if WARMUP_STEPS > 0:
        print(f"Warming up simulation ({WARMUP_STEPS} steps)...")
        warmup_steps_per_batch = 100
        for i in range(0, WARMUP_STEPS, warmup_steps_per_batch):
            for _ in range(warmup_steps_per_batch):
                sim.step()
                if args.scene == "galaxy":
                    sim.bodies = [b for b in sim.bodies if b.mass >= 1e30 or
                                  np.linalg.norm(b.position) <= 3e12]
                    sim.bodies = handle_collisions(sim.bodies)
            print(f"  {min(i + warmup_steps_per_batch, WARMUP_STEPS)}/{WARMUP_STEPS} steps", end="\r")
        print(f"\nWarmup complete. Recording {RECORD_FRAMES} frames...")
    else:
        print(f"Recording {RECORD_FRAMES} frames...")
 
    frames = []
    frame_count = 0
    captured = 0
 
    while captured < len(range(0, RECORD_FRAMES, RECORD_EVERY)):
        # handle quit events so window doesn't freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)
 
        step_scene()
        offset = get_offset()
        renderer.draw(sim.bodies, scale * renderer.zoom, offset, sim.time, integrator_names[integrator_index])
        renderer.tick(FPS)
 
        if frame_count % RECORD_EVERY == 0:
            # grab the current frame from the pygame surface
            raw = pygame.surfarray.array3d(renderer.screen)
            # pygame uses (width, height, channels), PIL expects (height, width, channels)
            img = Image.fromarray(raw.transpose(1, 0, 2))
            # scale down to reduce GIF file size
            # img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
            frames.append(img)
            captured += 1
            print(f"  Captured frame {captured}/{len(range(0, RECORD_FRAMES, RECORD_EVERY))}", end="\r")
 
        frame_count += 1
        if frame_count >= RECORD_FRAMES:
            break
 
    print(f"\nSaving {GIF_FILENAME}...")
    frames[0].save(
        GIF_FILENAME,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / GIF_FPS),
        loop=0,
    )
    print(f"Done. Saved to {GIF_FILENAME}")
    renderer.close()
    raise SystemExit(0)

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
    
    step_scene()
    offset = get_offset()
    renderer.draw(sim.bodies, scale * renderer.zoom, offset, sim.time, integrator_names[integrator_index])
    renderer.tick(FPS)

renderer.close()
