import numpy as np
from simulation.body import Body
from config import G

AU = 1.496e11   # metres per astronomical unit

def create_solar_system() -> list[Body]:
    M_sun = 1.989e30

    # Circular orbital speed: v = sqrt(G * M_sun / r)
    # Planets are placed on the positive x-axis and given velocity in the positive y direction
    # so they orbit counterclockwise when viewed from above.
    def orbital_velocity(r: float) -> float:
        return np.sqrt(G * M_sun / r)

    sun = Body(
        name="Sun",
        mass=M_sun,
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        color=(255, 255, 0),
        radius=6.0,
    )

    mercury_r = 0.387 * AU
    mercury = Body(
        name="Mercury",
        mass=3.285e23,
        position=np.array([mercury_r, 0.0]),
        velocity=np.array([0.0, orbital_velocity(mercury_r)]),
        color=(169, 169, 169),
        radius=2.0,
    )

    venus_r = 0.723 * AU
    venus = Body(
        name="Venus",
        mass=4.867e24,
        position=np.array([venus_r, 0.0]),
        velocity=np.array([0.0, orbital_velocity(venus_r)]),
        color=(255, 198, 100),
        radius=3.0,
    )

    earth_r = 1.0 * AU
    earth = Body(
        name="Earth",
        mass=5.972e24,
        position=np.array([earth_r, 0.0]),
        velocity=np.array([0.0, orbital_velocity(earth_r)]),
        color=(0, 100, 255),
        radius=3.0,
    )

    mars_r = 1.524 * AU
    mars = Body(
        name="Mars",
        mass=6.390e23,
        position=np.array([mars_r, 0.0]),
        velocity=np.array([0.0, orbital_velocity(mars_r)]),
        color=(188, 74, 60),
        radius=2.0,
    )

    jupiter_r = 5.203 * AU
    jupiter = Body(
        name="Jupiter",
        mass=1.898e27,
        position=np.array([jupiter_r, 0.0]),
        velocity=np.array([0.0, orbital_velocity(jupiter_r)]),
        color=(201, 144, 57),
        radius=6.0,
    )

    saturn_r = 9.537 * AU
    saturn = Body(
        name="Saturn",
        mass=5.683e26,
        position=np.array([saturn_r, 0.0]),
        velocity=np.array([0.0, orbital_velocity(saturn_r)]),
        color=(210, 180, 100),
        radius=5.0,
    )

    uranus_r = 19.191 * AU
    uranus = Body(
        name="Uranus",
        mass=8.681e25,
        position=np.array([uranus_r, 0.0]),
        velocity=np.array([0.0, orbital_velocity(uranus_r)]),
        color=(100, 220, 220),
        radius=4.0,
    )

    neptune_r = 30.069 * AU
    neptune = Body(
        name="Neptune",
        mass=1.024e26,
        position=np.array([neptune_r, 0.0]),
        velocity=np.array([0.0, orbital_velocity(neptune_r)]),
        color=(50, 100, 255),
        radius=4.0,
    )

    return [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]
