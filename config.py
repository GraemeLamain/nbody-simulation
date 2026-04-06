# Global settings and physical constants

# --- Window ---
WIDTH  = 1200   # window width in pixels
HEIGHT = 800    # window height in pixels
FPS    = 60     # target frames per second

# --- Simulation ---

# How much simulated time passes each integrator step, in seconds.
# 86400 = 24 hours = 1 day, so each call to integrator.step() advances
# the simulation clock by one day. Smaller values give more accurate orbits
# but need more steps to cover the same span of simulated time.
TIMESTEP = 86400.0

# Barnes-Hut opening angle (theta, θ).
# Controls how aggressively the tree approximates distant groups of bodies.
# Lower theta = fewer approximations = more accurate but slower.
# Higher theta = more approximations = faster but less accurate.
# 0.0 = exact pairwise limit (every node gets opened all the way down).
# 1.2 = standard value used in most cosmological N-body codes.
THETA = 1.2

# --- Physical Constants ---

# Newton's gravitational constant (SI units)
G = 6.674e-11

# One Astronomical Unit in metres - the average Earth-Sun distance
AU = 1.496e11

# Gravitational softening length in metres.
# Without this, F = G*m1*m2/r^2 blows up to infinity when two bodies get
# very close (r -> 0). Adding softening replaces r with (r + epsilon) in the
# denominator, keeping the force finite even during close encounters.
# Physically it's a bit like treating bodies as fuzzy blobs instead of
# true point masses.
# Galaxy scenes: keep this nonzero - stars can pass right through each other.
# Solar system: can safely be 0.0 since the planets stay well apart.
SOFTENING = 1e10         # ~1 billion metres, roughly the radius of the Sun
# SOFTENING = 0.0       # uncomment for solar system scenes
