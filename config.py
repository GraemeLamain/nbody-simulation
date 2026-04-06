# Global settings and physical constants

# --- Window ---
WIDTH  = 1200   # window width in pixels
HEIGHT = 800    # window height in pixels
FPS    = 60     # target frames per second

# --- Simulation ---

# Simulated time per integrator step in seconds. 86400 = 1 day.
# Smaller dt = more accurate orbits but more steps needed to cover the same time span.
TIMESTEP = 86400.0

# Barnes-Hut opening angle theta.
# Lower = more accurate but slower. Higher = faster but less accurate.
# 0.0 is exact (no approximations), 1.2 is the standard cosmological value.
THETA = 1.2

# --- Physical Constants ---
G  = 6.674e-11       # gravitational constant (SI)
AU = 1.496e11        # 1 astronomical unit in metres

# Softening length - prevents F blowing up when two bodies get very close.
# Basically treats bodies as fuzzy blobs instead of true point masses.
# Keep nonzero for galaxy scenes, can be 0 for solar system since planets stay well apart.
SOFTENING = 1e10
# SOFTENING = 0.0