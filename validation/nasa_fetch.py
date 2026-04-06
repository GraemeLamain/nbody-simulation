# HTTP requests for JPL Horizons API, unit conversion (AU to SI)
import re
import requests
import numpy as np
from datetime import datetime, timedelta
from simulation.body import Body
from scenes.solar_system import create_solar_system

AU_TO_M        = 1.496e11       # metres per AU
AU_DAY_TO_MS   = 1_731_481.0   # m/s per AU/day  (= 1.496e11 m / 86400 s)

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Barycenter just means the center of mass of that planet including its moons
# This is done so that we dont need to have a separate body for each moon but we still get the accuracy of having them in the simulation
BODY_IDS = {
    "Sun":     "10", # Sun's center
    "Mercury": "1",  # Mercury System Barycenter 
    "Venus":   "2",  # Venus System Barycenter
    "Earth":   "3",  # Earth-Moon Barycenter
    "Mars":    "4",  # Mars System Barycenter
    "Jupiter": "5",  # Jupiter System Barycenter
    "Saturn":  "6",  # Saturn System Barycenter
    "Uranus":  "7",  # Uranus System Barycenter
    "Neptune": "8",  # Neptune System Barycenter
}

# Masses, colors, and radii mirrored from scenes/solar_system.py
BODY_META = {
    "Sun":     {"mass": 1.989e30, "color": (255, 255,   0), "radius":  6.0},
    "Mercury": {"mass": 3.285e23, "color": (169, 169, 169), "radius":  2.0},
    "Venus":   {"mass": 4.867e24, "color": (255, 198, 100), "radius":  3.0},
    "Earth":   {"mass": 5.972e24, "color": (  0, 100, 255), "radius":  3.0},
    "Mars":    {"mass": 6.390e23, "color": (188,  74,  60), "radius":  2.0},
    "Jupiter": {"mass": 1.898e27, "color": (201, 144,  57), "radius":  6.0},
    "Saturn":  {"mass": 5.683e26, "color": (210, 180, 100), "radius":  5.0},
    "Uranus":  {"mass": 8.681e25, "color": (100, 220, 220), "radius":  4.0},
    "Neptune": {"mass": 1.024e26, "color": ( 50, 100, 255), "radius":  4.0},
}


def fetch_body_vectors(body_id: str, date: str) -> dict:
    """Query the NASA JPL Horizons API for position and velocity vectors.

    Args:
        body_id: JPL body ID string (e.g. "399" for Earth).
        date:    Start date in "YYYY-MM-DD" format.

    Returns:
        dict with keys:
            "position"  - np.ndarray [x, y] in metres
            "velocity"  - np.ndarray [vx, vy] in m/s
    """
    start_dt  = datetime.strptime(date, "%Y-%m-%d")
    stop_date = (start_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        "format":     "json",
        "COMMAND":    body_id,
        "OBJ_DATA":   "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER":     "@10",
        "REF_PLANE":  "ECLIPTIC",
        "START_TIME": date,
        "STOP_TIME":  stop_date,
        "STEP_SIZE":  "1d",
        "VEC_TABLE":  "2",
        "OUT_UNITS":  "AU-D",
    }

    response = requests.get(HORIZONS_URL, params=params, timeout=30)
    response.raise_for_status()

    result_text = response.json()["result"]

    # Extract data block between $$SOE and $$EOE markers
    soe = result_text.find("$$SOE")
    eoe = result_text.find("$$EOE")
    if soe == -1 or eoe == -1:
        raise ValueError(f"Could not find $$SOE/$$EOE markers in Horizons response for body {body_id}")

    data_block = result_text[soe + len("$$SOE"):eoe]

    # Parse X, Y, Z and VX, VY, VZ using regex
    # Horizons vector format:
    #   X = <val> Y = <val> Z = <val>
    #   VX= <val> VY= <val> VZ= <val>
    def extract(pattern):
        match = re.search(pattern, data_block)
        if not match:
            raise ValueError(f"Pattern '{pattern}' not found in Horizons data block")
        return float(match.group(1))

    x  = extract(r"X\s*=\s*([-\d.E+]+)")
    y  = extract(r"Y\s*=\s*([-\d.E+]+)")
    vx = extract(r"VX=\s*([-\d.E+]+)")
    vy = extract(r"VY=\s*([-\d.E+]+)")

    return {
        "position": np.array([x * AU_TO_M,      y * AU_TO_M]),
        "velocity": np.array([vx * AU_DAY_TO_MS, vy * AU_DAY_TO_MS]),
    }

def fetch_jpl_timeseries(body_id: str, start_date: str, days: int) -> list[np.ndarray]:
    """Fetch daily positions for a specific body over a set number of days."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    stop_date = (start_dt + timedelta(days=days)).strftime("%Y-%m-%d")

    params = {
        "format":     "json",
        "COMMAND":    body_id,
        "OBJ_DATA":   "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER":     "@10",
        "REF_PLANE":  "ECLIPTIC",
        "START_TIME": start_date,
        "STOP_TIME":  stop_date,
        "STEP_SIZE":  "1d",
        "VEC_TABLE":  "2",
        "OUT_UNITS":  "AU-D"
    }

    response = requests.get(HORIZONS_URL, params=params, timeout=30)
    response.raise_for_status()
    result_text = response.json()["result"]

    soe = result_text.find("$$SOE")
    eoe = result_text.find("$$EOE")
    data_block = result_text[soe + len("$$SOE"):eoe]

    # Find all occurrences of X and Y in the timeseries block
    matches = re.findall(r"X\s*=\s*([-\d.E+]+)\s*Y\s*=\s*([-\d.E+]+)", data_block)
    
    positions = []
    for x, y in matches:
        positions.append(np.array([float(x) * AU_TO_M, float(y) * AU_TO_M]))
        
    return positions


def create_solar_system_from_jpl(date: str) -> list[Body]:
    """Build a solar system body list using real JPL Horizons initial conditions.

    Positions and velocities are fetched from the Horizons API for the given date.
    The Sun is placed at the origin and all other bodies are expressed relative to it.
    Masses, colors, and radii are taken from the approximate solar system scene.

    Falls back to create_solar_system() for any body that fails to fetch.

    Args:
        date: Date string in "YYYY-MM-DD" format.

    Returns:
        list[Body] ordered Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.
    """
    body_names = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                  "Jupiter", "Saturn", "Uranus", "Neptune"]

    # Fetch vectors for all bodies
    raw: dict[str, dict] = {}
    fallback_needed: list[str] = []

    for name in body_names:
        try:
            print(f"  Fetching {name} from JPL Horizons...")
            raw[name] = fetch_body_vectors(BODY_IDS[name], date)
        except Exception as exc:
            print(f"  WARNING: Failed to fetch {name}: {exc}. Will use approximate fallback.")
            fallback_needed.append(name)

    # If Sun fetch failed we cannot centre anything - fall back entirely
    if "Sun" in fallback_needed:
        print("  Sun fetch failed - falling back to full approximate solar system.")
        return create_solar_system()

    sun_pos = raw["Sun"]["position"]
    sun_vel = raw["Sun"]["velocity"]

    bodies: list[Body] = []
    approx_bodies = {b.name: b for b in create_solar_system()}

    for name in body_names:
        meta = BODY_META[name]

        if name in fallback_needed:
            # Use approximate body from scenes/solar_system.py
            approx = approx_bodies[name]
            bodies.append(Body(
                name=name,
                mass=meta["mass"],
                position=approx.position.copy(),
                velocity=approx.velocity.copy(),
                color=meta["color"],
                radius=meta["radius"],
            ))
        else:
            # Shift to Sun-centred reference frame
            position = raw[name]["position"] - sun_pos
            velocity = raw[name]["velocity"] - sun_vel
            bodies.append(Body(
                name=name,
                mass=meta["mass"],
                position=position,
                velocity=velocity,
                color=meta["color"],
                radius=meta["radius"],
            ))

    return bodies
