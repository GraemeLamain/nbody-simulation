import copy
import numpy as np
import matplotlib.pyplot as plt
from simulation.simulation import Simulation
from simulation.physics import compute_gravity_naive
from validation.nasa_fetch import fetch_jpl_timeseries, BODY_IDS

SECONDS_PER_YEAR = 365.25 * 86400


def plot_energy_drift(time_series: list, energy_dict: dict, filename: str = "energy_drift.png") -> None:
    '''Plot total mechanical energy over time for each integrator.

    A good integrator keeps this flat - any drift means the integrator is losing
    or adding energy that shouldn't be there.
    '''
    years = [t / SECONDS_PER_YEAR for t in time_series]

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, energies in energy_dict.items():
        ax.plot(years, energies, label=name, linewidth=1.2)

    ax.set_xlabel("Simulated Time (years)")
    ax.set_ylabel("Total Energy (J)")
    ax.set_title("Energy Drift by Integrator")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.show()


def plot_orbital_deviation(time_series: list, sim_positions_dict: dict, jpl_positions: list, planet_name: str, filename: str = None) -> None:
    '''Plot distance between simulated and JPL reference positions over time.'''
    if filename is None:
        filename = f"{planet_name.lower()}_combined_deviation.png"

    years = [t / SECONDS_PER_YEAR for t in time_series]

    fig, ax = plt.subplots(figsize=(10, 5))
    
    for name, sim_positions in sim_positions_dict.items():
        deviations_km = [
            np.linalg.norm(s - j) / 1000.0
            for s, j in zip(sim_positions, jpl_positions)
        ]
        ax.plot(years, deviations_km, label=name, linewidth=1.2)

    ax.set_xlabel("Simulated Time (years)")
    ax.set_ylabel("Position Deviation (km)")
    ax.set_title(f"Orbital Deviation vs JPL - {planet_name}")
    # log scale is important here - Euler's error is so large it squishes RK4/Leapfrog on a linear plot
    ax.set_yscale("log") 
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.show()


def calculate_orbital_period(time_series: list, position_series: list) -> float:
    '''Detect orbital period by finding when the body crosses the x-axis (y: negative -> positive).'''
    crossings = []
    for i in range(1, len(position_series)):
        y_prev = position_series[i-1][1]
        y_curr = position_series[i][1]
        
        if y_prev < 0 and y_curr >= 0:
            # linear interpolation for sub-step precision
            fraction = abs(y_prev) / (abs(y_prev) + abs(y_curr))
            exact_time = time_series[i-1] + fraction * (time_series[i] - time_series[i-1])
            crossings.append(exact_time)
            
    if len(crossings) >= 2:
        return (crossings[1] - crossings[0]) / 86400.0
    return 0.0


def run_integrator_comparison(bodies_factory, integrators: dict, target_planet: str = "Earth", years: int = 2, dt: float = 86400.0) -> None:
    REAL_PERIODS = {
        "Mercury": 87.97,
        "Venus": 224.70,
        "Earth": 365.25,
        "Mars": 686.98,
        "Jupiter": 4332.59,
        "Saturn": 10759.22,
        "Uranus": 30688.50,
        "Neptune": 60182.00,
    }

    if target_planet not in REAL_PERIODS:
        raise ValueError(f"Unknown target planet: {target_planet}")

    total_steps = int(years * 365.25)
    start_date = "2026-03-03"
    
    print(f"\n--- Fetching JPL Timeseries for {target_planet} ---")
    jpl_positions = fetch_jpl_timeseries(BODY_IDS[target_planet], start_date, total_steps + 1)
    
    real_period = REAL_PERIODS[target_planet]
    print(f"JPL Target Period: ~{real_period} days\n")

    results = {}

    for name, integrator in integrators.items():
        print(f"Running {name} integrator...")

        sim = Simulation(
            bodies=bodies_factory(),
            integrator=copy.deepcopy(integrator),
            compute_forces=compute_gravity_naive,
            dt=dt,
        )

        time_series = []
        energy_series = []
        sim_positions = []
        
        target_idx = next(i for i, b in enumerate(sim.bodies) if b.name == target_planet)

        for step in range(total_steps):
            time_series.append(sim.time)
            energy_series.append(sim.get_energy())
            
            # track relative to the Sun so we're measuring orbital error, not system drift
            sun_pos = sim.bodies[0].position
            planet_pos = sim.bodies[target_idx].position
            sim_positions.append(planet_pos - sun_pos)

            sim.step()

        results[name] = (time_series, energy_series, sim_positions)
        
        sim_period = calculate_orbital_period(time_series, sim_positions)
        period_error = abs(sim_period - real_period) if sim_period else float('inf')
        
        deviations = [np.linalg.norm(s - j) / 1000.0 for s, j in zip(sim_positions, jpl_positions[:-1])]
        max_dev = max(deviations)
        
        print(f"  {name} Final Results:")
        if sim_period:
            print(f"    - Period: {sim_period:.2f} days (Error: {period_error:.2f} days)")
        else:
            print(f"    - Period: Orbit incomplete")
        print(f"    - Max Position Deviation from JPL: {max_dev:,.2f} km")
        print(f"    - Energy Drift: {energy_series[-1] - energy_series[0]:.3e} J\n")

    first_name = next(iter(results))
    shared_times = results[first_name][0]

    energy_dict = {name: data[1] for name, data in results.items()}
    positions_dict = {name: data[2] for name, data in results.items()}

    plot_energy_drift(shared_times, energy_dict)
    plot_orbital_deviation(shared_times, positions_dict, jpl_positions[:-1], target_planet)