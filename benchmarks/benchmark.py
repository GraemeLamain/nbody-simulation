# FPS and compute time: naive vs Barnes-Hut across particle counts
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from simulation.body import Body
from simulation.physics import compute_gravity_naive, compute_gravity_barnes_hut
from config import G

def make_bodies(n: int, seed: int = 42) -> list[Body]:
    """Generate n random bodies in a disk layout for benchmarking."""
    rng = np.random.default_rng(seed)
    BH_MASS = 1e33
    bodies = []

    # central mass so orbits are stable
    bodies.append(Body(
        name="center",
        mass=BH_MASS,
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
    ))

    angles = rng.uniform(0, 2 * np.pi, n - 1)
    radii  = rng.uniform(1e10, 5e11, n - 1)

    for i, (theta, r) in enumerate(zip(angles, radii)):
        pos = np.array([r * np.cos(theta), r * np.sin(theta)])
        v_mag = np.sqrt(G * BH_MASS / r)
        vel = np.array([-np.sin(theta), np.cos(theta)]) * v_mag
        bodies.append(Body(
            name=f"star_{i}",
            mass=1e20,
            position=pos,
            velocity=vel,
        ))

    return bodies


def time_method(fn, bodies: list[Body], runs: int = 3) -> float:
    """Return median wall-clock time in seconds over a number of runs."""
    times = []
    for _ in range(runs):
        # deep copy so forces don't accumulate across runs
        import copy
        b = copy.deepcopy(bodies)
        t0 = time.perf_counter()
        fn(b)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_benchmark():
    particle_counts = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 75000, 100000]

    # Barnes-Hut needs a warmup call so Numba JIT compiles before we start timing
    print("Warming up Numba JIT (this takes ~10 seconds on first run)...")
    warmup_bodies = make_bodies(50)
    compute_gravity_barnes_hut(warmup_bodies)
    compute_gravity_naive(warmup_bodies)
    print("Done. Starting benchmark...\n")

    results = {
        "Naive":      [],
        "Barnes-Hut": [],
    }

    for n in particle_counts:
        bodies = make_bodies(n)
        print(f"N = {n:>5}  ", end="", flush=True)

        t_naive = time_method(compute_gravity_naive, bodies)
        results["Naive"].append(t_naive)
        print(f"Naive: {t_naive*1000:7.2f}ms  ", end="", flush=True)

        t_bh = time_method(compute_gravity_barnes_hut, bodies)
        results["Barnes-Hut"].append(t_bh)
        print(f"Barnes-Hut: {t_bh*1000:7.2f}ms")

    # --- Plot ---
    colors = {
        "Naive":      "#F87171",
        "Barnes-Hut": "#34D399",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#080D1A")
    ax.set_facecolor("#0E1830")

    for name, times in results.items():
        ax.plot(particle_counts, [t * 1000 for t in times],
                label=name, color=colors[name], linewidth=2, marker="o", markersize=5)

    ax.set_xlabel("Number of Bodies (N)", color="#CBD5E1", fontsize=12)
    ax.set_ylabel("Time per Force Evaluation (ms)", color="#CBD5E1", fontsize=12)
    ax.set_title("Gravity Algorithm Performance Comparison", color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#111D35", edgecolor="#4FC3F7", labelcolor="white", fontsize=11)
    ax.tick_params(colors="#64748B")
    ax.spines["bottom"].set_color("#64748B")
    ax.spines["left"].set_color("#64748B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color="#4FC3F7")
    ax.set_yscale("log")

    fig.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150, facecolor=fig.get_facecolor())
    print("\nSaved: benchmark_results.png")
    plt.show()

    # also print a summary table
    print("\n--- Summary ---")
    print(f"{'N':>6}  {'Naive (ms)':>12}  {'Barnes-Hut (ms)':>16}  {'BH Speedup vs Naive':>20}")
    for i, n in enumerate(particle_counts):
        naive_t = results["Naive"][i] * 1000
        bh_t    = results["Barnes-Hut"][i] * 1000
        speedup = naive_t / bh_t if bh_t > 0 else 0
        print(f"{n:>6}  {naive_t:>12.2f}  {bh_t:>16.2f}  {speedup:>20.2f}x")


if __name__ == "__main__":
    run_benchmark()