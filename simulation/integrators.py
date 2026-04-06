# Different numerical integrators for stepping the position and velocity of bodies
import numpy as np
from abc import ABC, abstractmethod
from simulation.body import Body


class Integrator(ABC):
    @abstractmethod
    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        pass


class EulerIntegration(Integrator):
    '''Basic Euler method - 1 force evaluation per step.

    Simplest integrator: use the current slope and march forward.
    Error accumulates linearly over time and energy tends to drift up,
    so it's mostly useful for quick tests or short runs.
    '''

    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        compute_forces(bodies)

        for body in bodies:
            # a = F / m
            acceleration = body.force / body.mass
            # v = a * dt
            body.velocity += acceleration * dt
            # position uses end-of-step velocity, which is where the O(dt) error comes from
            body.position += body.velocity * dt


class RK4Integrator(Integrator):
    '''4th-order Runge-Kutta - 4 force evaluations per step.

    Samples the slope at 4 points across the interval and takes a weighted average.
    Much more accurate than Euler for the same dt, but costs 4x the force evaluations.
    Not symplectic so energy can still drift over very long runs, just much more slowly.
    '''

    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        # save starting state - we temporarily move bodies to probe points
        orig_pos = [np.copy(b.position) for b in bodies]
        orig_vel = [np.copy(b.velocity) for b in bodies]

        # k1: slopes at the start
        compute_forces(bodies)
        k1_acc = [np.copy(b.force / b.mass) for b in bodies]
        k1_r   = [np.copy(b.velocity)       for b in bodies]

        # k2: slopes at midpoint using k1
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + k1_r[i]   * (dt / 2)
            body.velocity = orig_vel[i] + k1_acc[i] * (dt / 2)
        compute_forces(bodies)
        k2_acc = [np.copy(b.force / b.mass) for b in bodies]
        k2_r   = [np.copy(b.velocity)       for b in bodies]
        for i, body in enumerate(bodies):
            body.position = np.copy(orig_pos[i])
            body.velocity = np.copy(orig_vel[i])

        # k3: slopes at midpoint using k2's better estimate
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + k2_r[i]   * (dt / 2)
            body.velocity = orig_vel[i] + k2_acc[i] * (dt / 2)
        compute_forces(bodies)
        k3_acc = [np.copy(b.force / b.mass) for b in bodies]
        k3_r   = [np.copy(b.velocity)       for b in bodies]
        for i, body in enumerate(bodies):
            body.position = np.copy(orig_pos[i])
            body.velocity = np.copy(orig_vel[i])

        # k4: slopes at the end using k3
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + k3_r[i]   * dt
            body.velocity = orig_vel[i] + k3_acc[i] * dt
        compute_forces(bodies)
        k4_acc = [np.copy(b.force / b.mass) for b in bodies]
        k4_r   = [np.copy(b.velocity)       for b in bodies]
        for i, body in enumerate(bodies):
            body.position = np.copy(orig_pos[i])
            body.velocity = np.copy(orig_vel[i])

        # weighted blend - midpoint estimates get 2x weight since they capture more curvature
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + (dt / 6) * (k1_r[i]   + 2*k2_r[i]   + 2*k3_r[i]   + k4_r[i])
            body.velocity = orig_vel[i] + (dt / 6) * (k1_acc[i] + 2*k2_acc[i] + 2*k3_acc[i] + k4_acc[i])


class LeapfrogIntegrator(Integrator):
    '''Velocity Verlet (Leapfrog) - 2 force evaluations per step. Symplectic.

    Staggers position and velocity updates so they leapfrog over each other.
    Time-reversible and symplectic, meaning it conserves energy much better
    than Euler or RK4 over long runs. Best choice for orbital simulations.
    '''

    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        # acceleration at current position
        compute_forces(bodies)
        acc_i = [np.copy(b.force / b.mass) for b in bodies]

        # half-kick velocity forward
        for i, body in enumerate(bodies):
            body.velocity = body.velocity + acc_i[i] * (dt / 2)

        # full drift position using half-step velocity
        for body in bodies:
            body.position = body.position + body.velocity * dt

        # acceleration at new position
        compute_forces(bodies)
        acc_f = [np.copy(b.force / b.mass) for b in bodies]

        # second half-kick to finish the step
        for i, body in enumerate(bodies):
            body.velocity = body.velocity + acc_f[i] * (dt / 2)