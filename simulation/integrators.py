# Base Integrator class + Euler, Rk4, Leapfrog subclasses
import numpy as np
from abc import ABC, abstractmethod     # This is just to make sure everything implements the step function but its not needed
from simulation.body import Body

class Integrator(ABC):
    '''Abstract base class for all integrators'''

    @abstractmethod
    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        '''Advance the simulation by one timestep.'''
        pass

class EulerIntegration(Integrator):
    '''Basic Euler Integration. O(n) per step, least accurate'''

    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        compute_forces(bodies)

        for body in bodies:
            # a = F / m  (Newton's second law — acceleration is force divided by mass)
            acceleration = body.force / body.mass

            # v_f = v_i + a_i * dt  (new velocity = old velocity + acceleration scaled by timestep)
            body.velocity += acceleration * dt
            # r_f = r_i + v_f * dt  (new position = old position + velocity scaled by timestep)
            # Note: uses the already-updated velocity (forward Euler), which introduces O(h) error per step
            body.position += body.velocity * dt

class RK4Integrator(Integrator):
    '''4th-order Runge-Kutta. 4 force evaluations per step, O(h^4) global error.

    Instead of using a single slope estimate like Euler, RK4 samples the slope at 4 points
    across the interval and takes a weighted average. This cancels out error terms up to h^4,
    making it dramatically more accurate for the same timestep size.
    '''

    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        # Store initial position and velocity
        orig_pos = [np.copy(b.position) for b in bodies]
        orig_vel = [np.copy(b.velocity) for b in bodies]

        # k1: slopes at the START of the interval (r_i, v_i)
        # k1_acc = F(r_i) / m      — acceleration at the current position
        # k1_r   = v_i             — velocity at the current position (dr/dt = v)
        compute_forces(bodies)
        k1_acc = [np.copy(b.force / b.mass) for b in bodies]
        k1_r   = [np.copy(b.velocity)       for b in bodies]

        # k2: slopes at the MIDPOINT, estimated using k1 to get there
        # k2_acc = F(r_i + k1_r * dt/2) / m    — acceleration if we took a half-step using k1's velocity
        # k2_r   = v_i + k1_acc * dt/2         — velocity if we took a half-step using k1's acceleration
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + k1_r[i]   * (dt / 2)
            body.velocity = orig_vel[i] + k1_acc[i] * (dt / 2)
        compute_forces(bodies)
        k2_acc = [np.copy(b.force / b.mass) for b in bodies]
        k2_r   = [np.copy(b.velocity)       for b in bodies]
        for i, body in enumerate(bodies):
            body.position = np.copy(orig_pos[i])
            body.velocity = np.copy(orig_vel[i])

        # k3: slopes at the MIDPOINT again, but now estimated using k2 (a better midpoint estimate)
        # k3_acc = F(r_i + k2_r * dt/2) / m    — acceleration using k2's improved midpoint position
        # k3_r   = v_i + k2_acc * dt/2         — velocity using k2's improved midpoint acceleration
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + k2_r[i]   * (dt / 2)
            body.velocity = orig_vel[i] + k2_acc[i] * (dt / 2)
        compute_forces(bodies)
        k3_acc = [np.copy(b.force / b.mass) for b in bodies]
        k3_r   = [np.copy(b.velocity)       for b in bodies]
        for i, body in enumerate(bodies):
            body.position = np.copy(orig_pos[i])
            body.velocity = np.copy(orig_vel[i])

        # k4: slopes at the END of the interval, estimated using k3 to get there
        # k4_acc = F(r_i + k3_r * dt) / m    — acceleration at the full-step position
        # k4_r   = v_i + k3_acc * dt         — velocity at the full-step point
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + k3_r[i]   * dt
            body.velocity = orig_vel[i] + k3_acc[i] * dt
        compute_forces(bodies)
        k4_acc = [np.copy(b.force / b.mass) for b in bodies]
        k4_r   = [np.copy(b.velocity)       for b in bodies]
        for i, body in enumerate(bodies):
            body.position = np.copy(orig_pos[i])
            body.velocity = np.copy(orig_vel[i])

        # Final weighted average of all four slope estimates
        # The midpoint slopes (k2, k3) are weighted double because they carry more information
        # about the curvature of the trajectory across the interval than the endpoints (k1, k4)
        # r_f = r_i + dt/6 * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        # v_f = v_i + dt/6 * (k1_acc + 2*k2_acc + 2*k3_acc + k4_acc)
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + (dt / 6) * (k1_r[i]   + 2*k2_r[i]   + 2*k3_r[i]   + k4_r[i])
            body.velocity = orig_vel[i] + (dt / 6) * (k1_acc[i] + 2*k2_acc[i] + 2*k3_acc[i] + k4_acc[i])

class LeapfrogIntegrator(Integrator):
    '''Velocity Verlet (Leapfrog). Symplectic — energy stays bounded over long timescales.
    2 force evaluations per step.

    The key insight: instead of updating position and velocity at the same time (like Euler),
    they are staggered — velocity is half-updated, then position is fully updated, then
    velocity finishes. This staggering makes the integrator time-reversible and symplectic,
    meaning it conserves a "shadow" Hamiltonian, so energy error oscillates rather than drifts.
    '''

    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        # First force evaluation at the current position r_i
        # a_i = F(r_i) / m  — acceleration based on where bodies are right now
        compute_forces(bodies)
        acc_i = [np.copy(b.force / b.mass) for b in bodies]

        # Half-kick: advance velocity by half a timestep using the current acceleration
        # v_{i+1/2} = v_i + a_i * dt/2
        # This gives us a velocity that is "centred" between the old and new positions
        for i, body in enumerate(bodies):
            body.velocity = body.velocity + acc_i[i] * (dt / 2)

        # Full drift: advance position by a full timestep using the half-step velocity
        # r_f = r_i + v_{i+1/2} * dt
        # Using the midpoint velocity here is what gives leapfrog its accuracy advantage over Euler
        for body in bodies:
            body.position = body.position + body.velocity * dt

        # Second force evaluation at the NEW position r_f
        # a_f = F(r_f) / m  — acceleration now that bodies have moved
        compute_forces(bodies)
        acc_f = [np.copy(b.force / b.mass) for b in bodies]

        # Second half-kick: finish advancing velocity using the new acceleration
        # v_f = v_{i+1/2} + a_f * dt/2
        # The two half-kicks together give a full velocity update, but split around the position update
        for i, body in enumerate(bodies):
            body.velocity = body.velocity + acc_f[i] * (dt / 2)
