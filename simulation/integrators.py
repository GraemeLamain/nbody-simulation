# Three numerical integrators for stepping positions and velocities forward in time.
#
# All three solve the same equations of motion from Newton's second law:
#
#   a = F / m           (force tells us the acceleration)
#   dv/dt = a           (acceleration changes velocity)
#   dr/dt = v           (velocity changes position)
#
# The difference is how carefully each one approximates those continuous equations
# over a single discrete timestep dt - and that determines how many force evaluations
# each step costs and how much the trajectory drifts over long runs.
import numpy as np
from abc import ABC, abstractmethod
from simulation.body import Body


class Integrator(ABC):
    '''Abstract base: any integrator must implement step().'''

    @abstractmethod
    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        '''Move all bodies forward by one timestep dt.'''
        pass


class EulerIntegration(Integrator):
    '''First-order Euler method. 1 force evaluation per step.

    The simplest numerical integrator there is: look at the slope right now
    (acceleration and velocity) and just march straight in that direction.

    Error is O(dt) per step, so it accumulates linearly with time.
    Euler is neither time-reversible nor symplectic, which means energy tends
    to slowly drift upward over long runs. Fine for quick tests or short simulations.
    '''

    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        # Evaluate gravity at the current positions.
        compute_forces(bodies)

        for body in bodies:
            # a = F / m  (Newton's 2nd law)
            acceleration = body.force / body.mass

            # Kick velocity by one full step using the acceleration we just computed.
            body.velocity += acceleration * dt

            # Drift position using the freshly-updated velocity.
            # This is the "forward Euler" quirk: position uses the end-of-step velocity
            # rather than a midpoint estimate, which is where the O(dt) error comes from.
            body.position += body.velocity * dt


class RK4Integrator(Integrator):
    '''4th-order Runge-Kutta. 4 force evaluations per step.

    RK4 samples the slope at four points across the interval and takes a
    weighted average. The weights (1, 2, 2, 1) are chosen so that error terms
    up to order dt^4 cancel out, leaving only O(dt^5) local error per step
    and O(dt^4) global error - much more accurate than Euler for the same dt.

    The cost is 4× as many force evaluations per step compared to Euler.
    RK4 is not symplectic, so energy can still drift over very long runs,
    but the drift is far slower than Euler due to the higher accuracy.
    '''

    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        # Save where everything is at the start of the step.
        # Each of the four slope evaluations temporarily moves the bodies
        # to a probe point, so we need to be able to restore them.
        orig_pos = [np.copy(b.position) for b in bodies]
        orig_vel = [np.copy(b.velocity) for b in bodies]

        # --- k1: slopes at the START of the step ---
        # Evaluate forces at r_i to get a_i = F(r_i)/m.
        # k1_r   = v_i         (dr/dt = v)
        # k1_acc = F(r_i) / m  (dv/dt = a)
        compute_forces(bodies)
        k1_acc = [np.copy(b.force / b.mass) for b in bodies]
        k1_r   = [np.copy(b.velocity)       for b in bodies]

        # --- k2: slopes at the MIDPOINT, using k1 to get there ---
        # Step each body halfway through the interval using k1's estimates,
        # then re-evaluate forces at that probe position.
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + k1_r[i]   * (dt / 2)
            body.velocity = orig_vel[i] + k1_acc[i] * (dt / 2)
        compute_forces(bodies)
        k2_acc = [np.copy(b.force / b.mass) for b in bodies]
        k2_r   = [np.copy(b.velocity)       for b in bodies]
        # Back to the start before the next probe.
        for i, body in enumerate(bodies):
            body.position = np.copy(orig_pos[i])
            body.velocity = np.copy(orig_vel[i])

        # --- k3: slopes at the MIDPOINT again, this time using k2's better estimate ---
        # k2 gave a better midpoint than k1 did, so we use k2 to get there
        # and re-evaluate - a corrected midpoint slope.
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + k2_r[i]   * (dt / 2)
            body.velocity = orig_vel[i] + k2_acc[i] * (dt / 2)
        compute_forces(bodies)
        k3_acc = [np.copy(b.force / b.mass) for b in bodies]
        k3_r   = [np.copy(b.velocity)       for b in bodies]
        for i, body in enumerate(bodies):
            body.position = np.copy(orig_pos[i])
            body.velocity = np.copy(orig_vel[i])

        # --- k4: slopes at the END of the step, using k3 to get there ---
        # Project a full step ahead using k3's slopes and evaluate forces there.
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + k3_r[i]   * dt
            body.velocity = orig_vel[i] + k3_acc[i] * dt
        compute_forces(bodies)
        k4_acc = [np.copy(b.force / b.mass) for b in bodies]
        k4_r   = [np.copy(b.velocity)       for b in bodies]
        for i, body in enumerate(bodies):
            body.position = np.copy(orig_pos[i])
            body.velocity = np.copy(orig_vel[i])

        # --- Final update: weighted blend of the four slope estimates ---
        # The midpoint estimates (k2, k3) carry twice the weight of the endpoints
        # because they capture more of the curvature across the interval.
        # This 1/6 : 2/6 : 2/6 : 1/6 weighting is what gives RK4 its 4th-order accuracy.
        #
        # r_new = r_i + (dt/6) * (k1_r   + 2*k2_r   + 2*k3_r   + k4_r)
        # v_new = v_i + (dt/6) * (k1_acc + 2*k2_acc + 2*k3_acc + k4_acc)
        for i, body in enumerate(bodies):
            body.position = orig_pos[i] + (dt / 6) * (k1_r[i]   + 2*k2_r[i]   + 2*k3_r[i]   + k4_r[i])
            body.velocity = orig_vel[i] + (dt / 6) * (k1_acc[i] + 2*k2_acc[i] + 2*k3_acc[i] + k4_acc[i])


class LeapfrogIntegrator(Integrator):
    '''Velocity Verlet (Leapfrog). 2 force evaluations per step. Symplectic.

    The trick is to stagger position and velocity updates - velocity takes two
    half-steps that "leapfrog" over the full position step in the middle.
    That staggering makes the method time-reversible and symplectic, meaning
    it preserves the geometric structure of the equations of motion.

    Error is O(dt^2) per step - better than Euler, cheaper than RK4.
    The recommended choice for long orbital simulations.
    '''

    def step(self, bodies: list[Body], dt: float, compute_forces) -> None:
        # --- First force evaluation: acceleration at the current position ---
        # a_i = F(r_i) / m
        compute_forces(bodies)
        acc_i = [np.copy(b.force / b.mass) for b in bodies]

        # --- First half-kick: nudge velocity forward by half a step ---
        # v_{i+1/2} = v_i + a_i * (dt/2)
        # This puts the velocity "halfway between" r_i and the upcoming r_{i+1}.
        # Using that midpoint velocity to update position is what buys the
        # 2nd-order accuracy over Euler.
        for i, body in enumerate(bodies):
            body.velocity = body.velocity + acc_i[i] * (dt / 2)

        # --- Full drift: advance position by a whole step ---
        # r_{i+1} = r_i + v_{i+1/2} * dt
        # We're using the half-step (midpoint) velocity here, not the start-of-step one.
        for body in bodies:
            body.position = body.position + body.velocity * dt

        # --- Second force evaluation: acceleration at the new position ---
        # a_{i+1} = F(r_{i+1}) / m
        # We need this to finish bringing velocity up to the end of the step.
        compute_forces(bodies)
        acc_f = [np.copy(b.force / b.mass) for b in bodies]

        # --- Second half-kick: bring velocity to the end of the step ---
        # v_{i+1} = v_{i+1/2} + a_{i+1} * (dt/2)
        # The two half-kicks sit symmetrically around the position update.
        # That symmetry is what makes the method time-reversible and keeps
        # the symplectic structure intact over long runs.
        for i, body in enumerate(bodies):
            body.velocity = body.velocity + acc_f[i] * (dt / 2)
