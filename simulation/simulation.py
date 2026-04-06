import numpy as np
from simulation.body import Body
from config import G

class Simulation:

    def __init__(self, bodies: list[Body], integrator, compute_forces, dt: float):
        self.bodies = bodies
        self.integrator = integrator
        self.compute_forces = compute_forces
        self.dt = dt
        self.time = 0.0

    def step(self) -> None:
        self.integrator.step(self.bodies, self.dt, self.compute_forces)
        self.time += self.dt

    def get_energy(self) -> float:
        '''Total mechanical energy (KE + PE) - used for validation.'''
        ke = sum(0.5 * b.mass * np.dot(b.velocity, b.velocity) for b in self.bodies)
        pe = 0.0

        for i in range(len(self.bodies)):
            for j in range(i+1, len(self.bodies)):
                r = np.linalg.norm(self.bodies[i].position - self.bodies[j].position)
                pe -= G * self.bodies[i].mass * self.bodies[j].mass / r
        return ke + pe