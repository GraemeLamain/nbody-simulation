### N-Body Simulation in Pygame ###

This project will be a real time 2D gravitational N-Body simulation of our Solar System. Using PyGame and NumPy, I will model the orbital physics of our solar system and validate the results against NASA JPL Horizons data to determine how accurate my simulation is compared to real world analysis. The simulation will be a natural expansion from Lab 3 where we showed the orbits of the Earth and the Moon, this time with more celestial bodies to increase complexity. In this project, I aim to compare the differences between Euler integration, RK4 integration and LeapFrog integration which is something we have not touched on in the course. Each method will have the same initial conditions and will be compared using their computational cost and their accuracy of the simulation. I would also like to implement and compare two approaches to gravitational force calculations, those being a naive pairwise algorithm and the Barnes-Hut algorithm which is known to speed up N-body computations. By implementing both of these methods I can compare them directly against each other in different scenarios to showcase when to use each method.

If time permits and all of that is completed ahead of time, there is an extension that can be made to this which would be to go from simulating the solar system, into simulating the collision of two galaxies. This expansion should utilize the same physics, however the challenge comes in simulating up to thousands of bodies and demonstrating the performance of the different algorithms at different levels of scale.

Command Line Flags for Starting:
- python main.py                    (Runs with approximate planet positions as normal)
- python main.py --jpl              (Runs with NASA JPL Horizons initial conditions)
- python main.py --compare          (Runs integrator comparison and exits)
- python main.py --compare --jpl    (Runs integrator comparison and comparison between simulated positions and JPL positions)
    - You should use this one almost always, the other one just works fine but it starts you far away from the actual positions so some numbers arent accurate
