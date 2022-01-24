import time
import math

from Car_Price_Prediction import Car_Price_Prediction
from Particle import Particle

class Swarm:

    INERTIAL_COEFFICIENT = 0.721 #: Inertia
    COGNITIVE_COEFFICIENT = 1.1193
    SOCIAL_COEFFICIENT = 1.1193

    def __init__(self, cpp: Car_Price_Prediction, swarm_size : int = 100):

        self.cpp = cpp
        self.param_bounds = self.cpp.bounds()

        self.gbest = self.cpp.generate_random_parameters(self.param_bounds)
        self.gbest_fit = self.cpp.evaluate(self.gbest)

        self.steps = []
        self.particles = []

        for i in range(swarm_size):
            self.particles.append(
                Particle(self.cpp, self.cpp.generate_random_parameters(self.cpp.bounds()), self.cpp.bounds()))

    def step(self):
        step: list[tuple[float, float]] = []

        for j in range(len(self.particles)):
            new_pos = self.particles[j].update_particle(self.gbest)
            np_fit = self.cpp.evaluate(new_pos)
            step.append((new_pos, np_fit))

        return step

    def run(self, run_time=10, steps=math.inf, target=0):
        """[summary]

        Args:
            run_time (int, optional): [description]. Defaults to 10.
            steps ([type], optional): [description]. Defaults to math.inf.
            target (int, optional): [description]. Defaults to 0.
        """
        end_time = time.time() + run_time
        step_count = 0
        log = []

        while time.time() < end_time and step_count < steps and self.gbest_fit > target:
            step = self.step()
            log.append(step)
            best_pos, best_fit = min(step, key=lambda p: p[1])

            if best_fit < self.cpp.evaluate(self.gbest):
                self.gbest = best_pos
                self.gbest_fit = best_fit

        return (self.gbest, self.gbest_fit, log)
